import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.schedulers import PopulationBasedTraining
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Dict, List, Tuple


# Настраиваемый трансформер для RL
class RLTransformer(nn.Module):
    def __init__(self, input_dim=50, output_dim=5, hidden_dim=128, num_layers=2, num_heads=4):
        super(RLTransformer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Входной слой
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Слои трансформера
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, 
                                                  dim_feedforward=hidden_dim*4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Выходной слой
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Расширяем размерность для позиционной последовательности (batch, seq_len=1, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # Проекция входа в скрытое пространство
        x = self.input_layer(x)
        
        # Применяем трансформер
        x = self.transformer_encoder(x)
        
        # Берем выход последнего токена
        x = x[:, -1, :]
        
        # Проекция в выходное пространство
        output = self.output_layer(x)
        
        return output


# Кастомная среда для обучения трансформера
class TransformerEnv(gym.Env):
    def __init__(self, env_config):
        super(TransformerEnv, self).__init__()
        
        # Получаем параметры из конфигурации
        self.input_dim = env_config.get("input_dim", 50)
        self.output_dim = env_config.get("output_dim", 5)
        self.max_steps = env_config.get("max_steps", 100)
        self.target_reward = env_config.get("target_reward", 10.0)
        
        # Определяем пространства наблюдений и действий
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.input_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.output_dim,), dtype=np.float32)
        
        # Целевой вектор для текущего эпизода
        self.target_vector = None
        self.current_step = 0
        self.total_reward = 0.0
        
        # Сгенерируем новый входной вектор и целевой вектор
        self.reset()
    
    def reset(self):
        # Генерация нового случайного входного вектора
        observation = np.random.normal(0, 1, size=(self.input_dim,)).astype(np.float32)
        
        # Генерация целевого вектора, который агент должен предсказать
        self.target_vector = np.random.uniform(-1.0, 1.0, size=(self.output_dim,)).astype(np.float32)
        
        # Сброс счетчика шагов и суммарной награды
        self.current_step = 0
        self.total_reward = 0.0
        
        return observation
    
    def step(self, action):
        # Вычисляем награду на основе близости предсказания к целевому вектору
        # Используем отрицательное евклидово расстояние как награду
        distance = np.linalg.norm(action - self.target_vector)
        reward = -distance
        
        # Другой вариант награды - косинусное сходство
        # cosine_sim = np.dot(action, self.target_vector) / (np.linalg.norm(action) * np.linalg.norm(self.target_vector))
        # reward = cosine_sim
        
        # Обновляем суммарную награду
        self.total_reward += reward
        
        # Увеличиваем счетчик шагов
        self.current_step += 1
        
        # Проверяем условия завершения
        done = False
        if self.total_reward >= self.target_reward or self.current_step >= self.max_steps:
            done = True
        
        # Генерируем новое наблюдение
        observation = np.random.normal(0, 1, size=(self.input_dim,)).astype(np.float32)
        
        # Дополнительная информация для отладки
        info = {
            "distance": distance,
            "current_step": self.current_step,
            "total_reward": self.total_reward,
            "target_vector": self.target_vector
        }
        
        return observation, reward, done, info


# Модель для RLlib, которая использует наш трансформер
class TransformerModel(nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        super(TransformerModel, self).__init__()
        
        # Получаем размерности входа и выхода из пространств
        input_dim = obs_space.shape[0]
        output_dim = action_space.shape[0]
        
        # Получаем гиперпараметры из конфигурации модели
        hidden_dim = model_config.get("custom_model_config", {}).get("hidden_dim", 128)
        num_layers = model_config.get("custom_model_config", {}).get("num_layers", 2)
        num_heads = model_config.get("custom_model_config", {}).get("num_heads", 4)
        
        # Создаем трансформер
        self.transformer = RLTransformer(
            input_dim=input_dim, 
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads
        )
        
        # Если используется для дискретных действий, добавим еще слой для логитов
        self.is_continuous = isinstance(action_space, spaces.Box)
        if not self.is_continuous:
            self.action_output = nn.Linear(output_dim, num_outputs)
        
        # Для оценки значения (value function)
        self.value_output = nn.Linear(hidden_dim, 1)
    
    def forward(self, input_dict, state, seq_lens):
        # Получаем входные данные
        obs = input_dict["obs"].float()
        
        # Прогоняем через трансформер
        transformer_output = self.transformer(obs)
        
        # Для непрерывных действий выход трансформера - это действия
        if self.is_continuous:
            # Ограничиваем выход до диапазона [-1, 1]
            action_out = torch.tanh(transformer_output)
        else:
            # Для дискретных действий получаем логиты
            action_out = self.action_output(transformer_output)
        
        # Для оценки значения берем скрытое состояние и проецируем в скаляр
        value_out = self.value_output(transformer_output[:, 0])
        
        return action_out, state, {"value_out": value_out}


# Функция для настройки обучения
def setup_training(input_dim=50, output_dim=5):
    ray.init(ignore_reinit_error=True)
    
    # Конфигурация среды
    env_config = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "max_steps": 100,
        "target_reward": 10.0
    }
    
    # Регистрируем кастомную среду
    tune.register_env("transformer_env", 
                     lambda config: TransformerEnv(config))
    
    # Настраиваем PBT-планировщик
    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="episode_reward_mean",
        mode="max",
        perturbation_interval=5,
        hyperparam_mutations={
            "lr": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            "gamma": [0.9, 0.95, 0.99],
            "train_batch_size": [1024, 2048, 4096],
            "sgd_minibatch_size": [64, 128, 256],
            "num_sgd_iter": [5, 10, 20],
            "model": {
                "custom_model_config": {
                    "hidden_dim": [64, 128, 256],
                    "num_layers": [1, 2, 3],
                    "num_heads": [2, 4, 8]
                }
            }
        }
    )
    
    # Функция для создания случайной политики
    def random_policy():
        return DQNConfig().environment(
            env="transformer_env", 
            env_config=env_config
        ).env_runners(
            explore=True,
            exploration_config={
                "type": "Random"  # Всегда выбираем случайные действия
            },
            num_env_runners=1
        )
    
    # Конфигурация алгоритма PPO с трансформерной моделью
    ppo_config = (
        PPOConfig()
        .environment(env="transformer_env", env_config=env_config)
        .framework("torch")
        .training(
            gamma=0.99,
            lr=3e-4,
            kl_coeff=0.2,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
            train_batch_size=4096,
            model={
                "custom_model": TransformerModel,
                "custom_model_config": {
                    "hidden_dim": 128,
                    "num_layers": 2,
                    "num_heads": 4
                }
            }
        )
        .api_stack( enable_rl_module_and_learner=False)
        .env_runners(num_env_runners=2)
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "1")))
        # Добавляем случайную политику
        .multi_agent(
            policies={
                "transformer_policy": PolicySpec(),
                "random_policy": PolicySpec(config=random_policy()),
            },
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: 
                "random_policy" if np.random.random() < 0.2 else "transformer_policy"
        )
    )
    
    # Запускаем оптимизацию с Ray Tune
    tuner = tune.Tuner(
        "PPO",
        run_config=tune.RunConfig(
            name="transformer_rl",
            stop={"episode_reward_mean": env_config["target_reward"]},
            checkpoint_config = tune.CheckpointConfig(
                checkpoint_frequency = 5,
                checkpoint_at_end = True
            ),
            storage_path=r"F:\work\code\for_git\transform_rl\ray_results",
            verbose=1
        ),
        tune_config=tune.TuneConfig(
            scheduler=pbt,
            num_samples=10,  # Число экспериментов для параллельной настройки
            # metric="episode_reward_mean",
            # mode="max"
        ),
        param_space=ppo_config.to_dict()
    )
    
    return tuner

# Функция для запуска обучения
def train_rl_transformer(input_dim=50, output_dim=5):
    tuner = setup_training(input_dim, output_dim)
    results = tuner.fit()
    
    # Получение лучшего результата
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
    print(f"Лучший результат: {best_result.metrics}")
    
    # Загрузка лучшей модели
    best_checkpoint = best_result.checkpoint
    print(f"Лучший чекпоинт: {best_checkpoint}")
    
    # Закрываем Ray
    ray.shutdown()
    
    return best_checkpoint

# Пример использования
if __name__ == "__main__":
    # Обучаем с настраиваемыми размерностями входа и выхода
    input_dim = 50  # Можно изменить
    output_dim = 5  # Можно изменить
    
    best_checkpoint = train_rl_transformer(input_dim, output_dim)
    print(f"Обучение завершено. Лучший результат в {best_checkpoint}")