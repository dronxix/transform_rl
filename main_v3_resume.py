import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.env.env_context import EnvContext
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune import CLIReporter
import gymnasium as gym
from gymnasium import spaces
import pickle
import os
from typing import Dict, Any, List, Tuple, Optional
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.checkpoints import get_checkpoint_info

os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

class PositionalEncoding(nn.Module):
    """Позиционное кодирование для трансформера"""
    
    def __init__(self, d_model: int, max_seq_length: int = 1000):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)


class ImprovedTransformerModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, custom_model_config=None, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs)
        nn.Module.__init__(self)

        # Объединяем model_config с custom_model_config для обратной совместимости
        custom_config = model_config.get("custom_model_config", {})
        full_config = {**model_config, **custom_config}

        # === ОСНОВНЫЕ ПАРАМЕТРЫ ===
        self.d_model = full_config.get("d_model", 256)
        self.nhead = full_config.get("nhead", 8)
        self.num_layers = full_config.get("num_layers", 6)
        self.dim_feedforward = full_config.get("dim_feedforward", 1024)
        self.dropout = full_config.get("dropout", 0.1)
        self.max_seq_length = full_config.get("max_seq_length", 100)

        # === ДОПОЛНИТЕЛЬНЫЕ ПАРАМЕТРЫ ===
        self.use_layer_norm = full_config.get("use_layer_norm", True)
        self.pre_norm = full_config.get("pre_norm", True)
        self.activation = full_config.get("activation", "gelu")
        self.attention_dropout = full_config.get("attention_dropout", 0.1)
        self.pos_encoding_type = full_config.get("pos_encoding_type", "learned")
        self.aggregation_method = full_config.get("aggregation_method", "mean_pooling")
        self.num_cls_tokens = full_config.get("num_cls_tokens", 1)
        self.hidden_sizes = full_config.get("hidden_sizes", [256])
        self.output_activation = full_config.get("output_activation", "none")
        self.use_output_norm = full_config.get("use_output_norm", False)
        self.init_std = full_config.get("init_std", 0.01)

        self.input_dim = obs_space.shape[1]
        
        # Поддержка spaces.Box для action_space
        if isinstance(action_space, spaces.Box):
            self.action_space_type = "box"
            self.action_dim = action_space.shape[0]
            self.action_low = torch.FloatTensor(action_space.low)
            self.action_high = torch.FloatTensor(action_space.high)
        else:
            self.action_space_type = "discrete"
            self.output_ranges = action_space.nvec if hasattr(action_space, 'nvec') else [action_space.n] * num_outputs

        # Сохраняем конфигурацию для отладки
        self.model_config = full_config

        # === ВХОДНАЯ ПРОЕКЦИЯ ===
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.d_model),
            nn.LayerNorm(self.d_model) if self.use_layer_norm else nn.Identity(),
            self._get_activation(),
            nn.Dropout(self.dropout)
        )

        # === ПОЗИЦИОННОЕ КОДИРОВАНИЕ ===
        if self.pos_encoding_type == "learned":
            self.pos_embedding = nn.Embedding(self.max_seq_length, self.d_model)
        elif self.pos_encoding_type == "sinusoidal":
            self.pos_encoding = PositionalEncoding(self.d_model, self.max_seq_length)

        # === УПРОЩЕННЫЕ ТРАНСФОРМЕРНЫЕ СЛОИ ===
        encoder_layers = []
        for _ in range(self.num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                activation=self.activation,
                batch_first=True,
                norm_first=self.pre_norm
            )
            encoder_layers.append(layer)
        self.transformer_layers = nn.ModuleList(encoder_layers)

        # === ВЫХОДНЫЕ СЛОИ В ЗАВИСИМОСТИ ОТ ТИПА ACTION SPACE ===
        if self.action_space_type == "box":
            # Для непрерывных действий: среднее и логарифм стандартного отклонения
            self.action_mean_head = nn.Sequential(
                nn.Linear(self.d_model, self.hidden_sizes[0]),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_sizes[0], self.action_dim)
            )
            
            # Параметризуемое стандартное отклонение
            self.action_log_std_head = nn.Sequential(
                nn.Linear(self.d_model, self.hidden_sizes[0]),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_sizes[0], self.action_dim)
            )
        else:
            # Для дискретных действий
            total_outputs = sum(self.output_ranges)
            self.action_head = nn.Sequential(
                nn.Linear(self.d_model, self.hidden_sizes[0]),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_sizes[0], total_outputs)
            )

        self.value_head = nn.Sequential(
            nn.Linear(self.d_model, self.hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_sizes[0], 1)
        )

        # Инициализация весов
        self._initialize_weights()

        self._value_out = None

    def create_padding_mask(self, obs, seq_lengths):
        """
        Создает маску для паддинга последовательностей
        """
        batch_size, max_len = obs.shape[:2]
        device = obs.device
        
        # Векторизованное создание маски
        positions = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)
        seq_lengths = seq_lengths.unsqueeze(1).expand(-1, max_len)
        mask = positions >= seq_lengths
        
        return mask

    def _get_activation(self):
        """Получение функции активации"""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
            "tanh": nn.Tanh()
        }
        return activations.get(self.activation, nn.ReLU())

    def _initialize_weights(self):
        """Консервативная инициализация весов модели"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, 'weight') and module.weight is not None:
                    # Xavier uniform инициализация
                    torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
                if hasattr(module, 'bias') and module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=self.init_std)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.constant_(module.bias, 0.0)
                torch.nn.init.constant_(module.weight, 1.0)

    def forward(self, input_dict, state, seq_lens):
        """Улучшенный прямой проход с поддержкой Box action space"""
        obs = input_dict["obs"]
        batch_size, max_len = obs.shape[:2]

        # Проверка входных данных на NaN/Inf
        if torch.isnan(obs).any() or torch.isinf(obs).any():
            print("Warning: NaN/Inf detected in input observations")
            obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

        # Получаем реальные длины последовательностей
        valid_mask = (obs.abs().sum(dim=-1) > 1e-6)
        seq_lengths = valid_mask.sum(dim=1)
        seq_lengths = torch.clamp(seq_lengths, min=1)

        # Входная проекция
        x = self.input_projection(obs)
        
        if torch.isnan(x).any():
            print("Warning: NaN detected after input projection")
            x = torch.nan_to_num(x, nan=0.0)

        # Позиционное кодирование
        if self.pos_encoding_type == "learned":
            positions = torch.arange(max_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
            positions = torch.clamp(positions, max=self.max_seq_length-1)
            pos_emb = self.pos_embedding(positions)
            x = x + pos_emb
        elif self.pos_encoding_type == "sinusoidal":
            x = self.pos_encoding(x)

        # Маска для паддинга
        padding_mask = self.create_padding_mask(obs, seq_lengths)

        # Пропускаем через трансформерные слои
        for i, layer in enumerate(self.transformer_layers):
            x_prev = x
            x = layer(x, src_key_padding_mask=padding_mask)
            
            if torch.isnan(x).any():
                print(f"Warning: NaN detected after transformer layer {i}")
                x = x_prev
                break

        # Агрегация
        valid_mask_float = (~padding_mask).float().unsqueeze(-1)
        seq_lengths_safe = seq_lengths.clamp(min=1).float().unsqueeze(-1).unsqueeze(-1)
        pooled = (x * valid_mask_float).sum(dim=1) / seq_lengths_safe.squeeze(-1)
        
        if torch.isnan(pooled).any():
            print("Warning: NaN detected in pooled representation")
            pooled = torch.zeros_like(pooled)

        # Генерируем действия в зависимости от типа action space
        if self.action_space_type == "box":
            # Для непрерывных действий возвращаем конкатенированные mean и log_std
            action_mean = self.action_mean_head(pooled)
            action_log_std = self.action_log_std_head(pooled)
            
            # Ограничиваем значения для стабильности
            action_mean = torch.tanh(action_mean)  # Нормализуем в [-1, 1]
            action_log_std = torch.clamp(action_log_std, min=-5, max=0)  # Ограничиваем log_std
            
            # Масштабируем mean к диапазону действий
            device = action_mean.device
            action_low = self.action_low.to(device)
            action_high = self.action_high.to(device)
            
            # Масштабируем от [-1, 1] к [low, high]
            action_mean = action_low + (action_mean + 1.0) * 0.5 * (action_high - action_low)
            
            # Конкатенируем mean и log_std для возврата
            action_logits = torch.cat([action_mean, action_log_std], dim=-1)
        else:
            # Для дискретных действий
            action_logits = self.action_head(pooled)
        
        if torch.isnan(action_logits).any():
            print("Warning: NaN detected in action logits, replacing with zeros")
            action_logits = torch.zeros_like(action_logits)

        # Вычисляем значение состояния
        value_out = self.value_head(pooled)
        if torch.isnan(value_out).any():
            print("Warning: NaN detected in value output, replacing with zeros")
            value_out = torch.zeros_like(value_out)
            
        self._value_out = value_out.squeeze(-1)

        return action_logits, state

    def value_function(self):
        """Возвращает значение состояния"""
        return self._value_out if self._value_out is not None else torch.zeros(1)

class CustomEnvironment(gym.Env):
    """
    Кастомная среда для обучения
    Принимает действия фиксированного размера, возвращает наблюдения переменной длины
    """
    
    def __init__(self, config: EnvContext):
        super().__init__()
        
        # Конфигурация среды
        self.max_sequence_length = config.get("max_sequence_length", 50)
        self.min_sequence_length = config.get("min_sequence_length", 5)
        self.observation_dim = config.get("observation_dim", 10)
        
        # Поддержка как дискретных, так и непрерывных действий
        self.action_type = config.get("action_type", "box")  # "box" или "discrete"
        
        # Определяем пространства
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.max_sequence_length, self.observation_dim),
            dtype=np.float32
        )
        
        if self.action_type == "box":
            # Непрерывное пространство действий
            self.action_dim = config.get("action_dim", 3)
            self.action_low = config.get("action_low", [-1.0] * self.action_dim)
            self.action_high = config.get("action_high", [1.0] * self.action_dim)
            
            self.action_space = spaces.Box(
                low=np.array(self.action_low, dtype=np.float32),
                high=np.array(self.action_high, dtype=np.float32),
                dtype=np.float32
            )
        else:
            # Дискретное пространство действий (оригинальный код)
            self.action_ranges = config.get("action_ranges", [5, 10, 3])
            self.action_space = spaces.MultiDiscrete(self.action_ranges)
        
        # Состояние среды
        self.current_sequence = None
        self.step_count = 0
        self.max_steps = config.get("max_steps", 100)
        
        # Псевдо-параметры задачи
        self.target_pattern = self._generate_target_pattern()
    
    def _generate_target_pattern(self):
        """Генерируем целевой паттерн для обучения (псевдокод)"""
        # Здесь может быть любая кастомная логика
        return np.random.randn(self.observation_dim)
    
    def _generate_observation(self):
        """Генерируем наблюдение переменной длины"""
        # Случайная длина последовательности
        seq_length = np.random.randint(self.min_sequence_length, self.max_sequence_length + 1)
        
        # Генерируем данные с учетом текущего состояния
        sequence = np.random.randn(seq_length, self.observation_dim).astype(np.float32)
        
        # Добавляем некоторую зависимость от целевого паттерна
        for i in range(seq_length):
            noise_level = 0.1 + 0.9 * (i / seq_length)  # Увеличиваем шум к концу
            sequence[i] += self.target_pattern * (1 - noise_level)
        
        # Паддинг до максимальной длины
        padded_sequence = np.zeros((self.max_sequence_length, self.observation_dim), dtype=np.float32)
        padded_sequence[:seq_length] = sequence
        
        return padded_sequence
    
    def reset(self, *, seed=None, options=None):
        """Сброс среды"""
        if seed is not None:
            np.random.seed(seed)
        
        self.step_count = 0
        self.current_sequence = self._generate_observation()
        
        return self.current_sequence, {}
    
    def step(self, action):
        """Выполнение шага в среде"""
        self.step_count += 1
        
        # Псевдокод для вычисления награды
        reward = self._calculate_reward(action)
        
        # Генерируем новое наблюдение
        self.current_sequence = self._generate_observation()
        
        # Проверяем условие завершения
        terminated = self.step_count >= self.max_steps
        truncated = False
        
        info = {
            "step_count": self.step_count,
            "action_taken": action
        }
        
        return self.current_sequence, reward, terminated, truncated, info
    
    def _calculate_reward(self, action):
        """Улучшенная функция награды для поддержки Box actions"""
        base_reward = 1.0
        
        if self.action_type == "box":
            # Для непрерывных действий
            # Целевые значения основаны на статистике наблюдений
            valid_obs = self.current_sequence[self.current_sequence.sum(axis=1) != 0]
            if len(valid_obs) > 0:
                mean_obs = np.mean(valid_obs, axis=0)
                target_actions = np.tanh(mean_obs[:self.action_dim])  # Нормализуем к [-1, 1]
                
                # Масштабируем к диапазону действий
                target_actions = (target_actions + 1.0) * 0.5  # К [0, 1]
                target_actions = self.action_low + target_actions * (np.array(self.action_high) - np.array(self.action_low))
                
                # Награда обратно пропорциональна расстоянию
                distances = np.abs(action - target_actions)
                normalized_distances = distances / (np.array(self.action_high) - np.array(self.action_low))
                action_reward = 1.0 - np.mean(normalized_distances)
            else:
                action_reward = 0.0
        else:
            # Оригинальная логика для дискретных действий
            action_rewards = []
            for i, (act, max_val) in enumerate(zip(action, self.action_ranges)):
                normalized_action = act / (max_val - 1)
                mean_obs = np.mean(self.current_sequence[self.current_sequence.sum(axis=1) != 0])
                target_normalized = (mean_obs + 1) / 2
                target_normalized = np.clip(target_normalized, 0, 1)
                distance = abs(normalized_action - target_normalized)
                action_reward = 1.0 - distance
                action_rewards.append(action_reward)
            
            action_reward = np.mean(action_rewards)
        
        total_reward = base_reward + action_reward
        total_reward += np.random.normal(0, 0.1)
        
        return float(total_reward)


def create_config():
    """Создание конфигурации для PPO"""
    
    # Конфигурация среды
    # env_config = {
    #     "max_sequence_length": 50,
    #     "min_sequence_length": 5,
    #     "observation_dim": 10,
    #     "num_actions": 3,
    #     "action_ranges": [5, 10, 3],
    #     "max_steps": 200
    # }
    env_config = {
        "max_sequence_length": 50,
        "min_sequence_length": 5,
        "observation_dim": 10,
        "action_type": "box",
        "action_dim": 3,
        "action_low": [-2.0, -1.0, -3.0],
        "action_high": [2.0, 1.0, 3.0],
        "max_steps": 200
    }
    
    # Конфигурация модели с расширенными параметрами для оптимизации
    model_config = {
        "custom_model": "transformer_model",
        "custom_model_config": {
            # === ОСНОВНЫЕ АРХИТЕКТУРНЫЕ ПАРАМЕТРЫ ===
            "d_model": 256,              # Размер эмбеддинга (128, 256, 512, 768, 1024)
            "nhead": 8,                  # Количество голов внимания (4, 8, 12, 16)
            "num_layers": 6,             # Количество слоев трансформера (2-12)
            "dim_feedforward": 1024,     # Размер FFN слоя (обычно 4*d_model)
            "dropout": 0.1,              # Dropout для регуляризации (0.0-0.3)
            "max_seq_length": 100,       # Максимальная длина последовательности
            
            # === УЛУЧШЕНИЯ АРХИТЕКТУРЫ ===
            "use_layer_norm": True,      # Использовать Layer Normalization
            "pre_norm": True,            # Pre-norm vs Post-norm архитектура
            "use_residual": True,        # Остаточные соединения
            "activation": "gelu",        # Функция активации (relu, gelu, swish)
            
            # === ПОЗИЦИОННОЕ КОДИРОВАНИЕ ===
            "pos_encoding_type": "learned",  # "sinusoidal", "learned", "rotary"
            "max_relative_position": 32,     # Для относительного позиционного кодирования
            
            # === ВНИМАНИЕ ===
            "attention_dropout": 0.1,    # Отдельный dropout для внимания
            "use_flash_attention": False,# Flash Attention для эффективности
            "attention_bias": False,     # Bias в слоях внимания
            
            # === АГРЕГАЦИЯ ===
            "aggregation_method": "cls_attention",  # "cls_attention", "mean_pooling", "max_pooling", "adaptive"
            "num_cls_tokens": 1,         # Количество CLS токенов
            "pooling_heads": 8,          # Головы внимания для агрегации
            
            # === ВЫХОДНЫЕ СЛОИ ===
            "hidden_sizes": [512, 256],  # Размеры скрытых слоев в головах
            "output_activation": "tanh", # Активация перед выходом
            "use_output_norm": True,     # Нормализация на выходе
            
            # === ИНИЦИАЛИЗАЦИЯ ===
            "init_std": 0.02,           # Стандартное отклонение для инициализации
            "init_method": "xavier",     # "xavier", "kaiming", "normal"
            
            # === СПЕЦИАЛЬНЫЕ ТЕХНИКИ ===
            "use_gradient_checkpointing": False,  # Экономия памяти
            "use_mixed_precision": False,         # Mixed precision training
        }
    }
    
    # Конфигурация PPO
    config = (PPOConfig()
                .environment(
                    env=CustomEnvironment, 
                    env_config=env_config)
                .env_runners(
                    num_env_runners=2, 
                    rollout_fragment_length=200,
                    num_gpus_per_env_runner=0.1
                    )
                .training(
                    train_batch_size=2000,
                    #sgd_minibatch_size=256,
                    #num_sgd_iter=10,
                    lr=3e-4,
                    #entropy_coeff=0.01,
                    #clip_param=0.2,
                    #vf_loss_coeff=0.5,
                    model=model_config
                )
                .api_stack(
                    enable_rl_module_and_learner=False,
                    enable_env_runner_and_connector_v2=False
                )
                .framework("torch")
                .debugging(log_level="INFO"))
    
    return config, env_config


def save_model_weights(trainer, checkpoint_path: str, filename: str = "model_weights.pkl"):
    """Улучшенная функция сохранения весов с поддержкой кастомных имен файлов"""
    
    # Создаем директорию если не существует
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Получаем политику из тренера
    policy = trainer.get_policy()
    model = policy.model
    
    # Расширенное сохранение состояния
    model_state = {
        'state_dict': model.state_dict(),
        'model_config': model.model_config,
        'obs_space': model.obs_space,
        'action_space': model.action_space,
        'num_outputs': model.num_outputs,
        'training_iteration': trainer.training_iteration,
        'timesteps_total': trainer._counters.get('num_env_steps_sampled', 0),
        'episode_reward_mean': trainer._counters.get('episode_reward_mean', 0),
        'save_timestamp': str(pd.Timestamp.now()),
        'ray_version': ray.__version__
    }
    
    weights_path = os.path.join(checkpoint_path, filename)
    with open(weights_path, 'wb') as f:
        pickle.dump(model_state, f)
    
    print(f"Веса модели сохранены в: {weights_path}")
    return weights_path


class StandaloneTransformer(nn.Module):
    """Автономная версия трансформера для инференса без Ray"""
    
    def __init__(self, model_state):
        super().__init__()
        
        # Восстанавливаем архитектуру модели
        self.obs_space = model_state['obs_space']
        self.action_space = model_state['action_space']
        self.num_outputs = model_state['num_outputs']
        
        # Создаем модель с теми же параметрами
        self.model = ImprovedTransformerModel(
            obs_space=self.obs_space,
            action_space=self.action_space,
            num_outputs=self.num_outputs,
            model_config=model_state['model_config'],
            name="standalone_transformer"
        )
        
        # Загружаем веса
        self.model.load_state_dict(model_state['state_dict'])
        self.model.eval()
    
    def predict(self, observation):
        """Предсказание действий для данного наблюдения с поддержкой Box actions"""
        with torch.no_grad():
            if isinstance(observation, np.ndarray):
                observation = torch.FloatTensor(observation)
            
            if len(observation.shape) == 2:
                observation = observation.unsqueeze(0)
            
            input_dict = {"obs": observation}
            action_logits, _ = self.model(input_dict, [], None)
            
            if self.model.action_space_type == "box":
                # Для непрерывных действий
                batch_size = action_logits.shape[0]
                action_dim = self.model.action_dim
                
                # Разделяем на mean и log_std
                action_mean = action_logits[:, :action_dim]
                action_log_std = action_logits[:, action_dim:]
                
                # Сэмплируем действия из нормального распределения
                action_std = torch.exp(action_log_std)
                dist = torch.distributions.Normal(action_mean, action_std)
                actions = dist.sample()
                
                # Ограничиваем действия к допустимому диапазону
                device = actions.device
                action_low = self.model.action_low.to(device)
                action_high = self.model.action_high.to(device)
                actions = torch.clamp(actions, action_low, action_high)
                
                return actions.squeeze(0).cpu().numpy()
            else:
                # Оригинальная логика для дискретных действий
                actions = []
                start_idx = 0
                
                if hasattr(self.action_space, 'nvec'):
                    action_ranges = self.action_space.nvec
                else:
                    action_ranges = [self.action_space.n] * self.num_outputs
                
                for action_range in action_ranges:
                    end_idx = start_idx + action_range
                    action_probs = F.softmax(action_logits[:, start_idx:end_idx], dim=-1)
                    action = torch.argmax(action_probs, dim=-1)
                    actions.append(action.item())
                    start_idx = end_idx
                
                return actions


def load_standalone_model(weights_path: str) -> StandaloneTransformer:
    """Загрузка модели для автономного использования"""
    with open(weights_path, 'rb') as f:
        model_state = pickle.load(f)
    
    return StandaloneTransformer(model_state)


def train_with_pbt():
    """Обучение с использованием Population Based Training"""
    
    # Инициализируем Ray
    ray.init(ignore_reinit_error=True)
    
    # Регистрируем улучшенную кастомную модель
    ModelCatalog.register_custom_model("transformer_model", ImprovedTransformerModel)
    
    # Создаем базовую конфигурацию
    base_config, env_config = create_config()
    
    # Настройка PBT с расширенным поиском гиперпараметров
    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="env_runners/episode_reward_mean",
        mode="max",
        perturbation_interval=15,  # Увеличил интервал для стабильности
        resample_probability=0.25,  # Вероятность полной повторной выборки
        hyperparam_mutations={
            # === PPO ПАРАМЕТРЫ ===
            "lr": lambda: np.random.uniform(1e-5, 5e-3),
            #"entropy_coeff": lambda: np.random.uniform(0.001, 0.1),
            #"clip_param": lambda: np.random.uniform(0.1, 0.5),
            "train_batch_size": lambda: np.random.choice([1000, 2000, 4000, 8000]),
            #"sgd_minibatch_size": lambda: np.random.choice([64, 128, 256, 512]),
            #"num_sgd_iter": lambda: np.random.choice([5, 10, 15, 20]),
            "gamma": lambda: np.random.uniform(0.95, 0.999),
            "lambda": lambda: np.random.uniform(0.9, 1.0),
            
            # === АРХИТЕКТУРНЫЕ ПАРАМЕТРЫ ===
            "model.custom_model_config.d_model": lambda: np.random.choice([128, 256, 384, 512]),
            "model.custom_model_config.nhead": lambda: np.random.choice([4, 8, 12, 16]),
            "model.custom_model_config.num_layers": lambda: np.random.choice([2, 4, 6, 8]),
            "model.custom_model_config.dropout": lambda: np.random.uniform(0.0, 0.3),
            "model.custom_model_config.attention_dropout": lambda: np.random.uniform(0.0, 0.2),
            
            # === СТРАТЕГИИ АГРЕГАЦИИ ===
            "model.custom_model_config.aggregation_method": lambda: np.random.choice([
                "cls_attention", "mean_pooling", "max_pooling"
            ]),
            "model.custom_model_config.num_cls_tokens": lambda: np.random.choice([1, 2, 3]),
            
            # === АКТИВАЦИИ И НОРМАЛИЗАЦИЯ ===
            "model.custom_model_config.activation": lambda: np.random.choice([
                "relu", "gelu", "swish"
            ]),
            "model.custom_model_config.pre_norm": lambda: np.random.choice([True, False]),
            "model.custom_model_config.output_activation": lambda: np.random.choice([
                "none", "tanh", "sigmoid"
            ]),
        },
        quantile_fraction=0.25,  # Топ 25% для селекции
        custom_explore_fn=None
    )
    
    # Настройка отчетности
    reporter = CLIReporter(
        metric_columns=["env_runners/episode_reward_mean", "training_iteration", "timesteps_total"],
        parameter_columns=["lr"]#, "entropy_coeff", "clip_param"]
    )
    
    # Запуск обучения
    analysis = tune.run(
        PPO,
        name="transformer_ppo_pbt",
        scheduler=pbt,
        config=base_config.to_dict(),
        num_samples=4,  # Размер популяции
        stop={"training_iteration": 100},
        progress_reporter=reporter,
        checkpoint_freq=10,
        keep_checkpoints_num=5,
        checkpoint_score_attr="env_runners/episode_reward_mean",
        storage_path=r"F:\work\code\for_git\transform_rl\new\ray_pth"
    )
    
    # Получаем лучший результат
    best_trial = analysis.get_best_trial("env_runners/episode_reward_mean", "max")
    best_checkpoint = analysis.get_best_checkpoint(best_trial, "env_runners/episode_reward_mean", "max")
    
    print(f"Лучший чекпоинт: {best_checkpoint}")
    
    # Загружаем лучшую модель и сохраняем веса
    trainer = PPO(config=base_config.to_dict())
    trainer.restore(best_checkpoint)
    
    weights_path = save_model_weights(trainer, r"F:\work\code\for_git\transform_rl\new\saved_models")
    
    return weights_path, analysis


def test_standalone_model(weights_path: str):
    """Тестирование автономной модели"""
    
    # Загружаем модель
    model = load_standalone_model(weights_path)
    
    # Создаем тестовое наблюдение
    test_obs = np.random.randn(30, 10).astype(np.float32)  # Последовательность длины 30
    
    # Паддинг до максимальной длины
    padded_obs = np.zeros((50, 10), dtype=np.float32)
    padded_obs[:30] = test_obs
    
    # Предсказание
    actions = model.predict(padded_obs)
    
    print("Тестовое наблюдение создано")
    print(f"Предсказанные действия: {actions}")
    
    return actions


class TrainingCheckpointManager:
    """Менеджер для управления чекпоинтами обучения"""
    
    def __init__(self, checkpoint_dir: str = "ray_checkpoints", weights_dir: str = "saved_models"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.weights_dir = Path(weights_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.weights_dir.mkdir(exist_ok=True)
        
        # Файл с метаданными обучения
        self.metadata_file = self.checkpoint_dir / "training_metadata.json"
    
    def save_training_metadata(self, config: Dict, analysis_data: Dict = None):
        """Сохранение метаданных обучения"""
        metadata = {
            "config": config,
            "analysis_data": analysis_data,
            "timestamp": str(pd.Timestamp.now()),
            "ray_version": ray.__version__
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def load_training_metadata(self) -> Optional[Dict]:
        """Загрузка метаданных обучения"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return None
    
    def find_latest_checkpoint(self, experiment_name: str = "transformer_ppo_pbt") -> Optional[str]:
        """Поиск последнего чекпоинта"""
        checkpoint_pattern = str(self.checkpoint_dir / experiment_name / "**" / "checkpoint_*")
        checkpoints = glob.glob(checkpoint_pattern, recursive=True)
        
        if not checkpoints:
            return None
            
        # Сортируем по времени модификации
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        return checkpoints[0]
    
    def find_best_checkpoint(self, analysis) -> Optional[str]:
        """Поиск лучшего чекпоинта из результатов Tune"""
        try:
            best_trial = analysis.get_best_trial("env_runners/episode_reward_mean", "max")
            return analysis.get_best_checkpoint(best_trial, "env_runners/episode_reward_mean", "max")
        except Exception as e:
            print(f"Ошибка при поиске лучшего чекпоинта: {e}")
            return None


def load_model_from_checkpoint(checkpoint_path: str, config: Dict) -> tuple:
    """Загрузка модели из чекпоинта Ray"""
    try:
        # Создаем тренер с конфигурацией
        trainer = PPO(config=config)
        
        # Восстанавливаем состояние из чекпоинта
        trainer.restore(checkpoint_path)
        
        print(f"Модель успешно загружена из чекпоинта: {checkpoint_path}")
        return trainer, True
        
    except Exception as e:
        print(f"Ошибка при загрузке чекпоинта {checkpoint_path}: {e}")
        return None, False


def initialize_model_weights(trainer, weights_path: str) -> bool:
    """Инициализация весов модели из сохраненного файла"""
    try:
        # Загружаем сохраненные веса
        with open(weights_path, 'rb') as f:
            model_state = pickle.load(f)
        
        # Получаем политику и модель
        policy = trainer.get_policy()
        model = policy.model
        
        # Загружаем веса в модель
        model.load_state_dict(model_state['state_dict'])
        
        print(f"Веса модели успешно загружены из: {weights_path}")
        return True
        
    except Exception as e:
        print(f"Ошибка при загрузке весов из {weights_path}: {e}")
        return False


def create_resume_config(base_config, resume_from_weights: str = None, 
                        resume_from_checkpoint: str = None, 
                        modify_lr: bool = True, lr_factor: float = 0.1):
    """Создание конфигурации для продолжения обучения"""
    
    # Копируем базовую конфигурацию
    resume_config = base_config.copy()
    
    if modify_lr and resume_from_checkpoint:
        # Снижаем learning rate для fine-tuning
        current_lr = resume_config.get("lr", 3e-4)
        new_lr = current_lr * lr_factor
        resume_config = resume_config.training(lr=new_lr)
        print(f"Learning rate изменен с {current_lr} на {new_lr} для fine-tuning")
    
    return resume_config


def train_with_pbt_resume(resume_from_checkpoint: str = None, 
                         resume_from_weights: str = None,
                         experiment_name: str = None,
                         modify_config: Dict = None):
    """Обучение с PBT с возможностью продолжения"""
    
    # Инициализируем Ray
    ray.init(ignore_reinit_error=True)
    
    # Регистрируем модель
    ModelCatalog.register_custom_model("transformer_model", ImprovedTransformerModel)
    
    # Менеджер чекпоинтов
    checkpoint_manager = TrainingCheckpointManager()
    
    # Создаем базовую конфигурацию
    base_config, env_config = create_config()
    
    # Применяем модификации конфигурации если есть
    if modify_config:
        for key, value in modify_config.items():
            if hasattr(base_config, key):
                setattr(base_config, key, value)
    
    # Определяем имя эксперимента
    if experiment_name is None:
        experiment_name = "transformer_ppo_pbt_resume" if resume_from_checkpoint else "transformer_ppo_pbt"
    
    # Логика продолжения обучения
    restore_config = None
    if resume_from_checkpoint:
        print(f"Продолжаем обучение с чекпоинта: {resume_from_checkpoint}")
        
        # Создаем конфигурацию для продолжения
        resume_config = create_resume_config(
            base_config, 
            resume_from_checkpoint=resume_from_checkpoint,
            modify_lr=True
        )
        
        # Настройка для восстановления в Tune
        restore_config = resume_from_checkpoint
        
    elif resume_from_weights:
        print(f"Инициализируем веса из файла: {resume_from_weights}")
        resume_config = base_config
    else:
        print("Начинаем обучение с нуля")
        resume_config = base_config
    
    # Callback для инициализации весов
    def on_trial_start(iteration, trials, trial, **info):
        if resume_from_weights and not resume_from_checkpoint:
            try:
                # Создаем временный тренер для загрузки весов
                temp_trainer = PPO(config=resume_config.to_dict())
                success = initialize_model_weights(temp_trainer, resume_from_weights)
                if success:
                    # Сохраняем инициализированную модель как чекпоинт
                    temp_checkpoint = temp_trainer.save()
                    print(f"Временный чекпоинт с загруженными весами: {temp_checkpoint}")
                    # Здесь можно добавить логику для использования этого чекпоинта
                temp_trainer.stop()
            except Exception as e:
                print(f"Ошибка при инициализации весов в trial: {e}")
    
    # Настройка PBT
    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="env_runners/episode_reward_mean",
        mode="max",
        perturbation_interval=15,
        resample_probability=0.25,
        hyperparam_mutations={
            "lr": lambda: np.random.uniform(1e-5, 5e-3),
            "train_batch_size": lambda: np.random.choice([1000, 2000, 4000, 8000]),
            "gamma": lambda: np.random.uniform(0.95, 0.999),
            "lambda": lambda: np.random.uniform(0.9, 1.0),
            "model.custom_model_config.d_model": lambda: np.random.choice([128, 256, 384, 512]),
            "model.custom_model_config.nhead": lambda: np.random.choice([4, 8, 12, 16]),
            "model.custom_model_config.num_layers": lambda: np.random.choice([2, 4, 6, 8]),
            "model.custom_model_config.dropout": lambda: np.random.uniform(0.0, 0.3),
            "model.custom_model_config.attention_dropout": lambda: np.random.uniform(0.0, 0.2),
            "model.custom_model_config.aggregation_method": lambda: np.random.choice([
                "cls_attention", "mean_pooling", "max_pooling"
            ]),
            "model.custom_model_config.activation": lambda: np.random.choice([
                "relu", "gelu", "swish"
            ]),
        },
        quantile_fraction=0.25,
    )
    
    # Настройка отчетности
    reporter = CLIReporter(
        metric_columns=["env_runners/episode_reward_mean", "training_iteration", "timesteps_total"],
        parameter_columns=["lr", "train_batch_size"]
    )
    
    # Параметры для Tune
    tune_config = {
        "config": resume_config.to_dict(),
        "name": experiment_name,
        "scheduler": pbt,
        "num_samples": 4,
        "stop": {"training_iteration": 100},
        "progress_reporter": reporter,
        "checkpoint_freq": 10,
        "keep_checkpoints_num": 5,
        "checkpoint_score_attr": "env_runners/episode_reward_mean",
        "storage_path": checkpoint_manager.checkpoint_dir,
    }
    
    # Добавляем restore если нужно
    if restore_config:
        tune_config["restore"] = restore_config
    
    # Запуск обучения
    try:
        analysis = tune.run(PPO, **tune_config)
        
        # Сохраняем метаданные
        checkpoint_manager.save_training_metadata(
            config=resume_config.to_dict(),
            analysis_data={
                "best_reward": analysis.best_result["env_runners/episode_reward_mean"],
                "experiment_name": experiment_name
            }
        )
        
        # Получаем лучший результат
        best_checkpoint = checkpoint_manager.find_best_checkpoint(analysis)
        
        if best_checkpoint:
            print(f"Лучший чекпоинт: {best_checkpoint}")
            
            # Загружаем лучшую модель и сохраняем веса
            trainer = PPO(config=resume_config.to_dict())
            trainer.restore(best_checkpoint)
            
            # Создаем уникальное имя для весов
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            weights_filename = f"model_weights_{experiment_name}_{timestamp}.pkl"
            weights_path = save_model_weights(trainer, str(checkpoint_manager.weights_dir), weights_filename)
            
            trainer.stop()
            
            return weights_path, analysis, best_checkpoint
        else:
            print("Не удалось найти лучший чекпоинт")
            return None, analysis, None
            
    except Exception as e:
        print(f"Ошибка во время обучения: {e}")
        raise

def auto_resume_training(checkpoint_dir: str = "ray_checkpoints", 
                        weights_dir: str = "saved_models",
                        experiment_name: str = "transformer_ppo_pbt"):
    """Автоматическое продолжение обучения с последнего чекпоинта"""
    
    checkpoint_manager = TrainingCheckpointManager(checkpoint_dir, weights_dir)
    
    # Ищем последний чекпоинт
    latest_checkpoint = checkpoint_manager.find_latest_checkpoint(experiment_name)
    
    if latest_checkpoint:
        print(f"Найден чекпоинт для продолжения: {latest_checkpoint}")
        return train_with_pbt_resume(
            resume_from_checkpoint=latest_checkpoint,
            experiment_name=f"{experiment_name}_resumed"
        )
    else:
        print("Чекпоинты не найдены, начинаем обучение с нуля")
        return train_with_pbt_resume(experiment_name=experiment_name)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Transformer PPO Training with Resume')
    parser.add_argument('--resume-checkpoint', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--resume-weights', type=str, help='Path to weights file to initialize from')
    parser.add_argument('--auto-resume', action='store_true', help='Automatically resume from latest checkpoint')
    parser.add_argument('--experiment-name', type=str, default='transformer_ppo_pbt', help='Name of experiment')
    
    args = parser.parse_args()
    
    print("Начинаем обучение трансформера с PPO и PBT...")
    
    try:
        if args.auto_resume:
            # Автоматическое продолжение
            result = auto_resume_training(experiment_name=args.experiment_name)
        elif args.resume_checkpoint:
            # Продолжение с конкретного чекпоинта
            result = train_with_pbt_resume(
                resume_from_checkpoint=args.resume_checkpoint,
                experiment_name=args.experiment_name
            )
        elif args.resume_weights:
            # Инициализация с весами
            result = train_with_pbt_resume(
                resume_from_weights=args.resume_weights,
                experiment_name=args.experiment_name
            )
        else:
            # Обучение с нуля
            result = train_with_pbt_resume(experiment_name=args.experiment_name)
        
        if result and len(result) >= 2:
            weights_path, analysis = result[:2]
            
            if weights_path:
                print(f"Обучение завершено. Веса сохранены в: {weights_path}")
                
                # Тестирование автономной модели
                print("\nТестируем автономную модель...")
                test_actions = test_standalone_model(weights_path)
                print("Все этапы выполнены успешно!")
            else:
                print("Обучение завершено, но не удалось сохранить веса")
        else:
            print("Обучение завершено с ошибками")
            
    except KeyboardInterrupt:
        print("\nОбучение прервано пользователем")
    except Exception as e:
        print(f"Ошибка во время обучения: {e}")
        raise
    finally:
        # Закрываем Ray
        ray.shutdown()
