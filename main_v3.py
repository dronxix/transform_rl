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
    """
    Улучшенная модель трансформера с дополнительными параметрами оптимизации
    """
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        # === ОСНОВНЫЕ ПАРАМЕТРЫ ===
        self.d_model = model_config.get("d_model", 256)
        self.nhead = model_config.get("nhead", 8)
        self.num_layers = model_config.get("num_layers", 6)
        self.dim_feedforward = model_config.get("dim_feedforward", 1024)
        self.dropout = model_config.get("dropout", 0.1)
        self.max_seq_length = model_config.get("max_seq_length", 100)
        
        # === ДОПОЛНИТЕЛЬНЫЕ ПАРАМЕТРЫ ===
        self.use_layer_norm = model_config.get("use_layer_norm", True)
        self.pre_norm = model_config.get("pre_norm", True)
        self.activation = model_config.get("activation", "gelu")
        self.attention_dropout = model_config.get("attention_dropout", 0.1)
        self.pos_encoding_type = model_config.get("pos_encoding_type", "learned")
        self.aggregation_method = model_config.get("aggregation_method", "cls_attention")
        self.num_cls_tokens = model_config.get("num_cls_tokens", 1)
        self.hidden_sizes = model_config.get("hidden_sizes", [512, 256])
        self.output_activation = model_config.get("output_activation", "tanh")
        self.use_output_norm = model_config.get("use_output_norm", True)
        self.init_std = model_config.get("init_std", 0.02)
        
        self.input_dim = obs_space.shape[1]
        self.output_ranges = action_space.nvec if hasattr(action_space, 'nvec') else [action_space.n] * num_outputs
        
        # === ВХОДНАЯ ПРОЕКЦИЯ ===
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.d_model),
            nn.LayerNorm(self.d_model) if self.use_layer_norm else nn.Identity(),
            self._get_activation()
        )
        
        # === ПОЗИЦИОННОЕ КОДИРОВАНИЕ ===
        if self.pos_encoding_type == "learned":
            self.pos_embedding = nn.Embedding(self.max_seq_length, self.d_model)
        elif self.pos_encoding_type == "sinusoidal":
            self.pos_encoding = PositionalEncoding(self.d_model, self.max_seq_length)
        
        # === УЛУЧШЕННЫЕ ТРАНСФОРМЕРНЫЕ СЛОИ ===
        encoder_layers = []
        for _ in range(self.num_layers):
            layer = self._create_transformer_layer()
            encoder_layers.append(layer)
        self.transformer_layers = nn.ModuleList(encoder_layers)
        
        # === CLS ТОКЕНЫ ===
        if self.aggregation_method == "cls_attention":
            self.cls_tokens = nn.Parameter(torch.randn(1, self.num_cls_tokens, self.d_model))
            self.cls_attention = nn.MultiheadAttention(
                embed_dim=self.d_model,
                num_heads=self.nhead,
                dropout=self.attention_dropout,
                batch_first=True
            )
        
        # === УЛУЧШЕННЫЕ ВЫХОДНЫЕ СЛОИ ===
        self.action_heads = nn.ModuleList([
            self._create_output_head(action_range) for action_range in self.output_ranges
        ])
        
        self.value_head = self._create_output_head(1, is_value=True)
        
        # Инициализация весов
        self._initialize_weights()
        
        self._value_out = None
    
    def _get_activation(self):
        """Получение функции активации"""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
            "tanh": nn.Tanh()
        }
        return activations.get(self.activation, nn.GELU())
    
    def _create_transformer_layer(self):
        """Создание улучшенного слоя трансформера"""
        if self.pre_norm:
            # Pre-LayerNorm архитектура (более стабильная)
            return PreNormTransformerLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                attention_dropout=self.attention_dropout,
                activation=self.activation
            )
        else:
            # Стандартная архитектура
            return nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                activation=self.activation,
                batch_first=True
            )
    
    def _create_output_head(self, output_size, is_value=False):
        """Создание улучшенной выходной головы"""
        layers = []
        
        # Входной размер
        current_size = self.d_model * self.num_cls_tokens if self.aggregation_method == "cls_attention" else self.d_model
        
        # Скрытые слои
        for hidden_size in self.hidden_sizes:
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.LayerNorm(hidden_size) if self.use_output_norm else nn.Identity(),
                self._get_activation(),
                nn.Dropout(self.dropout)
            ])
            current_size = hidden_size
        
        # Выходной слой
        layers.append(nn.Linear(current_size, output_size))
        
        # Активация на выходе (только для действий, не для value)
        if not is_value and self.output_activation != "none":
            if self.output_activation == "tanh":
                layers.append(nn.Tanh())
            elif self.output_activation == "sigmoid":
                layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Инициализация весов модели"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, 'weight') and module.weight is not None:
                    torch.nn.init.normal_(module.weight, std=self.init_std)
                if hasattr(module, 'bias') and module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, std=self.init_std)
    
    def forward(self, input_dict, state, seq_lens):
        """Улучшенный прямой проход"""
        obs = input_dict["obs"]
        batch_size, max_len = obs.shape[:2]
        
        # Получаем реальные длины последовательностей
        seq_lengths = (obs.sum(dim=-1) != 0).sum(dim=1)
        
        # Входная проекция
        x = self.input_projection(obs)
        
        # Позиционное кодирование
        if self.pos_encoding_type == "learned":
            positions = torch.arange(max_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.pos_embedding(positions)
            x = x + pos_emb
        elif self.pos_encoding_type == "sinusoidal":
            x = self.pos_encoding(x)
        
        # Маска для паддинга
        padding_mask = self.create_padding_mask(obs, seq_lengths)
        
        # Пропускаем через трансформерные слои
        for layer in self.transformer_layers:
            if isinstance(layer, PreNormTransformerLayer):
                x = layer(x, src_key_padding_mask=padding_mask)
            else:
                x = layer(x, src_key_padding_mask=padding_mask)
        
        # Агрегация
        if self.aggregation_method == "cls_attention":
            cls_tokens = self.cls_tokens.expand(batch_size, -1, -1)
            aggregated, _ = self.cls_attention(
                query=cls_tokens,
                key=x,
                value=x,
                key_padding_mask=padding_mask
            )
            pooled = aggregated.view(batch_size, -1)  # Flatten CLS tokens
        elif self.aggregation_method == "mean_pooling":
            # Средневзвешенное по реальной длине
            mask = (~padding_mask).float().unsqueeze(-1)
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1)
        elif self.aggregation_method == "max_pooling":
            pooled = x.masked_fill(padding_mask.unsqueeze(-1), float('-inf')).max(dim=1)[0]
        
        # Генерируем действия
        actions = []
        for head in self.action_heads:
            action_logits = head(pooled)
            actions.append(action_logits)
        
        action_logits = torch.cat(actions, dim=-1)
        
        # Вычисляем значение состояния
        self._value_out = self.value_head(pooled).squeeze(-1)
        
        return action_logits, state


class PreNormTransformerLayer(nn.Module):
    """Pre-LayerNorm трансформерный слой для лучшей стабильности обучения"""
    
    def __init__(self, d_model, nhead, dim_feedforward, dropout, attention_dropout, activation):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, nhead, dropout=attention_dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
    
    def _get_activation(self, activation):
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU()
        }
        return activations.get(activation, nn.GELU())
    
    def forward(self, x, src_key_padding_mask=None):
        # Pre-norm архитектура
        norm_x = self.norm1(x)
        attn_out, _ = self.attention(norm_x, norm_x, norm_x, key_padding_mask=src_key_padding_mask)
        x = x + self.dropout1(attn_out)
        
        norm_x = self.norm2(x)
        ffn_out = self.ffn(norm_x)
        x = x + ffn_out
        
        return x


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
        self.num_actions = config.get("num_actions", 3)
        self.action_ranges = config.get("action_ranges", [5, 10, 3])  # Диапазоны для каждого действия
        
        # Определяем пространства
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.max_sequence_length, self.observation_dim),
            dtype=np.float32
        )
        
        # Дискретное многомерное пространство действий
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
        """Псевдокод для вычисления награды"""
        # Пример: награда зависит от того, насколько хорошо действие
        # соответствует скрытому паттерну в данных
        
        base_reward = 1.0
        
        # Штраф/бонус за каждое действие
        action_rewards = []
        for i, (act, max_val) in enumerate(zip(action, self.action_ranges)):
            # Нормализуем действие к [0, 1]
            normalized_action = act / (max_val - 1)
            
            # Целевое значение зависит от среднего значения текущей последовательности
            mean_obs = np.mean(self.current_sequence[self.current_sequence.sum(axis=1) != 0])
            target_normalized = (mean_obs + 1) / 2  # Нормализуем к [0, 1]
            target_normalized = np.clip(target_normalized, 0, 1)
            
            # Награда обратно пропорциональна расстоянию до цели
            distance = abs(normalized_action - target_normalized)
            action_reward = 1.0 - distance
            action_rewards.append(action_reward)
        
        total_reward = base_reward + np.mean(action_rewards)
        
        # Добавляем случайный шум для исследования
        total_reward += np.random.normal(0, 0.1)
        
        return float(total_reward)


def create_config():
    """Создание конфигурации для PPO"""
    
    # Конфигурация среды
    env_config = {
        "max_sequence_length": 50,
        "min_sequence_length": 5,
        "observation_dim": 10,
        "num_actions": 3,
        "action_ranges": [5, 10, 3],
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
              .environment(env=CustomEnvironment, env_config=env_config)
              .rollouts(num_rollout_workers=2, rollout_fragment_length=200)
              .training(
                  train_batch_size=2000,
                  sgd_minibatch_size=256,
                  num_sgd_iter=10,
                  lr=3e-4,
                  entropy_coeff=0.01,
                  clip_param=0.2,
                  vf_loss_coeff=0.5,
                  model=model_config
              )
              .framework("torch")
              .debugging(log_level="INFO"))
    
    return config, env_config


def save_model_weights(trainer, checkpoint_path: str):
    """Сохранение весов модели для дальнейшего использования без Ray"""
    
    # Получаем политику из тренера
    policy = trainer.get_policy()
    model = policy.model
    
    # Сохраняем веса модели
    model_state = {
        'state_dict': model.state_dict(),
        'model_config': model.model_config,
        'obs_space': model.obs_space,
        'action_space': model.action_space,
        'num_outputs': model.num_outputs
    }
    
    weights_path = os.path.join(checkpoint_path, 'model_weights.pkl')
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
        self.model = TransformerModel(
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
        """Предсказание действий для данного наблюдения"""
        with torch.no_grad():
            # Подготавливаем входные данные
            if isinstance(observation, np.ndarray):
                observation = torch.FloatTensor(observation)
            
            if len(observation.shape) == 2:
                observation = observation.unsqueeze(0)  # Добавляем batch dimension
            
            input_dict = {"obs": observation}
            action_logits, _ = self.model(input_dict, [], None)
            
            # Преобразуем логиты в действия
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
        perturbation_interval=15,  # Увеличил интервал для стабильности
        resample_probability=0.25,  # Вероятность полной повторной выборки
        hyperparam_mutations={
            # === PPO ПАРАМЕТРЫ ===
            "lr": lambda: np.random.uniform(1e-5, 5e-3),
            "entropy_coeff": lambda: np.random.uniform(0.001, 0.1),
            "clip_param": lambda: np.random.uniform(0.1, 0.5),
            "train_batch_size": lambda: np.random.choice([1000, 2000, 4000, 8000]),
            "sgd_minibatch_size": lambda: np.random.choice([64, 128, 256, 512]),
            "num_sgd_iter": lambda: np.random.choice([5, 10, 15, 20]),
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
        metric_columns=["episode_reward_mean", "training_iteration", "timesteps_total"],
        parameter_columns=["lr", "entropy_coeff", "clip_param"]
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
        checkpoint_score_attr="episode_reward_mean",
        local_dir="./ray_results"
    )
    
    # Получаем лучший результат
    best_trial = analysis.get_best_trial("episode_reward_mean", "max")
    best_checkpoint = analysis.get_best_checkpoint(best_trial, "episode_reward_mean", "max")
    
    print(f"Лучший чекпоинт: {best_checkpoint}")
    
    # Загружаем лучшую модель и сохраняем веса
    trainer = PPO(config=base_config.to_dict())
    trainer.restore(best_checkpoint)
    
    weights_path = save_model_weights(trainer, "./saved_models")
    
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


if __name__ == "__main__":
    print("Начинаем обучение трансформера с PPO и PBT...")
    
    # Обучение модели
    weights_path, analysis = train_with_pbt()
    
    print(f"Обучение завершено. Веса сохранены в: {weights_path}")
    
    # Тестирование автономной модели
    print("\nТестируем автономную модель...")
    test_actions = test_standalone_model(weights_path)
    
    print("\nВсе этапы выполнены успешно!")
    
    # Закрываем Ray
    ray.shutdown()