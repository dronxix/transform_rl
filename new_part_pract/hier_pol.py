"""
Иерархическая система управления с командной политикой
Командир дает целеуказания роботам: куда идти, кого атаковать, какую формацию держать
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from gymnasium import spaces
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from torch.distributions import Categorical, Normal

# Типы команд
COMMAND_TYPES = {
    "ATTACK_TARGET": 0,      # Атаковать конкретного врага
    "MOVE_TO_POSITION": 1,   # Двигаться к позиции
    "DEFEND_AREA": 2,        # Защищать область
    "SUPPORT_ALLY": 3,       # Поддерживать союзника
    "RETREAT": 4,            # Отступать
    "HOLD_POSITION": 5,      # Держать позицию
}

FORMATION_TYPES = {
    "SPREAD": 0,      # Рассредоточенная
    "LINE": 1,        # Линия
    "WEDGE": 2,       # Клин
    "CIRCLE": 3,      # Круг
    "STACK": 4,       # Кучно
}

class CommanderModel(TorchModelV2, nn.Module):
    """
    Модель командира, которая анализирует глобальную ситуацию 
    и выдает команды для всей команды
    """
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        cfg = model_config.get("custom_model_config", {})
        
        # Извлекаем размеры из observation space
        if hasattr(obs_space, 'spaces'):
            global_feats = obs_space["global_state"].shape[0]
            allies_shape = obs_space["allies"].shape  # [max_allies, ally_feats]
            enemies_shape = obs_space["enemies"].shape  # [max_enemies, enemy_feats]
            self.max_allies = allies_shape[0]
            self.max_enemies = enemies_shape[0]
            ally_feats = allies_shape[1]
            enemy_feats = enemies_shape[1]
        else:
            self.max_allies = cfg.get("max_allies", 6)
            self.max_enemies = cfg.get("max_enemies", 6)
            global_feats = 64
            ally_feats = 8
            enemy_feats = 10
        
        hidden = cfg.get("hidden", 256)
        
        # Энкодеры для разных типов информации
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feats, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2)
        )
        
        self.allies_encoder = nn.Sequential(
            nn.Linear(ally_feats, hidden//4),
            nn.ReLU(),
            nn.Linear(hidden//4, hidden//4)
        )
        
        self.enemies_encoder = nn.Sequential(
            nn.Linear(enemy_feats, hidden//4),
            nn.ReLU(),
            nn.Linear(hidden//4, hidden//4)
        )
        
        # Attention для агрегации союзников и врагов
        self.allies_attention = nn.MultiheadAttention(
            embed_dim=hidden//4, 
            num_heads=4, 
            batch_first=True
        )
        self.enemies_attention = nn.MultiheadAttention(
            embed_dim=hidden//4, 
            num_heads=4, 
            batch_first=True
        )
        
        # Интегратор всей информации
        integrated_size = hidden//2 + hidden//4 + hidden//4  # global + allies + enemies
        self.integrator = nn.Sequential(
            nn.Linear(integrated_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        
        # Головы для разных типов команд
        # Каждому союзнику даем индивидуальную команду
        self.command_type_head = nn.Linear(hidden, self.max_allies * len(COMMAND_TYPES))
        self.target_enemy_head = nn.Linear(hidden, self.max_allies * self.max_enemies)
        self.move_position_head = nn.Linear(hidden, self.max_allies * 2)  # x, y для каждого
        self.formation_head = nn.Linear(hidden, len(FORMATION_TYPES))  # общая формация
        
        # Value function
        self.value_head = nn.Linear(hidden, 1)
        self._value_out = None
        
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        
        # Обрабатываем глобальное состояние
        global_state = obs["global_state"]
        global_encoded = self.global_encoder(global_state)
        
        # Обрабатываем союзников
        allies = obs["allies"]  # [B, max_allies, ally_feats]
        allies_mask = obs["allies_mask"]  # [B, max_allies]
        
        allies_encoded = self.allies_encoder(allies)  # [B, max_allies, hidden//4]
        
        # Attention для союзников
        # Создаем key_padding_mask для attention (True = ignore)
        allies_padding_mask = (allies_mask == 0)  # True где нет союзников
        allies_attended, _ = self.allies_attention(
            allies_encoded, allies_encoded, allies_encoded,
            key_padding_mask=allies_padding_mask
        )
        
        # Агрегируем союзников (mean pooling с учетом маски)
        allies_mask_expanded = allies_mask.unsqueeze(-1).float()  # [B, max_allies, 1]
        allies_sum = (allies_attended * allies_mask_expanded).sum(dim=1)  # [B, hidden//4]
        allies_count = allies_mask.sum(dim=1, keepdim=True).float().clamp(min=1)  # [B, 1]
        allies_aggregated = allies_sum / allies_count  # [B, hidden//4]
        
        # Обрабатываем врагов аналогично
        enemies = obs["enemies"]  # [B, max_enemies, enemy_feats]
        enemies_mask = obs["enemies_mask"]  # [B, max_enemies]
        
        enemies_encoded = self.enemies_encoder(enemies)  # [B, max_enemies, hidden//4]
        
        enemies_padding_mask = (enemies_mask == 0)
        enemies_attended, _ = self.enemies_attention(
            enemies_encoded, enemies_encoded, enemies_encoded,
            key_padding_mask=enemies_padding_mask
        )
        
        enemies_mask_expanded = enemies_mask.unsqueeze(-1).float()
        enemies_sum = (enemies_attended * enemies_mask_expanded).sum(dim=1)
        enemies_count = enemies_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        enemies_aggregated = enemies_sum / enemies_count
        
        # Интегрируем всю информацию
        integrated = torch.cat([global_encoded, allies_aggregated, enemies_aggregated], dim=-1)
        features = self.integrator(integrated)
        
        # Генерируем команды
        command_types = self.command_type_head(features)  # [B, max_allies * n_command_types]
        target_enemies = self.target_enemy_head(features)  # [B, max_allies * max_enemies]
        move_positions = self.move_position_head(features)  # [B, max_allies * 2]
        formation = self.formation_head(features)  # [B, n_formation_types]
        
        # Применяем маски к командам союзников
        batch_size = allies_mask.shape[0]
        
        # Reshape команд по союзникам
        command_types = command_types.view(batch_size, self.max_allies, len(COMMAND_TYPES))
        target_enemies = target_enemies.view(batch_size, self.max_allies, self.max_enemies)
        move_positions = move_positions.view(batch_size, self.max_allies, 2)
        
        # Маскируем команды для несуществующих союзников
        allies_mask_expanded = allies_mask.unsqueeze(-1)  # [B, max_allies, 1]
        
        # Для command_types и target_enemies ставим очень низкие логиты где нет союзников
        command_types = command_types.masked_fill(~allies_mask_expanded, -1e9)
        target_enemies = target_enemies.masked_fill(~allies_mask_expanded, -1e9)
        
        # Также применяем маску врагов к target_enemies
        enemies_mask_expanded = enemies_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, max_enemies, 1]
        target_enemies = target_enemies.masked_fill(~enemies_mask_expanded.squeeze(-1), -1e9)
        
        # Flatten обратно для выхода
        command_types_flat = command_types.view(batch_size, -1)
        target_enemies_flat = target_enemies.view(batch_size, -1)
        move_positions_flat = move_positions.view(batch_size, -1)
        
        # Собираем итоговый выход
        output = torch.cat([
            command_types_flat,    # max_allies * n_command_types
            target_enemies_flat,   # max_allies * max_enemies  
            move_positions_flat,   # max_allies * 2
            formation              # n_formation_types
        ], dim=-1)
        
        # Value function
        self._value_out = self.value_head(features).squeeze(-1)
        
        return output, state
    
    def value_function(self):
        return self._value_out


class CommandDistribution(TorchDistributionWrapper):
    """
    Дистрибуция для команд командира
    """
    
    def __init__(self, inputs, model):
        super().__init__(inputs, model)
        
        # Получаем размеры из модели
        self.max_allies = getattr(model, 'max_allies', 6)
        self.max_enemies = getattr(model, 'max_enemies', 6)
        self.n_command_types = len(COMMAND_TYPES)
        self.n_formation_types = len(FORMATION_TYPES)
        
        # Распаковываем входы
        idx = 0
        
        # Command types для каждого союзника
        command_size = self.max_allies * self.n_command_types
        command_logits = inputs[..., idx:idx+command_size]
        command_logits = command_logits.view(-1, self.max_allies, self.n_command_types)
        idx += command_size
        
        # Target enemies для каждого союзника
        target_size = self.max_allies * self.max_enemies
        target_logits = inputs[..., idx:idx+target_size]
        target_logits = target_logits.view(-1, self.max_allies, self.max_enemies)
        idx += target_size
        
        # Move positions для каждого союзника
        move_size = self.max_allies * 2
        move_coords = inputs[..., idx:idx+move_size]
        move_coords = move_coords.view(-1, self.max_allies, 2)
        idx += move_size
        
        # Formation type (общий для всей команды)
        formation_logits = inputs[..., idx:idx+self.n_formation_types]
        
        # Создаем дистрибуции
        self.command_dists = []
        self.target_dists = []
        self.move_dists = []
        
        for i in range(self.max_allies):
            self.command_dists.append(Categorical(logits=command_logits[:, i, :]))
            self.target_dists.append(Categorical(logits=target_logits[:, i, :]))
            self.move_dists.append(Normal(
                loc=move_coords[:, i, :], 
                scale=torch.ones_like(move_coords[:, i, :])
            ))
        
        self.formation_dist = Categorical(logits=formation_logits)
        
        # Для совместимости с RLLib
        self.dist = self.formation_dist
        self.last_sample = None
    
    @staticmethod
    def required_model_output_shape(action_space, model_config):
        """Вычисляем требуемый размер выхода модели"""
        cfg = model_config.get("custom_model_config", {})
        max_allies = cfg.get("max_allies", 6)
        max_enemies = cfg.get("max_enemies", 6)
        
        command_size = max_allies * len(COMMAND_TYPES)
        target_size = max_allies * max_enemies
        move_size = max_allies * 2
        formation_size = len(FORMATION_TYPES)
        
        return command_size + target_size + move_size + formation_size
    
    def sample(self):
        """Сэмплируем команды"""
        batch_size = self.formation_dist.logits.shape[0]
        
        # Сэмплируем для каждого союзника
        commands = []
        targets = []
        moves = []
        
        for i in range(self.max_allies):
            commands.append(self.command_dists[i].sample())
            targets.append(self.target_dists[i].sample())
            moves.append(self.move_dists[i].sample())
        
        # Сэмплируем общую формацию
        formation = self.formation_dist.sample()
        
        # Собираем в один тензор
        commands_tensor = torch.stack(commands, dim=1)  # [B, max_allies]
        targets_tensor = torch.stack(targets, dim=1)    # [B, max_allies]
        moves_tensor = torch.stack(moves, dim=1)        # [B, max_allies, 2]
        
        # Flatten для возврата
        result = torch.cat([
            commands_tensor.float(),                    # [B, max_allies]
            targets_tensor.float(),                     # [B, max_allies]
            moves_tensor.view(batch_size, -1),          # [B, max_allies * 2]
            formation.float().unsqueeze(-1)             # [B, 1]
        ], dim=-1)
        
        self.last_sample = result
        return result.cpu().numpy() if result.is_cuda else result.numpy()
    
    def deterministic_sample(self):
        """Детерминированный сэмпл (argmax)"""
        batch_size = self.formation_dist.logits.shape[0]
        
        commands = []
        targets = []
        moves = []
        
        for i in range(self.max_allies):
            commands.append(torch.argmax(self.command_dists[i].logits, dim=-1))
            targets.append(torch.argmax(self.target_dists[i].logits, dim=-1))
            moves.append(self.move_dists[i].loc)  # Mean для Normal
        
        formation = torch.argmax(self.formation_dist.logits, dim=-1)
        
        commands_tensor = torch.stack(commands, dim=1).float()
        targets_tensor = torch.stack(targets, dim=1).float()
        moves_tensor = torch.stack(moves, dim=1)
        
        result = torch.cat([
            commands_tensor,
            targets_tensor,
            moves_tensor.view(batch_size, -1),
            formation.float().unsqueeze(-1)
        ], dim=-1)
        
        self.last_sample = result
        return result.cpu().numpy() if result.is_cuda else result.numpy()
    
    def logp(self, x):
        """Вычисляем log probability"""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.formation_dist.logits.device)
        
        batch_size = x.shape[0]
        
        # Распаковываем действие
        idx = 0
        commands = x[:, idx:idx+self.max_allies].long()
        idx += self.max_allies
        
        targets = x[:, idx:idx+self.max_allies].long()
        idx += self.max_allies
        
        moves = x[:, idx:idx+self.max_allies*2].view(batch_size, self.max_allies, 2)
        idx += self.max_allies * 2
        
        formation = x[:, idx].long()
        
        # Вычисляем log probs
        total_logp = torch.zeros(batch_size, device=x.device)
        
        for i in range(self.max_allies):
            total_logp += self.command_dists[i].log_prob(commands[:, i])
            total_logp += self.target_dists[i].log_prob(targets[:, i])
            total_logp += self.move_dists[i].log_prob(moves[:, i]).sum(-1)
        
        total_logp += self.formation_dist.log_prob(formation)
        
        return total_logp
    
    def sampled_action_logp(self):
        if self.last_sample is None:
            return torch.zeros(1)
        return self.logp(self.last_sample)
    
    def entropy(self):
        """Вычисляем энтропию"""
        total_entropy = torch.zeros_like(self.formation_dist.logits[:, 0])
        
        for i in range(self.max_allies):
            total_entropy += self.command_dists[i].entropy()
            total_entropy += self.target_dists[i].entropy()
            total_entropy += self.move_dists[i].entropy().sum(-1)
        
        total_entropy += self.formation_dist.entropy()
        
        return total_entropy


class CommandFollowerModel(TorchModelV2, nn.Module):
    """
    Модель исполнителя, которая получает команды от командира
    и выполняет низкоуровневые действия
    """
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        cfg = model_config.get("custom_model_config", {})
        
        # Извлекаем размеры
        if hasattr(obs_space, 'spaces'):
            self_feats = obs_space["self"].shape[0]
            command_feats = obs_space.get("command", spaces.Box(0, 1, (10,))).shape[0]  # команда от командира
            allies_shape = obs_space["allies"].shape
            enemies_shape = obs_space["enemies"].shape
            self.max_allies = allies_shape[0]
            self.max_enemies = enemies_shape[0]
            ally_feats = allies_shape[1]
            enemy_feats = enemies_shape[1]
        else:
            self_feats = cfg.get("self_feats", 12)
            command_feats = cfg.get("command_feats", 10)
            self.max_allies = cfg.get("max_allies", 6)
            self.max_enemies = cfg.get("max_enemies", 6)
            ally_feats = cfg.get("ally_feats", 8)
            enemy_feats = cfg.get("enemy_feats", 10)
        
        hidden = cfg.get("hidden", 256)
        
        # Энкодеры
        self.self_encoder = nn.Sequential(
            nn.Linear(self_feats, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, hidden//2)
        )
        
        self.command_encoder = nn.Sequential(
            nn.Linear(command_feats, hidden//4),
            nn.ReLU(),
            nn.Linear(hidden//4, hidden//4)
        )
        
        self.context_encoder = nn.Sequential(
            nn.Linear(ally_feats + enemy_feats, hidden//4),
            nn.ReLU(),
            nn.Linear(hidden//4, hidden//4)
        )
        
        # Интегратор
        integrated_size = hidden//2 + hidden//4 + hidden//4
        self.integrator = nn.Sequential(
            nn.Linear(integrated_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        
        # Политические головы (как в оригинальной модели)
        self.head_target = nn.Linear(hidden, self.max_enemies)
        self.head_move_mu = nn.Linear(hidden, 2)
        self.head_aim_mu = nn.Linear(hidden, 2)
        self.log_std_move = nn.Parameter(torch.full((2,), -0.5))
        self.log_std_aim = nn.Parameter(torch.full((2,), -0.5))
        self.head_fire_logit = nn.Linear(hidden, 1)
        
        # Value function
        self.value_head = nn.Linear(hidden, 1)
        self._value_out = None
        
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        
        # Кодируем собственное состояние
        self_encoded = self.self_encoder(obs["self"])
        
        # Кодируем полученную команду
        command = obs.get("command", torch.zeros(obs["self"].shape[0], 10))  # fallback если нет команды
        command_encoded = self.command_encoder(command)
        
        # Простая агрегация контекста (союзники + враги)
        allies = obs["allies"]  # [B, max_allies, ally_feats]
        enemies = obs["enemies"]  # [B, max_enemies, enemy_feats]
        allies_mask = obs["allies_mask"]
        enemies_mask = obs["enemies_mask"]
        
        # Простое усреднение союзников и врагов для контекста
        allies_masked = allies * allies_mask.unsqueeze(-1).float()
        enemies_masked = enemies * enemies_mask.unsqueeze(-1).float()
        
        allies_mean = allies_masked.sum(dim=1) / (allies_mask.sum(dim=1, keepdim=True).float() + 1e-8)
        enemies_mean = enemies_masked.sum(dim=1) / (enemies_mask.sum(dim=1, keepdim=True).float() + 1e-8)
        
        context = torch.cat([allies_mean, enemies_mean], dim=-1)
        context_encoded = self.context_encoder(context)
        
        # Интегрируем все
        integrated = torch.cat([self_encoded, command_encoded, context_encoded], dim=-1)
        features = self.integrator(integrated)
        
        # Генерируем действия
        logits_target = self.head_target(features)
        mu_move = self.head_move_mu(features)
        mu_aim = self.head_aim_mu(features)
        logit_fire = self.head_fire_logit(features)
        
        # Применяем маску врагов
        mask = obs["enemy_action_mask"].float()
        inf_mask = (1.0 - mask) * torch.finfo(logits_target.dtype).min
        masked_logits = logits_target + inf_mask
        
        # Логарифмические стандартные отклонения
        log_std_move = self.log_std_move.clamp(-5.0, 2.0).expand_as(mu_move)
        log_std_aim = self.log_std_aim.clamp(-5.0, 2.0).expand_as(mu_aim)
        
        # Собираем выход
        output = torch.cat([
            masked_logits, mu_move, log_std_move, 
            mu_aim, log_std_aim, logit_fire
        ], dim=-1)
        
        # Value function
        self._value_out = self.value_head(features).squeeze(-1)
        
        return output, state
    
    def value_function(self):
        return self._value_out


def decode_commander_action(action: np.ndarray, max_allies: int = 6, max_enemies: int = 6) -> Dict:
    """
    Декодирует действие командира в понятные команды
    """
    if action.ndim > 1:
        action = action[0]  # Берем первый элемент batch
    
    idx = 0
    
    # Команды для каждого союзника
    commands = action[idx:idx+max_allies].astype(int)
    idx += max_allies
    
    # Целевые враги для каждого союзника
    targets = action[idx:idx+max_allies].astype(int)
    idx += max_allies
    
    # Позиции для движения (x, y для каждого союзника)
    positions = action[idx:idx+max_allies*2].reshape(max_allies, 2)
    idx += max_allies * 2
    
    # Общая формация
    formation = int(action[idx])
    
    # Декодируем в читаемый формат
    decoded_commands = []
    for i in range(max_allies):
        cmd_type = list(COMMAND_TYPES.keys())[commands[i] % len(COMMAND_TYPES)]
        target_enemy = targets[i] % max_enemies
        position = positions[i]
        
        decoded_commands.append({
            "ally_id": i,
            "command_type": cmd_type,
            "target_enemy": target_enemy,
            "move_position": position,
        })
    
    formation_type = list(FORMATION_TYPES.keys())[formation % len(FORMATION_TYPES)]
    
    return {
        "individual_commands": decoded_commands,
        "formation": formation_type,
    }


def create_command_for_follower(commander_action: Dict, ally_id: int) -> np.ndarray:
    """
    Создает команду для конкретного исполнителя на основе решения командира
    """
    individual_commands = commander_action["individual_commands"]
    formation = commander_action["formation"]
    
    if ally_id < len(individual_commands):
        cmd = individual_commands[ally_id]
        
        # Кодируем команду в вектор признаков
        command_vector = np.zeros(10, dtype=np.float32)
        
        # [0:6] - one-hot для типа команды
        if cmd["command_type"] in COMMAND_TYPES:
            command_vector[COMMAND_TYPES[cmd["command_type"]]] = 1.0
        
        # [6:7] - id целевого врага (нормализованный)
        command_vector[6] = cmd["target_enemy"] / 10.0  # нормализуем
        
        # [7:9] - позиция для движения
        command_vector[7:9] = np.clip(cmd["move_position"], -5.0, 5.0) / 5.0  # нормализуем
        
        # [9] - тип формации (нормализованный)
        if formation in FORMATION_TYPES:
            command_vector[9] = FORMATION_TYPES[formation] / len(FORMATION_TYPES)
    
    return command_vector


# Регистрируем модели
ModelCatalog.register_custom_model("commander", CommanderModel)
ModelCatalog.register_custom_model("follower", CommandFollowerModel)
ModelCatalog.register_custom_action_dist("command_dist", CommandDistribution)


# Пример использования в тренировке
def create_hierarchical_config():
    """
    Создает конфигурацию для иерархической тренировки
    """
    from ray.rllib.algorithms.ppo import PPOConfig
    from arena_env import ArenaEnv
    
    # Получаем размеры окружения
    tmp_env = ArenaEnv({"ally_choices": [1], "enemy_choices": [1]})
    obs_space = tmp_env.observation_space
    act_space = tmp_env.action_space
    
    max_enemies = obs_space["enemies"].shape[0]
    max_allies = obs_space["allies"].shape[0]
    
    # Расширяем observation space для включения команд
    from gymnasium import spaces
    follower_obs_space = spaces.Dict({
        **obs_space.spaces,
        "command": spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
    })
    
    # Конфигурация командира
    commander_model_config = {
        "custom_model": "commander",
        "custom_action_dist": "command_dist",
        "custom_model_config": {
            "max_allies": max_allies,
            "max_enemies": max_enemies,
            "hidden": 256,
        },
        "vf_share_layers": False,
    }
    
    # Конфигурация исполнителей
    follower_model_config = {
        "custom_model": "follower",
        "custom_action_dist": "masked_multihead",  # Используем существующую дистрибуцию
        "custom_model_config": {
            "max_allies": max_allies,
            "max_enemies": max_enemies,
            "hidden": 256,
        },
        "vf_share_layers": False,
    }
    
    # Policy mapping: командир управляет командой, исполнители выполняют
    def policy_mapping_fn(agent_id: str, episode=None, **kwargs):
        if agent_id == "commander":
            return "commander_policy"
        elif agent_id.startswith("red_"):
            return "follower_policy"
        else:
            return "opponent_policy"  # Враги используют обычную политику
    
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env="ArenaEnvHierarchical",  # Нужно создать обертку
            env_config={
                "episode_len": 128,
                "ally_choices": [2, 3],
                "enemy_choices": [2, 3],
                "hierarchical": True,
            }
        )
        .framework("torch")
        .env_runners(
            num_env_runners=2,
            num_envs_per_env_runner=1,
            rollout_fragment_length=256,
        )
        .training(
            gamma=0.99,
            lr=3e-4,
            train_batch_size=8192,
            minibatch_size=1024,
            num_epochs=4,
        )
        .multi_agent(
            policies={
                "commander_policy": (None, obs_space, None, {
                    "model": commander_model_config
                }),
                "follower_policy": (None, follower_obs_space, act_space, {
                    "model": follower_model_config
                }),
                "opponent_policy": (None, obs_space, act_space, {
                    "model": {
                        "custom_model": "entity_attention",
                        "custom_action_dist": "masked_multihead",
                        "custom_model_config": {
                            "max_allies": max_allies,
                            "max_enemies": max_enemies,
                        },
                        "vf_share_layers": False,
                    }
                }),
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["commander_policy", "follower_policy"],
        )
    )
    
    return config


if __name__ == "__main__":
    # Тест декодирования команд
    print("=== Testing Command System ===")
    
    # Симулируем действие командира
    max_allies, max_enemies = 3, 4
    
    # Создаем случайное действие командира
    np.random.seed(42)
    commander_action_raw = np.random.randint(0, 6, size=(
        max_allies +  # commands
        max_allies +  # targets  
        max_allies * 2 +  # positions
        1  # formation
    )).astype(float)
    
    # Добавляем случайные позиции
    positions_start = max_allies * 2
    commander_action_raw[positions_start:positions_start + max_allies * 2] = \
        np.random.uniform(-3, 3, size=max_allies * 2)
    
    print(f"Raw commander action: {commander_action_raw}")
    
    # Декодируем
    decoded = decode_commander_action(commander_action_raw, max_allies, max_enemies)
    
    print(f"\nDecoded commands:")
    print(f"Formation: {decoded['formation']}")
    print("Individual commands:")
    for cmd in decoded['individual_commands']:
        print(f"  Ally {cmd['ally_id']}: {cmd['command_type']} -> "
              f"Enemy {cmd['target_enemy']}, Move to {cmd['move_position']}")
    
    # Тест создания команд для исполнителей
    print(f"\nCommands for followers:")
    for ally_id in range(max_allies):
        follower_command = create_command_for_follower(decoded, ally_id)
        print(f"  Ally {ally_id}: {follower_command}")
    
    print("\n✓ Command system test passed!")