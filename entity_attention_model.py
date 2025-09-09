"""
ONNX-совместимая версия EntityAttentionModel для универсальных actions/obs
Автоматически адаптируется к любым форматам действий и наблюдений
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Union
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
import math
from collections import OrderedDict

class DynamicActionConfig:
    """Конфигурация для автоматического определения структуры действий"""
    
    def __init__(self, action_space=None, model_config=None):
        self.action_space = action_space
        self.model_config = model_config or {}
        self.action_spec = self._analyze_action_space()
        
    def _analyze_action_space(self) -> Dict[str, Any]:
        """Анализирует action_space и создает универсальную спецификацию"""
        if self.action_space is None:
            # Дефолтная конфигурация
            return {
                "discrete_actions": {"target": 6},
                "continuous_actions": {"move": 3, "aim": 3},
                "binary_actions": {"fire": 1},
                "total_output_size": 6 + 3 + 3 + 3 + 3 + 1  # target + move + move_std + aim + aim_std + fire
            }
        
        spec = {
            "discrete_actions": {},
            "continuous_actions": {},
            "binary_actions": {},
            "total_output_size": 0
        }
        
        if hasattr(self.action_space, 'spaces'):
            # Dict action space
            for name, space in self.action_space.spaces.items():
                if hasattr(space, 'n'):
                    # Discrete space
                    spec["discrete_actions"][name] = space.n
                    spec["total_output_size"] += space.n
                elif hasattr(space, 'shape'):
                    # Continuous space
                    action_dim = space.shape[0] if space.shape else 1
                    if name in ["fire", "shoot", "attack"] or action_dim == 1:
                        spec["binary_actions"][name] = 1
                        spec["total_output_size"] += 1
                    else:
                        spec["continuous_actions"][name] = action_dim
                        spec["total_output_size"] += action_dim * 2  # mu + log_std
        else:
            # Single action space - используем дефолт
            pass
            
        return spec

class DynamicObservationProcessor:
    """Процессор для автоматической обработки различных форматов наблюдений"""
    
    def __init__(self, obs_space=None, model_config=None):
        self.obs_space = obs_space
        self.model_config = model_config or {}
        self.obs_spec = self._analyze_observation_space()
        
    def _analyze_observation_space(self) -> Dict[str, Any]:
        """Анализирует observation_space и создает спецификацию"""
        if self.obs_space is None:
            return self._default_obs_spec()
            
        spec = {
            "self_features": 13,  # Дефолт для 3D
            "ally_features": 9,
            "enemy_features": 11,
            "global_features": 64,
            "max_allies": 6,
            "max_enemies": 6,
            "additional_features": {}
        }
        
        if hasattr(self.obs_space, 'spaces'):
            for name, space in self.obs_space.spaces.items():
                if name == "self":
                    spec["self_features"] = space.shape[0]
                elif name == "allies":
                    spec["max_allies"] = space.shape[0]
                    spec["ally_features"] = space.shape[1]
                elif name == "enemies":
                    spec["max_enemies"] = space.shape[0]
                    spec["enemy_features"] = space.shape[1]
                elif name == "global_state":
                    spec["global_features"] = space.shape[0]
                elif name not in ["allies_mask", "enemies_mask", "enemy_action_mask"]:
                    # Дополнительные признаки
                    spec["additional_features"][name] = space.shape
                    
        return spec
    
    def _default_obs_spec(self):
        """Дефолтная спецификация наблюдений для обратной совместимости"""
        return {
            "self_features": 13,
            "ally_features": 9,
            "enemy_features": 11,
            "global_features": 64,
            "max_allies": 6,
            "max_enemies": 6,
            "additional_features": {}
        }

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        
    def forward(self, x):
        pe = self.pe.to(x.device)
        return x + pe[:, :x.size(1), :]

class MLP(nn.Module):
    def __init__(self, dims: List[int], act=nn.GELU, out_act=None):
        super().__init__()
        layers = []
        for i in range(len(dims)-1):
            layers += [nn.Linear(dims[i], dims[i+1])]
            if i < len(dims)-2:
                layers += [act()]
        if out_act is not None:
            layers += [out_act()]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x): 
        return self.net(x)

class ONNXCompatibleMultiHeadAttention(nn.Module):
    """ONNX-совместимая реализация multi-head attention"""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query, key, value, key_padding_mask=None, need_weights=False):
        batch_size, seq_len, d_model = query.size()
        
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        Q = Q.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        
        output = self.w_o(attn_output)
        
        if need_weights:
            avg_weights = attn_weights.mean(dim=1)
            return output, avg_weights
        else:
            return output, None

class ONNXCompatibleAttnBlock(nn.Module):
    """ONNX-совместимый блок attention"""
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.0):
        super().__init__()
        self.mha = ONNXCompatibleMultiHeadAttention(d_model, nhead, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), 
            nn.GELU(), 
            nn.Linear(dim_ff, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.last_attn = None
        
    def forward(self, x, key_padding_mask):
        attn_out, attn_w = self.mha(x, x, x, key_padding_mask=key_padding_mask, need_weights=True)
        self.last_attn = attn_w
        x = self.ln1(x + attn_out)
        
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        
        return x

class UniversalActionHead(nn.Module):
    """Универсальная голова для любых типов действий"""
    
    def __init__(self, d_model: int, action_config: DynamicActionConfig, hidden: int = 256):
        super().__init__()
        self.action_config = action_config
        self.action_spec = action_config.action_spec
        
        # Создаем головы для разных типов действий
        self.discrete_heads = nn.ModuleDict()
        self.continuous_heads = nn.ModuleDict()
        self.continuous_logstd = nn.ParameterDict()
        self.binary_heads = nn.ModuleDict()
        
        # Дискретные действия
        for name, n_classes in self.action_spec["discrete_actions"].items():
            self.discrete_heads[name] = MLP([d_model, hidden, n_classes])
        
        # Непрерывные действия
        for name, action_dim in self.action_spec["continuous_actions"].items():
            self.continuous_heads[name] = MLP([d_model, hidden, action_dim])
            self.continuous_logstd[name] = nn.Parameter(torch.full((action_dim,), -0.5))
        
        # Бинарные действия
        for name, _ in self.action_spec["binary_actions"].items():
            self.binary_heads[name] = MLP([d_model, hidden, 1])
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Возвращает словарь с логитами для каждого действия"""
        outputs = {}
        
        # Дискретные действия
        for name, head in self.discrete_heads.items():
            outputs[f"{name}_logits"] = head(x)
        
        # Непрерывные действия
        for name, head in self.continuous_heads.items():
            mu = head(x)
            log_std = self.continuous_logstd[name].clamp(-5.0, 2.0).expand_as(mu)
            outputs[f"{name}_mu"] = mu
            outputs[f"{name}_log_std"] = log_std
        
        # Бинарные действия
        for name, head in self.binary_heads.items():
            outputs[f"{name}_logit"] = head(x)
        
        return outputs

class UniversalObservationEncoder(nn.Module):
    """Универсальный энкодер для различных форматов наблюдений"""
    
    def __init__(self, obs_processor: DynamicObservationProcessor, d_model: int = 128):
        super().__init__()
        self.obs_processor = obs_processor
        self.obs_spec = obs_processor.obs_spec
        self.d_model = d_model
        
        # Основные энкодеры
        self.self_enc = MLP([self.obs_spec["self_features"], d_model])
        self.ally_enc = MLP([self.obs_spec["ally_features"], d_model])
        self.enemy_enc = MLP([self.obs_spec["enemy_features"], d_model])
        
        # Энкодеры для дополнительных признаков
        self.additional_encoders = nn.ModuleDict()
        for name, shape in self.obs_spec["additional_features"].items():
            input_size = np.prod(shape) if isinstance(shape, tuple) else shape
            self.additional_encoders[name] = MLP([input_size, d_model])
    
    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Кодирует наблюдения в токены"""
        encoded = {}
        
        # Основные компоненты
        if "self" in obs:
            encoded["self"] = self.self_enc(obs["self"])
        
        if "allies" in obs:
            encoded["allies"] = self.ally_enc(obs["allies"])
        
        if "enemies" in obs:
            encoded["enemies"] = self.enemy_enc(obs["enemies"])
        
        # Дополнительные компоненты
        for name, encoder in self.additional_encoders.items():
            if name in obs:
                # Flatten если нужно
                x = obs[name]
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)
                encoded[name] = encoder(x)
        
        return encoded

class ONNXEntityAttentionModel(TorchModelV2, nn.Module):
    """Универсальная ONNX-совместимая модель для любых action/obs форматов"""
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        cfg = model_config.get("custom_model_config", {})
        d_model = int(cfg.get("d_model", 160))
        nhead = int(cfg.get("nhead", 8))
        layers = int(cfg.get("layers", 2))
        ff = int(cfg.get("ff", 320))
        hidden = int(cfg.get("hidden", 256))
        self.logstd_min = float(cfg.get("logstd_min", -5.0))
        self.logstd_max = float(cfg.get("logstd_max", 2.0))

        # Создаем конфигурации для универсальной обработки
        self.action_config = DynamicActionConfig(action_space, model_config)
        self.obs_processor = DynamicObservationProcessor(obs_space, model_config)
        
        print(f"🔧 Universal Model Configuration:")
        print(f"   Actions: {self.action_config.action_spec}")
        print(f"   Observations: {self.obs_processor.obs_spec}")

        # Сохраняем для экспорта
        self.d_model = d_model
        self.nhead = nhead
        self.layers = layers
        self.max_allies = self.obs_processor.obs_spec["max_allies"]
        self.max_enemies = self.obs_processor.obs_spec["max_enemies"]

        # Универсальный энкодер наблюдений
        self.obs_encoder = UniversalObservationEncoder(self.obs_processor, d_model)
        
        # Attention система
        max_seq_len = max(self.max_allies + self.max_enemies + 1, 64)
        self.posenc = PositionalEncoding(d_model, max_len=max_seq_len)
        self.blocks = nn.ModuleList([
            ONNXCompatibleAttnBlock(d_model, nhead, ff) for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.last_attn = None

        # Универсальная голова действий
        self.action_head = UniversalActionHead(d_model, self.action_config, hidden)
        
        # Value function - адаптируется к глобальным признакам
        global_feats = self.obs_processor.obs_spec["global_features"]
        additional_feats = sum(
            np.prod(shape) if isinstance(shape, tuple) else shape 
            for shape in self.obs_processor.obs_spec["additional_features"].values()
        )
        total_value_feats = global_feats + additional_feats
        
        self.value_net = MLP([total_value_feats, hidden, 1])
        self._value_out: Optional[torch.Tensor] = None

    def _ensure_tensor_device(self, tensor, target_device):
        """Безопасное перемещение тензора на нужное устройство"""
        if isinstance(tensor, torch.Tensor):
            return tensor.to(target_device)
        elif isinstance(tensor, np.ndarray):
            return torch.from_numpy(tensor).to(target_device)
        else:
            return torch.tensor(tensor).to(target_device)

    def _ensure_obs_device_consistency(self, obs):
        """Убеждаемся что все наблюдения на одном устройстве"""
        target_device = next(self.parameters()).device
        
        obs_fixed = {}
        for key, value in obs.items():
            if isinstance(value, (torch.Tensor, np.ndarray)):
                obs_fixed[key] = self._ensure_tensor_device(value, target_device)
                
                if key in ["allies_mask", "enemies_mask", "enemy_action_mask"]:
                    obs_fixed[key] = obs_fixed[key].long()
                else:
                    obs_fixed[key] = obs_fixed[key].float()
            else:
                obs_fixed[key] = value
                
        return obs_fixed, target_device

    def _build_attention_sequence(self, encoded_obs: Dict[str, torch.Tensor], obs: Dict[str, torch.Tensor], target_device):
        """Строит последовательность для attention с учетом всех типов наблюдений"""
        sequence_parts = []
        mask_parts = []
        batch_size = next(iter(encoded_obs.values())).size(0)
        
        # Self токен (всегда первый)
        if "self" in encoded_obs:
            sequence_parts.append(encoded_obs["self"].unsqueeze(1))
            mask_parts.append(torch.zeros(batch_size, 1, dtype=torch.bool, device=target_device))
        
        # Allies
        if "allies" in encoded_obs:
            sequence_parts.append(encoded_obs["allies"])
            allies_mask = obs.get("allies_mask", torch.ones(batch_size, self.max_allies, device=target_device))
            mask_parts.append(~(allies_mask > 0))
        
        # Enemies
        if "enemies" in encoded_obs:
            sequence_parts.append(encoded_obs["enemies"])
            enemies_mask = obs.get("enemies_mask", torch.ones(batch_size, self.max_enemies, device=target_device))
            mask_parts.append(~(enemies_mask > 0))
        
        # Дополнительные наблюдения как отдельные токены
        for name, encoded_feat in encoded_obs.items():
            if name not in ["self", "allies", "enemies"]:
                # Добавляем как single token
                if encoded_feat.dim() == 2:  # [B, D]
                    sequence_parts.append(encoded_feat.unsqueeze(1))
                    mask_parts.append(torch.zeros(batch_size, 1, dtype=torch.bool, device=target_device))
                elif encoded_feat.dim() == 3:  # [B, L, D]
                    sequence_parts.append(encoded_feat)
                    seq_len = encoded_feat.size(1)
                    mask_parts.append(torch.zeros(batch_size, seq_len, dtype=torch.bool, device=target_device))
        
        # Объединяем
        if sequence_parts:
            x = torch.cat(sequence_parts, dim=1)
            pad_mask = torch.cat(mask_parts, dim=1)
        else:
            # Fallback - создаем пустую последовательность
            x = torch.zeros(batch_size, 1, self.d_model, device=target_device)
            pad_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=target_device)
        
        return x, pad_mask

    def _flatten_action_outputs(self, action_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Преобразует словарь выходов действий в плоский тензор для совместимости с RLLib"""
        flat_parts = []
        
        # Собираем в определенном порядке для консистентности
        # 1. Дискретные действия
        for name in sorted(self.action_config.action_spec["discrete_actions"].keys()):
            if f"{name}_logits" in action_outputs:
                flat_parts.append(action_outputs[f"{name}_logits"])
        
        # 2. Непрерывные действия (mu, затем log_std)
        for name in sorted(self.action_config.action_spec["continuous_actions"].keys()):
            if f"{name}_mu" in action_outputs:
                flat_parts.append(action_outputs[f"{name}_mu"])
                flat_parts.append(action_outputs[f"{name}_log_std"])
        
        # 3. Бинарные действия
        for name in sorted(self.action_config.action_spec["binary_actions"].keys()):
            if f"{name}_logit" in action_outputs:
                flat_parts.append(action_outputs[f"{name}_logit"])
        
        if flat_parts:
            return torch.cat(flat_parts, dim=-1)
        else:
            # Fallback - возвращаем пустой тензор
            batch_size = 1  # Попытаемся определить из context
            return torch.zeros(batch_size, 1, device=next(self.parameters()).device)

    def forward(self, input_dict, state, seq_lens):
        raw_obs = input_dict["obs"]
        
        if isinstance(raw_obs, dict):
            obs = raw_obs
        else:
            obs = dict(raw_obs)
        
        # Убеждаемся что все тензоры на правильном устройстве
        obs, target_device = self._ensure_obs_device_consistency(obs)
        
        try:
            # Кодируем наблюдения
            encoded_obs = self.obs_encoder(obs)
            
            # Строим последовательность для attention
            x, pad_mask = self._build_attention_sequence(encoded_obs, obs, target_device)
            
            # Positional encoding + attention блоки
            x = self.posenc(x)
            for blk in self.blocks:
                x = blk(x, key_padding_mask=pad_mask)
            self.last_attn = self.blocks[-1].last_attn if self.blocks else None
            x = self.norm(x)

            # Self-токен как агрегат (первый токен)
            h = x[:, 0, :]

            # Генерируем действия через универсальную голову
            action_outputs = self.action_head(h)
            
            # Применяем маски для дискретных действий если есть
            if "target_logits" in action_outputs and "enemy_action_mask" in obs:
                mask = obs["enemy_action_mask"].float()
                inf_mask = (1.0 - mask) * torch.finfo(action_outputs["target_logits"].dtype).min
                action_outputs["target_logits"] = action_outputs["target_logits"] + inf_mask

            # Преобразуем в плоский формат для RLLib
            out = self._flatten_action_outputs(action_outputs)

            # Value function - собираем все доступные глобальные признаки
            value_inputs = []
            if "global_state" in obs:
                value_inputs.append(obs["global_state"])
            
            # Добавляем дополнительные признаки для value function
            for name in self.obs_processor.obs_spec["additional_features"]:
                if name in obs and name != "global_state":
                    feat = obs[name]
                    if feat.dim() > 2:
                        feat = feat.view(feat.size(0), -1)
                    value_inputs.append(feat)
            
            if value_inputs:
                value_input = torch.cat(value_inputs, dim=-1)
            else:
                # Fallback - используем self признаки
                value_input = obs.get("self", torch.zeros(h.size(0), 1, device=target_device))
            
            v = self.value_net(value_input).squeeze(-1)
            self._value_out = v
            
            return out, state
            
        except Exception as e:
            print(f"ERROR in Universal model forward: {e}")
            print(f"Observation keys: {list(obs.keys())}")
            print(f"Observation shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in obs.items()]}")
            raise

    def value_function(self):
        return self._value_out

    def get_action_spec(self):
        """Возвращает спецификацию действий для дистрибуции"""
        return self.action_config.action_spec

    def get_obs_spec(self):
        """Возвращает спецификацию наблюдений"""
        return self.obs_processor.obs_spec

# Регистрируем универсальную модель
ModelCatalog.register_custom_model("entity_attention", ONNXEntityAttentionModel)
ModelCatalog.register_custom_model("onnx_entity_attention", ONNXEntityAttentionModel)
ModelCatalog.register_custom_model("universal_entity_attention", ONNXEntityAttentionModel)