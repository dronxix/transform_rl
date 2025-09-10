"""
ONNX-совместимая версия EntityAttentionModel для 3D пространства
Заменяет nn.MultiheadAttention на кастомную реализацию
Обновлено для работы с 3D координатами и действиями
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        
    def forward(self, x):  # [B,L,D]
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
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
                
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
        
        # Отдельные проекции вместо in_proj
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query, key, value, key_padding_mask=None, need_weights=False):
        batch_size, seq_len, d_model = query.size()
        
        # Проекции Q, K, V
        Q = self.w_q(query)  # [B, L, D]
        K = self.w_k(key)    # [B, L, D]
        V = self.w_v(value)  # [B, L, D]
        
        # Reshape для multi-head: [B, L, D] -> [B, H, L, D/H]
        Q = Q.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, L, L]
        
        # Применяем маску если есть
        if key_padding_mask is not None:
            # key_padding_mask: [B, L], True = ignore
            # Расширяем для всех голов: [B, 1, 1, L]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)  # [B, H, L, L]
        attn_weights = self.dropout(attn_weights)
        
        # Применяем внимание к values
        attn_output = torch.matmul(attn_weights, V)  # [B, H, L, D/H]
        
        # Concat heads: [B, H, L, D/H] -> [B, L, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        
        # Финальная проекция
        output = self.w_o(attn_output)
        
        if need_weights:
            # Усредняем веса по головам для совместимости
            avg_weights = attn_weights.mean(dim=1)  # [B, L, L]
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
        # Self-attention
        attn_out, attn_w = self.mha(x, x, x, key_padding_mask=key_padding_mask, need_weights=True)
        self.last_attn = attn_w
        x = self.ln1(x + attn_out)
        
        # Feed-forward
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        
        return x

class ONNXEntityAttentionModel(TorchModelV2, nn.Module):
    """ONNX-совместимая версия EntityAttentionModel для 3D пространства"""
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        cfg = model_config.get("custom_model_config", {})
        d_model = int(cfg.get("d_model", 160))
        nhead   = int(cfg.get("nhead", 8))
        layers  = int(cfg.get("layers", 2))
        ff      = int(cfg.get("ff", 320))
        hidden  = int(cfg.get("hidden", 256))
        self.logstd_min = float(cfg.get("logstd_min", -5.0))
        self.logstd_max = float(cfg.get("logstd_max", 2.0))

        # Извлекаем размеры для 3D
        if hasattr(obs_space, 'spaces'):
            self_feats = obs_space["self"].shape[0]
            allies_shape = obs_space["allies"].shape
            enemies_shape = obs_space["enemies"].shape
            self.max_allies = allies_shape[0]
            self.max_enemies = enemies_shape[0]
            ally_feats = allies_shape[1]
            enemy_feats = enemies_shape[1]
            global_feats = obs_space["global_state"].shape[0]
        else:
            self.max_allies = int(cfg.get("max_allies", 6))
            self.max_enemies = int(cfg.get("max_enemies", 6))
            # Обновленные размеры для 3D
            self_feats = 13   # Было 12, теперь +1 для Z координаты
            ally_feats = 9    # Было 8, теперь +1 для Z координаты
            enemy_feats = 11  # Было 10, теперь +1 для Z координаты
            global_feats = 64

        # Сохраняем для экспорта
        self.d_model = d_model
        self.nhead = nhead
        self.layers = layers

        # Энкодеры для 3D данных
        self.self_enc  = MLP([self_feats, d_model])
        self.ally_enc  = MLP([ally_feats, d_model])
        self.enemy_enc = MLP([enemy_feats, d_model])

        # Дополнительный энкодер для 3D пространственных признаков
        self.spatial_3d_enc = MLP([3, d_model // 4])  # Для кодирования 3D позиций отдельно
        
        # ONNX-совместимые блоки
        self.posenc = PositionalEncoding(d_model, max_len=max(self.max_allies + self.max_enemies + 1, 64))
        self.blocks = nn.ModuleList([
            ONNXCompatibleAttnBlock(d_model, nhead, ff) for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.last_attn = None

        # Политические головы для 3D действий
        self.head_target     = MLP([d_model, hidden, self.max_enemies])
        
        # 3D движение (x, y, z)
        self.head_move_mu    = MLP([d_model, hidden, 3])  # Теперь 3D
        self.log_std_move    = nn.Parameter(torch.full((3,), -0.5))  # 3D log std
        
        # 3D прицеливание (x, y, z)
        self.head_aim_mu     = MLP([d_model, hidden, 3])  # Теперь 3D
        self.log_std_aim     = nn.Parameter(torch.full((3,), -0.5))  # 3D log std
        
        self.head_fire_logit = MLP([d_model, hidden, 1])

        # Дополнительные головы для 3D информации
        self.head_3d_awareness = MLP([d_model, hidden // 2, 3])  # Понимание 3D окружения
        
        # Централизованная value с учетом 3D пространства
        self.value_net = MLP([global_feats + 3, hidden, 1])  # +3 для дополнительной 3D информации
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

    def _extract_3d_spatial_features(self, obs, target_device):
        """Извлекает и обрабатывает 3D пространственные признаки"""
        batch_size = obs["self"].shape[0]
        
        # Извлекаем 3D позицию из self наблюдений (первые 3 компонента)
        self_3d_pos = obs["self"][:, :3]  # [B, 3] - x, y, z позиция
        
        # Кодируем 3D позицию отдельно
        spatial_encoding = self.spatial_3d_enc(self_3d_pos)  # [B, d_model//4]
        
        # Вычисляем 3D метрики окружения
        allies_3d = obs["allies"][:, :, :3]  # [B, max_allies, 3]
        enemies_3d = obs["enemies"][:, :, :3]  # [B, max_enemies, 3]
        
        # Средние расстояния в 3D
        allies_mask = obs["allies_mask"].float().unsqueeze(-1)  # [B, max_allies, 1]
        enemies_mask = obs["enemies_mask"].float().unsqueeze(-1)  # [B, max_enemies, 1]
        
        # Расстояния до союзников и врагов в 3D
        allies_distances = torch.norm(allies_3d, dim=-1, keepdim=True)  # [B, max_allies, 1]
        enemies_distances = torch.norm(enemies_3d, dim=-1, keepdim=True)  # [B, max_enemies, 1]
        
        # Средние расстояния с учетом масок
        avg_ally_dist = (allies_distances * allies_mask).sum(dim=1) / (allies_mask.sum(dim=1) + 1e-8)
        avg_enemy_dist = (enemies_distances * enemies_mask).sum(dim=1) / (enemies_mask.sum(dim=1) + 1e-8)
        
        # Высота относительно поля (z-координата)
        relative_height = self_3d_pos[:, 2:3] / 6.0  # Нормализуем к высоте поля
        
        # Дополнительные 3D признаки
        spatial_features = torch.cat([
            avg_ally_dist,     # [B, 1]
            avg_enemy_dist,    # [B, 1] 
            relative_height    # [B, 1]
        ], dim=-1)  # [B, 3]
        
        return spatial_encoding, spatial_features

    def forward(self, input_dict, state, seq_lens):
        raw_obs = input_dict["obs"]
        
        if isinstance(raw_obs, dict):
            obs = raw_obs
        else:
            obs = dict(raw_obs)
        
        # Убеждаемся что все тензоры на правильном устройстве
        obs, target_device = self._ensure_obs_device_consistency(obs)
        
        try:
            # Извлекаем 3D пространственные признаки
            spatial_encoding, spatial_features = self._extract_3d_spatial_features(obs, target_device)
            
            # Энкодинг токенов с учетом 3D
            self_tok   = self.self_enc(obs["self"])
            allies_tok = self.ally_enc(obs["allies"])
            enemies_tok= self.enemy_enc(obs["enemies"])
            
            # Добавляем пространственное кодирование к self токену
            self_tok = self_tok + torch.cat([spatial_encoding, torch.zeros(spatial_encoding.shape[0], 
                                           self_tok.shape[1] - spatial_encoding.shape[1], 
                                           device=target_device)], dim=1)
            
            x = torch.cat([self_tok.unsqueeze(1), allies_tok, enemies_tok], dim=1)

            # Паддинг-маска
            B = x.size(0)
            pad_self = torch.zeros(B, 1, dtype=torch.bool, device=target_device)
            am = obs["allies_mask"] > 0
            em = obs["enemies_mask"] > 0
            pad_mask = torch.cat([pad_self, ~am, ~em], dim=1)

            # Positional encoding + attention блоки
            x = self.posenc(x)
            for blk in self.blocks:
                x = blk(x, key_padding_mask=pad_mask)
            self.last_attn = self.blocks[-1].last_attn if self.blocks else None
            x = self.norm(x)

            # Self-токен как агрегат
            h = x[:, 0, :]

            # Политические головы для 3D действий
            logits_target = self.head_target(h)
            mask = obs["enemy_action_mask"].float()
            inf_mask = (1.0 - mask) * torch.finfo(logits_target.dtype).min
            masked_logits = logits_target + inf_mask

            # 3D движение
            mu_move = self.head_move_mu(h)  # [B, 3] для x, y, z
            log_std_move = self.log_std_move.clamp(self.logstd_min, self.logstd_max).expand_as(mu_move)
            
            # 3D прицеливание
            mu_aim = self.head_aim_mu(h)   # [B, 3] для x, y, z
            log_std_aim = self.log_std_aim.clamp(self.logstd_min, self.logstd_max).expand_as(mu_aim)
            
            # Огонь
            logit_fire = self.head_fire_logit(h)
            
            # 3D осведомленность (дополнительная информация о пространстве)
            awareness_3d = self.head_3d_awareness(h)

            # Склейка (увеличенный размер для 3D)
            out = torch.cat([
                masked_logits,      # target selection
                mu_move,           # 3D movement (3 values)
                log_std_move,      # 3D movement std (3 values)
                mu_aim,            # 3D aiming (3 values)
                log_std_aim,       # 3D aiming std (3 values)
                logit_fire,        # fire decision (1 value)
                awareness_3d       # 3D spatial awareness (3 values)
            ], dim=-1)

            # Централизованная V с 3D информацией
            global_with_3d = torch.cat([obs["global_state"], spatial_features], dim=-1)
            v = self.value_net(global_with_3d).squeeze(-1)
            self._value_out = v
            
            return out, state
            
        except Exception as e:
            print(f"ERROR in 3D ONNX model forward: {e}")
            print(f"Observation shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in obs.items()]}")
            raise

    def value_function(self):
        return self._value_out

    def get_3d_action_dim(self):
        """Возвращает размерность 3D действий для совместимости"""
        # target + move(3) + move_std(3) + aim(3) + aim_std(3) + fire + awareness(3)
        return self.max_enemies + 3 + 3 + 3 + 3 + 1 + 3

# Регистрируем ONNX-совместимую 3D модель
ModelCatalog.register_custom_model("onnx_entity_attention", ONNXEntityAttentionModel)
ModelCatalog.register_custom_model("entity_attention_3d", ONNXEntityAttentionModel)  # Алиас для 3D