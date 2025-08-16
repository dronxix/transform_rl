"""
ONNX-совместимая версия EntityAttentionModel
Заменяет nn.MultiheadAttention на кастомную реализацию
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
    """ONNX-совместимая версия EntityAttentionModel"""
    
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

        # Извлекаем размеры
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
            self_feats = 12
            ally_feats = 8
            enemy_feats = 10
            global_feats = 64

        # Сохраняем для экспорта
        self.d_model = d_model
        self.nhead = nhead
        self.layers = layers

        # Энкодеры
        self.self_enc  = MLP([self_feats, d_model])
        self.ally_enc  = MLP([ally_feats, d_model])
        self.enemy_enc = MLP([enemy_feats, d_model])

        # ONNX-совместимые блоки
        self.posenc = PositionalEncoding(d_model, max_len=max(self.max_allies + self.max_enemies + 1, 64))
        self.blocks = nn.ModuleList([
            ONNXCompatibleAttnBlock(d_model, nhead, ff) for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.last_attn = None

        # Политические головы
        self.head_target     = MLP([d_model, hidden, self.max_enemies])
        self.head_move_mu    = MLP([d_model, hidden, 2])
        self.head_aim_mu     = MLP([d_model, hidden, 2])
        self.log_std_move    = nn.Parameter(torch.full((2,), -0.5))
        self.log_std_aim     = nn.Parameter(torch.full((2,), -0.5))
        self.head_fire_logit = MLP([d_model, hidden, 1])

        # Централизованная value
        self.value_net = MLP([global_feats, hidden, 1])
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

    def forward(self, input_dict, state, seq_lens):
        raw_obs = input_dict["obs"]
        
        if isinstance(raw_obs, dict):
            obs = raw_obs
        else:
            obs = dict(raw_obs)
        
        # Убеждаемся что все тензоры на правильном устройстве
        obs, target_device = self._ensure_obs_device_consistency(obs)
        
        try:
            # Энкодинг токенов
            self_tok   = self.self_enc(obs["self"])
            allies_tok = self.ally_enc(obs["allies"])
            enemies_tok= self.enemy_enc(obs["enemies"])
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

            # Политические головы
            logits_target = self.head_target(h)
            mask = obs["enemy_action_mask"].float()
            inf_mask = (1.0 - mask) * torch.finfo(logits_target.dtype).min
            masked_logits = logits_target + inf_mask

            mu_move = self.head_move_mu(h)
            mu_aim  = self.head_aim_mu(h)
            log_std_move = self.log_std_move.clamp(self.logstd_min, self.logstd_max).expand_as(mu_move)
            log_std_aim  = self.log_std_aim .clamp(self.logstd_min, self.logstd_max).expand_as(mu_aim)
            logit_fire   = self.head_fire_logit(h)

            # Склейка
            out = torch.cat([masked_logits, mu_move, log_std_move, mu_aim, log_std_aim, logit_fire], dim=-1)

            # Централизованная V
            v = self.value_net(obs["global_state"]).squeeze(-1)
            self._value_out = v
            
            return out, state
            
        except Exception as e:
            print(f"ERROR in ONNX model forward: {e}")
            raise

    def value_function(self):
        return self._value_out

# Регистрируем ONNX-совместимую модель
ModelCatalog.register_custom_model("onnx_entity_attention", ONNXEntityAttentionModel)