"""
EntityAttentionModel — трансформер с:
- паддинг-масками (ally/enemy mask),
- отдельные головы действий (target/move/aim/fire),
- централизованная value по global_state,
- сохранение карты внимания (last_attn) для TensorBoard.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog

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
        return x + self.pe[:, :x.size(1), :]

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
    def forward(self, x): return self.net(x)

class AttnBlock(nn.Module):
    """Один блок self-attention с возвратом карт внимания."""
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_ff), nn.GELU(), nn.Linear(dim_ff, d_model))
        self.ln2 = nn.LayerNorm(d_model)
        self.last_attn = None  # [B,H,L,L]
    def forward(self, x, key_padding_mask):
        attn_out, attn_w = self.mha(x, x, x, key_padding_mask=key_padding_mask, need_weights=True, average_attn_weights=False)
        self.last_attn = attn_w
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))
        return x

class EntityAttentionModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
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

        self_feats   = obs_space["self"].shape[0]
        allies_shape = obs_space["allies"].shape
        enemies_shape= obs_space["enemies"].shape
        self.max_allies  = int(allies_shape[0])
        self.max_enemies = int(enemies_shape[0])
        ally_feats  = int(allies_shape[1])
        enemy_feats = int(enemies_shape[1])
        global_feats= int(obs_space["global_state"].shape[0])

        # Энкодеры
        self.self_enc  = MLP([self_feats, d_model])
        self.ally_enc  = MLP([ally_feats, d_model])
        self.enemy_enc = MLP([enemy_feats, d_model])

        self.posenc = PositionalEncoding(d_model, max_len=max(self.max_allies + self.max_enemies + 1, 64))
        self.blocks = nn.ModuleList([AttnBlock(d_model, nhead, ff) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.last_attn = None  # [B,H,L,L] — снимок последнего блока

        # Политические головы
        self.head_target     = MLP([d_model, hidden, self.max_enemies])  # logits по врагам
        self.head_move_mu    = MLP([d_model, hidden, 2])
        self.head_aim_mu     = MLP([d_model, hidden, 2])
        self.log_std_move    = nn.Parameter(torch.full((2,), -0.5))
        self.log_std_aim     = nn.Parameter(torch.full((2,), -0.5))
        self.head_fire_logit = MLP([d_model, hidden, 1])

        # Централизованная value
        self.value_net = MLP([global_feats, hidden, 1])

        self._value_out: Optional[torch.Tensor] = None
        self.obs_space_struct = {
            "self": obs_space["self"],
            "allies": obs_space["allies"],
            "enemies": obs_space["enemies"],
            "global_state": obs_space["global_state"]
        }

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        # Токены: [self | allies... | enemies...]
        self_tok   = self.self_enc(obs["self"])                # [B,d]
        allies_tok = self.ally_enc(obs["allies"])              # [B,Na,d]
        enemies_tok= self.enemy_enc(obs["enemies"])            # [B,Ne,d]
        x = torch.cat([self_tok.unsqueeze(1), allies_tok, enemies_tok], dim=1)  # [B,1+Na+Ne,d]

        # Паддинг-маска: True = masked (игнор)
        B = x.size(0)
        pad_self = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        am = obs["allies_mask"] > 0
        em = obs["enemies_mask"] > 0
        pad_mask = torch.cat([pad_self, ~am, ~em], dim=1)

        x = self.posenc(x)
        for blk in self.blocks:
            x = blk(x, key_padding_mask=pad_mask)
        self.last_attn = self.blocks[-1].last_attn  # [B,H,L,L]
        x = self.norm(x)

        # Self-токен как «агрегат»
        h = x[:, 0, :]

        # Target с маской доступных врагов
        logits_target = self.head_target(h)
        mask = obs["enemy_action_mask"].float()
        inf_mask = (1.0 - mask) * torch.finfo(logits_target.dtype).min
        masked_logits = logits_target + inf_mask

        # Непрерывные головы
        mu_move = self.head_move_mu(h)
        mu_aim  = self.head_aim_mu(h)
        log_std_move = self.log_std_move.clamp(self.logstd_min, self.logstd_max).expand_as(mu_move)
        log_std_aim  = self.log_std_aim .clamp(self.logstd_min, self.logstd_max).expand_as(mu_aim)
        logit_fire   = self.head_fire_logit(h)

        # Склейка в один тензор (ожидает кастомный action_dist)
        out = torch.cat([masked_logits, mu_move, log_std_move, mu_aim, log_std_aim, logit_fire], dim=-1)

        # Централизованная V
        v = self.value_net(obs["global_state"]).squeeze(-1)
        self._value_out = v
        return out, state

    def value_function(self):
        return self._value_out

ModelCatalog.register_custom_model("entity_attention", EntityAttentionModel)
