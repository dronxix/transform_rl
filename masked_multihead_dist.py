"""
Кастомный TorchDistributionWrapper - ИСПРАВЛЕН для Ray 2.48:
- Добавлен required_model_output_shape
- Исправлена проблема с last_sample  
- Добавлен метод kl для KL divergence
- ИСПРАВЛЕНИЕ: Возвращаем Dict вместо плоского array
"""

import torch
from torch.distributions import Categorical, Normal, Bernoulli
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models import ModelCatalog
import gymnasium as gym

class MaskedTargetMoveAimFire(TorchDistributionWrapper):
    def __init__(self, inputs, model):
        super().__init__(inputs, model)
        # Получаем max_enemies из модели или конфига
        if hasattr(model, 'max_enemies'):
            self.ne = model.max_enemies
        else:
            # Fallback - пытаемся извлечь из action_space
            self.ne = 6  # По умолчанию
            
        idx = 0
        logits_t = inputs[..., idx:idx+self.ne]; idx += self.ne
        mu_move  = inputs[..., idx:idx+2];        idx += 2
        logstd_mv= inputs[..., idx:idx+2];        idx += 2
        mu_aim   = inputs[..., idx:idx+2];        idx += 2
        logstd_am= inputs[..., idx:idx+2];        idx += 2
        logit_fr = inputs[..., idx:idx+1];        idx += 1

        self.cat = Categorical(logits=logits_t)
        self.mv  = Normal(mu_move, logstd_mv.exp())
        self.am  = Normal(mu_aim,  logstd_am.exp())
        self.fr  = Bernoulli(logits=logit_fr)
        
        # ИСПРАВЛЕНИЕ: Инициализируем last_sample
        self.last_sample = None
        
        # ИСПРАВЛЕНИЕ: Добавляем атрибут dist для совместимости
        self.dist = self.cat

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        """
        КРИТИЧНО: Этот метод ОБЯЗАТЕЛЕН для Ray 2.48!
        Возвращает размер выхода модели для данного action_space.
        """
        # Извлекаем max_enemies из конфига модели
        custom_config = model_config.get("custom_model_config", {})
        
        # Пытаемся получить max_enemies разными способами
        max_enemies = None
        
        # 1. Из custom_model_config
        if "max_enemies" in custom_config:
            max_enemies = custom_config["max_enemies"]
        
        # 2. Из action_space (если это Dict с target)
        elif hasattr(action_space, 'spaces') and 'target' in action_space.spaces:
            if hasattr(action_space.spaces['target'], 'n'):
                max_enemies = action_space.spaces['target'].n
        
        # 3. Fallback на стандартное значение
        if max_enemies is None:
            max_enemies = 6
            print(f"Warning: max_enemies not found in config, using default: {max_enemies}")
        
        # Рассчитываем размер выхода: target + move + aim + fire
        # target: max_enemies logits
        # move: mu(2) + log_std(2) 
        # aim: mu(2) + log_std(2)
        # fire: logit(1)
        output_size = max_enemies + 2 + 2 + 2 + 2 + 1
        
        return output_size

    def _convert_to_dict(self, flat_action):
        """Преобразует плоское действие в Dict формат"""
        if isinstance(flat_action, dict):
            return flat_action  # Уже в правильном формате
            
        # Извлекаем компоненты из плоского действия
        target = flat_action[..., 0:1]  # [B, 1]
        move = flat_action[..., 1:3]    # [B, 2] 
        aim = flat_action[..., 3:5]     # [B, 2]
        fire = flat_action[..., 5:6]    # [B, 1]
        
        return {
            "target": target.squeeze(-1).long(),  # [B] - дискретное
            "move": move,                         # [B, 2] - непрерывное
            "aim": aim,                          # [B, 2] - непрерывное  
            "fire": fire.squeeze(-1).long(),     # [B] - дискретное
        }

    def sample(self):
        t = self.cat.sample().unsqueeze(-1).float()
        mv = torch.tanh(self.mv.rsample())
        am = torch.tanh(self.am.rsample())
        fr = self.fr.sample().float()
        flat_action = torch.cat([t, mv, am, fr], dim=-1)
        
        # ИСПРАВЛЕНИЕ: Сохраняем плоское действие для логирования
        self.last_sample = flat_action
        
        # ИСПРАВЛЕНИЕ: Возвращаем Dict формат
        return self._convert_to_dict(flat_action)

    def deterministic_sample(self):
        t = torch.argmax(self.cat.logits, dim=-1).unsqueeze(-1).float()
        mv = torch.tanh(self.mv.loc)
        am = torch.tanh(self.am.loc)
        fr = (self.fr.logits > 0).float()
        flat_action = torch.cat([t, mv, am, fr], dim=-1)
        
        # ИСПРАВЛЕНИЕ: Сохраняем плоское действие для логирования
        self.last_sample = flat_action
        
        # ИСПРАВЛЕНИЕ: Возвращаем Dict формат
        return self._convert_to_dict(flat_action)

    def logp(self, x):
        """x может быть как Dict, так и плоским тензором"""
        if isinstance(x, dict):
            # Преобразуем Dict обратно в плоский формат для вычисления logp
            target = x["target"].float().unsqueeze(-1)  # [B, 1]
            move = x["move"]                            # [B, 2]
            aim = x["aim"]                             # [B, 2] 
            fire = x["fire"].float().unsqueeze(-1)     # [B, 1]
            x = torch.cat([target, move, aim, fire], dim=-1)
        
        eps = 1e-6
        t_idx = x[..., 0].long()
        mv = torch.clamp(x[..., 1:3], -1+eps, 1-eps)
        am = torch.clamp(x[..., 3:5], -1+eps, 1-eps)
        fr = x[..., 5]

        # inverse tanh
        z_mv = 0.5 * (torch.log1p(mv) - torch.log1p(-mv))
        z_am = 0.5 * (torch.log1p(am) - torch.log1p(-am))

        lp_t  = self.cat.log_prob(t_idx)
        lp_mv = self.mv.log_prob(z_mv).sum(-1) - torch.log(1 - mv.pow(2) + eps).sum(-1)
        lp_am = self.am.log_prob(z_am).sum(-1) - torch.log(1 - am.pow(2) + eps).sum(-1)
        p = torch.sigmoid(self.fr.logits.squeeze(-1))
        lp_fr = torch.where(fr > 0.5, torch.log(p + eps), torch.log(1 - p + eps))
        return lp_t + lp_mv + lp_am + lp_fr

    def sampled_action_logp(self):
        """
        ИСПРАВЛЕНИЕ: Переопределяем метод для корректной работы с last_sample
        """
        if self.last_sample is None:
            # Если sample() еще не вызывался, делаем это
            self.sample()
        return self.logp(self.last_sample)

    def kl(self, other):
        """
        ИСПРАВЛЕНИЕ: Реализуем KL divergence для кастомной дистрибуции
        """
        if not isinstance(other, MaskedTargetMoveAimFire):
            # Если другая дистрибуция не того же типа, возвращаем 0
            return torch.zeros_like(self.cat.logits[..., 0])
        
        # Вычисляем KL для каждого компонента
        kl_cat = torch.distributions.kl.kl_divergence(self.cat, other.cat)
        kl_mv = torch.distributions.kl.kl_divergence(self.mv, other.mv).sum(-1)
        kl_am = torch.distributions.kl.kl_divergence(self.am, other.am).sum(-1)
        kl_fr = torch.distributions.kl.kl_divergence(self.fr, other.fr).squeeze(-1)
        
        # Суммируем все компоненты
        return kl_cat + kl_mv + kl_am + kl_fr

    def entropy(self):
        cat_H = self.cat.entropy()
        mv_H  = self.mv.entropy().sum(-1)
        am_H  = self.am.entropy().sum(-1)
        p = torch.sigmoid(self.fr.logits.squeeze(-1))
        bern_H = -(p*torch.log(p+1e-8) + (1-p)*torch.log(1-p+1e-8))
        return cat_H + mv_H + am_H + bern_H

ModelCatalog.register_custom_action_dist("masked_multihead", MaskedTargetMoveAimFire)