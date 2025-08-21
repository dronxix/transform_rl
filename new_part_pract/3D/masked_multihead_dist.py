"""
Кастомный TorchDistributionWrapper для 3D пространства - ИСПРАВЛЕН для Ray 2.48:
- Возвращает numpy arrays вместо тензоров для совместимости с Ray
- Правильная обработка Dict действий для 3D (target, move_3d, aim_3d, fire)
- Поддержка 3D движения и прицеливания
"""

import numpy as np
import torch
from torch.distributions import Categorical, Normal, Bernoulli
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models import ModelCatalog

class MaskedTargetMoveAimFire3D(TorchDistributionWrapper):
    """3D версия дистрибуции для target + 3D_move + 3D_aim + fire"""
    
    def __init__(self, inputs, model):
        super().__init__(inputs, model)
        # Получаем max_enemies из модели или конфига
        if hasattr(model, 'max_enemies'):
            self.ne = model.max_enemies
        else:
            # Fallback - пытаемся извлечь из action_space
            self.ne = 6  # По умолчанию
        
        # Парсим входы для 3D действий
        idx = 0
        
        # Target selection (дискретный выбор врага)
        logits_t = inputs[..., idx:idx+self.ne]; idx += self.ne
        
        # 3D движение: mu и log_std для x, y, z
        mu_move = inputs[..., idx:idx+3]; idx += 3      # 3D движение (x, y, z)
        logstd_mv = inputs[..., idx:idx+3]; idx += 3    # 3D log_std для движения
        
        # 3D прицеливание: mu и log_std для x, y, z  
        mu_aim = inputs[..., idx:idx+3]; idx += 3       # 3D прицеливание (x, y, z)
        logstd_am = inputs[..., idx:idx+3]; idx += 3    # 3D log_std для прицеливания
        
        # Fire decision (бинарный)
        logit_fr = inputs[..., idx:idx+1]; idx += 1
        
        # Дополнительные 3D компоненты (если есть)
        if inputs.shape[-1] > idx:
            # Возможно есть дополнительные компоненты от 3D awareness
            self.has_3d_awareness = True
            self.awareness_3d = inputs[..., idx:idx+3] if inputs.shape[-1] >= idx+3 else None
        else:
            self.has_3d_awareness = False
            self.awareness_3d = None

        # Создаем дистрибуции
        self.cat = Categorical(logits=logits_t)
        self.mv = Normal(mu_move, logstd_mv.exp())      # 3D движение
        self.am = Normal(mu_aim, logstd_am.exp())       # 3D прицеливание
        self.fr = Bernoulli(logits=logit_fr)
        
        # Инициализируем last_sample
        self.last_sample = None
        
        # Добавляем атрибут dist для совместимости
        self.dist = self.cat

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        """
        КРИТИЧНО: Этот метод ОБЯЗАТЕЛЕН для Ray 2.48!
        Возвращает размер выхода модели для 3D действий.
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
        
        # Рассчитываем размер выхода для 3D: 
        # target + move_3d + move_std_3d + aim_3d + aim_std_3d + fire + [awareness_3d]
        # target: max_enemies logits
        # move_3d: mu(3) + log_std(3) 
        # aim_3d: mu(3) + log_std(3)
        # fire: logit(1)
        # awareness_3d: optional(3) - добавляется моделью при необходимости
        
        output_size = max_enemies + 3 + 3 + 3 + 3 + 1  # = max_enemies + 13
        
        # Проверяем есть ли дополнительные 3D компоненты
        if custom_config.get("include_3d_awareness", False):
            output_size += 3
        
        return output_size

    def _convert_to_numpy_dict_3d(self, tensor_action):
        """Преобразует тензор действия в Dict с numpy arrays для 3D"""
        # Убеждаемся что работаем с CPU tensor
        if tensor_action.is_cuda:
            tensor_action = tensor_action.cpu()
        
        # Если батч размерности нет, добавляем
        if tensor_action.dim() == 1:
            tensor_action = tensor_action.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False
        
        # Извлекаем компоненты для 3D
        target = tensor_action[..., 0].long()
        move_3d = tensor_action[..., 1:4]  # 3D движение [x, y, z]
        aim_3d = tensor_action[..., 4:7]   # 3D прицеливание [x, y, z]  
        fire = tensor_action[..., 7].long()
        
        # Конвертируем в numpy
        result = {
            "target": target.numpy(),
            "move": move_3d.numpy(),    # 3D массив
            "aim": aim_3d.numpy(),      # 3D массив
            "fire": fire.numpy(),
        }
        
        # Если был single sample, убираем батч размерность
        if squeeze_batch:
            result = {k: v[0] for k, v in result.items()}
        
        return result

    def sample(self):
        """Возвращает сэмпл в формате Dict с numpy arrays для 3D"""
        t = self.cat.sample().unsqueeze(-1).float()
        
        # 3D движение с tanh для ограничения в [-1, 1]
        mv = torch.tanh(self.mv.rsample())  # [batch, 3] для x, y, z
        
        # 3D прицеливание с tanh для ограничения в [-1, 1]  
        am = torch.tanh(self.am.rsample())  # [batch, 3] для x, y, z
        
        # Fire decision
        fr = self.fr.sample().float()
        
        # Объединяем в плоский тензор
        flat_action = torch.cat([t, mv, am, fr], dim=-1)
        
        # Сохраняем для логирования
        self.last_sample = flat_action
        
        # Конвертируем в numpy dict для 3D
        return self._convert_to_numpy_dict_3d(flat_action)

    def deterministic_sample(self):
        """Возвращает детерминированный сэмпл в формате Dict с numpy arrays для 3D"""
        t = torch.argmax(self.cat.logits, dim=-1).unsqueeze(-1).float()
        
        # 3D движение - используем mean (loc)
        mv = torch.tanh(self.mv.loc)  # [batch, 3]
        
        # 3D прицеливание - используем mean (loc)
        am = torch.tanh(self.am.loc)  # [batch, 3]
        
        # Fire decision - используем логику > 0
        fr = (self.fr.logits > 0).float()
        
        # Объединяем в плоский тензор
        flat_action = torch.cat([t, mv, am, fr], dim=-1)
        
        # Сохраняем для логирования
        self.last_sample = flat_action
        
        # Конвертируем в numpy dict для 3D
        return self._convert_to_numpy_dict_3d(flat_action)

    def logp(self, x):
        """x может быть как Dict с numpy/tensor для 3D, так и плоским тензором"""
        # Преобразуем в тензор если нужно
        if isinstance(x, dict):
            # Конвертируем numpy в tensor если нужно
            target = x["target"]
            move = x["move"]    # Теперь должно быть 3D
            aim = x["aim"]      # Теперь должно быть 3D
            fire = x["fire"]
            
            # Конвертируем в тензоры
            if isinstance(target, np.ndarray):
                target = torch.from_numpy(target).float()
            if isinstance(move, np.ndarray):
                move = torch.from_numpy(move).float()
            if isinstance(aim, np.ndarray):
                aim = torch.from_numpy(aim).float()
            if isinstance(fire, np.ndarray):
                fire = torch.from_numpy(fire).float()
            
            # Убеждаемся что размерности правильные для 3D
            if target.dim() == 1:
                target = target.unsqueeze(-1)
            if fire.dim() == 1:
                fire = fire.unsqueeze(-1)
            
            # Убеждаемся что move и aim имеют 3 компонента
            if move.shape[-1] < 3:
                # Дополняем до 3D если нужно
                padding = torch.zeros(move.shape[:-1] + (3 - move.shape[-1],), device=move.device)
                move = torch.cat([move, padding], dim=-1)
            elif move.shape[-1] > 3:
                # Обрезаем до 3D если слишком много
                move = move[..., :3]
                
            if aim.shape[-1] < 3:
                # Дополняем до 3D если нужно
                padding = torch.zeros(aim.shape[:-1] + (3 - aim.shape[-1],), device=aim.device)
                aim = torch.cat([aim, padding], dim=-1)
            elif aim.shape[-1] > 3:
                # Обрезаем до 3D если слишком много
                aim = aim[..., :3]
                
            x = torch.cat([target, move, aim, fire], dim=-1)
            
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # Убеждаемся что x - тензор на правильном устройстве
        if x.device != self.cat.logits.device:
            x = x.to(self.cat.logits.device)
        
        eps = 1e-6
        t_idx = x[..., 0].long()
        mv = torch.clamp(x[..., 1:4], -1+eps, 1-eps)  # 3D движение [x, y, z]
        am = torch.clamp(x[..., 4:7], -1+eps, 1-eps)  # 3D прицеливание [x, y, z]
        fr = x[..., 7]

        # inverse tanh для 3D
        z_mv = 0.5 * (torch.log1p(mv) - torch.log1p(-mv))  # [batch, 3]
        z_am = 0.5 * (torch.log1p(am) - torch.log1p(-am))  # [batch, 3]

        # Вычисляем log probabilities
        lp_t = self.cat.log_prob(t_idx)
        
        # 3D движение: сумма по всем 3 компонентам
        lp_mv = self.mv.log_prob(z_mv).sum(-1) - torch.log(1 - mv.pow(2) + eps).sum(-1)
        
        # 3D прицеливание: сумма по всем 3 компонентам  
        lp_am = self.am.log_prob(z_am).sum(-1) - torch.log(1 - am.pow(2) + eps).sum(-1)
        
        # Fire probability
        p = torch.sigmoid(self.fr.logits.squeeze(-1))
        lp_fr = torch.where(fr > 0.5, torch.log(p + eps), torch.log(1 - p + eps))
        
        return lp_t + lp_mv + lp_am + lp_fr

    def sampled_action_logp(self):
        """Переопределяем метод для корректной работы с last_sample"""
        if self.last_sample is None:
            # Если sample() еще не вызывался, делаем это
            self.sample()
        return self.logp(self.last_sample)

    def kl(self, other):
        """Реализуем KL divergence для кастомной 3D дистрибуции"""
        if not isinstance(other, MaskedTargetMoveAimFire3D):
            # Если другая дистрибуция не того же типа, возвращаем 0
            return torch.zeros_like(self.cat.logits[..., 0])
        
        # Вычисляем KL для каждого компонента
        kl_cat = torch.distributions.kl.kl_divergence(self.cat, other.cat)
        
        # KL для 3D компонентов (сумма по всем 3 осям)
        kl_mv = torch.distributions.kl.kl_divergence(self.mv, other.mv).sum(-1)
        kl_am = torch.distributions.kl.kl_divergence(self.am, other.am).sum(-1)
        
        kl_fr = torch.distributions.kl.kl_divergence(self.fr, other.fr).squeeze(-1)
        
        # Суммируем все компоненты
        return kl_cat + kl_mv + kl_am + kl_fr

    def entropy(self):
        """Вычисляем энтропию для 3D дистрибуции"""
        cat_H = self.cat.entropy()
        
        # Энтропия для 3D компонентов (сумма по всем 3 осям)
        mv_H = self.mv.entropy().sum(-1)   # 3D движение
        am_H = self.am.entropy().sum(-1)   # 3D прицеливание
        
        # Fire entropy
        p = torch.sigmoid(self.fr.logits.squeeze(-1))
        bern_H = -(p*torch.log(p+1e-8) + (1-p)*torch.log(1-p+1e-8))
        
        return cat_H + mv_H + am_H + bern_H

    def get_3d_action_breakdown(self, action_tensor):
        """Утилита для разбора 3D действий (для отладки)"""
        if action_tensor.dim() == 1:
            action_tensor = action_tensor.unsqueeze(0)
        
        breakdown = {
            "target": action_tensor[..., 0].long(),
            "move_x": action_tensor[..., 1],
            "move_y": action_tensor[..., 2], 
            "move_z": action_tensor[..., 3],
            "aim_x": action_tensor[..., 4],
            "aim_y": action_tensor[..., 5],
            "aim_z": action_tensor[..., 6],
            "fire": action_tensor[..., 7] > 0.5,
        }
        
        return breakdown


# Дополнительная дистрибуция для обратной совместимости с 2D
class MaskedTargetMoveAimFire(TorchDistributionWrapper):
    """Оригинальная 2D версия для обратной совместимости"""
    
    def __init__(self, inputs, model):
        super().__init__(inputs, model)
        # Получаем max_enemies из модели или конфига
        if hasattr(model, 'max_enemies'):
            self.ne = model.max_enemies
        else:
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
        
        # Инициализируем last_sample
        self.last_sample = None
        
        # Добавляем атрибут dist для совместимости
        self.dist = self.cat

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        """2D версия расчета размера выхода"""
        custom_config = model_config.get("custom_model_config", {})
        max_enemies = custom_config.get("max_enemies", 6)
        
        # 2D: target + move(2) + move_std(2) + aim(2) + aim_std(2) + fire
        output_size = max_enemies + 2 + 2 + 2 + 2 + 1
        return output_size

    def _convert_to_numpy_dict(self, tensor_action):
        """Преобразует тензор действия в Dict с numpy arrays для 2D"""
        if tensor_action.is_cuda:
            tensor_action = tensor_action.cpu()
        
        if tensor_action.dim() == 1:
            tensor_action = tensor_action.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False
        
        # Извлекаем компоненты для 2D
        target = tensor_action[..., 0].long()
        move = tensor_action[..., 1:3]    # 2D движение
        aim = tensor_action[..., 3:5]     # 2D прицеливание
        fire = tensor_action[..., 5].long()
        
        result = {
            "target": target.numpy(),
            "move": move.numpy(),
            "aim": aim.numpy(),
            "fire": fire.numpy(),
        }
        
        if squeeze_batch:
            result = {k: v[0] for k, v in result.items()}
        
        return result

    def sample(self):
        """Возвращает сэмпл в формате Dict с numpy arrays для 2D"""
        t = self.cat.sample().unsqueeze(-1).float()
        mv = torch.tanh(self.mv.rsample())
        am = torch.tanh(self.am.rsample())
        fr = self.fr.sample().float()
        
        flat_action = torch.cat([t, mv, am, fr], dim=-1)
        self.last_sample = flat_action
        
        return self._convert_to_numpy_dict(flat_action)

    def deterministic_sample(self):
        """Возвращает детерминированный сэмпл в формате Dict с numpy arrays для 2D"""
        t = torch.argmax(self.cat.logits, dim=-1).unsqueeze(-1).float()
        mv = torch.tanh(self.mv.loc)
        am = torch.tanh(self.am.loc)
        fr = (self.fr.logits > 0).float()
        
        flat_action = torch.cat([t, mv, am, fr], dim=-1)
        self.last_sample = flat_action
        
        return self._convert_to_numpy_dict(flat_action)

    def logp(self, x):
        """Логирование для 2D версии"""
        if isinstance(x, dict):
            target = x["target"]
            move = x["move"]
            aim = x["aim"]
            fire = x["fire"]
            
            if isinstance(target, np.ndarray):
                target = torch.from_numpy(target).float()
            if isinstance(move, np.ndarray):
                move = torch.from_numpy(move).float()
            if isinstance(aim, np.ndarray):
                aim = torch.from_numpy(aim).float()
            if isinstance(fire, np.ndarray):
                fire = torch.from_numpy(fire).float()
            
            if target.dim() == 1:
                target = target.unsqueeze(-1)
            if fire.dim() == 1:
                fire = fire.unsqueeze(-1)
                
            x = torch.cat([target, move, aim, fire], dim=-1)
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        if x.device != self.cat.logits.device:
            x = x.to(self.cat.logits.device)
        
        eps = 1e-6
        t_idx = x[..., 0].long()
        mv = torch.clamp(x[..., 1:3], -1+eps, 1-eps)  # 2D
        am = torch.clamp(x[..., 3:5], -1+eps, 1-eps)  # 2D
        fr = x[..., 5]

        z_mv = 0.5 * (torch.log1p(mv) - torch.log1p(-mv))
        z_am = 0.5 * (torch.log1p(am) - torch.log1p(-am))

        lp_t  = self.cat.log_prob(t_idx)
        lp_mv = self.mv.log_prob(z_mv).sum(-1) - torch.log(1 - mv.pow(2) + eps).sum(-1)
        lp_am = self.am.log_prob(z_am).sum(-1) - torch.log(1 - am.pow(2) + eps).sum(-1)
        p = torch.sigmoid(self.fr.logits.squeeze(-1))
        lp_fr = torch.where(fr > 0.5, torch.log(p + eps), torch.log(1 - p + eps))
        
        return lp_t + lp_mv + lp_am + lp_fr

    def sampled_action_logp(self):
        if self.last_sample is None:
            self.sample()
        return self.logp(self.last_sample)

    def kl(self, other):
        if not isinstance(other, MaskedTargetMoveAimFire):
            return torch.zeros_like(self.cat.logits[..., 0])
        
        kl_cat = torch.distributions.kl.kl_divergence(self.cat, other.cat)
        kl_mv = torch.distributions.kl.kl_divergence(self.mv, other.mv).sum(-1)
        kl_am = torch.distributions.kl.kl_divergence(self.am, other.am).sum(-1)
        kl_fr = torch.distributions.kl.kl_divergence(self.fr, other.fr).squeeze(-1)
        
        return kl_cat + kl_mv + kl_am + kl_fr

    def entropy(self):
        cat_H = self.cat.entropy()
        mv_H  = self.mv.entropy().sum(-1)
        am_H  = self.am.entropy().sum(-1)
        p = torch.sigmoid(self.fr.logits.squeeze(-1))
        bern_H = -(p*torch.log(p+1e-8) + (1-p)*torch.log(1-p+1e-8))
        return cat_H + mv_H + am_H + bern_H


# Автоматический выбор дистрибуции на основе размерности
def create_adaptive_action_distribution(inputs, model):
    """
    Автоматически выбирает 2D или 3D дистрибуцию на основе размера входа
    """
    input_size = inputs.shape[-1]
    
    # Получаем max_enemies
    max_enemies = getattr(model, 'max_enemies', 6)
    
    # Вычисляем ожидаемые размеры
    size_2d = max_enemies + 2 + 2 + 2 + 2 + 1  # target + move_2d + aim_2d + fire
    size_3d = max_enemies + 3 + 3 + 3 + 3 + 1  # target + move_3d + aim_3d + fire
    
    if input_size >= size_3d:
        print(f"Using 3D action distribution (input_size={input_size}, expected_3d={size_3d})")
        return MaskedTargetMoveAimFire3D(inputs, model)
    else:
        print(f"Using 2D action distribution (input_size={input_size}, expected_2d={size_2d})")
        return MaskedTargetMoveAimFire(inputs, model)


# Регистрируем дистрибуции
ModelCatalog.register_custom_action_dist("masked_multihead", MaskedTargetMoveAimFire)
ModelCatalog.register_custom_action_dist("masked_multihead_3d", MaskedTargetMoveAimFire3D)
ModelCatalog.register_custom_action_dist("masked_multihead_adaptive", create_adaptive_action_distribution)