"""
Универсальный TorchDistributionWrapper для любых форматов действий
Автоматически адаптируется к структуре actions_dict
"""

import numpy as np
import torch
from torch.distributions import Categorical, Normal, Bernoulli
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models import ModelCatalog
from typing import Dict, List, Any, Union, Optional
import math

class UniversalActionDistribution(TorchDistributionWrapper):
    """Универсальная дистрибуция для любых форматов действий"""
    
    def __init__(self, inputs, model):
        super().__init__(inputs, model)
        
        # Получаем спецификацию действий из модели
        if hasattr(model, 'get_action_spec'):
            self.action_spec = model.get_action_spec()
        else:
            # Fallback на дефолт
            self.action_spec = {
                "discrete_actions": {"target": 6},
                "continuous_actions": {"move": 3, "aim": 3},
                "binary_actions": {"fire": 1}
            }
        
        print(f"🎯 Universal Action Distribution initialized with spec: {self.action_spec}")
        
        # Парсим входы согласно спецификации
        self.parsed_inputs = self._parse_inputs(inputs)
        
        # Создаем дистрибуции
        self.discrete_dists = {}
        self.continuous_dists = {}
        self.binary_dists = {}
        
        self._create_distributions()
        
        # Для совместимости с RLLib
        self.dist = list(self.discrete_dists.values())[0] if self.discrete_dists else None
        self.last_sample = None

    def _parse_inputs(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Парсит плоский входной тензор согласно спецификации действий"""
        parsed = {}
        idx = 0
        
        # Дискретные действия
        for name, n_classes in self.action_spec["discrete_actions"].items():
            parsed[f"{name}_logits"] = inputs[..., idx:idx+n_classes]
            idx += n_classes
        
        # Непрерывные действия (mu + log_std)
        for name, action_dim in self.action_spec["continuous_actions"].items():
            parsed[f"{name}_mu"] = inputs[..., idx:idx+action_dim]
            idx += action_dim
            parsed[f"{name}_log_std"] = inputs[..., idx:idx+action_dim]
            idx += action_dim
        
        # Бинарные действия
        for name, _ in self.action_spec["binary_actions"].items():
            parsed[f"{name}_logit"] = inputs[..., idx:idx+1]
            idx += 1
        
        return parsed

    def _create_distributions(self):
        """Создает дистрибуции для каждого типа действий"""
        
        # Дискретные дистрибуции
        for name in self.action_spec["discrete_actions"]:
            logits_key = f"{name}_logits"
            if logits_key in self.parsed_inputs:
                self.discrete_dists[name] = Categorical(logits=self.parsed_inputs[logits_key])
        
        # Непрерывные дистрибуции
        for name in self.action_spec["continuous_actions"]:
            mu_key = f"{name}_mu"
            std_key = f"{name}_log_std"
            if mu_key in self.parsed_inputs and std_key in self.parsed_inputs:
                mu = self.parsed_inputs[mu_key]
                log_std = self.parsed_inputs[std_key]
                std = log_std.exp()
                self.continuous_dists[name] = Normal(mu, std)
        
        # Бинарные дистрибуции
        for name in self.action_spec["binary_actions"]:
            logit_key = f"{name}_logit"
            if logit_key in self.parsed_inputs:
                self.binary_dists[name] = Bernoulli(logits=self.parsed_inputs[logit_key])

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        """Вычисляем требуемый размер выхода модели"""
        
        # Пытаемся получить из конфига модели
        custom_config = model_config.get("custom_model_config", {})
        
        # Если есть явная спецификация действий
        if "action_spec" in custom_config:
            spec = custom_config["action_spec"]
            total = 0
            total += sum(spec.get("discrete_actions", {}).values())
            total += sum(v * 2 for v in spec.get("continuous_actions", {}).values())  # *2 для mu + log_std
            total += len(spec.get("binary_actions", {}))
            return total
        
        # Анализируем action_space
        if hasattr(action_space, 'spaces'):
            total = 0
            for name, space in action_space.spaces.items():
                if hasattr(space, 'n'):
                    # Discrete
                    total += space.n
                elif hasattr(space, 'shape'):
                    # Continuous
                    action_dim = space.shape[0] if space.shape else 1
                    if name in ["fire", "shoot", "attack"] or action_dim == 1:
                        total += 1  # Binary
                    else:
                        total += action_dim * 2  # mu + log_std
            return total
        
        # Fallback на дефолт (3D версия)
        max_enemies = custom_config.get("max_enemies", 6)
        return max_enemies + 3*2 + 3*2 + 1  # target + move(mu+std) + aim(mu+std) + fire

    def _convert_to_numpy_dict(self, tensor_dict: Dict[str, torch.Tensor], squeeze_batch: bool = False) -> Union[Dict[str, np.ndarray], Dict[str, Union[int, float, np.ndarray]]]:
        """Преобразует тензоры в numpy dict для возврата"""
        result = {}
        
        for name, tensor in tensor_dict.items():
            if tensor.is_cuda:
                tensor = tensor.cpu()
            
            numpy_val = tensor.numpy()
            
            if squeeze_batch and numpy_val.ndim > 0:
                numpy_val = numpy_val[0] if numpy_val.shape[0] == 1 else numpy_val
            
            # Преобразуем scalar arrays в Python типы
            if numpy_val.ndim == 0:
                result[name] = numpy_val.item()
            elif numpy_val.shape == (1,):
                result[name] = numpy_val[0]
            else:
                result[name] = numpy_val
        
        return result

    def _sample_all_actions(self, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """Сэмплирует все действия"""
        sampled = {}
        
        # Дискретные действия
        for name, dist in self.discrete_dists.items():
            if deterministic:
                sampled[name] = torch.argmax(dist.logits, dim=-1)
            else:
                sampled[name] = dist.sample()
        
        # Непрерывные действия
        for name, dist in self.continuous_dists.items():
            if deterministic:
                sampled[name] = torch.tanh(dist.loc)  # mean
            else:
                sampled[name] = torch.tanh(dist.rsample())  # с градиентами
        
        # Бинарные действия
        for name, dist in self.binary_dists.items():
            if deterministic:
                sampled[name] = (dist.logits > 0).long()
            else:
                sampled[name] = dist.sample().long()
        
        return sampled

    def sample(self):
        """Возвращает сэмпл в формате Dict с numpy arrays"""
        sampled_dict = self._sample_all_actions(deterministic=False)
        
        # Сохраняем для логирования
        self.last_sample = sampled_dict
        
        # Преобразуем в numpy dict
        batch_size = next(iter(sampled_dict.values())).size(0)
        squeeze_batch = batch_size == 1
        
        return self._convert_to_numpy_dict(sampled_dict, squeeze_batch)

    def deterministic_sample(self):
        """Возвращает детерминированный сэмпл"""
        sampled_dict = self._sample_all_actions(deterministic=True)
        
        # Сохраняем для логирования
        self.last_sample = sampled_dict
        
        # Преобразуем в numpy dict
        batch_size = next(iter(sampled_dict.values())).size(0)
        squeeze_batch = batch_size == 1
        
        return self._convert_to_numpy_dict(sampled_dict, squeeze_batch)

    def logp(self, x):
        """Вычисляем log probability для универсальных действий"""
        
        # Конвертируем в тензоры если нужно
        if isinstance(x, dict):
            action_tensors = {}
            device = next(iter(self.parsed_inputs.values())).device
            
            for name, value in x.items():
                if isinstance(value, np.ndarray):
                    action_tensors[name] = torch.from_numpy(value).to(device)
                elif isinstance(value, (int, float)):
                    action_tensors[name] = torch.tensor(value, device=device)
                elif isinstance(value, torch.Tensor):
                    action_tensors[name] = value.to(device)
                else:
                    action_tensors[name] = torch.tensor(value, device=device)
        else:
            # Если x - плоский тензор, нужно его распарсить обратно
            # Это сложнее, поэтому лучше работать с dict форматом
            raise ValueError("logp expects dict format for universal actions")
        
        total_logp = 0.0
        batch_size = None
        
        # Дискретные действия
        for name, dist in self.discrete_dists.items():
            if name in action_tensors:
                action = action_tensors[name]
                if action.dim() == 0:
                    action = action.unsqueeze(0)
                if batch_size is None:
                    batch_size = action.size(0)
                total_logp = total_logp + dist.log_prob(action.long())
        
        # Непрерывные действия
        for name, dist in self.continuous_dists.items():
            if name in action_tensors:
                action = action_tensors[name]
                if action.dim() == 1 and action.size(0) != batch_size:
                    action = action.unsqueeze(0)
                
                # Inverse tanh для действий в [-1, 1]
                eps = 1e-6
                action_clamped = torch.clamp(action, -1+eps, 1-eps)
                z = 0.5 * (torch.log1p(action_clamped) - torch.log1p(-action_clamped))
                
                logp_gaussian = dist.log_prob(z).sum(-1)
                logp_tanh_correction = -torch.log(1 - action_clamped.pow(2) + eps).sum(-1)
                
                total_logp = total_logp + logp_gaussian + logp_tanh_correction
        
        # Бинарные действия
        for name, dist in self.binary_dists.items():
            if name in action_tensors:
                action = action_tensors[name]
                if action.dim() == 0:
                    action = action.unsqueeze(0)
                if action.dim() == 2 and action.size(1) == 1:
                    action = action.squeeze(-1)
                
                p = torch.sigmoid(dist.logits.squeeze(-1))
                logp_binary = torch.where(
                    action > 0.5, 
                    torch.log(p + eps), 
                    torch.log(1 - p + eps)
                )
                total_logp = total_logp + logp_binary
        
        # Убеждаемся что результат имеет правильную размерность
        if isinstance(total_logp, (int, float)):
            if batch_size is not None:
                total_logp = torch.full((batch_size,), total_logp, 
                                     device=next(iter(self.parsed_inputs.values())).device)
            else:
                total_logp = torch.tensor(total_logp)
        
        return total_logp

    def sampled_action_logp(self):
        """Переопределяем метод для корректной работы с last_sample"""
        if self.last_sample is None:
            self.sample()
        return self.logp(self.last_sample)

    def kl(self, other):
        """Реализуем KL divergence"""
        if not isinstance(other, UniversalActionDistribution):
            # Если другая дистрибуция не того же типа, возвращаем 0
            batch_size = next(iter(self.parsed_inputs.values())).size(0)
            return torch.zeros(batch_size, device=next(iter(self.parsed_inputs.values())).device)
        
        total_kl = 0.0
        
        # KL для дискретных действий
        for name in self.discrete_dists:
            if name in other.discrete_dists:
                kl_discrete = torch.distributions.kl.kl_divergence(
                    self.discrete_dists[name], other.discrete_dists[name]
                )
                total_kl = total_kl + kl_discrete
        
        # KL для непрерывных действий
        for name in self.continuous_dists:
            if name in other.continuous_dists:
                kl_continuous = torch.distributions.kl.kl_divergence(
                    self.continuous_dists[name], other.continuous_dists[name]
                ).sum(-1)
                total_kl = total_kl + kl_continuous
        
        # KL для бинарных действий
        for name in self.binary_dists:
            if name in other.binary_dists:
                kl_binary = torch.distributions.kl.kl_divergence(
                    self.binary_dists[name], other.binary_dists[name]
                ).squeeze(-1)
                total_kl = total_kl + kl_binary
        
        return total_kl

    def entropy(self):
        """Вычисляем энтропию для всех действий"""
        total_entropy = 0.0
        
        # Энтропия дискретных действий
        for dist in self.discrete_dists.values():
            total_entropy = total_entropy + dist.entropy()
        
        # Энтропия непрерывных действий
        for dist in self.continuous_dists.values():
            total_entropy = total_entropy + dist.entropy().sum(-1)
        
        # Энтропия бинарных действий
        for dist in self.binary_dists.values():
            p = torch.sigmoid(dist.logits.squeeze(-1))
            binary_entropy = -(p*torch.log(p+1e-8) + (1-p)*torch.log(1-p+1e-8))
            total_entropy = total_entropy + binary_entropy
        
        return total_entropy

    def get_action_breakdown(self) -> Dict[str, Any]:
        """Утилита для анализа действий (для отладки)"""
        return {
            "action_spec": self.action_spec,
            "discrete_actions": list(self.discrete_dists.keys()),
            "continuous_actions": list(self.continuous_dists.keys()),
            "binary_actions": list(self.binary_dists.keys()),
            "total_distributions": len(self.discrete_dists) + len(self.continuous_dists) + len(self.binary_dists)
        }


# Специализированные дистрибуции для обратной совместимости
class MaskedTargetMoveAimFire3D(UniversalActionDistribution):
    """3D версия для target + move + aim + fire (обратная совместимость)"""
    
    def __init__(self, inputs, model):
        # Устанавливаем специфичную спецификацию для 3D
        self.action_spec = {
            "discrete_actions": {"target": getattr(model, 'max_enemies', 6)},
            "continuous_actions": {"move": 3, "aim": 3},  # 3D
            "binary_actions": {"fire": 1}
        }
        super().__init__(inputs, model)

class MaskedTargetMoveAimFire(UniversalActionDistribution):
    """2D версия для target + move + aim + fire (обратная совместимость)"""
    
    def __init__(self, inputs, model):
        # Устанавливаем специфичную спецификацию для 2D
        self.action_spec = {
            "discrete_actions": {"target": getattr(model, 'max_enemies', 6)},
            "continuous_actions": {"move": 2, "aim": 2},  # 2D
            "binary_actions": {"fire": 1}
        }
        super().__init__(inputs, model)


def create_adaptive_action_distribution(inputs, model):
    """
    Автоматически создает подходящую дистрибуцию на основе модели
    """
    
    # Пытаемся получить спецификацию из модели
    if hasattr(model, 'get_action_spec'):
        return UniversalActionDistribution(inputs, model)
    
    # Fallback - анализируем размер входа
    input_size = inputs.shape[-1]
    max_enemies = getattr(model, 'max_enemies', 6)
    
    # Вычисляем ожидаемые размеры
    size_2d = max_enemies + 2*2 + 2*2 + 1  # target + move_2d + aim_2d + fire
    size_3d = max_enemies + 3*2 + 3*2 + 1  # target + move_3d + aim_3d + fire
    
    if input_size >= size_3d:
        print(f"🎯 Auto-detected 3D action space (size={input_size})")
        return MaskedTargetMoveAimFire3D(inputs, model)
    else:
        print(f"🎯 Auto-detected 2D action space (size={input_size})")
        return MaskedTargetMoveAimFire(inputs, model)


# Регистрируем дистрибуции
ModelCatalog.register_custom_action_dist("universal_action_dist", UniversalActionDistribution)
ModelCatalog.register_custom_action_dist("masked_multihead", MaskedTargetMoveAimFire)
ModelCatalog.register_custom_action_dist("masked_multihead_3d", MaskedTargetMoveAimFire3D)
ModelCatalog.register_custom_action_dist("masked_multihead_adaptive", create_adaptive_action_distribution)


# Утилиты для работы с универсальными действиями
class ActionSpecAnalyzer:
    """Анализатор для автоматического определения спецификации действий"""
    
    @staticmethod
    def analyze_action_space(action_space) -> Dict[str, Any]:
        """Анализирует action_space и возвращает спецификацию"""
        spec = {
            "discrete_actions": {},
            "continuous_actions": {},
            "binary_actions": {},
            "total_output_size": 0
        }
        
        if hasattr(action_space, 'spaces'):
            for name, space in action_space.spaces.items():
                if hasattr(space, 'n'):
                    # Discrete space
                    spec["discrete_actions"][name] = space.n
                    spec["total_output_size"] += space.n
                elif hasattr(space, 'shape'):
                    # Continuous space
                    action_dim = space.shape[0] if space.shape else 1
                    if action_dim == 1 and name in ["fire", "shoot", "attack", "use"]:
                        spec["binary_actions"][name] = 1
                        spec["total_output_size"] += 1
                    else:
                        spec["continuous_actions"][name] = action_dim
                        spec["total_output_size"] += action_dim * 2  # mu + log_std
        
        return spec
    
    @staticmethod
    def print_action_analysis(action_space, model_config=None):
        """Печатает анализ action_space для отладки"""
        spec = ActionSpecAnalyzer.analyze_action_space(action_space)
        
        print("🔍 Action Space Analysis:")
        print(f"   Discrete actions: {spec['discrete_actions']}")
        print(f"   Continuous actions: {spec['continuous_actions']}")
        print(f"   Binary actions: {spec['binary_actions']}")
        print(f"   Total output size needed: {spec['total_output_size']}")
        
        if model_config:
            custom_config = model_config.get("custom_model_config", {})
            if "max_enemies" in custom_config:
                print(f"   Max enemies from config: {custom_config['max_enemies']}")
        
        return spec


def test_universal_action_distribution():
    """Тест универсальной системы действий"""
    print("🧪 Testing Universal Action Distribution...")
    
    # Создаем мок модель
    class MockModel:
        def __init__(self, action_spec):
            self._action_spec = action_spec
            self.max_enemies = action_spec["discrete_actions"].get("target", 6)
        
        def get_action_spec(self):
            return self._action_spec
    
    # Тест 1: Базовая 3D конфигурация
    action_spec_3d = {
        "discrete_actions": {"target": 4},
        "continuous_actions": {"move": 3, "aim": 3},
        "binary_actions": {"fire": 1}
    }
    
    model_3d = MockModel(action_spec_3d)
    
    # Создаем тестовые входы
    batch_size = 2
    total_size = 4 + 3*2 + 3*2 + 1  # target + move(mu+std) + aim(mu+std) + fire
    test_inputs = torch.randn(batch_size, total_size)
    
    # Создаем дистрибуцию
    dist_3d = UniversalActionDistribution(test_inputs, model_3d)
    
    print(f"✅ 3D Distribution created:")
    print(f"   Action breakdown: {dist_3d.get_action_breakdown()}")
    
    # Тестируем семплирование
    sample = dist_3d.sample()
    print(f"   Sample keys: {list(sample.keys())}")
    print(f"   Sample types: {[(k, type(v)) for k, v in sample.items()]}")
    
    # Тест 2: Кастомная конфигурация
    custom_spec = {
        "discrete_actions": {"weapon": 3, "formation": 5},
        "continuous_actions": {"velocity": 2, "direction": 1},
        "binary_actions": {"shield": 1, "boost": 1}
    }
    
    model_custom = MockModel(custom_spec)
    custom_size = 3 + 5 + 2*2 + 1*2 + 1 + 1  # weapon + formation + vel(mu+std) + dir(mu+std) + shield + boost
    test_inputs_custom = torch.randn(batch_size, custom_size)
    
    dist_custom = UniversalActionDistribution(test_inputs_custom, model_custom)
    sample_custom = dist_custom.sample()
    
    print(f"✅ Custom Distribution created:")
    print(f"   Action breakdown: {dist_custom.get_action_breakdown()}")
    print(f"   Sample keys: {list(sample_custom.keys())}")
    
    # Тест логирования
    logp = dist_custom.logp(sample_custom)
    print(f"   Log probability: {logp.shape}")
    
    print("✅ Universal action distribution tests passed!")


if __name__ == "__main__":
    test_universal_action_distribution()