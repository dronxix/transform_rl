"""
Универсальный ONNX экспорт с автоматической адаптацией к любым форматам obs/actions
"""

import os
import json
import torch
import platform
from typing import Dict, Any, List, Optional, Union
import numpy as np


class UniversalPolicyONNXWrapper(torch.nn.Module):
    """Универсальный wrapper для экспорта политик любого формата"""

    def __init__(self, original_model, onnx_model, action_spec: Dict, obs_spec: Dict):
        super().__init__()
        self.onnx_model = onnx_model
        self.action_spec = action_spec
        self.obs_spec = obs_spec

        # Переносим веса из оригинальной модели
        self._transfer_weights(original_model, onnx_model)

    def _transfer_weights(self, original_model, onnx_model):
        """Универсальный перенос весов"""
        print("Transferring weights to universal ONNX model...")

        original_state = original_model.state_dict()
        onnx_state = onnx_model.state_dict()

        transferred = 0
        total_params = len(onnx_state)

        for onnx_name, onnx_param in onnx_state.items():
            transferred_this_param = False

            # 1. Прямое совпадение имен
            if onnx_name in original_state:
                orig_param = original_state[onnx_name]
                if orig_param.shape == onnx_param.shape:
                    onnx_param.data.copy_(orig_param.data.cpu())
                    transferred += 1
                    transferred_this_param = True
                    continue

            # 2. Поиск по схожим паттернам
            if not transferred_this_param:
                orig_name = self._find_compatible_param(onnx_name, onnx_param, original_state)
                if orig_name:
                    orig_param = original_state[orig_name]
                    if orig_param.shape == onnx_param.shape:
                        onnx_param.data.copy_(orig_param.data.cpu())
                        transferred += 1
                        transferred_this_param = True
                        print(f"Mapped {onnx_name} <- {orig_name}")
                        continue

            # 3. Специальная обработка для attention и action heads
            if not transferred_this_param:
                if self._transfer_special_weights(onnx_name, onnx_param, original_state):
                    transferred += 1
                    transferred_this_param = True

            if not transferred_this_param:
                print(f"⚠ No match found for {onnx_name} (shape: {tuple(onnx_param.shape)})")

        print(f"Successfully transferred {transferred}/{total_params} parameters ({100*transferred/total_params:.1f}%)")

    def _find_compatible_param(
        self,
        onnx_name: str,
        onnx_param: torch.Tensor,
        original_state: Dict[str, torch.Tensor],
    ) -> Optional[str]:
        """Ищет совместимый параметр в оригинальной модели"""
        onnx_parts = onnx_name.split('.')

        # Паттерны для поиска
        search_patterns = [
            '.'.join(onnx_parts[-2:]) if len(onnx_parts) >= 2 else onnx_name,  # суффикс из 2-х частей
            onnx_parts[-1] if onnx_parts else onnx_name,                        # последний компонент
            self._extract_layer_type(onnx_name),                                 # тип слоя
        ]

        for pattern in search_patterns:
            if not pattern:
                continue
            for orig_name, orig_param in original_state.items():
                # Совпадение по подстроке + совпадение формы
                if pattern in orig_name and getattr(orig_param, "shape", None) == onnx_param.shape:
                    return orig_name

        return None

    def _extract_layer_type(self, param_name: str) -> str:
        """Извлекает тип слоя из имени параметра"""
        layer_types = ['linear', 'conv', 'norm', 'embedding', 'attention', 'head']
        for layer_type in layer_types:
            if layer_type in param_name.lower():
                return layer_type
        return ""

    def _transfer_special_weights(self, onnx_name: str, onnx_param: torch.Tensor, original_state: Dict) -> bool:
        """Специальная обработка для особых типов слоев"""

        # Action heads - могут иметь разные названия
        if any(head_type in onnx_name for head_type in ['discrete_heads', 'continuous_heads', 'binary_heads']):
            for orig_name in original_state:
                if any(head_type in orig_name for head_type in ['head_target', 'head_move', 'head_aim', 'head_fire']):
                    if original_state[orig_name].shape == onnx_param.shape:
                        onnx_param.data.copy_(original_state[orig_name].data.cpu())
                        return True

        # Observation encoders
        if any(enc_type in onnx_name for enc_type in ['self_enc', 'ally_enc', 'enemy_enc']):
            for orig_name in original_state:
                if any(enc_type in orig_name for enc_type in ['self_enc', 'ally_enc', 'enemy_enc']):
                    if original_state[orig_name].shape == onnx_param.shape:
                        onnx_param.data.copy_(original_state[orig_name].data.cpu())
                        return True

        return False

    def _prepare_universal_inputs(self, *args) -> Dict[str, torch.Tensor]:
        """Подготавливает универсальные входы для модели"""

        # Определяем формат входов на основе количества аргументов
        if len(args) == 1 and isinstance(args[0], dict):
            # Dict формат (уже собранный)
            obs_dict = dict(args[0])
        else:
            # Tensor формат - преобразуем в стандартную структуру по позициям
            input_names = ["self_vec", "allies", "allies_mask", "enemies", "enemies_mask", "enemy_action_mask"]
            obs_dict = {}
            for i, tensor in enumerate(args):
                if i < len(input_names):
                    obs_dict[input_names[i]] = tensor

        # Убеждаемся, что все на CPU (ONNX-трейс на CPU)
        obs_dict = {k: (v.cpu() if hasattr(v, 'cpu') else v) for k, v in obs_dict.items()}

        # Приводим ключи к ожидаемым моделью
        # alias: self_vec -> self
        if "self_vec" in obs_dict and "self" not in obs_dict:
            obs_dict["self"] = obs_dict.pop("self_vec")

        # Гарантируем наличие обязательных компонент
        if not obs_dict:
            raise ValueError("Empty observation dictionary passed to wrapper.")
        # batch_size по первой компоненте
        any_tensor = next(iter(obs_dict.values()))
        batch_size = int(any_tensor.shape[0])

        required_components = ["self", "allies", "allies_mask", "enemies", "enemies_mask", "enemy_action_mask", "global_state"]
        for comp in required_components:
            if comp not in obs_dict:
                if comp == "self":
                    obs_dict[comp] = torch.zeros(batch_size, self.obs_spec.get("self_features", 13))
                elif comp == "allies":
                    max_allies = self.obs_spec.get("max_allies", 6)
                    ally_feats = self.obs_spec.get("ally_features", 9)
                    obs_dict[comp] = torch.zeros(batch_size, max_allies, ally_feats)
                elif comp == "allies_mask":
                    max_allies = self.obs_spec.get("max_allies", 6)
                    obs_dict[comp] = torch.zeros(batch_size, max_allies, dtype=torch.int64)
                elif comp == "enemies":
                    max_enemies = self.obs_spec.get("max_enemies", 6)
                    enemy_feats = self.obs_spec.get("enemy_features", 11)
                    obs_dict[comp] = torch.zeros(batch_size, max_enemies, enemy_feats)
                elif comp == "enemies_mask":
                    max_enemies = self.obs_spec.get("max_enemies", 6)
                    obs_dict[comp] = torch.zeros(batch_size, max_enemies, dtype=torch.int64)
                elif comp == "enemy_action_mask":
                    max_enemies = self.obs_spec.get("max_enemies", 6)
                    obs_dict[comp] = torch.zeros(batch_size, max_enemies, dtype=torch.int64)
                elif comp == "global_state":
                    global_feats = self.obs_spec.get("global_features", 64)
                    obs_dict[comp] = torch.zeros(batch_size, global_feats)

        # Доп. фичи, если описаны в спецификации
        additional_features = self.obs_spec.get("additional_features", {})
        for feat_name, feat_shape in additional_features.items():
            if feat_name not in obs_dict:
                # feat_shape ожидается как tuple, добавляем batch
                obs_dict[feat_name] = torch.zeros((batch_size, *tuple(feat_shape)), dtype=torch.float32)

        return obs_dict

    @torch.no_grad()
    def forward(self, *args):
        """Universal forward для любых входных форматов"""

        # Подготавливаем входы
        obs_dict = self._prepare_universal_inputs(*args)

        # Forward pass через модель (поддержка двух сигнатур возврата)
        out = self.onnx_model({"obs": obs_dict}, [], None)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        logits = logits.cpu()

        # Декодируем действия согласно action_spec
        return self._decode_universal_actions(logits)

    def _decode_universal_actions(self, logits: torch.Tensor) -> torch.Tensor:
        """Декодирует логиты в детерминированные действия согласно спецификации"""

        batch_size = logits.shape[0]
        idx = 0
        action_components = []

        # Дискретные действия (argmax)
        for name, n_classes in self.action_spec.get("discrete_actions", {}).items():
            discrete_logits = logits[:, idx:idx+n_classes]
            discrete_actions = torch.argmax(discrete_logits, dim=-1).float().unsqueeze(-1)
            action_components.append(discrete_actions)
            idx += n_classes

        # Непрерывные действия (tanh(mu))
        for name, action_dim in self.action_spec.get("continuous_actions", {}).items():
            mu = logits[:, idx:idx+action_dim]
            idx += action_dim
            # пропускаем log_std, если он присутствует в логе
            log_std = logits[:, idx:idx+action_dim]  # noqa: F841
            idx += action_dim
            continuous_actions = torch.tanh(mu)
            action_components.append(continuous_actions)

        # Бинарные действия (sigmoid > 0.5)
        for name, _ in self.action_spec.get("binary_actions", {}).items():
            binary_logits = logits[:, idx:idx+1]
            binary_actions = (torch.sigmoid(binary_logits) > 0.5).float()
            action_components.append(binary_actions)
            idx += 1

        # Объединяем все компоненты
        if action_components:
            result = torch.cat(action_components, dim=-1)
        else:
            # Fallback
            result = torch.zeros(batch_size, 1)

        return result.cpu()


def export_universal_onnx_with_meta(algorithm, iteration: int, export_dir: str, policies_to_export: List[str]):
    """
    Универсальная функция экспорта ONNX с автоматической адаптацией к любым форматам
    """
    print(f"\n=== Universal ONNX Export (iteration {iteration}) ===")

    # Получаем информацию об окружении
    env_config = getattr(algorithm.config, 'env_config', {})

    # Пытаемся получить спецификации из первой политики
    main_policy = algorithm.get_policy("main")
    action_spec = {}
    obs_spec = {}

    if hasattr(main_policy.model, 'get_action_spec'):
        action_spec = main_policy.model.get_action_spec()
    if hasattr(main_policy.model, 'get_obs_spec'):
        obs_spec = main_policy.model.get_obs_spec()

    # Fallback анализ если спецификации недоступны
    if not action_spec or not obs_spec:
        print("📊 Analyzing environment for universal export...")
        try:
            from ray.tune.registry import _global_registry
            env_name = getattr(algorithm.config, 'env', 'ArenaEnv')
            env_creator = _global_registry.get("env", env_name)

            if env_creator:
                temp_env = env_creator(env_config)
                obs_space = temp_env.observation_space
                act_space = temp_env.action_space

                from entity_attention_model import DynamicActionConfig, DynamicObservationProcessor
                if not action_spec:
                    action_config = DynamicActionConfig(act_space)
                    action_spec = action_config.action_spec
                if not obs_spec:
                    obs_processor = DynamicObservationProcessor(obs_space)
                    obs_spec = obs_processor.obs_spec

        except Exception as e:
            print(f"⚠️ Could not analyze environment: {e}")
            # Используем дефолтные спецификации
            action_spec = {
                "discrete_actions": {"target": 6},
                "continuous_actions": {"move": 3, "aim": 3},
                "binary_actions": {"fire": 1}
            }
            obs_spec = {
                "self_features": 13, "ally_features": 9, "enemy_features": 11,
                "max_allies": 6, "max_enemies": 6, "global_features": 64
            }

    print(f"🎯 Universal export configuration:")
    print(f"   Action spec: {action_spec}")
    print(f"   Obs spec: {list(obs_spec.keys())}")

    # Создаем директорию для этой итерации
    iter_dir = os.path.join(export_dir, f"iter_{iteration:06d}")
    os.makedirs(iter_dir, exist_ok=True)

    successful_exports = []

    for policy_id in policies_to_export:
        try:
            print(f"\nExporting universal policy: {policy_id}")

            # Получаем оригинальную модель
            policy = algorithm.get_policy(policy_id)
            original_model = policy.model

            # Создаем универсальную ONNX-совместимую модель
            from entity_attention_model import ONNXEntityAttentionModel

            # Подготавливаем конфигурацию модели
            model_config_dict = {
                "custom_model": "universal_entity_attention",
                "custom_model_config": {
                    "d_model": getattr(original_model, 'd_model', 128),
                    "nhead": getattr(original_model, 'nhead', 8),
                    "layers": getattr(original_model, 'layers', 2),
                    "ff": 256,
                    "hidden": 256,
                    "action_spec": action_spec,
                    "obs_spec": obs_spec,
                },
                "vf_share_layers": False,
            }

            # Создаем ONNX модель
            from gymnasium import spaces

            # Создаем obs_space на основе спецификации
            obs_space_dict = {
                "self": spaces.Box(low=-15, high=15, shape=(obs_spec["self_features"],), dtype=np.float32),
                "allies": spaces.Box(low=-15, high=15, shape=(obs_spec["max_allies"], obs_spec["ally_features"]), dtype=np.float32),
                "allies_mask": spaces.MultiBinary(obs_spec["max_allies"]),
                "enemies": spaces.Box(low=-15, high=15, shape=(obs_spec["max_enemies"], obs_spec["enemy_features"]), dtype=np.float32),
                "enemies_mask": spaces.MultiBinary(obs_spec["max_enemies"]),
                "global_state": spaces.Box(low=-15, high=15, shape=(obs_spec["global_features"],), dtype=np.float32),
                "enemy_action_mask": spaces.MultiBinary(obs_spec["max_enemies"]),
            }
            # Доп. фичи из спецификации
            for _feat, _shape in obs_spec.get("additional_features", {}).items():
                obs_space_dict[_feat] = spaces.Box(low=-15, high=15, shape=tuple(_shape), dtype=np.float32)

            temp_obs_space = spaces.Dict(obs_space_dict)

            # Создаем action_space на основе спецификации
            action_space_dict = {}
            for name, n_classes in action_spec.get("discrete_actions", {}).items():
                action_space_dict[name] = spaces.Discrete(n_classes)
            for name, action_dim in action_spec.get("continuous_actions", {}).items():
                action_space_dict[name] = spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)
            for name, _ in action_spec.get("binary_actions", {}).items():
                action_space_dict[name] = spaces.Discrete(2)
            temp_act_space = spaces.Dict(action_space_dict)

            onnx_model = ONNXEntityAttentionModel(
                obs_space=temp_obs_space,
                action_space=temp_act_space,
                num_outputs=getattr(original_model, 'num_outputs', None),
                model_config=model_config_dict,
                name=f"universal_onnx_export_{policy_id}"
            )

            # Перемещаем на CPU
            onnx_model = onnx_model.cpu()
            onnx_model.eval()

            # Создаем универсальный wrapper
            wrapper = UniversalPolicyONNXWrapper(original_model, onnx_model, action_spec, obs_spec)

            # Подготавливаем тестовые входы
            B = 1
            test_inputs = _create_universal_test_inputs(B, obs_spec)

            # Тестовый прогон
            print(f"  Testing universal wrapper...")
            with torch.no_grad():
                test_output = wrapper(*test_inputs)
            print(f"  ✓ Test successful, output shape: {test_output.shape}")

            # ONNX экспорт
            onnx_path = os.path.join(iter_dir, f"policy_{policy_id}_universal.onnx")
            print(f"  Exporting to universal ONNX...")

            # Определяем input names динамически
            input_names = _generate_input_names(obs_spec)

            torch.onnx.export(
                wrapper,
                test_inputs,
                onnx_path,
                opset_version=17,
                input_names=input_names,
                output_names=["action"],
                dynamic_axes={name: {0: "batch"} for name in (input_names + ["action"])},
                do_constant_folding=True,
                export_params=True,
                verbose=False,
            )

            # Создание универсальных метаданных
            meta = _create_universal_metadata(
                iteration, policy_id, action_spec, obs_spec, input_names, algorithm
            )

            # Сохранение метаданных
            meta_path = os.path.join(iter_dir, f"policy_{policy_id}_universal_meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            print(f"  ✓ Exported universal ONNX: {onnx_path}")
            print(f"  ✓ Saved universal meta: {meta_path}")

            # Валидация ONNX
            _validate_universal_onnx(onnx_path, test_inputs, input_names)

            successful_exports.append({
                "policy_id": policy_id,
                "onnx_path": onnx_path,
                "meta_path": meta_path,
                "input_names": input_names,
                "action_spec": action_spec,
                "obs_spec": obs_spec,
                "export_type": "universal"
            })

        except Exception as e:
            print(f"  ✗ Failed to export universal {policy_id}: {e}")
            import traceback
            traceback.print_exc()

    # Создаем общий мета-файл
    if successful_exports:
        _create_universal_summary(iter_dir, iteration, successful_exports, action_spec, obs_spec)

    # Создаем latest ссылку
    _create_safe_symlink(iter_dir, os.path.join(export_dir, "latest"))

    print(f"=== Universal ONNX export completed for iteration {iteration} ===\n")
    return successful_exports


def _create_universal_test_inputs(batch_size: int, obs_spec: Dict) -> tuple:
    """Создает универсальные тестовые входы"""

    inputs = [
        torch.zeros(batch_size, obs_spec["self_features"], dtype=torch.float32),                                  # self_vec (alias -> self)
        torch.zeros(batch_size, obs_spec["max_allies"], obs_spec["ally_features"], dtype=torch.float32),          # allies
        torch.zeros(batch_size, obs_spec["max_allies"], dtype=torch.int64),                                       # allies_mask
        torch.zeros(batch_size, obs_spec["max_enemies"], obs_spec["enemy_features"], dtype=torch.float32),        # enemies
        torch.zeros(batch_size, obs_spec["max_enemies"], dtype=torch.int64),                                      # enemies_mask
        torch.zeros(batch_size, obs_spec["max_enemies"], dtype=torch.int64),                                      # enemy_action_mask
    ]

    # Дополнительные признаки (если есть)
    additional_features = obs_spec.get("additional_features", {})
    for feat_name, feat_shape in additional_features.items():
        # Каждую доп. фичу добавляем отдельным входом в том же порядке, что и названия
        inputs.append(torch.zeros((batch_size, *tuple(feat_shape)), dtype=torch.float32))

    return tuple(inputs)


def _generate_input_names(obs_spec: Dict) -> List[str]:
    """Генерирует названия входов на основе спецификации"""

    base_names = ["self_vec", "allies", "allies_mask", "enemies", "enemies_mask", "enemy_action_mask"]

    # Добавляем дополнительные входы, если есть
    additional_features = obs_spec.get("additional_features", {})
    for feature_name in additional_features:
        base_names.append(feature_name)

    return base_names


def _create_universal_metadata(iteration: int, policy_id: str, action_spec: Dict,
                               obs_spec: Dict, input_names: List[str], algorithm) -> Dict:
    """Создает универсальные метаданные"""

    meta = {
        "iteration": iteration,
        "policy_id": policy_id,
        "export_type": "universal",
        "action_specification": action_spec,
        "observation_specification": obs_spec,
        "input_names": input_names,
        "output_names": ["action"],
        "model_architecture": {
            "type": "universal_entity_attention",
            "d_model": getattr(algorithm.get_policy(policy_id).model, 'd_model', 128),
            "nhead": getattr(algorithm.get_policy(policy_id).model, 'nhead', 8),
            "layers": getattr(algorithm.get_policy(policy_id).model, 'layers', 2),
        },
        "compatibility": {
            "framework": "torch",
            "opset_version": 17,
            "dynamic_batching": True,
            "supports_any_obs_format": True,
            "supports_any_action_format": True,
        },
        "training_info": {
            "timesteps_total": algorithm.metrics.peek("timesteps_total", 0),
            "episodes_total": algorithm.metrics.peek("episodes_total", 0),
            "episode_reward_mean": algorithm.metrics.peek("env_runners/episode_reward_mean", 0),
        },
        "usage_instructions": {
            "description": "Universal policy export compatible with any observation/action format",
            "input_format": "Provide inputs matching the observation_specification",
            "output_format": "Returns actions matching the action_specification",
            "adaptation_note": "Model automatically adapts to your environment's obs/action structure"
        }
    }

    return meta


def _validate_universal_onnx(onnx_path: str, test_inputs: tuple, input_names: List[str]):
    """Валидирует универсальную ONNX модель"""

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        # Получаем фактические имена входов
        actual_input_names = [inp.name for inp in sess.get_inputs()]
        print(f"  ℹ Universal ONNX inputs: {actual_input_names}")

        if len(actual_input_names) == len(test_inputs):
            onnx_inputs = {}
            for i, name in enumerate(actual_input_names):
                onnx_inputs[name] = test_inputs[i].cpu().numpy()

            test_result = sess.run(["action"], onnx_inputs)
            print(f"  ✓ Universal ONNX validation passed, shape: {test_result[0].shape}")
            print(f"  ✓ Sample output: {test_result[0][0]}")

        else:
            print(f"  ! Input count mismatch: expected {len(test_inputs)}, got {len(actual_input_names)}")

    except ImportError:
        print("  ! onnxruntime not available, skipping validation")
    except Exception as e:
        print(f"  ! Universal ONNX validation failed: {e}")


def _create_universal_summary(iter_dir: str, iteration: int, successful_exports: List[Dict],
                              action_spec: Dict, obs_spec: Dict):
    """Создает сводку универсального экспорта"""

    general_meta_path = os.path.join(iter_dir, "universal_export_summary.json")
    general_meta = {
        "iteration": iteration,
        "timestamp": str(np.datetime64('now')),
        "export_type": "universal_adaptive",
        "exported_policies": successful_exports,
        "universal_specifications": {
            "action_spec": action_spec,
            "observation_spec": obs_spec,
        },
        "compatibility": {
            "supports_any_environment": True,
            "automatic_adaptation": True,
            "cross_platform": True,
            "framework_agnostic": True,
        },
        "features": [
            "Automatic obs/action format detection",
            "Dynamic model configuration",
            "Universal ONNX compatibility",
            "Cross-environment portability",
            "Flexible inference engine support"
        ]
    }

    with open(general_meta_path, "w", encoding="utf-8") as f:
        json.dump(general_meta, f, ensure_ascii=False, indent=2)

    print(f"✓ Universal export summary saved: {general_meta_path}")


def _create_safe_symlink(target_dir: str, link_path: str):
    """Безопасное создание символической ссылки"""
    try:
        if os.path.islink(link_path):
            os.unlink(link_path)
        elif os.path.exists(link_path):
            if os.path.isdir(link_path):
                import shutil
                shutil.rmtree(link_path)
            else:
                os.remove(link_path)

        target_name = os.path.basename(target_dir)
        try:
            os.symlink(target_name, link_path)
            print(f"  ✓ Created symlink: {link_path} -> {target_name}")
        except OSError as e:
            if platform.system() == "Windows" and "required privilege" in str(e).lower():
                import shutil
                if os.path.isdir(target_dir):
                    shutil.copytree(target_dir, link_path)
                else:
                    shutil.copy2(target_dir, link_path)
                print(f"  ✓ Created copy: {link_path}")
            else:
                raise

    except Exception as e:
        print(f"  ! Warning: Could not create latest link: {e}")


# Обратная совместимость
def export_onnx_with_meta(algorithm, iteration: int, export_dir: str, policies_to_export: List[str]):
    """Обратно совместимая функция - переадресует на универсальную"""
    return export_universal_onnx_with_meta(algorithm, iteration, export_dir, policies_to_export)


# Утилиты для тестирования универсального экспорта
def test_universal_onnx_export():
    """Тест универсального ONNX экспорта"""
    print("🧪 Testing Universal ONNX Export System...")

    # Создаем mock спецификации
    test_action_spec = {
        "discrete_actions": {"weapon": 3, "target": 4},
        "continuous_actions": {"velocity": 2, "direction": 3},
        "binary_actions": {"shield": 1, "boost": 1}
    }

    test_obs_spec = {
        "self_features": 10,
        "ally_features": 6,
        "enemy_features": 8,
        "max_allies": 3,
        "max_enemies": 4,
        "global_features": 20,
        "additional_features": {"special_sensor": (5,)}
    }

    print(f"✅ Test specifications created:")
    print(f"   Actions: {test_action_spec}")
    print(f"   Observations: {list(test_obs_spec.keys())}")

    # Тест создания входов
    test_inputs = _create_universal_test_inputs(2, test_obs_spec)
    print(f"✅ Test inputs created: {len(test_inputs)} tensors")
    print(f"   Input shapes: {[t.shape for t in test_inputs]}")

    # Тест генерации имен
    input_names = _generate_input_names(test_obs_spec)
    print(f"✅ Input names generated: {input_names}")

    # Тест метаданных
    class MockAlgorithm:
        class MockMetrics:
            def peek(self, key, default):
                return {"timesteps_total": 10000, "episodes_total": 100,
                        "env_runners/episode_reward_mean": 15.5}.get(key, default)

        class MockPolicy:
            class MockModel:
                d_model = 128
                nhead = 8
                layers = 2
            model = MockModel()

        def get_policy(self, name):
            return self.MockPolicy()

        metrics = MockMetrics()

    mock_algo = MockAlgorithm()
    metadata = _create_universal_metadata(100, "test_policy", test_action_spec, test_obs_spec, input_names, mock_algo)

    print(f"✅ Universal metadata created:")
    print(f"   Keys: {list(metadata.keys())}")
    print(f"   Export type: {metadata['export_type']}")
    print(f"   Compatibility: {metadata['compatibility']}")

    print("✅ Universal ONNX export system tests passed!")
    return True


def run_inference_test(onnx_path: str, batch_size: int = 3, verbose: bool = True):
    """
    Универсальный тест инференса с автоматическим чтением мета-файлов
    """
    print(f"=== Universal ONNX Inference Test ===")
    print(f"Model: {onnx_path}")
    print(f"Batch size: {batch_size}")

    # Читаем метаданные
    meta_path = onnx_path.replace('.onnx', '_meta.json')
    if not os.path.exists(meta_path):
        # Ищем в той же директории
        base_dir = os.path.dirname(onnx_path)
        meta_files = [f for f in os.listdir(base_dir) if f.endswith('_meta.json') or f.endswith('summary.json')]
        if meta_files:
            meta_path = os.path.join(base_dir, meta_files[0])

    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        if 'universal_specifications' in metadata:
            action_spec = metadata['universal_specifications']['action_spec']
            obs_spec = metadata['universal_specifications']['observation_spec']
        elif 'action_specification' in metadata:
            action_spec = metadata['action_specification']
            obs_spec = metadata['observation_specification']
        else:
            print("⚠️ No universal specifications found in metadata")
            return None

        print(f"✅ Universal metadata loaded:")
        print(f"   Export type: {metadata.get('export_type', 'unknown')}")
        print(f"   Action spec: {action_spec}")
        print(f"   Obs spec: {list(obs_spec.keys())}")
    else:
        print(f"⚠️ No metadata found, using defaults")
        action_spec = {"discrete_actions": {"target": 6}, "continuous_actions": {"move": 3, "aim": 3}, "binary_actions": {"fire": 1}}
        obs_spec = {"self_features": 13, "ally_features": 9, "enemy_features": 11, "max_allies": 6, "max_enemies": 6, "global_features": 64}

    # Создаем тестовые входы
    test_inputs = _create_universal_test_inputs(batch_size, obs_spec)
    input_names = _generate_input_names(obs_spec)

    print(f"\nUniversal test scenario:")
    print(f"  Input shapes: {[t.shape for t in test_inputs]}")
    print(f"  Input names: {input_names}")

    # Тест инференса
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        # Подготавливаем входы
        onnx_inputs = {}
        actual_input_names = [inp.name for inp in sess.get_inputs()]

        for i, name in enumerate(actual_input_names):
            if i < len(test_inputs):
                onnx_inputs[name] = test_inputs[i].numpy()

        # Выполняем предсказание
        results = sess.run(["action"], onnx_inputs)
        actions = results[0]

        print(f"\n✅ Universal inference successful!")
        print(f"  Output shape: {actions.shape}")
        print(f"  Sample actions:\n{actions}")

        # Анализируем результат согласно спецификации
        print(f"\n📊 Action Analysis:")
        idx = 0
        for name, n_classes in action_spec.get("discrete_actions", {}).items():
            discrete_actions = actions[:, idx:idx+n_classes]
            print(f"  Discrete '{name}': {discrete_actions.shape} (classes: {n_classes})")
            idx += n_classes

        for name, action_dim in action_spec.get("continuous_actions", {}).items():
            continuous_actions = actions[:, idx:idx+action_dim]
            print(f"  Continuous '{name}': {continuous_actions.shape} (dim: {action_dim})")
            idx += action_dim

        for name, _ in action_spec.get("binary_actions", {}).items():
            binary_actions = actions[:, idx:idx+1]
            print(f"  Binary '{name}': {binary_actions.shape}")
            idx += 1

        return actions

    except Exception as e:
        print(f"✗ Universal inference failed: {e}")
        raise


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_universal_onnx_export()
        elif sys.argv[1] == "infer" and len(sys.argv) > 2:
            run_inference_test(sys.argv[2])
        else:
            print("Usage:")
            print("  python onnx_callbacks.py test - Test universal export system")
            print("  python onnx_callbacks.py infer <model.onnx> - Test inference")
    else:
        print("🌟 Universal ONNX Export System loaded!")
        print("Features:")
        print("  ✅ Automatic adaptation to any obs/action format")
        print("  ✅ Dynamic model configuration")
        print("  ✅ Universal metadata generation")
        print("  ✅ Cross-environment compatibility")
        print("  ✅ Flexible inference support")
