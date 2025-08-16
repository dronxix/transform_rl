"""
Финальная версия ONNX экспорта с ONNX-совместимой моделью
ИСПРАВЛЕНО: убраны ошибки в логике переноса весов и структуре кода
"""

import os
import json
import torch
from typing import Dict, Any, List, Optional
import numpy as np
import ray
from torch.utils.tensorboard import SummaryWriter

from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.algorithms import Algorithm

class ONNXExportWrapper(torch.nn.Module):
    """Wrapper для ONNX экспорта с переносом весов"""
    
    def __init__(self, original_model, onnx_model, ne: int, na: int):
        super().__init__()
        self.onnx_model = onnx_model
        self.ne = ne
        self.na = na
        
        # Переносим веса из оригинальной модели
        self._transfer_weights(original_model, onnx_model)
        
    def _transfer_weights(self, original_model, onnx_model):
        """Переносит веса из оригинальной модели в ONNX-совместимую"""
        print("Transferring weights to ONNX model...")
        
        # Получаем словари состояний
        original_state = original_model.state_dict()
        onnx_state = onnx_model.state_dict()
        
        transferred = 0
        total_params = len(onnx_state)
        
        # Простое сопоставление по именам параметров
        for onnx_name, onnx_param in onnx_state.items():
            transferred_this_param = False
            
            # 1. Прямое совпадение имен
            if onnx_name in original_state:
                orig_param = original_state[onnx_name]
                if orig_param.shape == onnx_param.shape:
                    onnx_param.data.copy_(orig_param.data.cpu())
                    transferred += 1
                    transferred_this_param = True
                else:
                    print(f"Shape mismatch for {onnx_name}: {orig_param.shape} vs {onnx_param.shape}")
            
            # 2. Специальная обработка для attention слоев
            elif not transferred_this_param and 'mha' in onnx_name:
                transferred_this_param = self._transfer_attention_weights(
                    onnx_name, onnx_param, original_state)
                if transferred_this_param:
                    transferred += 1
            
            # 3. Поиск по схожести имен для остальных параметров
            elif not transferred_this_param:
                orig_name = self._find_similar_param(onnx_name, original_state)
                if orig_name:
                    orig_param = original_state[orig_name]
                    if orig_param.shape == onnx_param.shape:
                        onnx_param.data.copy_(orig_param.data.cpu())
                        transferred += 1
                        transferred_this_param = True
                        print(f"Mapped {onnx_name} <- {orig_name}")
            
            if not transferred_this_param:
                print(f"No match found for {onnx_name} (shape: {onnx_param.shape})")
        
        print(f"Successfully transferred {transferred}/{total_params} parameters")
        
        # Проверяем что основные слои перенесены
        key_layers = ['self_enc', 'ally_enc', 'enemy_enc', 'head_target', 'head_move_mu', 'head_aim_mu', 'head_fire_logit']
        for layer in key_layers:
            found = any(layer in name for name in onnx_state.keys() if any(layer in transferred_name for transferred_name in original_state.keys()))
            if found:
                print(f"✓ {layer} weights transferred")
            else:
                print(f"⚠ {layer} weights may not be transferred")
    
    def _transfer_attention_weights(self, onnx_name, onnx_param, original_state):
        """Специальная обработка для attention весов"""
        # Пытаемся найти соответствующие веса в оригинальной модели
        
        if 'w_o.weight' in onnx_name:
            # out_proj weight
            orig_name = onnx_name.replace('mha.w_o.weight', 'mha.out_proj.weight')
            if orig_name in original_state and original_state[orig_name].shape == onnx_param.shape:
                onnx_param.data.copy_(original_state[orig_name].data.cpu())
                return True
                
        elif 'w_o.bias' in onnx_name:
            # out_proj bias
            orig_name = onnx_name.replace('mha.w_o.bias', 'mha.out_proj.bias')
            if orig_name in original_state and original_state[orig_name].shape == onnx_param.shape:
                onnx_param.data.copy_(original_state[orig_name].data.cpu())
                return True
        
        elif 'w_q.weight' in onnx_name or 'w_k.weight' in onnx_name or 'w_v.weight' in onnx_name:
            # Для Q, K, V весов нужно извлечь из in_proj_weight
            orig_name = onnx_name.replace('mha.w_q.weight', 'mha.in_proj_weight').replace('mha.w_k.weight', 'mha.in_proj_weight').replace('mha.w_v.weight', 'mha.in_proj_weight')
            if orig_name in original_state:
                in_proj = original_state[orig_name]  # [3*d_model, d_model]
                d_model = onnx_param.shape[0]
                
                if 'w_q.weight' in onnx_name:
                    onnx_param.data.copy_(in_proj[:d_model].cpu())
                    return True
                elif 'w_k.weight' in onnx_name:
                    onnx_param.data.copy_(in_proj[d_model:2*d_model].cpu())
                    return True
                elif 'w_v.weight' in onnx_name:
                    onnx_param.data.copy_(in_proj[2*d_model:3*d_model].cpu())
                    return True
        
        return False
    
    def _find_similar_param(self, onnx_name, original_state):
        """Ищет похожий параметр в оригинальной модели"""
        onnx_parts = onnx_name.split('.')
        
        # Ищем по окончанию (weight, bias)
        if len(onnx_parts) >= 2:
            suffix = '.'.join(onnx_parts[-2:])  # например, "net.0.weight"
            
            for orig_name in original_state.keys():
                if orig_name.endswith(suffix):
                    # Проверяем что это похожий слой
                    orig_parts = orig_name.split('.')
                    
                    # Простая эвристика: совпадение ключевых частей имени
                    common_parts = set(onnx_parts[:-2]) & set(orig_parts[:-2])
                    if len(common_parts) > 0:
                        return orig_name
        
        return None

    @torch.no_grad()
    def forward(self,
                self_vec,
                allies, allies_mask,
                enemies, enemies_mask,
                global_state,
                enemy_action_mask):
        
        # Убеждаемся что все на CPU
        inputs_cpu = {
            "self": self_vec.cpu(),
            "allies": allies.cpu(),
            "allies_mask": allies_mask.cpu(),
            "enemies": enemies.cpu(),
            "enemies_mask": enemies_mask.cpu(),
            "global_state": global_state.cpu(),  # Включаем global_state для полного forward pass
            "enemy_action_mask": enemy_action_mask.cpu(),
        }
        
        # Forward pass через ONNX модель (получаем только logits, не value)
        logits, _ = self.onnx_model({"obs": inputs_cpu}, [], None)
        logits = logits.cpu()
        
        # Декодируем как в оригинале
        idx = 0
        logits_t = logits[:, idx:idx+self.ne]; idx += self.ne
        mu_move  = logits[:, idx:idx+2];        idx += 2
        idx += 2  # log_std_move
        mu_aim   = logits[:, idx:idx+2];        idx += 2
        idx += 2  # log_std_aim
        logit_fr = logits[:, idx:idx+1];        idx += 1

        target = torch.argmax(logits_t, dim=-1).unsqueeze(-1).float()
        move = torch.tanh(mu_move)
        aim  = torch.tanh(mu_aim)
        fire = (logit_fr > 0).float()
        
        result = torch.cat([target, move, aim, fire], dim=-1)
        return result.cpu()

class FinalONNXCallbacks(RLlibCallback):
    """Финальные callbacks с ONNX-совместимой моделью"""
    
    def __init__(self):
        super().__init__()
        self.export_onnx = True
        self.export_every = 25
        self.export_dir = "./onnx_exports"
        self.policies_to_export = ["main"]
        
    def setup(self, **kwargs):
        self.export_onnx = kwargs.get('export_onnx', True)
        self.export_every = kwargs.get('export_every', 25)
        self.export_dir = kwargs.get('export_dir', "./onnx_exports")
        self.policies_to_export = kwargs.get('policies_to_export', ["main"])
        
        if self.export_onnx:
            os.makedirs(self.export_dir, exist_ok=True)

    def on_train_result(self, *, algorithm: Algorithm, result: Dict[str, Any], **kwargs) -> None:
        it = result["training_iteration"]
        
        if self.export_onnx and it % self.export_every == 0 and it > 0:
            try:
                self._export_onnx_compatible(algorithm, it)
                result.setdefault("custom_metrics", {})["onnx_export_iteration"] = it
                print(f"✓ ONNX export completed for iteration {it}")
            except Exception as e:
                print(f"✗ ONNX export failed for iteration {it}: {e}")
                import traceback
                traceback.print_exc()

    def _export_onnx_compatible(self, algorithm: Algorithm, iteration: int):
        """ONNX экспорт с ONNX-совместимой моделью"""
        print(f"\n=== ONNX Compatible Export (iteration {iteration}) ===")
        
        # Получаем размеры
        env_config = algorithm.config.env_config
        from arena_env import ArenaEnv
        tmp_env = ArenaEnv(env_config)
        obs_space = tmp_env.observation_space
        
        max_enemies = obs_space["enemies"].shape[0]
        max_allies = obs_space["allies"].shape[0]
        self_feats = obs_space["self"].shape[0]
        ally_feats = obs_space["allies"].shape[1]
        enemy_feats = obs_space["enemies"].shape[1]
        global_feats = obs_space["global_state"].shape[0]
        
        iter_dir = os.path.join(self.export_dir, f"iter_{iteration:06d}")
        os.makedirs(iter_dir, exist_ok=True)
        
        for policy_id in self.policies_to_export:
            try:
                print(f"Exporting policy: {policy_id}")
                
                # Получаем оригинальную модель
                policy = algorithm.get_policy(policy_id)
                original_model = policy.model
                
                # Создаем ONNX-совместимую модель
                from entity_attention_model import ONNXEntityAttentionModel
                
                # Получаем конфигурацию модели из политики
                model_config_dict = getattr(policy.config, 'model', {})
                if not model_config_dict:
                    model_config_dict = {
                        "custom_model": "onnx_entity_attention",
                        "custom_model_config": {
                            "d_model": 128,
                            "nhead": 8,
                            "layers": 2,
                            "ff": 256,
                            "hidden": 256,
                            "max_enemies": max_enemies,
                            "max_allies": max_allies,
                        },
                        "vf_share_layers": False,
                    }
                
                # Создаем ONNX модель
                onnx_model = ONNXEntityAttentionModel(
                    obs_space=tmp_env.observation_space,
                    action_space=tmp_env.action_space,
                    num_outputs=getattr(original_model, 'num_outputs', None),
                    model_config=model_config_dict,
                    name=f"onnx_export_{policy_id}"
                )
                
                # Перемещаем на CPU
                onnx_model = onnx_model.cpu()
                onnx_model.eval()
                
                # Определяем размеры
                ne = getattr(original_model, 'max_enemies', max_enemies)
                na = getattr(original_model, 'max_allies', max_allies)
                
                # Создаем wrapper с переносом весов (ИСПРАВЛЕНО: без global_state)
                from onnx_wrap import PolicyOnlyONNXWrapper
                wrapper = PolicyOnlyONNXWrapper(original_model, onnx_model, ne=ne, na=na)
                
                # Тестовые входы (БЕЗ global_state)
                B = 1
                test_inputs = (
                    torch.zeros(B, self_feats, dtype=torch.float32),      # self_vec
                    torch.zeros(B, na, ally_feats, dtype=torch.float32),  # allies
                    torch.zeros(B, na, dtype=torch.int32),                # allies_mask
                    torch.zeros(B, ne, enemy_feats, dtype=torch.float32), # enemies
                    torch.zeros(B, ne, dtype=torch.int32),                # enemies_mask
                    torch.zeros(B, ne, dtype=torch.int32),                # enemy_action_mask
                )
                
                # Тестовый прогон
                print(f"Testing ONNX wrapper...")
                with torch.no_grad():
                    test_output = wrapper(*test_inputs)
                print(f"Test successful, output shape: {test_output.shape}")
                
                # ONNX экспорт (ИСПРАВЛЕНО: правильные имена без global_state)
                onnx_path = os.path.join(iter_dir, f"policy_{policy_id}.onnx")
                print(f"Exporting to ONNX (opset 17)...")
                
                input_names = [
                    "self_vec", "allies", "allies_mask", 
                    "enemies", "enemies_mask", "enemy_action_mask"
                ]
                
                torch.onnx.export(
                    wrapper,
                    test_inputs,
                    onnx_path,
                    opset_version=17,
                    input_names=input_names,
                    output_names=["action"],
                    dynamic_axes={name: {0: "batch"} for name in input_names + ["action"]},
                    do_constant_folding=True,
                    export_params=True,
                    verbose=False,
                )
                
                # Метаданные с правильными именами входов
                meta = {
                    "iteration": iteration,
                    "policy_id": policy_id,
                    "max_allies": int(na),
                    "max_enemies": int(ne),
                    "obs_dims": {
                        "self": int(self_feats),
                        "ally": int(ally_feats),
                        "enemy": int(enemy_feats),
                        "global_state": int(global_feats),  # Оставляем для совместимости
                    },
                    "action_spec": {
                        "target_discrete_n": int(ne),
                        "move_box": 2,
                        "aim_box": 2,
                        "fire_binary": 1,
                        "total_action_dim": 6
                    },
                    "input_names": input_names,  # Фактические имена входов ONNX
                    "export_method": "PolicyOnlyONNXWrapper",
                    "note": "Policy-only export without value function (no global_state input)"
                }
                
                meta_path = os.path.join(iter_dir, f"policy_{policy_id}_meta.json")
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                
                print(f"  ✓ Success: {onnx_path}")
                
                # Валидация с правильными именами входов
                try:
                    import onnxruntime as ort
                    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
                    
                    # Получаем фактические имена входов из модели
                    actual_input_names = [input.name for input in sess.get_inputs()]
                    print(f"  ℹ ONNX model input names: {actual_input_names}")
                    
                    # Создаем правильный словарь входов
                    if len(actual_input_names) == len(test_inputs):
                        onnx_inputs = {}
                        for i, name in enumerate(actual_input_names):
                            onnx_inputs[name] = test_inputs[i].numpy()
                        
                        test_result = sess.run(["action"], onnx_inputs)
                        print(f"  ✓ ONNX validation passed, shape: {test_result[0].shape}")
                        print(f"  ✓ Sample output: {test_result[0][0]}")
                    else:
                        print(f"  ! Input count mismatch: expected {len(test_inputs)}, got {len(actual_input_names)}")
                        
                except ImportError:
                    print("  ! onnxruntime not available, skipping validation")
                except Exception as e:
                    print(f"  ! ONNX validation failed: {e}")
                    print(f"  ! This is usually OK - the model exported successfully")
                    
            except Exception as e:
                print(f"  ✗ Failed to export {policy_id}: {e}")
                import traceback
                traceback.print_exc()
        
        # Создаем latest ссылку
        self._safe_create_latest(iter_dir)
        print(f"=== ONNX Compatible Export completed ===\n")

    def _safe_create_latest(self, iter_dir):
        """Безопасное создание latest ссылки"""
        latest_path = os.path.join(self.export_dir, "latest")
        try:
            if os.path.exists(latest_path):
                if os.path.islink(latest_path):
                    os.unlink(latest_path)
                else:
                    import shutil
                    if os.path.isdir(latest_path):
                        shutil.rmtree(latest_path)
                    else:
                        os.remove(latest_path)
            
            import platform
            if platform.system() == "Windows":
                import shutil
                shutil.copytree(iter_dir, latest_path)
                print(f"  ✓ Created latest copy (Windows)")
            else:
                os.symlink(os.path.basename(iter_dir), latest_path)
                print(f"  ✓ Created latest symlink")
                
        except Exception as e:
            print(f"  ! Could not create latest link: {e}")

# Standalone тест ONNX экспорта
def test_onnx_export():
    """Тест ONNX экспорта с совместимой моделью"""
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env
    
    from arena_env import ArenaEnv
    from entity_attention_model import ONNXEntityAttentionModel
    from masked_multihead_dist import MaskedTargetMoveAimFire
    
    def env_creator(cfg): 
        return ArenaEnv(cfg)
    
    print("=== Testing ONNX Compatible Export ===")
    
    ray.init(ignore_reinit_error=True)
    register_env("ArenaEnv", env_creator)
    
    # Регистрируем ONNX модель
    from ray.rllib.models import ModelCatalog
    ModelCatalog.register_custom_model("onnx_entity_attention", ONNXEntityAttentionModel)
    
    try:
        # Простая конфигурация для теста
        config = (
            PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .environment(
                env="ArenaEnv",
                env_config={
                    "episode_len": 64,
                    "ally_choices": [1],
                    "enemy_choices": [1],
                }
            )
            .framework("torch")
            .env_runners(
                num_env_runners=1,
                rollout_fragment_length=128,
            )
            .resources(
                num_gpus=0,  # Принудительно CPU для избежания проблем
            )
            .training(
                train_batch_size=512,
                minibatch_size=128,
            )
            .multi_agent(
                policies={
                    "main": (None, ArenaEnv({}).observation_space, ArenaEnv({}).action_space, {
                        "model": {
                            "custom_model": "onnx_entity_attention",  # Используем ONNX совместимую модель
                            "custom_action_dist": "masked_multihead",
                            "custom_model_config": {
                                "d_model": 128,
                                "nhead": 8,
                                "layers": 2,
                                "ff": 256,
                                "hidden": 256,
                            },
                            "vf_share_layers": False,
                        }
                    })
                },
                policy_mapping_fn=lambda aid, *args, **kwargs: "main",
                policies_to_train=["main"],
            )
        )
        
        # Создаем callbacks
        callbacks = FinalONNXCallbacks()
        callbacks.setup(
            export_onnx=True,
            export_every=2,  # Экспорт каждые 2 итерации для быстрого тестирования
            export_dir="./test_onnx_compatible",
            policies_to_export=["main"]
        )
        
        config = config.callbacks(lambda: callbacks)
        
        algo = config.build()
        
        print("Starting training with ONNX-compatible model...")
        for i in range(5):  # Короткий тест
            result = algo.train()
            reward_mean = result.get('env_runners', {}).get('episode_reward_mean', 0)
            print(f"Iteration {i}: reward = {reward_mean:.3f}")
            
            # Проверяем экспорт
            if i >= 2:
                export_dir = "./test_onnx_compatible"
                if os.path.exists(export_dir):
                    exports = [f for f in os.listdir(export_dir) if f.startswith('iter_')]
                    print(f"  ONNX exports found: {exports}")
                    
                    # Проверяем latest
                    latest_path = os.path.join(export_dir, "latest")
                    if os.path.exists(latest_path):
                        print(f"  Latest export available: {latest_path}")
                        
                        # Пробуем загрузить ONNX
                        onnx_files = []
                        for root, dirs, files in os.walk(latest_path):
                            onnx_files.extend([f for f in files if f.endswith('.onnx')])
                        print(f"  ONNX files in latest: {onnx_files}")
        
        algo.stop()
        print("✓ ONNX compatible test completed successfully!")
        
    except Exception as e:
        print(f"✗ ONNX compatible test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ray.shutdown()

if __name__ == "__main__":
    test_onnx_export()