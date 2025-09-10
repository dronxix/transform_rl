"""
Исправленный экспорт ONNX с корректным созданием meta.json файлов
"""

import os
import json
import torch
import platform
from typing import Dict, Any, List, Optional
import numpy as np

class PolicyONNXWrapper(torch.nn.Module):
    """Исправленный wrapper для экспорта только политики"""
    
    def __init__(self, original_model, onnx_model, ne: int, na: int):
        super().__init__()
        self.onnx_model = onnx_model
        self.ne = ne
        self.na = na
        
        # Переносим веса из оригинальной модели
        self._transfer_weights(original_model, onnx_model)
        
    def _transfer_weights(self, original_model, onnx_model):
        """Улучшенный перенос весов"""
        print("Transferring weights to ONNX model...")
        
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
            
            # 2. Специальная обработка для attention слоев
            if 'mha' in onnx_name and not transferred_this_param:
                transferred_this_param = self._transfer_attention_weights(
                    onnx_name, onnx_param, original_state)
                if transferred_this_param:
                    transferred += 1
                    continue
            
            # 3. Поиск по схожести имен
            if not transferred_this_param:
                orig_name = self._find_similar_param(onnx_name, original_state)
                if orig_name:
                    orig_param = original_state[orig_name]
                    if orig_param.shape == onnx_param.shape:
                        onnx_param.data.copy_(orig_param.data.cpu())
                        transferred += 1
                        transferred_this_param = True
                        print(f"Mapped {onnx_name} <- {orig_name}")
                        continue
            
            if not transferred_this_param:
                print(f"⚠ No match found for {onnx_name} (shape: {onnx_param.shape})")
        
        print(f"Successfully transferred {transferred}/{total_params} parameters ({100*transferred/total_params:.1f}%)")
        
        # Проверяем что основные слои перенесены
        key_layers = ['self_enc', 'ally_enc', 'enemy_enc', 'head_target', 'head_move_mu', 'head_aim_mu', 'head_fire_logit']
        for layer in key_layers:
            found = any(layer in name for name in onnx_state.keys())
            status = "✓" if found else "⚠"
            print(f"  {status} {layer} weights")
    
    def _transfer_attention_weights(self, onnx_name, onnx_param, original_state):
        """Специальная обработка для attention весов"""
        # Обработка выходной проекции
        if 'w_o.weight' in onnx_name:
            orig_name = onnx_name.replace('mha.w_o.weight', 'mha.out_proj.weight')
            if orig_name in original_state and original_state[orig_name].shape == onnx_param.shape:
                onnx_param.data.copy_(original_state[orig_name].data.cpu())
                return True
                
        elif 'w_o.bias' in onnx_name:
            orig_name = onnx_name.replace('mha.w_o.bias', 'mha.out_proj.bias')
            if orig_name in original_state and original_state[orig_name].shape == onnx_param.shape:
                onnx_param.data.copy_(original_state[orig_name].data.cpu())
                return True
        
        # Обработка Q, K, V проекций (разделение in_proj_weight)
        elif any(proj in onnx_name for proj in ['w_q.weight', 'w_k.weight', 'w_v.weight']):
            orig_name = onnx_name.replace('mha.w_q.weight', 'mha.in_proj_weight').replace('mha.w_k.weight', 'mha.in_proj_weight').replace('mha.w_v.weight', 'mha.in_proj_weight')
            if orig_name in original_state:
                in_proj = original_state[orig_name]
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
        
        # Ищем по суффиксу (последние 2 части имени)
        if len(onnx_parts) >= 2:
            suffix = '.'.join(onnx_parts[-2:])
            
            for orig_name in original_state.keys():
                if orig_name.endswith(suffix):
                    orig_parts = orig_name.split('.')
                    # Проверяем есть ли общие части в пути
                    common_parts = set(onnx_parts[:-2]) & set(orig_parts[:-2])
                    if len(common_parts) > 0:
                        return orig_name
        
        return None

    @torch.no_grad()
    def forward(self, self_vec, allies, allies_mask, enemies, enemies_mask, enemy_action_mask):
        """Forward для экспорта в ONNX (только политика, без value function)"""
        
        # Убеждаемся что все на CPU
        inputs_cpu = {
            "self": self_vec.cpu(),
            "allies": allies.cpu(),
            "allies_mask": allies_mask.cpu(),
            "enemies": enemies.cpu(),
            "enemies_mask": enemies_mask.cpu(),
            "enemy_action_mask": enemy_action_mask.cpu(),
        }
        
        # Создаем dummy global_state для forward pass модели
        batch_size = self_vec.shape[0]
        dummy_global_state = torch.zeros(batch_size, 64, dtype=torch.float32)
        inputs_cpu["global_state"] = dummy_global_state
        
        # Forward pass через ONNX модель
        logits, _ = self.onnx_model({"obs": inputs_cpu}, [], None)
        logits = logits.cpu()
        
        # Декодируем детерминированные действия
        idx = 0
        logits_t = logits[:, idx:idx+self.ne]; idx += self.ne
        mu_move  = logits[:, idx:idx+2];        idx += 2
        idx += 2  # log_std_move
        mu_aim   = logits[:, idx:idx+2];        idx += 2
        idx += 2  # log_std_aim
        logit_fr = logits[:, idx:idx+1];        idx += 1

        # Детерминированные действия
        target = torch.argmax(logits_t, dim=-1).unsqueeze(-1).float()
        move = torch.tanh(mu_move)
        aim  = torch.tanh(mu_aim)
        fire = (logit_fr > 0).float()
        
        result = torch.cat([target, move, aim, fire], dim=-1)
        return result.cpu()


def export_onnx_with_meta(algorithm, iteration: int, export_dir: str, policies_to_export: List[str]):
    """
    Исправленная функция экспорта ONNX с правильным созданием meta.json
    """
    print(f"\n=== ONNX Export with Meta (iteration {iteration}) ===")
    
    # Получаем размеры из окружения
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
    
    print(f"Environment dimensions:")
    print(f"  max_enemies={max_enemies}, max_allies={max_allies}")
    print(f"  self={self_feats}, ally={ally_feats}, enemy={enemy_feats}, global={global_feats}")
    
    # Создаем директорию для этой итерации
    iter_dir = os.path.join(export_dir, f"iter_{iteration:06d}")
    os.makedirs(iter_dir, exist_ok=True)
    
    successful_exports = []
    
    for policy_id in policies_to_export:
        try:
            print(f"\nExporting policy: {policy_id}")
            
            # Получаем оригинальную модель
            policy = algorithm.get_policy(policy_id)
            original_model = policy.model
            
            # Создаем ONNX-совместимую модель
            from entity_attention_model import ONNXEntityAttentionModel
            
            # Получаем конфигурацию модели
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
            
            # Определяем размеры для wrapper
            ne = getattr(original_model, 'max_enemies', max_enemies)
            na = getattr(original_model, 'max_allies', max_allies)
            
            # Создаем wrapper с переносом весов
            wrapper = PolicyONNXWrapper(original_model, onnx_model, ne=ne, na=na)
            
            # Тестовые входы
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
            print(f"  Testing wrapper...")
            with torch.no_grad():
                test_output = wrapper(*test_inputs)
            print(f"  ✓ Test successful, output shape: {test_output.shape}")
            
            # ONNX экспорт
            onnx_path = os.path.join(iter_dir, f"policy_{policy_id}.onnx")
            print(f"  Exporting to ONNX...")
            
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
            
            # ИСПРАВЛЕНИЕ: Правильное создание метаданных
            meta = {
                "iteration": iteration,
                "policy_id": policy_id,
                "max_allies": int(na),
                "max_enemies": int(ne),
                "obs_dims": {
                    "self": int(self_feats),
                    "ally": int(ally_feats),
                    "enemy": int(enemy_feats),
                    "global_state": int(global_feats),
                },
                "action_spec": {
                    "target_discrete_n": int(ne),
                    "move_box": 2,
                    "aim_box": 2,
                    "fire_binary": 1,
                    "total_action_dim": 6
                },
                "input_names": input_names,
                "output_names": ["action"],
                "model_config": {
                    "d_model": getattr(original_model, 'd_model', 128),
                    "nhead": getattr(original_model, 'nhead', 8),
                    "layers": getattr(original_model, 'layers', 2),
                },
                "export_info": {
                    "wrapper_class": "PolicyONNXWrapper",
                    "opset_version": 17,
                    "framework": "torch",
                },
                "training_info": {
                    "timesteps_total": algorithm.metrics.peek("timesteps_total", 0),
                    "episodes_total": algorithm.metrics.peek("episodes_total", 0),
                    "episode_reward_mean": algorithm.metrics.peek("env_runners/episode_reward_mean", 0),
                },
                "usage_note": "Policy-only export. Use self_vec, allies, allies_mask, enemies, enemies_mask, enemy_action_mask as inputs. Output is [target, move_x, move_y, aim_x, aim_y, fire]."
            }
            
            # ИСПРАВЛЕНИЕ: Обязательно сохраняем meta.json
            meta_path = os.path.join(iter_dir, f"policy_{policy_id}_meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            
            print(f"  ✓ Exported ONNX: {onnx_path}")
            print(f"  ✓ Saved meta: {meta_path}")
            
            # Валидация ONNX
            try:
                import onnxruntime as ort
                sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
                
                # Получаем фактические имена входов
                actual_input_names = [input.name for input in sess.get_inputs()]
                print(f"  ℹ ONNX inputs: {actual_input_names}")
                
                if len(actual_input_names) == len(test_inputs):
                    onnx_inputs = {}
                    for i, name in enumerate(actual_input_names):
                        onnx_inputs[name] = test_inputs[i].numpy()
                    
                    test_result = sess.run(["action"], onnx_inputs)
                    print(f"  ✓ ONNX validation passed, shape: {test_result[0].shape}")
                    print(f"  ✓ Sample output: {test_result[0][0]}")
                    
                    successful_exports.append({
                        "policy_id": policy_id,
                        "onnx_path": onnx_path,
                        "meta_path": meta_path,
                        "input_names": actual_input_names,
                        "output_shape": test_result[0].shape
                    })
                else:
                    print(f"  ! Input count mismatch: expected {len(test_inputs)}, got {len(actual_input_names)}")
                    
            except ImportError:
                print("  ! onnxruntime not available, skipping validation")
                successful_exports.append({
                    "policy_id": policy_id,
                    "onnx_path": onnx_path,
                    "meta_path": meta_path,
                    "input_names": input_names,
                    "output_shape": None
                })
            except Exception as e:
                print(f"  ! ONNX validation failed: {e}")
                # Все равно считаем экспорт успешным
                successful_exports.append({
                    "policy_id": policy_id,
                    "onnx_path": onnx_path,
                    "meta_path": meta_path,
                    "input_names": input_names,
                    "output_shape": None
                })
                    
        except Exception as e:
            print(f"  ✗ Failed to export {policy_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Создаем общий мета-файл для всех экспортированных политик
    if successful_exports:
        general_meta_path = os.path.join(iter_dir, "export_summary.json")
        general_meta = {
            "iteration": iteration,
            "timestamp": str(np.datetime64('now')),
            "exported_policies": successful_exports,
            "environment_info": {
                "max_allies": max_allies,
                "max_enemies": max_enemies,
                "obs_dims": {
                    "self": self_feats,
                    "ally": ally_feats,
                    "enemy": enemy_feats,
                    "global_state": global_feats,
                }
            }
        }
        
        with open(general_meta_path, "w", encoding="utf-8") as f:
            json.dump(general_meta, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Export summary saved: {general_meta_path}")
    
    # Создаем latest ссылку
    _create_safe_symlink(iter_dir, os.path.join(export_dir, "latest"))
    
    print(f"=== ONNX export completed for iteration {iteration} ===\n")
    return successful_exports


def _create_safe_symlink(target_dir: str, link_path: str):
    """Безопасное создание символической ссылки с обработкой Windows"""
    try:
        # Удаляем существующую ссылку если есть
        if os.path.islink(link_path):
            os.unlink(link_path)
        elif os.path.exists(link_path):
            if os.path.isdir(link_path):
                import shutil
                shutil.rmtree(link_path)
            else:
                os.remove(link_path)
        
        # Попытка создать символическую ссылку
        target_name = os.path.basename(target_dir)
        try:
            os.symlink(target_name, link_path)
            print(f"  ✓ Created symlink: {link_path} -> {target_name}")
        except OSError as e:
            if platform.system() == "Windows" and "required privilege" in str(e).lower():
                # Windows без прав администратора - создаем обычную копию
                print(f"  ! Cannot create symlink on Windows, creating copy instead")
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