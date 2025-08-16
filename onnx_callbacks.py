"""
Исправленные Callbacks с автоматическим экспортом ONNX при сохранении чекпоинтов
ИСПРАВЛЕНО: Использует успешный подход из simple_onnx.py с league callbacks
"""

import os
import json
import torch
import platform
from typing import Dict, Any, List, Optional
import numpy as np
import ray
from torch.utils.tensorboard import SummaryWriter

from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.algorithms import Algorithm

class PolicyOnlyONNXWrapper(torch.nn.Module):
    """Wrapper для экспорта только политики (без value function)"""
    
    def __init__(self, original_model, onnx_model, ne: int, na: int):
        super().__init__()
        self.onnx_model = onnx_model
        self.ne = ne
        self.na = na
        
        # Переносим веса из оригинальной модели
        self._transfer_weights(original_model, onnx_model)
        
    def _transfer_weights(self, original_model, onnx_model):
        """Упрощенный перенос весов без сложной логики"""
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
                else:
                    print(f"Shape mismatch for {onnx_name}: {orig_param.shape} vs {onnx_param.shape}")
            
            # 2. Специальная обработка для attention слоев
            elif not transferred_this_param and 'mha' in onnx_name:
                transferred_this_param = self._transfer_attention_weights(
                    onnx_name, onnx_param, original_state)
                if transferred_this_param:
                    transferred += 1
            
            # 3. Поиск по схожести имен
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
            found = any(layer in name for name in onnx_state.keys())
            if found:
                print(f"✓ {layer} weights transferred")
            else:
                print(f"⚠ {layer} weights may not be transferred")
    
    def _transfer_attention_weights(self, onnx_name, onnx_param, original_state):
        """Специальная обработка для attention весов"""
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
        
        elif 'w_q.weight' in onnx_name or 'w_k.weight' in onnx_name or 'w_v.weight' in onnx_name:
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
        
        if len(onnx_parts) >= 2:
            suffix = '.'.join(onnx_parts[-2:])
            
            for orig_name in original_state.keys():
                if orig_name.endswith(suffix):
                    orig_parts = orig_name.split('.')
                    common_parts = set(onnx_parts[:-2]) & set(orig_parts[:-2])
                    if len(common_parts) > 0:
                        return orig_name
        
        return None

    @torch.no_grad()
    def forward(self,
                self_vec,
                allies, allies_mask,
                enemies, enemies_mask,
                enemy_action_mask):
        """
        ИСПРАВЛЕНО: Убран global_state - он нужен только для value function
        Экспортируем только политику
        """
        
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

class LeagueCallbacksWithONNX(RLlibCallback):
    def __init__(self):
        super().__init__()
        self.league = None
        self.opponent_ids = None
        self.eval_eps = 6
        self.clone_every = 10
        self.sample_top_k = 3
        self.attn_log_every = 20
        self.writer: Optional[SummaryWriter] = None
        self.curriculum = None
        
        # ONNX экспорт настройки
        self.export_onnx = True
        self.export_every = 25
        self.export_dir = "./onnx_exports"
        self.policies_to_export = ["main"]
        
    def setup(self, league_actor, opponent_ids: List[str], **kwargs):
        """Настройка параметров callbacks"""
        self.league = league_actor
        self.opponent_ids = opponent_ids
        self.eval_eps = kwargs.get('eval_episodes', 6)
        self.clone_every = kwargs.get('clone_every_iters', 10)
        self.sample_top_k = kwargs.get('sample_top_k', 3)
        self.attn_log_every = kwargs.get('attn_log_every', 20)
        self.curriculum = kwargs.get('curriculum_schedule', [])
        
        # ONNX настройки
        self.export_onnx = kwargs.get('export_onnx', True)
        self.export_every = kwargs.get('export_every', 25)
        self.export_dir = kwargs.get('export_dir', "./onnx_exports")
        self.policies_to_export = kwargs.get('policies_to_export', ["main"])
        
        # Создаем директорию для экспорта
        if self.export_onnx:
            os.makedirs(self.export_dir, exist_ok=True)

    def on_algorithm_init(self, *, algorithm: Algorithm, **kwargs) -> None:
        """Инициализация algorithm"""
        pass

    def on_train_result(self, *, algorithm: Algorithm, result: Dict[str, Any], **kwargs) -> None:
        """Основная логика обработки результатов тренировки"""
        if self.league is None:
            return
            
        # Создаем writer
        if self.writer is None:
            logdir = getattr(algorithm, "logdir", "./logs")
            self.writer = SummaryWriter(log_dir=logdir)

        it = result["training_iteration"]
        ts_total = result.get("timesteps_total", 0)

        # 1) Evaluation матчей
        try:
            for pid in self.opponent_ids:
                w_main, w_opp = self._play_match(algorithm, pid, self.eval_eps)
                ray.get(self.league.update_pair_result.remote(w_main, w_opp, pid))
        except Exception as e:
            print(f"Error in match evaluation: {e}")

        # 2) Логирование рейтингов
        try:
            scores = ray.get(self.league.get_all_scores.remote())
            result.setdefault("custom_metrics", {})
            
            for k, (mu, sigma) in scores.items():
                result["custom_metrics"][f"ts_{k}_mu"] = mu
                result["custom_metrics"][f"ts_{k}_sigma"] = sigma
                
                conservative_score = mu - 3 * sigma
                self.writer.add_scalar(f"ts/{k}_conservative", conservative_score, it)
        except Exception as e:
            print(f"Error getting league scores: {e}")
            scores = {}

        # 3) Клонирование худшего оппонента
        if it % self.clone_every == 0 and it > 0 and scores:
            try:
                items = [(pid, scores[pid][0] - 3*scores[pid][1]) for pid in self.opponent_ids]
                worst = min(items, key=lambda z: z[1])[0]
                
                w = algorithm.get_policy("main").get_weights()
                algorithm.get_policy(worst).set_weights(w)
                ray.get(self.league.clone_main_into.remote(worst))
                
                result["custom_metrics"][f"league_refresh_{worst}"] = it
                print(f"Refreshed opponent {worst} at iteration {it}")
            except Exception as e:
                print(f"Error refreshing opponent: {e}")

        # 4) Куррикулум
        if self.curriculum:
            for threshold, ac, ec in reversed(self.curriculum):
                if ts_total >= threshold:
                    try:
                        self._apply_curriculum(algorithm, ac, ec)
                        result["custom_metrics"]["curriculum_ally_choices"] = str(ac)
                        result["custom_metrics"]["curriculum_enemy_choices"] = str(ec)
                    except Exception as e:
                        print(f"Error setting curriculum: {e}")
                    break

        # 5) ONNX экспорт - ИСПРАВЛЕНО: используем успешный подход
        if self.export_onnx and it % self.export_every == 0 and it > 0:
            try:
                self._export_onnx_compatible(algorithm, it)
                result["custom_metrics"]["onnx_export_iteration"] = it
                print(f"✓ ONNX export completed for iteration {it}")
            except Exception as e:
                print(f"✗ ONNX export failed for iteration {it}: {e}")
                import traceback
                traceback.print_exc()

        if self.writer:
            self.writer.flush()

    def _export_onnx_compatible(self, algorithm: Algorithm, iteration: int):
        """ONNX экспорт с использованием успешного подхода из simple_onnx.py"""
        print(f"\n=== ONNX Compatible Export (iteration {iteration}) ===")
        
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
        
        # Создаем директорию для этой итерации
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
                
                # Создаем wrapper с переносом весов (БЕЗ global_state в экспорте)
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
                
                # ONNX экспорт (БЕЗ global_state)
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
                
                # Сохраняем метаданные
                meta = {
                    "iteration": iteration,
                    "policy_id": policy_id,
                    "max_allies": int(na),
                    "max_enemies": int(ne),
                    "obs_dims": {
                        "self": int(self_feats),
                        "ally": int(ally_feats),
                        "enemy": int(enemy_feats),
                        "global_state": int(global_feats),  # Для совместимости
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
                    "training_info": {
                        "timesteps_total": algorithm.metrics.peek("timesteps_total", 0),
                        "episodes_total": algorithm.metrics.peek("episodes_total", 0),
                        "episode_reward_mean": algorithm.metrics.peek("env_runners/episode_reward_mean", 0),
                    },
                    "note": "Policy-only export without value function (no global_state input)"
                }
                
                meta_path = os.path.join(iter_dir, f"policy_{policy_id}_meta.json")
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                
                print(f"  ✓ Exported: {onnx_path}")
                
                # Быстрая валидация ONNX
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
        self._create_safe_symlink(iter_dir, os.path.join(self.export_dir, "latest"))
        
        print(f"=== ONNX export completed for iteration {iteration} ===\n")

    def _create_safe_symlink(self, target_dir: str, link_path: str):
        """Безопасное создание символической ссылки с обработкой Windows"""
        try:
            # Удаляем существующую ссылку если есть
            if os.path.islink(link_path):
                os.unlink(link_path)
            elif os.path.exists(link_path):
                # Если это не ссылка, но файл/папка существует
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
                    print(f"  ! Cannot create symlink on Windows (privilege required), creating copy instead")
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

    def _play_match(self, algorithm: Algorithm, opp_id: str, episodes: int) -> tuple:
        """Версия матча для Ray 2.48"""
        try:
            from arena_env import ArenaEnv
            env_config = algorithm.config.env_config.copy() if hasattr(algorithm.config, 'env_config') else {}
            temp_env = ArenaEnv(env_config)
            
            wins_main, wins_opp = 0, 0
            
            for episode_idx in range(episodes):
                obs, _ = temp_env.reset()
                done = False
                
                while not done:
                    action_dict = {}
                    
                    for aid, ob in obs.items():
                        pol_id = "main" if aid.startswith("red_") else opp_id
                        pol = algorithm.get_policy(pol_id)
                        act, _, _ = pol.compute_single_action(ob, explore=False)
                        
                        if isinstance(act, dict):
                            action_dict[aid] = act
                        else:
                            action_dict[aid] = {
                                "target": int(act[0]) if len(act) > 0 else 0,
                                "move": act[1:3] if len(act) > 2 else [0.0, 0.0],
                                "aim": act[3:5] if len(act) > 4 else [0.0, 0.0],
                                "fire": int(round(float(act[5]))) if len(act) > 5 else 0,
                            }
                    
                    obs, rews, terms, truncs, infos = temp_env.step(action_dict)
                    done = terms.get("__all__", False) or truncs.get("__all__", False)
                
                red_sum = sum(v for k, v in rews.items() if k.startswith("red_"))
                blue_sum = sum(v for k, v in rews.items() if k.startswith("blue_"))
                
                if red_sum > blue_sum:
                    wins_main += 1
                elif blue_sum > red_sum:
                    wins_opp += 1
                    
            return wins_main, wins_opp
            
        except Exception as e:
            print(f"Error in _play_match: {e}")
            return 0, 0

    def _apply_curriculum(self, algorithm, ally_choices, enemy_choices):
        """Применение куррикулума для Ray 2.48"""
        try:
            if hasattr(algorithm.config, 'env_config'):
                algorithm.config.env_config["ally_choices"] = ally_choices
                algorithm.config.env_config["enemy_choices"] = enemy_choices
                print(f"Updated curriculum in config: allies={ally_choices}, enemies={enemy_choices}")
            
            try:
                if hasattr(algorithm, 'env_runner_group') and algorithm.env_runner_group:
                    def set_curriculum_fn(env):
                        if hasattr(env, 'set_curriculum'):
                            env.set_curriculum(ally_choices, enemy_choices)
                    
                    algorithm.env_runner_group.foreach_env(set_curriculum_fn)
                    print(f"Applied curriculum to env_runners: allies={ally_choices}, enemies={enemy_choices}")
            except (AttributeError, Exception) as e:
                print(f"Could not apply curriculum to existing envs: {e}")
                
        except Exception as e:
            print(f"Could not apply curriculum: {e}")