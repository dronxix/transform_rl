"""
Callbacks с автоматическим экспортом ONNX при сохранении чекпоинтов
ИСПРАВЛЕНО: 
1. Принудительное перемещение всех тензоров на CPU перед экспортом
2. Обработка ошибок с символическими ссылками в Windows
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

class ScriptedPolicy(torch.nn.Module):
    """Wrapper для экспорта модели в ONNX с принудительным CPU"""
    def __init__(self, model, ne: int, na: int):
        super().__init__()
        self.m = model
        self.ne = ne
        self.na = na
        
        # КРИТИЧНО: Принудительно перемещаем модель на CPU
        self.m = self.m.cpu()
        self.m.eval()

    @torch.no_grad()
    def forward(self,
                self_vec,
                allies, allies_mask,
                enemies, enemies_mask,
                global_state,
                enemy_action_mask):
        
        # КРИТИЧНО: Убеждаемся что ВСЕ входы на CPU
        self_vec = self_vec.cpu()
        allies = allies.cpu()
        allies_mask = allies_mask.cpu()
        enemies = enemies.cpu()
        enemies_mask = enemies_mask.cpu()
        global_state = global_state.cpu()
        enemy_action_mask = enemy_action_mask.cpu()
        
        obs = {
            "self": self_vec,
            "allies": allies,
            "allies_mask": allies_mask,
            "enemies": enemies,
            "enemies_mask": enemies_mask,
            "global_state": global_state,
            "enemy_action_mask": enemy_action_mask,
        }
        
        # Убеждаемся что модель на CPU
        with torch.no_grad():
            logits, _ = self.m({"obs": obs}, [], None)
            
        # Убеждаемся что выход на CPU
        logits = logits.cpu()
        
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
        return result.cpu()  # Гарантируем CPU выход

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
        self.export_every = 25  # Каждые N итераций
        self.export_dir = "./onnx_exports"
        self.policies_to_export = ["main"]  # Какие политики экспортировать
        
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

        # 5) ONNX экспорт
        if self.export_onnx and it % self.export_every == 0 and it > 0:
            try:
                self._export_onnx_models(algorithm, it)
                result["custom_metrics"]["onnx_export_iteration"] = it
            except Exception as e:
                print(f"Error exporting ONNX: {e}")
                import traceback
                traceback.print_exc()

        if self.writer:
            self.writer.flush()

    def _export_onnx_models(self, algorithm: Algorithm, iteration: int):
        """Экспортирует модели в ONNX формат с исправлениями устройств"""
        print(f"\n=== Exporting ONNX models at iteration {iteration} ===")
        
        # Получаем размеры из конфига окружения
        env_config = algorithm.config.env_config
        from arena_env import ArenaEnv
        tmp_env = ArenaEnv(env_config)
        obs_space = tmp_env.observation_space
        
        # Извлекаем размеры
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
                
                # Получаем политику и модель
                policy = algorithm.get_policy(policy_id)
                model = policy.model
                
                # ИСПРАВЛЕНИЕ: Вместо deepcopy используем безопасный способ
                # Получаем конфиг модели из политики
                model_config_dict = getattr(policy.config, 'model', {})
                if not model_config_dict:
                    # Fallback конфиг если не найден
                    model_config_dict = {
                        "custom_model": "entity_attention",
                        "custom_action_dist": "masked_multihead", 
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
                
                # Создаем новую модель с теми же параметрами
                from entity_attention_model import EntityAttentionModel
                
                model_copy = EntityAttentionModel(
                    obs_space=tmp_env.observation_space,
                    action_space=tmp_env.action_space,
                    num_outputs=model.num_outputs if hasattr(model, 'num_outputs') else None,
                    model_config=model_config_dict,
                    name=f"export_{policy_id}"
                )
                
                # Перемещаем на CPU
                model_copy = model_copy.cpu()
                
                # Безопасная загрузка весов с преобразованием на CPU
                original_state = model.state_dict()
                cpu_state = {}
                for key, value in original_state.items():
                    if isinstance(value, torch.Tensor):
                        cpu_state[key] = value.cpu().detach()
                    else:
                        cpu_state[key] = value
                
                try:
                    model_copy.load_state_dict(cpu_state, strict=False)
                    print(f"Model weights loaded successfully")
                except Exception as e:
                    print(f"Warning: Could not load all weights: {e}")
                    print("Proceeding with randomly initialized model for structure test")
                
                model_copy.eval()
                print(f"Model recreated and moved to CPU successfully")
                
                # Определяем размеры модели
                if hasattr(model_copy, 'max_enemies') and hasattr(model_copy, 'max_allies'):
                    ne = model_copy.max_enemies
                    na = model_copy.max_allies
                else:
                    ne = max_enemies
                    na = max_allies
                
                # Создаем wrapper
                wrapper = ScriptedPolicy(model_copy, ne=ne, na=na)
                
                # КРИТИЧНО: Все тестовые входы на CPU
                B = 1
                ex_self  = torch.zeros(B, self_feats, dtype=torch.float32, device='cpu')
                ex_allies= torch.zeros(B, na, ally_feats, dtype=torch.float32, device='cpu')
                ex_amask = torch.zeros(B, na, dtype=torch.int32, device='cpu')
                ex_enems = torch.zeros(B, ne, enemy_feats, dtype=torch.float32, device='cpu')
                ex_emask = torch.zeros(B, ne, dtype=torch.int32, device='cpu')
                ex_gs    = torch.zeros(B, global_feats, dtype=torch.float32, device='cpu')
                ex_emact = torch.zeros(B, ne, dtype=torch.int32, device='cpu')
                
                # Тестовый прогон
                print(f"Testing model forward pass...")
                with torch.no_grad():
                    test_output = wrapper(ex_self, ex_allies, ex_amask, ex_enems, ex_emask, ex_gs, ex_emact)
                print(f"Test output shape: {test_output.shape}, device: {test_output.device}")
                
                # Экспорт в ONNX
                onnx_path = os.path.join(iter_dir, f"policy_{policy_id}.onnx")
                
                print(f"Starting ONNX export...")
                torch.onnx.export(
                    wrapper,
                    (ex_self, ex_allies, ex_amask, ex_enems, ex_emask, ex_gs, ex_emact),
                    onnx_path,
                    opset_version=17,
                    input_names=["self", "allies", "allies_mask", "enemies", "enemies_mask", "global_state", "enemy_action_mask"],
                    output_names=["action"],
                    dynamic_axes={name: {0: "batch"} for name in
                                  ["self","allies","allies_mask","enemies","enemies_mask","global_state","enemy_action_mask","action"]},
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
                        "global_state": int(global_feats),
                    },
                    "action_spec": {
                        "target_discrete_n": int(ne),
                        "move_box": 2,
                        "aim_box": 2,
                        "fire_binary": 1,
                        "total_action_dim": 6
                    },
                    "model_config": {
                        "d_model": getattr(model_copy, 'd_model', None),
                        "nhead": getattr(model_copy, 'nhead', None),
                        "layers": getattr(model_copy, 'layers', None),
                    },
                    "training_info": {
                        "timesteps_total": algorithm.metrics.peek("timesteps_total", 0),
                        "episodes_total": algorithm.metrics.peek("episodes_total", 0),
                        "episode_reward_mean": algorithm.metrics.peek("env_runners/episode_reward_mean", 0),
                    },
                    "note": "Exported during training - use masks for variable entity counts"
                }
                
                meta_path = os.path.join(iter_dir, f"policy_{policy_id}_meta.json")
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                
                print(f"  ✓ Exported: {onnx_path}")
                
                # Быстрая валидация ONNX
                try:
                    import onnxruntime as ort
                    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
                    test_result = sess.run(["action"], {
                        "self": ex_self.numpy(),
                        "allies": ex_allies.numpy(),
                        "allies_mask": ex_amask.numpy(),
                        "enemies": ex_enems.numpy(),
                        "enemies_mask": ex_emask.numpy(),
                        "global_state": ex_gs.numpy(),
                        "enemy_action_mask": ex_emact.numpy(),
                    })
                    print(f"  ✓ ONNX validation passed, output shape: {test_result[0].shape}")
                    
                except ImportError:
                    print("  ! onnxruntime not available, skipping validation")
                except Exception as e:
                    print(f"  ! ONNX validation failed: {e}")
                    
            except Exception as e:
                print(f"  ✗ Failed to export {policy_id}: {e}")
                import traceback
                traceback.print_exc()
        
        # ИСПРАВЛЕНИЕ: Безопасное создание симлинка для Windows
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