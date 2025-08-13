"""
Callbacks - ПРАВИЛЬНО обновлены для Ray 2.48.0
Изменения:
- Использование env_runners вместо workers
- Правильные API для эпизодов
- Обновленная работа с метриками
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import ray
import torch
from torch.utils.tensorboard import SummaryWriter
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID
from ray.rllib.algorithms import Algorithm
from ray.rllib.env.env_runner import EnvRunner

class LeagueCallbacks(DefaultCallbacks):
    def __init__(self, 
                 league_actor, 
                 opponent_ids: List[str],
                 eval_episodes=6, 
                 clone_every_iters=10, 
                 sample_top_k=3,
                 attn_log_every=20,
                 curriculum_schedule=None):
        super().__init__()
        self.league = league_actor
        self.opponent_ids = opponent_ids
        self.eval_eps = int(eval_episodes)
        self.clone_every = int(clone_every_iters)
        self.sample_top_k = int(sample_top_k)
        self.attn_log_every = int(attn_log_every)
        self.writer: Optional[SummaryWriter] = None
        
        # Куррикулум: [(timestep_threshold, ally_choices, enemy_choices)]
        self.curriculum = curriculum_schedule or [
            (0,           [1],         [1]),
            (1_000_000,   [1, 2],      [1, 2]),
            (5_000_000,   [1, 2, 3],   [1, 2, 3]),
            (10_000_000,  [1, 2, 3, 4],[1, 2, 3, 4]),
        ]

    def on_episode_start(
        self,
        *,
        env_runner: Optional[EnvRunner] = None,  # Ray 2.48 - env_runner вместо worker
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        episode: Episode,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Выбор соперника на старте эпизода."""
        opp = ray.get(self.league.get_opponent_weighted.remote(self.sample_top_k))
        episode.user_data["opp_id"] = opp
        episode.user_data["battle_start"] = True

    def on_episode_step(
        self,
        *,
        env_runner: Optional[EnvRunner] = None,  # Ray 2.48
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        episode: Episode,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Сбор метрик во время эпизода."""
        # Ray 2.48 - обновленный способ доступа к infos
        last_info = episode.last_info_for()  # получаем последний info
        if last_info and isinstance(last_info, dict):
            # Сохраняем максимальные значения за эпизод
            for key in ["invalid_target", "oob_move", "oob_aim"]:
                if key in last_info:
                    current = episode.user_data.get(f"max_{key}", 0)
                    episode.user_data[f"max_{key}"] = max(current, last_info[key])

    def on_episode_end(
        self,
        *,
        env_runner: Optional[EnvRunner] = None,  # Ray 2.48
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        episode: Episode,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Финализация метрик эпизода."""
        # Ray 2.48 - правильный доступ к custom metrics
        episode.custom_metrics["invalid_target"] = episode.user_data.get("max_invalid_target", 0)
        episode.custom_metrics["oob_move"] = episode.user_data.get("max_oob_move", 0)
        episode.custom_metrics["oob_aim"] = episode.user_data.get("max_oob_aim", 0)
        
        # Добавляем информацию о сопернике
        opp_id = episode.user_data.get("opp_id", self.opponent_ids[0])
        if opp_id in self.opponent_ids:
            episode.custom_metrics["opponent_id"] = self.opponent_ids.index(opp_id)

    def _play_match(self, algorithm: Algorithm, opp_id: str, episodes: int) -> tuple[int, int]:
        """Играет матч между main и оппонентом для оценки."""
        # Ray 2.48 - обновленный доступ к env_runners
        local_env_runner = algorithm.env_runner_group.local_env_runner
        if local_env_runner is None:
            # Fallback на старый API если новый не работает
            local_env_runner = algorithm.workers.local_worker()
            
        env = local_env_runner.env
        
        wins_main, wins_opp = 0, 0
        
        for _ in range(episodes):
            obs, _ = env.reset()
            done = False
            
            while not done:
                action_dict = {}
                
                for aid, ob in obs.items():
                    pol_id = "main" if aid.startswith("red_") else opp_id
                    pol = algorithm.get_policy(pol_id)
                    
                    # Ray 2.48 - compute_single_action все еще поддерживается
                    act, _, _ = pol.compute_single_action(ob, explore=False)
                    
                    action_dict[aid] = {
                        "target": int(act[0]),
                        "move": act[1:3],
                        "aim": act[3:5],
                        "fire": int(round(float(act[5]))),
                    }
                
                obs, rews, terms, truncs, infos = env.step(action_dict)
                done = terms.get("__all__", False) or truncs.get("__all__", False)
            
            # Подсчет победителя
            red_sum = sum(v for k, v in rews.items() if k.startswith("red_"))
            blue_sum = sum(v for k, v in rews.items() if k.startswith("blue_"))
            
            if red_sum > blue_sum:
                wins_main += 1
            elif blue_sum > red_sum:
                wins_opp += 1
                
        return wins_main, wins_opp

    def on_train_result(
        self,
        *,
        algorithm: Algorithm,
        result: Dict[str, Any],
        **kwargs,
    ) -> None:
        """Обработка результатов тренировки."""
        # Создаем TensorBoard writer
        if self.writer is None:
            logdir = result.get("logdir", None) or getattr(algorithm, "logdir", "./logs")
            self.writer = SummaryWriter(log_dir=logdir)

        it = result["training_iteration"]
        ts_total = int(result.get("timesteps_total", 0))

        # 1) Оценка через TrueSkill
        for pid in self.opponent_ids:
            try:
                w_main, w_opp = self._play_match(algorithm, pid, self.eval_eps)
                self.league.update_pair_result.remote(w_main, w_opp, pid)
            except Exception as e:
                print(f"Error in match evaluation against {pid}: {e}")
                continue

        # Получаем и логируем рейтинги
        try:
            scores = ray.get(self.league.get_all_scores.remote())
            for k, (mu, sigma) in scores.items():
                # Ray 2.48 - сохраняем в custom_metrics вместо прямого result
                result.setdefault("custom_metrics", {})
                result["custom_metrics"][f"ts/{k}_mu"] = mu
                result["custom_metrics"][f"ts/{k}_sigma"] = sigma
                conservative_score = mu - 3 * sigma
                self.writer.add_scalar(f"ts/{k}_conservative", conservative_score, it)
        except Exception as e:
            print(f"Error getting league scores: {e}")

        # 2) Освежение худшего оппонента
        if it % self.clone_every == 0 and it > 0:
            try:
                items = [(pid, scores[pid][0] - 3*scores[pid][1]) for pid in self.opponent_ids]
                worst = min(items, key=lambda z: z[1])[0]
                
                # Клонируем веса main в худшего оппонента
                w = algorithm.get_policy("main").get_weights()
                algorithm.get_policy(worst).set_weights(w)
                self.league.clone_main_into.remote(worst)
                
                result.setdefault("custom_metrics", {})
                result["custom_metrics"][f"league/refresh_{worst}"] = it
                print(f"Refreshed opponent {worst} at iteration {it}")
            except Exception as e:
                print(f"Error refreshing opponent: {e}")

        # 3) Логирование attention maps
        if self.attn_log_every > 0 and it % self.attn_log_every == 0:
            try:
                pol = algorithm.get_policy("main")
                
                # Ray 2.48 - правильный доступ к env_runner
                try:
                    env_runner = algorithm.env_runner_group.local_env_runner
                except:
                    env_runner = algorithm.workers.local_worker()
                    
                env = env_runner.env
                
                # Получаем одно наблюдение
                obs, _ = env.reset()
                for aid, ob in obs.items():
                    if aid.startswith("red_"):
                        _ = pol.compute_single_action(ob, explore=False)
                        break
                
                # Извлекаем attention map
                model = pol.model
                attn = getattr(model, "last_attn", None)
                
                if attn is not None and attn.numel() > 0:
                    with torch.no_grad():
                        # [B, H, L, L] -> [B, L, L]
                        attn_map = attn.mean(dim=1)
                        # Берем первый батч -> [1, L, L]
                        attn_img = attn_map[0:1].cpu().numpy()
                        
                    # TensorBoard ожидает [N, C, H, W]
                    if len(attn_img.shape) == 3:
                        # Добавляем канал если нужно
                        attn_img = np.expand_dims(attn_img, axis=1)
                    
                    self.writer.add_image("attention/last", attn_img[0], it, dataformats="CHW")
            except Exception as e:
                print(f"Error logging attention: {e}")

        # 4) Применение куррикулума
        for threshold, ac, ec in reversed(self.curriculum):
            if ts_total >= threshold:
                try:
                    # Ray 2.48 - обновленный способ применения к env_runners
                    def set_curriculum_fn(env):
                        if hasattr(env, 'set_curriculum'):
                            env.set_curriculum(ac, ec)
                    
                    # Пробуем новый API
                    try:
                        algorithm.env_runner_group.foreach_env(set_curriculum_fn)
                    except:
                        # Fallback на старый API
                        algorithm.workers.foreach_env(set_curriculum_fn)
                    
                    result.setdefault("custom_metrics", {})
                    result["custom_metrics"]["curriculum/ally_choices"] = str(ac)
                    result["custom_metrics"]["curriculum/enemy_choices"] = str(ec)
                    result["custom_metrics"]["curriculum/threshold"] = threshold
                except Exception as e:
                    print(f"Error setting curriculum: {e}")
                break

        # Сохраняем логи
        if self.writer:
            self.writer.flush()
