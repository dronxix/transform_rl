"""
Callbacks для Ray 2.48+ - исправленная версия с правильным доступом к окружению
"""

from typing import Dict, Any, List, Optional
import numpy as np
import ray
import torch
from torch.utils.tensorboard import SummaryWriter

from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.algorithms import Algorithm

class LeagueCallbacks(RLlibCallback):
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
        
    def setup(self, league_actor, opponent_ids: List[str], **kwargs):
        """Настройка параметров callbacks"""
        self.league = league_actor
        self.opponent_ids = opponent_ids
        self.eval_eps = kwargs.get('eval_episodes', 6)
        self.clone_every = kwargs.get('clone_every_iters', 10)
        self.sample_top_k = kwargs.get('sample_top_k', 3)
        self.attn_log_every = kwargs.get('attn_log_every', 20)
        self.curriculum = kwargs.get('curriculum_schedule', [])

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

        if self.writer:
            self.writer.flush()

    def _play_match(self, algorithm: Algorithm, opp_id: str, episodes: int) -> tuple:
        """Исправленная версия матча для Ray 2.48"""
        try:
            # Создаем временное окружение для матчей
            from arena_env import ArenaEnv
            
            # Получаем конфиг окружения из алгоритма
            env_config = algorithm.config.env_config.copy() if hasattr(algorithm.config, 'env_config') else {}
            
            # Создаем временное окружение
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
                        
                        # Получаем действие от политики
                        act, _, _ = pol.compute_single_action(ob, explore=False)
                        
                        # Преобразуем в правильный формат если нужно
                        if isinstance(act, dict):
                            action_dict[aid] = act
                        else:
                            # Если действие в виде массива, преобразуем в dict
                            action_dict[aid] = {
                                "target": int(act[0]) if len(act) > 0 else 0,
                                "move": act[1:3] if len(act) > 2 else [0.0, 0.0],
                                "aim": act[3:5] if len(act) > 4 else [0.0, 0.0],
                                "fire": int(round(float(act[5]))) if len(act) > 5 else 0,
                            }
                    
                    obs, rews, terms, truncs, infos = temp_env.step(action_dict)
                    done = terms.get("__all__", False) or truncs.get("__all__", False)
                
                # Подсчитываем победы
                red_sum = sum(v for k, v in rews.items() if k.startswith("red_"))
                blue_sum = sum(v for k, v in rews.items() if k.startswith("blue_"))
                
                if red_sum > blue_sum:
                    wins_main += 1
                elif blue_sum > red_sum:
                    wins_opp += 1
                    
            return wins_main, wins_opp
            
        except Exception as e:
            print(f"Error in _play_match: {e}")
            import traceback
            traceback.print_exc()
            return 0, 0

    def _apply_curriculum(self, algorithm, ally_choices, enemy_choices):
        """Применение куррикулума для Ray 2.48"""
        try:
            # В Ray 2.48 env_runners заменили rollout_workers
            # Но у нас нет прямого доступа к окружениям через них
            # Поэтому будем менять конфиг алгоритма
            
            # Обновляем конфиг
            if hasattr(algorithm.config, 'env_config'):
                algorithm.config.env_config["ally_choices"] = ally_choices
                algorithm.config.env_config["enemy_choices"] = enemy_choices
                print(f"Updated curriculum in config: allies={ally_choices}, enemies={enemy_choices}")
            
            # Попытка применить к существующим env_runners если возможно
            try:
                if hasattr(algorithm, 'env_runner_group') and algorithm.env_runner_group:
                    # Это может не сработать, так как API изменился
                    def set_curriculum_fn(env):
                        if hasattr(env, 'set_curriculum'):
                            env.set_curriculum(ally_choices, enemy_choices)
                    
                    algorithm.env_runner_group.foreach_env(set_curriculum_fn)
                    print(f"Applied curriculum to env_runners: allies={ally_choices}, enemies={enemy_choices}")
            except (AttributeError, Exception) as e:
                print(f"Could not apply curriculum to existing envs: {e}")
                
        except Exception as e:
            print(f"Could not apply curriculum: {e}")