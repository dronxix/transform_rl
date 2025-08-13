"""
Callbacks - ПРАВИЛЬНО обновлены для Ray 2.48.0
Критические исправления:
- RLlibCallback вместо DefaultCallbacks
- SingleAgentEpisode/MultiAgentEpisode вместо Episode
- env_runner вместо worker
- Правильные импорты и API
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import ray
import torch
from torch.utils.tensorboard import SummaryWriter

# ПРАВИЛЬНЫЕ импорты для Ray 2.48.0
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.evaluation.episode_v2 import EpisodeV2  # для совместимости со старым API
from ray.rllib.env import EnvRunner
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.core.rl_module import RLModule
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import PolicyID
from ray.rllib.algorithms import Algorithm

class LeagueCallbacks(RLlibCallback):  # Наследуем от RLlibCallback!
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
        
    def setup(self, 
              league_actor, 
              opponent_ids: List[str],
              eval_episodes=6, 
              clone_every_iters=10, 
              sample_top_k=3,
              attn_log_every=20,
              curriculum_schedule=None):
        """Настройка параметров callbacks после создания."""
        self.league = league_actor
        self.opponent_ids = opponent_ids
        self.eval_eps = eval_episodes
        self.clone_every = clone_every_iters
        self.sample_top_k = sample_top_k
        self.attn_log_every = attn_log_every
        self.curriculum = curriculum_schedule or [
            (0,           [1],         [1]),
            (1_000_000,   [1, 2],      [1, 2]),
            (5_000_000,   [1, 2, 3],   [1, 2, 3]),
            (10_000_000,  [1, 2, 3, 4],[1, 2, 3, 4]),
        ]

    def on_algorithm_init(self, *, algorithm: Algorithm, **kwargs) -> None:
        """Вызывается при инициализации алгоритма."""
        # Инициализация может происходить здесь если нужно
        pass

    def on_episode_created(
        self,
        *,
        episode: Union[SingleAgentEpisode, MultiAgentEpisode, EpisodeV2],
        env_runner: Optional[EnvRunner] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[Any] = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
        **kwargs,
    ) -> None:
        """Вызывается при создании эпизода (до reset)."""
        # Выбираем соперника
        if self.league is not None:
            opp = ray.get(self.league.get_opponent_weighted.remote(self.sample_top_k))
            
            # Для новых типов эпизодов
            if isinstance(episode, (SingleAgentEpisode, MultiAgentEpisode)):
                episode.custom_data["opp_id"] = opp
                episode.custom_data["max_invalid_target"] = 0
                episode.custom_data["max_oob_move"] = 0
                episode.custom_data["max_oob_aim"] = 0
            # Для старого API (EpisodeV2)
            elif hasattr(episode, 'user_data'):
                episode.user_data["opp_id"] = opp
                episode.user_data["max_invalid_target"] = 0
                episode.user_data["max_oob_move"] = 0
                episode.user_data["max_oob_aim"] = 0

    def on_episode_start(
        self,
        *,
        episode: Union[SingleAgentEpisode, MultiAgentEpisode, EpisodeV2],
        env_runner: Optional[EnvRunner] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[Any] = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
        **kwargs,
    ) -> None:
        """Вызывается после reset среды."""
        # Можем добавить дополнительную логику если нужно
        pass

    def on_episode_step(
        self,
        *,
        episode: Union[SingleAgentEpisode, MultiAgentEpisode, EpisodeV2],
        env_runner: Optional[EnvRunner] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[Any] = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
        **kwargs,
    ) -> None:
        """Вызывается на каждом шаге эпизода."""
        # Собираем метрики из infos
        if isinstance(episode, MultiAgentEpisode):
            # Для MultiAgentEpisode в новом API
            for agent_id in episode.agent_ids:
                info = episode.get_last_info(agent_id)
                if info and isinstance(info, dict):
                    for key in ["invalid_target", "oob_move", "oob_aim"]:
                        if key in info:
                            current = episode.custom_data.get(f"max_{key}", 0)
                            episode.custom_data[f"max_{key}"] = max(current, info[key])
                            
        elif isinstance(episode, SingleAgentEpisode):
            # Для SingleAgentEpisode
            info = episode.get_last_info()
            if info and isinstance(info, dict):
                for key in ["invalid_target", "oob_move", "oob_aim"]:
                    if key in info:
                        current = episode.custom_data.get(f"max_{key}", 0)
                        episode.custom_data[f"max_{key}"] = max(current, info[key])
                        
        elif hasattr(episode, 'last_info_for'):
            # Старый API (EpisodeV2)
            for agent_id in episode.get_agents():
                info = episode.last_info_for(agent_id)
                if info and isinstance(info, dict):
                    for key in ["invalid_target", "oob_move", "oob_aim"]:
                        if key in info:
                            current = episode.user_data.get(f"max_{key}", 0)
                            episode.user_data[f"max_{key}"] = max(current, info[key])

    def on_episode_end(
        self,
        *,
        episode: Union[SingleAgentEpisode, MultiAgentEpisode, EpisodeV2],
        env_runner: Optional[EnvRunner] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[Any] = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
        **kwargs,
    ) -> None:
        """Финализация метрик эпизода."""
        # Сохраняем custom metrics
        if isinstance(episode, (SingleAgentEpisode, MultiAgentEpisode)):
            # Новый API - используем custom_data
            data = episode.custom_data
            
            # Логируем метрики через metrics_logger если доступен
            if metrics_logger:
                metrics_logger.log_value("invalid_target", data.get("max_invalid_target", 0))
                metrics_logger.log_value("oob_move", data.get("max_oob_move", 0))
                metrics_logger.log_value("oob_aim", data.get("max_oob_aim", 0))
            
            # Также сохраняем в episode для совместимости
            if hasattr(episode, 'custom_metrics'):
                episode.custom_metrics["invalid_target"] = data.get("max_invalid_target", 0)
                episode.custom_metrics["oob_move"] = data.get("max_oob_move", 0)
                episode.custom_metrics["oob_aim"] = data.get("max_oob_aim", 0)
                
        elif hasattr(episode, 'user_data'):
            # Старый API (EpisodeV2)
            episode.custom_metrics["invalid_target"] = episode.user_data.get("max_invalid_target", 0)
            episode.custom_metrics["oob_move"] = episode.user_data.get("max_oob_move", 0)
            episode.custom_metrics["oob_aim"] = episode.user_data.get("max_oob_aim", 0)

    def _play_match(self, algorithm: Algorithm, opp_id: str, episodes: int) -> tuple[int, int]:
        """Играет матч между main и оппонентом для оценки."""
        # Получаем env_runner (с fallback на старое API)
        try:
            # Ray 2.48 - новый способ
            if hasattr(algorithm, 'env_runner_group'):
                env_runner = algorithm.env_runner_group.local_env_runner
            else:
                # Fallback на workers если env_runner_group не доступен
                env_runner = algorithm.workers.local_worker()
        except:
            print("Warning: Could not get env_runner, skipping match evaluation")
            return 0, 0
            
        env = env_runner.env
        
        wins_main, wins_opp = 0, 0
        
        for _ in range(episodes):
            obs, _ = env.reset()
            done = False
            
            while not done:
                action_dict = {}
                
                for aid, ob in obs.items():
                    pol_id = "main" if aid.startswith("red_") else opp_id
                    pol = algorithm.get_policy(pol_id)
                    
                    # compute_single_action все еще работает
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
        # Проверяем что league инициализирован
        if self.league is None:
            return
            
        # Создаем TensorBoard writer
        if self.writer is None:
            logdir = result.get("logdir", None) or getattr(algorithm, "logdir", "./logs")
            self.writer = SummaryWriter(log_dir=logdir)

        it = result["training_iteration"]
        
        # Получаем timesteps_total из правильного места
        ts_total = result.get("timesteps_total", 0)
        if ts_total == 0:
            # Ray 2.48 - может быть во вложенной структуре
            ts_total = result.get("env_runners", {}).get("timesteps_total", 0)

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
                # Сохраняем в custom_metrics
                result.setdefault("custom_metrics", {})
                result["custom_metrics"][f"ts_{k}_mu"] = mu
                result["custom_metrics"][f"ts_{k}_sigma"] = sigma
                
                conservative_score = mu - 3 * sigma
                self.writer.add_scalar(f"ts/{k}_conservative", conservative_score, it)
        except Exception as e:
            print(f"Error getting league scores: {e}")

        # 2) Освежение худшего оппонента
        if it % self.clone_every == 0 and it > 0 and scores:
            try:
                items = [(pid, scores[pid][0] - 3*scores[pid][1]) for pid in self.opponent_ids]
                worst = min(items, key=lambda z: z[1])[0]
                
                # Клонируем веса
                w = algorithm.get_policy("main").get_weights()
                algorithm.get_policy(worst).set_weights(w)
                self.league.clone_main_into.remote(worst)
                
                result.setdefault("custom_metrics", {})
                result["custom_metrics"][f"league_refresh_{worst}"] = it
                print(f"Refreshed opponent {worst} at iteration {it}")
            except Exception as e:
                print(f"Error refreshing opponent: {e}")

        # 3) Логирование attention maps
        if self.attn_log_every > 0 and it % self.attn_log_every == 0:
            try:
                self._log_attention(algorithm, it)
            except Exception as e:
                print(f"Error logging attention: {e}")

        # 4) Применение куррикулума
        if self.curriculum:
            for threshold, ac, ec in reversed(self.curriculum):
                if ts_total >= threshold:
                    try:
                        self._apply_curriculum(algorithm, ac, ec)
                        result.setdefault("custom_metrics", {})
                        result["custom_metrics"]["curriculum_ally_choices"] = str(ac)
                        result["custom_metrics"]["curriculum_enemy_choices"] = str(ec)
                    except Exception as e:
                        print(f"Error setting curriculum: {e}")
                    break

        # Сохраняем логи
        if self.writer:
            self.writer.flush()
            
    def _log_attention(self, algorithm, iteration):
        """Логирует attention maps в TensorBoard."""
        pol = algorithm.get_policy("main")
        
        # Получаем env_runner
        try:
            if hasattr(algorithm, 'env_runner_group'):
                env_runner = algorithm.env_runner_group.local_env_runner
            else:
                env_runner = algorithm.workers.local_worker()
        except:
            return
            
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
                attn_img = np.expand_dims(attn_img, axis=1)
            
            self.writer.add_image("attention/last", attn_img[0], iteration, dataformats="CHW")
            
    def _apply_curriculum(self, algorithm, ally_choices, enemy_choices):
        """Применяет куррикулум к средам."""
        def set_curriculum_fn(env):
            if hasattr(env, 'set_curriculum'):
                env.set_curriculum(ally_choices, enemy_choices)
        
        # Пробуем разные способы доступа к env_runners
        try:
            if hasattr(algorithm, 'env_runner_group'):
                algorithm.env_runner_group.foreach_env(set_curriculum_fn)
            elif hasattr(algorithm, 'workers'):
                algorithm.workers.foreach_env(set_curriculum_fn)
        except Exception as e:
            print(f"Could not apply curriculum: {e}")
