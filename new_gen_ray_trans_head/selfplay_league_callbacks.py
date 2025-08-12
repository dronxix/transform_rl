"""
Callbacks:
- Выбор соперника через Ray-актор (без глобалок).
- Апдейт TrueSkill + лог в TensorBoard.
- Периодическое «освежение» худшего оппонента (клонирование весов main).
- Лог карт внимания модели в TB (раз в attn_log_every итераций).
- Куррикулум по timesteps_total: ally_choices/enemy_choices меняются с порогами.
- Сбор custom_metrics (invalid_target, oob_*).
"""

from typing import Dict, Any, List, Optional
import numpy as np
import ray
import torch
from torch.utils.tensorboard import SummaryWriter
from ray.rllib.algorithms.callbacks import DefaultCallbacks

class LeagueCallbacks(DefaultCallbacks):
    def __init__(self, league_actor, opponent_ids: List[str],
                 eval_episodes=6, clone_every_iters=10, sample_top_k=3,
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
        # [(timestep_threshold, ally_choices, enemy_choices)] — ОТСОРТИРОВАННЫЙ список
        # !!! Исправлено: просто числа без «именованных» присваиваний
        self.curriculum = curriculum_schedule or [
            (0,           [1],        [1]),
            (1_000_000,   [1, 2],     [1, 2]),
            (5_000_000,   [1, 2, 3],  [1, 2, 3]),
            (10_000_000,  [1, 2, 3, 4],[1, 2, 3, 4]),
        ]

    # ---- выбор соперника на старте эпизода ----
    def on_episode_start(self, *, worker, base_env, policies, episode, **kwargs):
        opp = ray.get(self.league.get_opponent_weighted.remote(self.sample_top_k))
        episode.user_data["opp_id"] = opp

    # ---- custom metrics из infos ----
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        inv = 0; oobm = 0; ooba = 0
        for aid in episode.get_agents():
            info = episode.last_info_for(aid)
            if not info: continue
            inv  = max(inv,  info.get("invalid_target", 0))
            oobm = max(oobm, info.get("oob_move", 0))
            ooba = max(ooba, info.get("oob_aim", 0))
        episode.custom_metrics["invalid_target"] = inv
        episode.custom_metrics["oob_move"] = oobm
        episode.custom_metrics["oob_aim"] = ooba

    # ---- мини-матчи для оценки TrueSkill ----
    def _play_match(self, algorithm, opp_id: str, episodes: int) -> tuple[int, int]:
        worker = algorithm.workers.local_worker()
        env = worker.env
        wins_main, wins_opp = 0, 0
        for _ in range(episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action_dict = {}
                for aid, ob in obs.items():
                    pol_id = "main" if aid.startswith("red_") else opp_id
                    pol = algorithm.get_policy(pol_id)
                    act, _, _ = pol.compute_single_action(ob, explore=False)
                    action_dict[aid] = {
                        "target": int(act[0]),
                        "move":   act[1:3],
                        "aim":    act[3:5],
                        "fire":   int(round(float(act[5]))),
                    }
                obs, rews, terms, truncs, infos = env.step(action_dict)
                done = terms.get("__all__", False) or truncs.get("__all__", False)
            red_sum = sum(v for k, v in rews.items() if k.startswith("red_"))
            blue_sum= sum(v for k, v in rews.items() if k.startswith("blue_"))
            if red_sum > blue_sum: wins_main += 1
            elif blue_sum > red_sum: wins_opp += 1
        return wins_main, wins_opp

    # ---- сбор результатов тренировки ----
    def on_train_result(self, *, algorithm, result: Dict[str, Any], **kwargs):
        # Создаём TB writer один раз
        if self.writer is None:
            logdir = result.get("logdir", None) or getattr(algorithm, "logdir", "./logs")
            self.writer = SummaryWriter(log_dir=logdir)

        it = result["training_iteration"]
        ts_total = int(result.get("timesteps_total", 0))

        # 1) TrueSkill: матч против каждого оппонента + лог
        for pid in self.opponent_ids:
            w_main, w_opp = self._play_match(algorithm, pid, self.eval_eps)
            self.league.update_pair_result.remote(w_main, w_opp, pid)

        scores = ray.get(self.league.get_all_scores.remote())
        for k, (mu, sigma) in scores.items():
            result[f"ts/{k}_mu"] = mu
            result[f"ts/{k}_sigma"] = sigma
            self.writer.add_scalar(f"ts/{k}_conservative", mu - 3*sigma, it)

        # 2) «Освежение» худшего оппонента
        if it % self.clone_every == 0 and it > 0:
            items = [(pid, scores[pid][0] - 3*scores[pid][1]) for pid in self.opponent_ids]
            worst = min(items, key=lambda z: z[1])[0]
            w = algorithm.get_policy("main").get_weights()
            algorithm.get_policy(worst).set_weights(w)
            self.league.clone_main_into.remote(worst)
            result[f"league/refresh_{worst}"] = it

        # 3) Лог последней карты внимания (раз в N итераций)
        if self.attn_log_every > 0 and it % self.attn_log_every == 0:
            pol = algorithm.get_policy("main")
            env = algorithm.workers.local_worker().env
            obs, _ = env.reset()
            for aid, ob in obs.items():
                if aid.startswith("red_"):
                    _ = pol.compute_single_action(ob, explore=False)
                    break
            model = pol.model
            attn = getattr(model, "last_attn", None)  # [B,H,L,L]
            if attn is not None:
                with torch.no_grad():
                    attn_map = attn.mean(dim=1)  # усредняем по головам H -> [B,L,L]
                    attn_img = attn_map[0:1].cpu().numpy()  # [1,L,L]
                # TensorBoard: ожидает [N,C,H,W]; добавим канал C=1
                self.writer.add_image("attention/last", attn_img, it, dataformats="NCHW")

        # 4) Куррикулум по timesteps_total
        for threshold, ac, ec in reversed(self.curriculum):
            if ts_total >= threshold:
                # применим на локальной среде; при желании можно сделать foreach_worker
                algorithm.workers.local_worker().foreach_env(lambda e: e.set_curriculum(ac, ec))
                result["curriculum/ally_choices"] = ac
                result["curriculum/enemy_choices"] = ec
                break

        self.writer.flush()
