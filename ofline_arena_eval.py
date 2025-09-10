"""
Офлайн-арена: круговой турнир между чекпоинтами.
- Для пары (i, j) запускает N эпизодов, считает победы/поражения.
- Обновляет TrueSkill и ELO.
- Сохраняет CSV и логи в TensorBoard.
"""

import os, glob, time, csv
import trueskill as ts
from typing import List, Tuple
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from torch.utils.tensorboard import SummaryWriter

from arena_env import ArenaEnv

TS_ENV = ts.TrueSkill(draw_probability=0.0)

def env_creator(cfg): return ArenaEnv(cfg)

def play_match(algo, cp_i: str, cp_j: str, episodes: int = 8) -> Tuple[int,int]:
    # Готовим веса: main <- cp_i, opponent_0 <- cp_j
    algo.restore(cp_i)
    w_red = algo.get_policy("main").get_weights()
    algo.restore(cp_j)
    w_blue = algo.get_policy("main").get_weights()
    algo.restore(cp_i)
    algo.get_policy("main").set_weights(w_red)
    algo.get_policy("opponent_0").set_weights(w_blue)

    worker = algo.workers.local_worker()
    env = worker.env
    wins_i, wins_j = 0, 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action_dict = {}
            for aid, ob in obs.items():
                pol_id = "main" if aid.startswith("red_") else "opponent_0"
                pol = algo.get_policy(pol_id)
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
        if red_sum > blue_sum: wins_i += 1
        elif blue_sum > red_sum: wins_j += 1
    return wins_i, wins_j

def simple_elo_update(elo_a, elo_b, score_a, k=32):
    ea = 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))
    elo_a += k * (score_a - ea)
    elo_b += k * ((1 - score_a) - (1 - ea))
    return elo_a, elo_b

def run_tournament(checkpoints: List[str], episodes_per_pair: int = 6, logdir="./arena_logs"):
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir=logdir)

    ray.init(ignore_reinit_error=True)
    register_env("ArenaEnv", env_creator)

    algo = (
        PPOConfig()
        .environment(env="ArenaEnv", env_config={"episode_len": 128, "ally_choices":[1,2], "enemy_choices":[1,2]})
        .framework("torch")
        .rollouts(num_rollout_workers=0)
        .training(model={"vf_share_layers": False})
        .multi_agent(
            policies={
                "main": (None, ArenaEnv({}).observation_space, ArenaEnv({}).action_space, {}),
                "opponent_0": (None, ArenaEnv({}).observation_space, ArenaEnv({}).action_space, {}),
            },
            policy_mapping_fn=lambda aid, *args, **kw: "main" if aid.startswith("red_") else "opponent_0",
            policies_to_train=[]
        )
        .resources(num_gpus=0)
        .build()
    )

    n = len(checkpoints)
    ratings = {cp: TS_ENV.create_rating() for cp in checkpoints}
    elos = {cp: 1000.0 for cp in checkpoints}

    start = time.time()
    for i in range(n):
        for j in range(i+1, n):
            cp_i, cp_j = checkpoints[i], checkpoints[j]
            w_i, w_j = play_match(algo, cp_i, cp_j, episodes=episodes_per_pair)

            # TrueSkill
            for _ in range(w_i):
                ratings[cp_i], ratings[cp_j] = ts.rate_1vs1(ratings[cp_i], ratings[cp_j], env=TS_ENV)
            for _ in range(w_j):
                ratings[cp_j], ratings[cp_i] = ts.rate_1vs1(ratings[cp_j], ratings[cp_i], env=TS_ENV)

            # ELO
            for _ in range(w_i): elos[cp_i], elos[cp_j] = simple_elo_update(elos[cp_i], elos[cp_j], 1)
            for _ in range(w_j): elos[cp_j], elos[cp_i] = simple_elo_update(elos[cp_j], elos[cp_i], 1)

            total = w_i + w_j
            writer.add_scalar(f"pairs/{os.path.basename(cp_i)}_vs_{os.path.basename(cp_j)}",
                              (w_i/(total+1e-9)), i*n+j)

    # CSV сводка
    csv_path = os.path.join(logdir, "tournament_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["checkpoint","trueskill_mu","trueskill_sigma","elo"])
        for cp in checkpoints:
            r = ratings[cp]
            wr.writerow([cp, f"{r.mu:.3f}", f"{r.sigma:.3f}", f"{elos[cp]:.1f}"])

    # Топ по консервативному TS (mu-3σ)
    sorted_ts = sorted(checkpoints, key=lambda c: ratings[c].mu - 3*ratings[c].sigma, reverse=True)
    for rank, cp in enumerate(sorted_ts):
        r = ratings[cp]
        writer.add_scalar("rank_ts/conservative", r.mu - 3*r.sigma, rank)
        writer.add_text("rank_ts/checkpoint", f"{rank+1}. {cp}", rank)
    writer.flush(); writer.close()

    print("Saved:", csv_path, "| Took:", time.time()-start, "sec")
    ray.shutdown()

if __name__ == "__main__":
    ckpts = sorted(glob.glob("./rllib_league_results/checkpoint_*"))
    run_tournament(ckpts, episodes_per_pair=6, logdir="./arena_logs")
