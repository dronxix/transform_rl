"""
Главный скрипт обучения:
- онлайновая лига (Ray-актор), взвешенный выбор соперника,
- переключатель ALGO_VARIANT: 'ppo' | 'gspo' | 'grpo',
- большие батчи, чекпоинты каждые 5М шагов,
- куррикулум по timesteps_total,
- TensorBoard-логи (в каталоге результатов).
"""

import os
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from arena_env import ArenaEnv
from entity_attention_model import EntityAttentionModel   # noqa: F401
from masked_multihead_dist import MaskedTargetMoveAimFire # noqa: F401
from league_state import LeagueState
from selfplay_league_callbacks import LeagueCallbacks
from gspo_grpo_policy import GSPOTorchPolicy, GRPOTorchPolicy

def env_creator(cfg): return ArenaEnv(cfg)

def main():
    ray.init(ignore_reinit_error=True)

    register_env("ArenaEnv", env_creator)
    ModelCatalog.register_custom_model("entity_attention", EntityAttentionModel)
    ModelCatalog.register_custom_action_dist("masked_multihead", MaskedTargetMoveAimFire)

    opponent_ids = [f"opponent_{i}" for i in range(6)]
    league = LeagueState.remote(opponent_ids)

    # Выбор варианта алгоритма через переменную окружения
    algo_variant = os.environ.get("ALGO_VARIANT", "gspo").lower()
    if   algo_variant == "gspo": policy_cls = GSPOTorchPolicy; vf_coeff=1.0; ent_coeff=0.003
    elif algo_variant == "grpo": policy_cls = GRPOTorchPolicy; vf_coeff=0.5; ent_coeff=0.004
    else:                        policy_cls = None;            vf_coeff=1.0; ent_coeff=0.003

    def policy_mapping_fn(agent_id: str, episode, worker, **kwargs):
        # Красные — всегда main, синие — оппонент, выбранный Callback'ом
        if agent_id.startswith("red_"):
            return "main"
        opp = episode.user_data.get("opp_id")
        if opp is None:  # Fallback на первом шаге
            opp = ray.get(league.get_opponent_weighted.remote(3))
            episode.user_data["opp_id"] = opp
        return opp

    # Получим spaces (через временный env)
    tmp_env = ArenaEnv({"ally_choices":[1,2], "enemy_choices":[1,2], "episode_len":128})
    obs_space = tmp_env.observation_space
    act_space = tmp_env.action_space

    # Политики: main (trainable) и пул оппонентов (заморожены)
    policies = {}
    policies["main"] = (
        policy_cls,  # None -> стандартный PPO
        obs_space,
        act_space,
        {
            "model": {
                "custom_model": "entity_attention",
                "custom_action_dist": "masked_multihead",
                "custom_model_config": {
                    "d_model": 160, "nhead": 8, "layers": 2, "ff": 320, "hidden": 256,
                    "logstd_min": -5.0, "logstd_max": 2.0,
                },
            },
            "vf_loss_coeff": vf_coeff,
            "grpo_ema_beta": 0.99,  # используется только в GRPO
        },
    )
    for pid in opponent_ids:
        policies[pid] = (
            None,
            obs_space,
            act_space,
            {
                "model": {
                    "custom_model": "entity_attention",
                    "custom_action_dist": "masked_multihead",
                    "custom_model_config": {
                        "d_model": 160, "nhead": 8, "layers": 2, "ff": 320, "hidden": 256,
                        "logstd_min": -5.0, "logstd_max": 2.0,
                    },
                },
            },
        )

    # Куррикулум: (порог по timesteps_total, ally_choices, enemy_choices)
    curriculum = [
        (0,           [1],        [1]),
        (2_000_000,   [1, 2],     [1, 2]),
        (10_000_000,  [1, 2, 3],  [1, 2, 3]),
        (25_000_000,  [1, 2, 3, 4],[1, 2, 3, 4]),
    ]

    config = (
        PPOConfig()
        .environment(env="ArenaEnv", env_config={
            "episode_len": 128,
            "ally_choices":  [1],   # старт с простых боёв
            "enemy_choices": [1],
            "max_allies": 6,
            "max_enemies": 6,
            "assert_invalid_actions": True,
        })
        .framework("torch")
        .rollouts(
            num_rollout_workers=8,
            rollout_fragment_length=256,
            batch_mode="truncate_episodes",
        )
        .training(
            gamma=0.99, lr=3e-4,
            sgd_minibatch_size=16384,
            train_batch_size=262144,  # 256k
            num_sgd_iter=4,
            use_gae=True, lambda_=0.95,
            clip_param=0.15, vf_clip_param=10.0,
            entropy_coeff=ent_coeff,
            model={"vf_share_layers": False},
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["main"],
        )
        .callbacks(lambda: LeagueCallbacks(
            league_actor=league,
            opponent_ids=opponent_ids,
            eval_episodes=6,
            clone_every_iters=10,
            sample_top_k=3,
            attn_log_every=20,
            curriculum_schedule=curriculum,
        ))
        .resources(num_gpus=1)
    )

    algo = config.build()

    # Инициализируем оппонентов копией main
    w = algo.get_policy("main").get_weights()
    for pid in opponent_ids:
        algo.get_policy(pid).set_weights(w)

    results_dir = os.path.abspath("./rllib_league_results"); os.makedirs(results_dir, exist_ok=True)

    # Сохраняем чекпоинты каждые ~10 итераций + по порогам timesteps_total
    next_ckpt = 5_000_000
    for i in range(2_000):  # много итераций (под долгие прогоны)
        res = algo.train()
        if i % 10 == 0:
            print(f"[{i}] rew_mean={res['episode_reward_mean']:.3f} "
                  f"ts_mu={res.get('ts/main_mu',0):.2f} steps={res.get('timesteps_total',0)}")
            cp = algo.save(results_dir)
            print("Checkpoint:", cp)
        if res.get("timesteps_total", 0) >= next_ckpt:
            cp = algo.save(results_dir)
            print(f"Saved milestone at {next_ckpt}:", cp)
            next_ckpt += 5_000_000

    algo.stop()
    ray.shutdown()

if __name__ == "__main__":
    main()
