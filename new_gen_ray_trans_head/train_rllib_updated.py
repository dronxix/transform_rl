"""
Главный скрипт обучения - ПРАВИЛЬНО обновлен для Ray 2.48.0
Изменения:
- rollouts -> env_runners
- Использование новых API Ray 2.48
- Правильная конфигурация multi-agent
"""

import os
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from arena_env import ArenaEnv
from entity_attention_model import EntityAttentionModel   
from masked_multihead_dist import MaskedTargetMoveAimFire 
from league_state import LeagueState
from selfplay_league_callbacks import LeagueCallbacks
from gspo_grpo_policy import GSPOTorchPolicy, GRPOTorchPolicy

def env_creator(cfg): 
    return ArenaEnv(cfg)

def main():
    # Ray 2.48 - новые параметры инициализации
    ray.init(
        ignore_reinit_error=True,
        num_cpus=None,  # автоопределение
        num_gpus=None,  # автоопределение
        include_dashboard=True,
        dashboard_host="0.0.0.0"
    )

    register_env("ArenaEnv", env_creator)
    ModelCatalog.register_custom_model("entity_attention", EntityAttentionModel)
    ModelCatalog.register_custom_action_dist("masked_multihead", MaskedTargetMoveAimFire)

    opponent_ids = [f"opponent_{i}" for i in range(6)]
    league = LeagueState.remote(opponent_ids)

    # Выбор варианта алгоритма
    algo_variant = os.environ.get("ALGO_VARIANT", "gspo").lower()
    if algo_variant == "gspo": 
        policy_cls = GSPOTorchPolicy
        vf_coeff = 1.0
        ent_coeff = 0.003
    elif algo_variant == "grpo": 
        policy_cls = GRPOTorchPolicy
        vf_coeff = 0.5
        ent_coeff = 0.004
    else:
        policy_cls = None
        vf_coeff = 1.0
        ent_coeff = 0.003

    def policy_mapping_fn(agent_id: str, episode, worker, **kwargs):
        if agent_id.startswith("red_"):
            return "main"
        opp = episode.user_data.get("opp_id")
        if opp is None:
            opp = ray.get(league.get_opponent_weighted.remote(3))
            episode.user_data["opp_id"] = opp
        return opp

    # Получим spaces
    tmp_env = ArenaEnv({
        "ally_choices": [1, 2], 
        "enemy_choices": [1, 2], 
        "episode_len": 128
    })
    obs_space = tmp_env.observation_space
    act_space = tmp_env.action_space

    # Конфигурация политик
    main_policy_config = {
        "model": {
            "custom_model": "entity_attention",
            "custom_action_dist": "masked_multihead",
            "custom_model_config": {
                "d_model": 160,
                "nhead": 8,
                "layers": 2,
                "ff": 320,
                "hidden": 256,
                "logstd_min": -5.0,
                "logstd_max": 2.0,
            },
        },
    }
    
    # Добавляем специфичные параметры для кастомных политик
    if policy_cls:
        main_policy_config["vf_loss_coeff"] = vf_coeff
        if algo_variant == "grpo":
            main_policy_config["grpo_ema_beta"] = 0.99

    policies = {
        "main": (
            policy_cls,
            obs_space,
            act_space,
            main_policy_config
        )
    }

    # Оппоненты
    for pid in opponent_ids:
        policies[pid] = (
            None,  # стандартный PPO
            obs_space,
            act_space,
            {
                "model": {
                    "custom_model": "entity_attention",
                    "custom_action_dist": "masked_multihead",
                    "custom_model_config": {
                        "d_model": 160,
                        "nhead": 8,
                        "layers": 2,
                        "ff": 320,
                        "hidden": 256,
                        "logstd_min": -5.0,
                        "logstd_max": 2.0,
                    },
                },
            },
        )

    # Куррикулум
    curriculum = [
        (0,           [1],         [1]),
        (2_000_000,   [1, 2],      [1, 2]),
        (10_000_000,  [1, 2, 3],   [1, 2, 3]),
        (25_000_000,  [1, 2, 3, 4],[1, 2, 3, 4]),
    ]

    # Создаем колбеки
    callbacks = LeagueCallbacks(
        league_actor=league,
        opponent_ids=opponent_ids,
        eval_episodes=6,
        clone_every_iters=10,
        sample_top_k=3,
        attn_log_every=20,
        curriculum_schedule=curriculum,
    )

    # Ray 2.48 - ПРАВИЛЬНАЯ конфигурация с env_runners
    config = (
        PPOConfig()
        .environment(
            env="ArenaEnv",
            env_config={
                "episode_len": 128,
                "ally_choices": [1],
                "enemy_choices": [1],
                "max_allies": 6,
                "max_enemies": 6,
                "assert_invalid_actions": True,
            }
        )
        .framework("torch")
        # Ray 2.48 - env_runners вместо rollouts!
        .env_runners(
            num_env_runners=8,  # вместо num_rollout_workers
            num_envs_per_env_runner=1,  # вместо num_envs_per_worker
            rollout_fragment_length=256,
            batch_mode="truncate_episodes",
            # Новые параметры для env_runners
            explore=True,
            exploration_config={
                "type": "StochasticSampling",
            },
            # Включаем новую систему коннекторов
            enable_connectors=True,
            env_runner_cls=None,  # используем дефолтный
        )
        .training(
            gamma=0.99,
            lr=3e-4,
            train_batch_size=262144,  # 256k
            sgd_minibatch_size=16384,
            num_sgd_iter=4,
            use_gae=True,
            lambda_=0.95,
            clip_param=0.15,
            vf_clip_param=10.0,
            entropy_coeff=ent_coeff,
            model={
                "vf_share_layers": False,
                # Ray 2.48 - новые параметры модели
                "uses_new_env_runners": True,
            },
            # Ray 2.48 - используем новый Learner API где возможно
            _enable_new_api_stack=False,  # пока отключаем полный переход
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["main"],
            # Ray 2.48 - обновленные параметры
            count_steps_by="agent_steps",
            observation_fn=None,  # используем дефолтную
            # Новый параметр для батчинга
            policy_map_capacity=100,
        )
        .callbacks(callbacks)
        .resources(
            num_gpus=1,
            # Ray 2.48 - новые параметры ресурсов
            num_cpus_per_env_runner=1,  # вместо num_cpus_per_worker
            num_gpus_per_env_runner=0,  # вместо num_gpus_per_worker
            num_learner_workers=0,  # 0 = обучение на драйвере
            num_gpus_per_learner_worker=1 if algo_variant else 0,
        )
        .debugging(
            log_level="INFO",
            seed=42,
        )
        .fault_tolerance(
            recreate_failed_env_runners=True,  # вместо recreate_failed_workers
            restart_failed_sub_environments=True,
            num_consecutive_env_runner_failures_tolerance=3,
        )
        .reporting(
            # Ray 2.48 - новые параметры отчетности
            min_time_s_per_iteration=None,
            min_sample_timesteps_per_iteration=1000,
            metrics_num_episodes_for_smoothing=100,
        )
    )

    algo = config.build()

    # Инициализируем оппонентов
    w = algo.get_policy("main").get_weights()
    for pid in opponent_ids:
        algo.get_policy(pid).set_weights(w)

    results_dir = os.path.abspath("./rllib_league_results")
    os.makedirs(results_dir, exist_ok=True)

    # Обучение с сохранением чекпоинтов
    next_ckpt = 5_000_000
    
    for i in range(2_000):
        try:
            res = algo.train()
            
            # Логирование
            if i % 10 == 0:
                # Ray 2.48 - новые ключи метрик
                rew_mean = res.get('env_runners', {}).get('episode_reward_mean', 0)
                if rew_mean == 0:  # fallback на старые ключи
                    rew_mean = res.get('episode_reward_mean', 0)
                    
                print(f"[{i}] rew_mean={rew_mean:.3f} "
                      f"ts_mu={res.get('custom_metrics', {}).get('ts/main_mu', 0):.2f} "
                      f"steps={res.get('timesteps_total', 0)}")
                
                # Ray 2.48 - обновленный метод сохранения
                checkpoint = algo.save(checkpoint_dir=results_dir)
                print(f"Checkpoint saved: {checkpoint}")
            
            # Milestone checkpoints
            if res.get("timesteps_total", 0) >= next_ckpt:
                checkpoint = algo.save(checkpoint_dir=results_dir)
                print(f"Milestone checkpoint at {next_ckpt}: {checkpoint}")
                next_ckpt += 5_000_000
                
        except Exception as e:
            print(f"Error during training iteration {i}: {e}")
            # Попытка восстановления
            try:
                algo.restore(checkpoint)
                print("Restored from last checkpoint")
            except:
                print("Could not restore, continuing...")
                continue

    algo.stop()
    ray.shutdown()

if __name__ == "__main__":
    main()
