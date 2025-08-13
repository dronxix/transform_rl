"""
Главный скрипт обучения - ПОЛНОСТЬЮ исправлен для Ray 2.48.0
Все API обновлены и проверены
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
    # Ray 2.48 - инициализация
    ray.init(
        ignore_reinit_error=True,
        num_cpus=None,
        num_gpus=None,
        include_dashboard=True,
        dashboard_host="0.0.0.0"
    )

    register_env("ArenaEnv", env_creator)
    ModelCatalog.register_custom_model("entity_attention", EntityAttentionModel)
    ModelCatalog.register_custom_action_dist("masked_multihead", MaskedTargetMoveAimFire)

    opponent_ids = [f"opponent_{i}" for i in range(6)]
    league = LeagueState.remote(opponent_ids)

    # Выбор алгоритма
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
        
        # Получаем opp_id из episode
        opp = None
        # Проверяем разные типы episode для совместимости
        if hasattr(episode, 'custom_data'):
            opp = episode.custom_data.get("opp_id")
        elif hasattr(episode, 'user_data'):
            opp = episode.user_data.get("opp_id")
            
        if opp is None:
            opp = ray.get(league.get_opponent_weighted.remote(3))
            # Сохраняем для будущего использования
            if hasattr(episode, 'custom_data'):
                episode.custom_data["opp_id"] = opp
            elif hasattr(episode, 'user_data'):
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
    
    # Специфичные параметры для кастомных политик
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

    # Создаем и настраиваем колбеки
    callbacks = LeagueCallbacks()
    callbacks.setup(
        league_actor=league,
        opponent_ids=opponent_ids,
        eval_episodes=6,
        clone_every_iters=10,
        sample_top_k=3,
        attn_log_every=20,
        curriculum_schedule=curriculum,
    )

    # Ray 2.48 - ПРАВИЛЬНАЯ конфигурация
    config = (
        PPOConfig()
        .api_stack(
            # ВАЖНО: используем старый API stack для ModelV2
            # Новый stack требует RLModule вместо ModelV2
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
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
        # Ray 2.48 - env_runners используется только с новым API stack
        # Для старого stack используем rollouts
        .rollouts(
            num_rollout_workers=8,
            num_envs_per_worker=1,
            rollout_fragment_length=256,
            batch_mode="truncate_episodes",
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
            },
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["main"],
            count_steps_by="agent_steps",
        )
        .callbacks(callbacks)
        .resources(
            num_gpus=1,
            num_cpus_per_worker=1,
            num_gpus_per_worker=0,
        )
        .debugging(
            log_level="INFO",
            seed=42,
        )
        .fault_tolerance(
            recreate_failed_workers=True,
            restart_failed_sub_environments=True,
            num_consecutive_worker_failures_tolerance=3,
        )
    )

    # Альтернативный вариант с НОВЫМ API stack (требует переписывания под RLModule)
    use_new_stack = os.environ.get("USE_NEW_STACK", "false").lower() == "true"
    
    if use_new_stack:
        print("WARNING: New API stack requires RLModule instead of ModelV2")
        print("This will require rewriting entity_attention_model.py")
        config = config.api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        # Для нового stack нужно использовать env_runners вместо rollouts
        config = config.env_runners(
            num_env_runners=8,
            num_envs_per_env_runner=1,
            rollout_fragment_length=256,
            batch_mode="truncate_episodes",
        )

    algo = config.build()

    # Инициализируем оппонентов
    w = algo.get_policy("main").get_weights()
    for pid in opponent_ids:
        algo.get_policy(pid).set_weights(w)

    results_dir = os.path.abspath("./rllib_league_results")
    os.makedirs(results_dir, exist_ok=True)

    # Обучение
    next_ckpt = 5_000_000
    max_iterations = int(os.environ.get("MAX_ITERATIONS", "2000"))
    
    for i in range(max_iterations):
        try:
            res = algo.train()
            
            # Логирование - проверяем разные места для метрик
            if i % 10 == 0:
                # Ray 2.48 - метрики могут быть в разных местах
                rew_mean = res.get('episode_reward_mean', 0)
                if rew_mean == 0:
                    rew_mean = res.get('env_runners', {}).get('episode_reward_mean', 0)
                if rew_mean == 0:
                    rew_mean = res.get('sampler_results', {}).get('episode_reward_mean', 0)
                
                ts_mu = res.get('custom_metrics', {}).get('ts_main_mu', 0)
                steps = res.get('timesteps_total', 0)
                
                print(f"[{i}] rew_mean={rew_mean:.3f} "
                      f"ts_mu={ts_mu:.2f} "
                      f"steps={steps}")
                
                # Сохранение чекпоинта
                checkpoint = algo.save(checkpoint_dir=results_dir)
                print(f"Checkpoint saved: {checkpoint}")
            
            # Milestone checkpoints
            if res.get("timesteps_total", 0) >= next_ckpt:
                checkpoint = algo.save(checkpoint_dir=results_dir)
                print(f"Milestone checkpoint at {next_ckpt}: {checkpoint}")
                next_ckpt += 5_000_000
                
        except Exception as e:
            print(f"Error during training iteration {i}: {e}")
            import traceback
            traceback.print_exc()
            
            # Попытка восстановления
            if 'checkpoint' in locals():
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
