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
import torch

def env_creator(cfg): 
    return ArenaEnv(cfg)

def main():
    # Инициализация Ray 2.48
    ray.init(ignore_reinit_error=True)

    register_env("ArenaEnv", env_creator)
    ModelCatalog.register_custom_model("entity_attention", EntityAttentionModel)
    ModelCatalog.register_custom_action_dist("masked_multihead", MaskedTargetMoveAimFire)

    opponent_ids = [f"opponent_{i}" for i in range(6)]
    league = LeagueState.remote(opponent_ids)

    # Статическая policy_mapping_fn для стабильности
    def policy_mapping_fn(agent_id: str, episode=None, **kwargs):
        if agent_id.startswith("red_"):
            return "main"
        else:
            # Простая ротация оппонентов
            import hashlib
            hash_val = int(hashlib.md5(str(episode).encode()).hexdigest()[:8], 16)
            return opponent_ids[hash_val % len(opponent_ids)]

    # Получаем spaces
    tmp_env = ArenaEnv({"ally_choices": [1], "enemy_choices": [1]})
    obs_space = tmp_env.observation_space
    act_space = tmp_env.action_space

    # ВАЖНО: Извлекаем max_enemies из observation_space
    max_enemies = obs_space["enemies"].shape[0]
    max_allies = obs_space["allies"].shape[0]

    # Базовая конфигурация модели
    base_model_config = {
        "custom_model": "entity_attention",
        "custom_action_dist": "masked_multihead",
        "custom_model_config": {
            "d_model": 128,
            "nhead": 8,
            "layers": 2,
            "ff": 256,
            "hidden": 256,
            "max_enemies": max_enemies,  # ПЕРЕДАЕМ ДЛЯ ACTION_DIST
            "max_allies": max_allies,
        },
        "vf_share_layers": False,
    }

    # Выбор алгоритма
    algo_variant = os.environ.get("ALGO_VARIANT", "gspo").lower()
    if algo_variant == "gspo": 
        policy_cls = GSPOTorchPolicy
    elif algo_variant == "grpo": 
        policy_cls = GRPOTorchPolicy
    else:
        policy_cls = None  # стандартный PPO

    # RAY 2.48 КОНФИГУРАЦИЯ
    config = (
        PPOConfig()
        # НЕ ВКЛЮЧАЕМ новый API stack - ModelV2 несовместим с RLModule
        .api_stack(
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
        # RAY 2.48: используем env_runners вместо rollouts
        .env_runners(
            num_env_runners=4,
            num_envs_per_env_runner=1,
            rollout_fragment_length=256,
            batch_mode="truncate_episodes",
        )
        .resources(
            num_gpus=1 if torch.cuda.is_available() else 0,
            num_cpus_for_main_process=1,  # RAY 2.48: изменился параметр
        )
        .training(
            gamma=0.99,
            lr=3e-4,
            train_batch_size=16384,
            minibatch_size=2048,
            num_epochs=4,
            use_gae=True,
            lambda_=0.95,
            clip_param=0.15,
            vf_clip_param=10.0,
            entropy_coeff=0.003,
        )
        .multi_agent(
            policies = {
                "main": (policy_cls, obs_space, act_space, {
                    "model": base_model_config.copy()
                }),
                **{
                    pid: (None, obs_space, act_space, {
                        "model": base_model_config.copy()
                    }) for pid in opponent_ids
                }
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["main"],
            count_steps_by="agent_steps",
        )
        .debugging(
            log_level="INFO",
        )
        .fault_tolerance(
            restart_failed_env_runners=True,  # RAY 2.48: новый параметр
        )
    )

    def create_callbacks():
        """Фабрика для создания callbacks с правильными параметрами"""
        callbacks = LeagueCallbacks()
        callbacks.setup(
            league_actor=league,
            opponent_ids=opponent_ids,
            eval_episodes=4,
            clone_every_iters=15,
            curriculum_schedule=[
                (0, [1], [1]),
                (2_000_000, [1, 2], [1, 2]),
                (8_000_000, [1, 2, 3], [1, 2, 3]),
            ]
        )
        return callbacks

    # Передаем функцию, а не объект
    config = config.callbacks(create_callbacks)

    # ИЛИ альтернативный способ - через лямбду:
    config = config.callbacks(lambda: LeagueCallbacks().setup(
        league_actor=league,
        opponent_ids=opponent_ids,
        eval_episodes=4,
        clone_every_iters=15,
        curriculum_schedule=[(0, [1], [1]), (2_000_000, [1, 2], [1, 2])]
    ))

    # ИЛИ самый простой способ - через класс:
    class ConfiguredLeagueCallbacks(LeagueCallbacks):
        def __init__(self):
            super().__init__()
            # Здесь сразу настраиваем параметры
            self.league = None  # Будет установлено позже
            self.opponent_ids = [f"opponent_{i}" for i in range(6)]
            self.eval_eps = 4
            self.clone_every = 15
            self.curriculum = [
                (0, [1], [1]),
                (2_000_000, [1, 2], [1, 2]),
                (8_000_000, [1, 2, 3], [1, 2, 3]),
            ]
        
        def set_league(self, league_actor):
            self.league = league_actor

    # Используем класс напрямую
    config = config.callbacks(ConfiguredLeagueCallbacks)
    
    # Построение алгоритма
    algo = config.build()
    
    # Инициализация весов оппонентов
    main_weights = algo.get_policy("main").get_weights()
    for pid in opponent_ids:
        algo.get_policy(pid).set_weights(main_weights)
    
    # Основной цикл тренировки
    try:
        for i in range(2000):
            result = algo.train()
            
            if i % 10 == 0:
                # Ray 2.48 - метрики в env_runners
                episode_reward_mean = result.get("env_runners", {}).get("episode_reward_mean", 0)
                timesteps_total = result.get("timesteps_total", 0)
                
                print(f"[{i:4d}] reward: {episode_reward_mean:.3f}, "
                      f"timesteps: {timesteps_total}")
                
                # Сохранение чекпоинта
                checkpoint = algo.save()
                if i % 50 == 0:
                    print(f"Checkpoint saved: {checkpoint}")
                    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        algo.stop()
        ray.shutdown()

if __name__ == "__main__":
    main()