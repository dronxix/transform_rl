"""
Чистый скрипт тренировки Arena Multi-Agent системы
ИСПРАВЛЕНИЕ: Добавлена обработка final_checkpoint и правильная работа с Ray 2.48
"""

import os
import sys
import argparse
import ray
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

# Импорты основных модулей
from arena_env import ArenaEnv
from entity_attention_model import ONNXEntityAttentionModel 
from masked_multihead_dist import MaskedTargetMoveAimFire 
from league_state import LeagueState
from gspo_grpo_policy import GSPOTorchPolicy, GRPOTorchPolicy
from callbacks import FixedLeagueCallbacksWithONNXAndRecording

def env_creator(cfg): 
    return ArenaEnv(cfg)

def main():
    """Основная функция тренировки"""
    
    # Парсинг аргументов
    parser = argparse.ArgumentParser(description="Arena Multi-Agent Training")
    parser.add_argument("--iterations", type=int, default=1000, help="Training iterations")
    parser.add_argument("--algo", choices=["ppo", "gspo", "grpo"], default="gspo", help="Algorithm")
    parser.add_argument("--test", action="store_true", help="Quick test mode")
    args = parser.parse_args()
    
    print(f"🚀 Starting Arena Training ({args.algo.upper()})")
    print(f"   Iterations: {args.iterations}")
    
    # Инициализация Ray
    ray.init(ignore_reinit_error=True)
    
    try:
        # Регистрация компонентов
        register_env("ArenaEnv", env_creator)
        ModelCatalog.register_custom_model("entity_attention", ONNXEntityAttentionModel)
        ModelCatalog.register_custom_action_dist("masked_multihead", MaskedTargetMoveAimFire)
        
        # Получение размеров окружения
        tmp_env = ArenaEnv({"ally_choices": [1], "enemy_choices": [1]})
        obs_space = tmp_env.observation_space
        act_space = tmp_env.action_space
        max_enemies = obs_space["enemies"].shape[0]
        max_allies = obs_space["allies"].shape[0]
        
        print(f"🏟️ Environment: {max_allies} allies vs {max_enemies} enemies")
        
        # Создание League
        opponent_ids = [f"opponent_{i}" for i in range(4)]
        league = LeagueState.remote(opponent_ids)
        
        # Выбор класса политики
        if args.algo == "gspo":
            policy_cls = GSPOTorchPolicy
        elif args.algo == "grpo":
            policy_cls = GRPOTorchPolicy
        else:
            policy_cls = None  # Стандартный PPO
        
        # Конфигурация модели
        model_config = {
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
        
        # Policy mapping
        def policy_mapping_fn(agent_id: str, episode=None, **kwargs):
            if agent_id.startswith("red_"):
                return "main"
            else:
                # Простая ротация оппонентов
                import hashlib
                hash_val = int(hashlib.md5(str(episode).encode()).hexdigest()[:8], 16)
                return opponent_ids[hash_val % len(opponent_ids)]
        
        # Создание политик
        policies = {
            "main": (policy_cls, obs_space, act_space, {"model": model_config}),
        }
        
        # Добавляем оппонентов
        for pid in opponent_ids:
            policies[pid] = (None, obs_space, act_space, {"model": model_config})
        
        # Настройки для тестового режима
        if args.test:
            num_workers = 0
            train_batch_size = 512
            iterations = 5
            export_every = 2
            print("🧪 Test mode: reduced settings")
        else:
            num_workers = 4
            train_batch_size = 16384
            iterations = args.iterations
            export_every = 25
        
        # Конфигурация PPO
        config = (
            PPOConfig()
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
                    "max_allies": max_allies,
                    "max_enemies": max_enemies,
                }
            )
            .framework("torch")
            .env_runners(
                num_env_runners=num_workers,
                num_envs_per_env_runner=1,
                rollout_fragment_length=256,
            )
            .resources(
                num_gpus=1 if torch.cuda.is_available() else 0,
                num_cpus_for_main_process=1,
            )
            .training(
                gamma=0.99,
                lr=3e-4,
                train_batch_size=train_batch_size,
                minibatch_size=train_batch_size // 8,
                num_epochs=4,
                use_gae=True,
                lambda_=0.95,
                clip_param=0.15,
                entropy_coeff=0.003,
            )
            .multi_agent(
                policies=policies,
                policy_mapping_fn=policy_mapping_fn,
                policies_to_train=["main"],
            )
        )
        
        # ИСПРАВЛЕНИЕ: Создание callbacks БЕЗ использования метода .callbacks()
        # Это решает проблему с worker processes
        callbacks = FixedLeagueCallbacksWithONNXAndRecording()
        callbacks.setup(
            league_actor=league,
            opponent_ids=opponent_ids,
            eval_episodes=2 if args.test else 4,
            clone_every_iters=3 if args.test else 15,
            export_onnx=True,
            export_every=export_every,
            export_dir="./onnx_exports",
            policies_to_export=["main"],
            record_battles=True,
            recording_frequency=1 if args.test else 5,
            recordings_dir="./battle_recordings",
        )
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Правильная установка callbacks для Ray 2.48
        # Метод .callbacks() не работает корректно с worker processes
        config.callbacks_class = FixedLeagueCallbacksWithONNXAndRecording
        
        # Построение алгоритма
        print("🔧 Building algorithm...")
        algo = config.build()
        
        # ИСПРАВЛЕНИЕ: Устанавливаем callbacks ПОСЛЕ создания алгоритма
        algo.callbacks = callbacks
        
        # Инициализация весов оппонентов
        main_weights = algo.get_policy("main").get_weights()
        for pid in opponent_ids:
            algo.get_policy(pid).set_weights(main_weights)
        
        print("✅ Algorithm ready, starting training...")
        
        # Цикл тренировки
        best_reward = float('-inf')
        final_checkpoint = None
        
        for i in range(iterations):
            try:
                result = algo.train()
                
                # Метрики
                env_runners = result.get("env_runners", {})
                reward = env_runners.get("episode_reward_mean", 0)
                timesteps = result.get("timesteps_total", 0)
                
                # Логирование
                if i % 5 == 0 or args.test:
                    print(f"[{i:4d}] Reward: {reward:7.3f}, Timesteps: {timesteps:,}")
                    
                    # League метрики
                    custom = result.get("custom_metrics", {})
                    ts_metrics = {k: v for k, v in custom.items() if k.startswith("ts_main")}
                    if ts_metrics:
                        mu = ts_metrics.get("ts_main_mu", 0)
                        sigma = ts_metrics.get("ts_main_sigma", 0)
                        print(f"       TrueSkill: μ={mu:.3f}, σ={sigma:.3f}")
                
                # Отслеживание лучшего результата
                if reward > best_reward:
                    best_reward = reward
                
                # Сохранение чекпоинтов
                checkpoint_freq = 3 if args.test else 50
                if i % checkpoint_freq == 0 and i > 0:
                    checkpoint_result = algo.save()
                    # ИСПРАВЛЕНИЕ: Правильная обработка Checkpoint объекта в Ray 2.48+
                    try:
                        if hasattr(checkpoint_result, 'to_directory'):
                            checkpoint_path = checkpoint_result.to_directory()
                        elif hasattr(checkpoint_result, 'as_directory'):
                            checkpoint_path = checkpoint_result.as_directory()
                        elif hasattr(checkpoint_result, 'checkpoint'):
                            checkpoint_path = checkpoint_result.checkpoint
                        elif isinstance(checkpoint_result, str):
                            checkpoint_path = checkpoint_result
                        else:
                            checkpoint_path = str(checkpoint_result)
                    except Exception as e:
                        print(f"Warning: Could not extract checkpoint path: {e}")
                        checkpoint_path = f"checkpoint_iter_{i}"
                    
                    print(f"💾 Checkpoint saved at iteration {i}")
                    print(f"   Best reward: {best_reward:.3f}")
                    final_checkpoint = checkpoint_path
                
            except KeyboardInterrupt:
                print("\n⏹️ Training interrupted")
                break
            except Exception as e:
                print(f"❌ Error at iteration {i}: {e}")
                if args.test:
                    break
                continue
        
        # Финальное сохранение
        if final_checkpoint is None:
            checkpoint_result = algo.save()
            # ИСПРАВЛЕНИЕ: Правильная обработка Checkpoint объекта в Ray 2.48+
            try:
                if hasattr(checkpoint_result, 'to_directory'):
                    final_checkpoint = checkpoint_result.to_directory()
                elif hasattr(checkpoint_result, 'as_directory'):
                    final_checkpoint = checkpoint_result.as_directory()
                elif hasattr(checkpoint_result, 'checkpoint'):
                    final_checkpoint = checkpoint_result.checkpoint
                elif isinstance(checkpoint_result, str):
                    final_checkpoint = checkpoint_result
                else:
                    final_checkpoint = str(checkpoint_result)
            except Exception as e:
                print(f"Warning: Could not extract final checkpoint path: {e}")
                final_checkpoint = f"final_checkpoint_iter_{i + 1}"
        
        print(f"\n🏁 Training completed!")
        try:
            print(f"   Final checkpoint: {os.path.basename(final_checkpoint)}")
            print(f"   Checkpoint directory: {os.path.dirname(final_checkpoint)}")
        except Exception as e:
            print(f"   Final checkpoint saved successfully")
            print(f"   Warning: Could not display path: {e}")
        print(f"   Best reward: {best_reward:.3f}")
        print(f"   Total iterations: {i + 1}")
        
        # Финальный экспорт ONNX
        if not args.test:
            try:
                from onnx_callbacks import export_onnx_with_meta
                exports = export_onnx_with_meta(
                    algorithm=algo,
                    iteration=i + 1,
                    export_dir="./onnx_exports",
                    policies_to_export=["main"]
                )
                if exports:
                    print(f"   ONNX exported: {len(exports)} models")
            except Exception as e:
                print(f"   ONNX export failed: {e}")
        
        # Показываем созданные файлы
        print(f"\n📁 Generated files:")
        if os.path.exists("./onnx_exports"):
            onnx_files = []
            for root, dirs, files in os.walk("./onnx_exports"):
                onnx_files.extend([f for f in files if f.endswith('.onnx')])
            if onnx_files:
                print(f"   ONNX models: ./onnx_exports/ ({len(onnx_files)} files)")
        
        if os.path.exists("./battle_recordings"):
            recording_files = []
            for root, dirs, files in os.walk("./battle_recordings"):
                recording_files.extend([f for f in files if f.endswith(('.json', '.html'))])
            if recording_files:
                print(f"   Battle recordings: ./battle_recordings/ ({len(recording_files)} files)")
        
        try:
            print(f"   Checkpoints: {os.path.dirname(final_checkpoint)}")
        except:
            print(f"   Checkpoints: ./checkpoints/ (saved successfully)")
        
        # Очистка
        algo.stop()
        
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        ray.shutdown()
    
    print(f"\n🎉 Training session completed successfully!")
    return 0

def quick_test():
    """Быстрый тест системы"""
    
    print("🧪 Running quick system test...")
    
    # Проверка импортов
    try:
        import ray
        from arena_env import ArenaEnv
        from entity_attention_model import ONNXEntityAttentionModel
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Проверка Ray
    try:
        ray.init(ignore_reinit_error=True, log_to_driver=False)
        print("✅ Ray initialization successful")
        ray.shutdown()
    except Exception as e:
        print(f"❌ Ray failed: {e}")
        return False
    
    # Проверка окружения
    try:
        env = ArenaEnv({"ally_choices": [1], "enemy_choices": [1], "episode_len": 10})
        obs, _ = env.reset()
        print(f"✅ Environment working (obs keys: {list(obs.keys())})")
    except Exception as e:
        print(f"❌ Environment failed: {e}")
        return False
    
    print("✅ Quick test passed - system looks good!")
    return True

if __name__ == "__main__":
    # Обработка специальных команд
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick-test":
            success = quick_test()
            sys.exit(0 if success else 1)
        elif sys.argv[1] == "--help-examples":
            print("""
🎮 Arena Training Examples:

1. Quick test (5 iterations):
   python train.py --test

2. Basic training:
   python train.py --iterations 100

3. GSPO algorithm:
   python train.py --algo gspo --iterations 500

4. GRPO algorithm:
   python train.py --algo grpo --iterations 300

5. System check:
   python train.py --quick-test

📁 Output locations:
   - Checkpoints: ./rllib_league_results/
   - ONNX models: ./onnx_exports/
   - Battle recordings: ./battle_recordings/
   - Logs: ./logs/

🔍 Monitor with TensorBoard:
   tensorboard --logdir ./logs
            """)
            sys.exit(0)
    
    # Обычный запуск
    sys.exit(main())