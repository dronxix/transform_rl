"""
Обновленный главный скрипт тренировки с всеми исправлениями:
1. Исправленный ONNX экспорт с meta.json файлами
2. Запись и визуализация боев
3. Иерархическая система команд (опционально)
4. Улучшенная обработка ошибок
"""

import os
import sys
import argparse
import ray
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

# Импортируем все наши модули
from arena_env import ArenaEnv
from entity_attention_model import ONNXEntityAttentionModel 
from masked_multihead_dist import MaskedTargetMoveAimFire 
from league_state import LeagueState
from gspo_grpo_policy import GSPOTorchPolicy, GRPOTorchPolicy

# Исправленные модули
from updated_callbacks import FixedLeagueCallbacksWithONNXAndRecording
from battle_data_recorder import BattleRecorder, RecordingArenaWrapper

# Опциональная иерархическая система
try:
    from hierarchical_command_policy import CommanderModel, CommandFollowerModel, CommandDistribution
    HIERARCHICAL_AVAILABLE = True
except ImportError:
    HIERARCHICAL_AVAILABLE = False
    print("⚠️ Hierarchical command system not available")

def env_creator(cfg): 
    return ArenaEnv(cfg)

def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description="Advanced Arena Training with ONNX Export and Battle Recording")
    
    # Основные параметры тренировки
    parser.add_argument("--iterations", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--algo", choices=["ppo", "gspo", "grpo"], default="gspo", help="Algorithm variant")
    parser.add_argument("--hierarchical", action="store_true", help="Use hierarchical command system")
    
    # Параметры окружения
    parser.add_argument("--max-allies", type=int, default=6, help="Maximum allies")
    parser.add_argument("--max-enemies", type=int, default=6, help="Maximum enemies")
    parser.add_argument("--episode-len", type=int, default=128, help="Episode length")
    
    # Параметры league
    parser.add_argument("--opponents", type=int, default=6, help="Number of opponent policies")
    parser.add_argument("--eval-episodes", type=int, default=4, help="Episodes per evaluation")
    parser.add_argument("--clone-every", type=int, default=15, help="Clone opponent every N iterations")
    
    # ONNX экспорт
    parser.add_argument("--export-onnx", action="store_true", default=True, help="Enable ONNX export")
    parser.add_argument("--export-every", type=int, default=25, help="Export ONNX every N iterations")
    parser.add_argument("--export-dir", type=str, default="./onnx_exports", help="ONNX export directory")
    
    # Запись боев
    parser.add_argument("--record-battles", action="store_true", default=True, help="Record battles for visualization")
    parser.add_argument("--recording-freq", type=int, default=5, help="Record every N-th evaluation match")
    parser.add_argument("--recordings-dir", type=str, default="./battle_recordings", help="Battle recordings directory")
    
    # Производительность
    parser.add_argument("--num-workers", type=int, default=4, help="Number of env runners")
    parser.add_argument("--train-batch-size", type=int, default=16384, help="Training batch size")
    parser.add_argument("--minibatch-size", type=int, default=2048, help="Minibatch size")
    
    # Логирование
    parser.add_argument("--log-dir", type=str, default="./logs", help="Logging directory")
    parser.add_argument("--checkpoint-freq", type=int, default=50, help="Checkpoint frequency")
    
    return parser.parse_args()

def setup_models_and_env(args):
    """Регистрация моделей и окружений"""
    
    register_env("ArenaEnv", env_creator)
    ModelCatalog.register_custom_model("entity_attention", ONNXEntityAttentionModel)
    ModelCatalog.register_custom_action_dist("masked_multihead", MaskedTargetMoveAimFire)
    
    # Регистрируем иерархические модели если доступны
    if args.hierarchical and HIERARCHICAL_AVAILABLE:
        ModelCatalog.register_custom_model("commander", CommanderModel)
        ModelCatalog.register_custom_model("follower", CommandFollowerModel)
        ModelCatalog.register_custom_action_dist("command_dist", CommandDistribution)
        print("✅ Hierarchical models registered")
    elif args.hierarchical:
        print("❌ Hierarchical system requested but not available, falling back to standard")
        args.hierarchical = False

def create_algorithm_config(args, obs_space, act_space, max_enemies, max_allies, opponent_ids):
    """Создание конфигурации алгоритма"""
    
    # Выбор класса политики
    if args.algo == "gspo":
        policy_cls = GSPOTorchPolicy
        print("🎯 Using GSPO (Group Advantage)")
    elif args.algo == "grpo":
        policy_cls = GRPOTorchPolicy
        print("🎯 Using GRPO (Group Return)")
    else:
        policy_cls = None
        print("🎯 Using standard PPO")
    
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
            "max_enemies": max_enemies,
            "max_allies": max_allies,
        },
        "vf_share_layers": False,
    }
    
    # Policy mapping function
    if args.hierarchical:
        def policy_mapping_fn(agent_id: str, episode=None, **kwargs):
            if agent_id == "commander":
                return "commander_policy"
            elif agent_id.startswith("red_"):
                return "follower_policy" 
            else:
                # Простая ротация оппонентов
                import hashlib
                hash_val = int(hashlib.md5(str(episode).encode()).hexdigest()[:8], 16)
                return opponent_ids[hash_val % len(opponent_ids)]
    else:
        def policy_mapping_fn(agent_id: str, episode=None, **kwargs):
            if agent_id.startswith("red_"):
                return "main"
            else:
                # Простая ротация оппонентов
                import hashlib
                hash_val = int(hashlib.md5(str(episode).encode()).hexdigest()[:8], 16)
                return opponent_ids[hash_val % len(opponent_ids)]
    
    # Создание политик
    if args.hierarchical:
        from gymnasium import spaces
        import numpy as np
        
        # Расширенное observation space для follower (включает команды)
        follower_obs_space = spaces.Dict({
            **obs_space.spaces,
            "command": spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        })
        
        policies = {
            "commander_policy": (None, obs_space, None, {
                "model": {
                    "custom_model": "commander",
                    "custom_action_dist": "command_dist",
                    "custom_model_config": base_model_config["custom_model_config"].copy(),
                    "vf_share_layers": False,
                }
            }),
            "follower_policy": (policy_cls, follower_obs_space, act_space, {
                "model": {
                    "custom_model": "follower",
                    "custom_action_dist": "masked_multihead",
                    "custom_model_config": base_model_config["custom_model_config"].copy(),
                    "vf_share_layers": False,
                }
            }),
            **{
                pid: (None, obs_space, act_space, {
                    "model": base_model_config.copy()
                }) for pid in opponent_ids
            }
        }
        
        policies_to_train = ["commander_policy", "follower_policy"]
        
    else:
        policies = {
            "main": (policy_cls, obs_space, act_space, {
                "model": base_model_config.copy()
            }),
            **{
                pid: (None, obs_space, act_space, {
                    "model": base_model_config.copy()
                }) for pid in opponent_ids
            }
        }
        
        policies_to_train = ["main"]
    
    # Создание конфигурации
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env="ArenaEnv",
            env_config={
                "episode_len": args.episode_len,
                "ally_choices": [1],
                "enemy_choices": [1],
                "max_allies": args.max_allies,
                "max_enemies": args.max_enemies,
                "assert_invalid_actions": True,
                "hierarchical": args.hierarchical,
            }
        )
        .framework("torch")
        .env_runners(
            num_env_runners=args.num_workers,
            num_envs_per_env_runner=1,
            rollout_fragment_length=256,
            batch_mode="truncate_episodes",
        )
        .resources(
            num_gpus=1 if torch.cuda.is_available() else 0,
            num_cpus_for_main_process=1,
        )
        .training(
            gamma=0.99,
            lr=3e-4,
            train_batch_size=args.train_batch_size,
            minibatch_size=args.minibatch_size,
            num_epochs=4,
            use_gae=True,
            lambda_=0.95,
            clip_param=0.15,
            vf_clip_param=10.0,
            entropy_coeff=0.003,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=policies_to_train,
            count_steps_by="agent_steps",
        )
        .fault_tolerance(
            restart_failed_env_runners=True,
        )
    )
    
    return config

def create_callbacks(args, league, opponent_ids):
    """Создание callbacks с настройками"""
    
    def create_callbacks_fn():
        callbacks = FixedLeagueCallbacksWithONNXAndRecording()
        callbacks.setup(
            league_actor=league,
            opponent_ids=opponent_ids,
            eval_episodes=args.eval_episodes,
            clone_every_iters=args.clone_every,
            curriculum_schedule=[
                (0, [1], [1]),
                (2_000_000, [1, 2], [1, 2]),
                (8_000_000, [1, 2, 3], [1, 2, 3]),
            ],            
            # ONNX экспорт
            export_onnx=args.export_onnx,
            export_every=args.export_every,
            export_dir=args.export_dir,
            policies_to_export=["main"] if not args.hierarchical else ["commander_policy", "follower_policy"],
            
            # Запись боев
            record_battles=args.record_battles,
            recording_frequency=args.recording_freq,
            recordings_dir=args.recordings_dir,
        )
        return callbacks
    
    return create_callbacks_fn

def main():
    """Основная функция"""
    
    # Парсинг аргументов
    args = parse_arguments()
    
    print("🚀 Starting Advanced Arena Training")
    print(f"   Algorithm: {args.algo.upper()}")
    print(f"   Hierarchical: {args.hierarchical}")
    print(f"   ONNX Export: {args.export_onnx}")
    print(f"   Battle Recording: {args.record_battles}")
    print(f"   Iterations: {args.iterations}")
    
    # Создание директорий
    os.makedirs(args.log_dir, exist_ok=True)
    if args.export_onnx:
        os.makedirs(args.export_dir, exist_ok=True)
    if args.record_battles:
        os.makedirs(args.recordings_dir, exist_ok=True)
    
    # Инициализация Ray
    ray.init(ignore_reinit_error=True)
    
    try:
        # Настройка моделей и окружений
        setup_models_and_env(args)
        
        # Создание League State
        opponent_ids = [f"opponent_{i}" for i in range(args.opponents)]
        league = LeagueState.remote(opponent_ids)
        
        # Получение размеров окружения
        tmp_env = ArenaEnv({"ally_choices": [1], "enemy_choices": [1]})
        obs_space = tmp_env.observation_space
        act_space = tmp_env.action_space
        max_enemies = obs_space["enemies"].shape[0]
        max_allies = obs_space["allies"].shape[0]
        
        print(f"🏟️ Arena setup: {max_allies} allies vs {max_enemies} enemies")
        
        # Создание конфигурации алгоритма
        config = create_algorithm_config(args, obs_space, act_space, max_enemies, max_allies, opponent_ids)
        
        # Добавление callbacks
        callbacks_fn = create_callbacks(args, league, opponent_ids)
        config = config.callbacks(callbacks_fn)
        
        # Построение алгоритма
        print("🔧 Building algorithm...")
        algo = config.build()
        
        # Инициализация весов оппонентов
        if args.hierarchical:
            # Для иерархической системы инициализируем отдельно
            commander_weights = algo.get_policy("commander_policy").get_weights()
            follower_weights = algo.get_policy("follower_policy").get_weights()
            
            for pid in opponent_ids:
                # Оппоненты используют стандартную модель
                algo.get_policy(pid).set_weights(follower_weights)
        else:
            main_weights = algo.get_policy("main").get_weights()
            for pid in opponent_ids:
                algo.get_policy(pid).set_weights(main_weights)
        
        print("✅ Algorithm built and initialized")
        
        # Основной цикл тренировки
        print(f"🎮 Starting training for {args.iterations} iterations...")
        
        best_reward = float('-inf')
        
        for i in range(args.iterations):
            try:
                result = algo.train()
                
                # Получение метрик
                env_runners_metrics = result.get("env_runners", {})
                episode_reward_mean = env_runners_metrics.get("episode_reward_mean", 0)
                timesteps_total = result.get("timesteps_total", 0)
                
                # Логирование прогресса
                if i % 10 == 0:
                    print(f"[{i:4d}] Reward: {episode_reward_mean:.3f}, Timesteps: {timesteps_total:,}")
                    
                    # Дополнительные метрики
                    custom_metrics = result.get("custom_metrics", {})
                    if custom_metrics:
                        league_metrics = [(k, v) for k, v in custom_metrics.items() if k.startswith("ts_")]
                        if league_metrics:
                            print(f"       League: {dict(league_metrics[:3])}")  # Показываем первые 3
                
                # Сохранение лучшей модели
                if episode_reward_mean > best_reward:
                    best_reward = episode_reward_mean
                    
                # Сохранение чекпоинтов
                if i % args.checkpoint_freq == 0 and i > 0:
                    checkpoint = algo.save()
                    print(f"💾 Checkpoint saved: {checkpoint}")
                    
                    # Дополнительная информация в логах
                    print(f"    Best reward so far: {best_reward:.3f}")
                    
                    # Статистика league если доступна
                    try:
                        scores = ray.get(league.get_all_scores.remote())
                        main_score = scores.get("main", (0, 0))
                        print(f"    Main policy TrueSkill: μ={main_score[0]:.3f}, σ={main_score[1]:.3f}")
                    except:
                        pass
                
            except KeyboardInterrupt:
                print("\n⏹️ Training interrupted by user")
                break
            except Exception as e:
                print(f"❌ Training error at iteration {i}: {e}")
                import traceback
                traceback.print_exc()
                
                # Продолжаем если ошибка не критична
                continue
        
        # Финальное сохранение
        final_checkpoint = algo.save()
        print(f"🏁 Final checkpoint saved: {final_checkpoint}")
        
        # Финальная статистика
        print(f"\n📊 Training Summary:")
        print(f"   Total iterations: {i + 1}")
        print(f"   Best reward: {best_reward:.3f}")
        print(f"   Final timesteps: {timesteps_total:,}")
        
        # Статистика League
        try:
            scores = ray.get(league.get_all_scores.remote())
            print(f"\n🏆 Final League Standings:")
            for policy_id, (mu, sigma) in scores.items():
                conservative_score = mu - 3 * sigma
                print(f"   {policy_id}: μ={mu:.3f}, σ={sigma:.3f}, conservative={conservative_score:.3f}")
        except Exception as e:
            print(f"   Could not retrieve league scores: {e}")
        
        # Экспорт финальных моделей
        if args.export_onnx:
            try:
                print(f"\n🔧 Exporting final ONNX models...")
                from updated_callbacks import export_onnx_with_meta
                
                policies_to_export = ["main"] if not args.hierarchical else ["commander_policy", "follower_policy"]
                successful_exports = export_onnx_with_meta(
                    algorithm=algo,
                    iteration=i + 1,
                    export_dir=args.export_dir,
                    policies_to_export=policies_to_export
                )
                
                if successful_exports:
                    print(f"✅ Exported {len(successful_exports)} final models")
                    for export in successful_exports:
                        print(f"   {export['policy_id']}: {export['onnx_path']}")
                else:
                    print("⚠️ No final models were exported")
                    
            except Exception as e:
                print(f"❌ Final ONNX export failed: {e}")
        
        # Экспорт финальных записей боев
        if args.record_battles:
            try:
                print(f"\n🎬 Exporting final battle recordings...")
                
                # Проверяем есть ли записи у callbacks
                callback_instance = algo.callbacks._callbacks[0] if hasattr(algo, 'callbacks') and algo.callbacks._callbacks else None
                if callback_instance and hasattr(callback_instance, 'battle_recorder') and callback_instance.battle_recorder:
                    recorder = callback_instance.battle_recorder
                    
                    # Экспортируем для веб-визуализации
                    html_path = recorder.export_for_web_visualizer()
                    if html_path:
                        print(f"✅ Battle replay exported: {html_path}")
                        
                        # Показываем статистику записей
                        stats = recorder.get_summary_statistics()
                        if "total_battles" in stats:
                            print(f"   Total battles recorded: {stats['total_battles']}")
                            print(f"   Average duration: {stats['average_duration']:.1f}s")
                            if "win_rate_by_team" in stats:
                                print(f"   Win rates: {stats['win_rate_by_team']}")
                    else:
                        print("⚠️ No battle recordings to export")
                else:
                    print("⚠️ Battle recorder not found in callbacks")
                    
            except Exception as e:
                print(f"❌ Battle recording export failed: {e}")
        
        # Очистка
        algo.stop()
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Results saved in:")
        print(f"   Checkpoints: {final_checkpoint}")
        if args.export_onnx:
            print(f"   ONNX models: {args.export_dir}")
        if args.record_battles:
            print(f"   Battle recordings: {args.recordings_dir}")
        print(f"   Logs: {args.log_dir}")
        
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        ray.shutdown()
    
    return 0

def run_quick_test():
    """Быстрый тест системы с минимальными параметрами"""
    
    print("🧪 Running quick test of the training system...")
    
    # Подготавливаем тестовые аргументы
    test_args = argparse.Namespace(
        iterations=5,
        algo="ppo",
        hierarchical=False,
        max_allies=3,
        max_enemies=3,
        episode_len=32,
        opponents=2,
        eval_episodes=1,
        clone_every=3,
        export_onnx=True,
        export_every=2,
        export_dir="./test_onnx",
        record_battles=True,
        recording_freq=1,
        recordings_dir="./test_recordings",
        num_workers=0,  # Только локальный worker для теста
        train_batch_size=512,
        minibatch_size=256,
        log_dir="./test_logs",
        checkpoint_freq=3
    )
    
    print("🚀 Starting quick test...")
    
    # Запускаем тренировку с тестовыми параметрами
    ray.init(ignore_reinit_error=True)
    
    try:
        setup_models_and_env(test_args)
        
        opponent_ids = [f"opponent_{i}" for i in range(test_args.opponents)]
        league = LeagueState.remote(opponent_ids)
        
        tmp_env = ArenaEnv({"ally_choices": [1], "enemy_choices": [1]})
        obs_space = tmp_env.observation_space
        act_space = tmp_env.action_space
        max_enemies = obs_space["enemies"].shape[0]
        max_allies = obs_space["allies"].shape[0]
        
        config = create_algorithm_config(test_args, obs_space, act_space, max_enemies, max_allies, opponent_ids)
        callbacks_fn = create_callbacks(test_args, league, opponent_ids)
        config = config.callbacks(callbacks_fn)
        
        algo = config.build()
        
        # Инициализируем веса
        main_weights = algo.get_policy("main").get_weights()
        for pid in opponent_ids:
            algo.get_policy(pid).set_weights(main_weights)
        
        print("✅ Algorithm built successfully")
        
        # Запускаем несколько итераций
        for i in range(test_args.iterations):
            try:
                result = algo.train()
                reward = result.get("env_runners", {}).get("episode_reward_mean", 0)
                print(f"Test iteration {i}: reward={reward:.3f}")
            except Exception as e:
                print(f"Error in test iteration {i}: {e}")
                
        algo.stop()
        print("✅ Quick test completed successfully!")
        
        # Проверяем что файлы созданы
        if os.path.exists(test_args.export_dir):
            onnx_files = [f for f in os.listdir(test_args.export_dir) if f.endswith('.onnx')]
            meta_files = [f for f in os.listdir(test_args.export_dir) if f.endswith('.json')]
            print(f"📄 Created {len(onnx_files)} ONNX files and {len(meta_files)} meta files")
            
        if os.path.exists(test_args.recordings_dir):
            recording_files = [f for f in os.listdir(test_args.recordings_dir) if f.endswith('.json') or f.endswith('.html')]
            print(f"🎬 Created {len(recording_files)} recording files")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        ray.shutdown()

if __name__ == "__main__":
    # Проверяем аргументы командной строки
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Режим быстрого теста
        success = run_quick_test()
        sys.exit(0 if success else 1)
    else:
        # Обычная тренировка
        sys.exit(main())


# Дополнительные утилиты для проверки системы
def check_system_requirements():
    """Проверка системных требований"""
    
    print("🔍 Checking system requirements...")
    
    # Проверка Python версии
    import sys
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Проверка основных библиотек
    required_packages = [
        ("torch", "PyTorch"),
        ("ray", "Ray"),
        ("numpy", "NumPy"),
        ("gymnasium", "Gymnasium"),
    ]
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} not found")
            return False
    
    # Проверка опциональных библиотек
    optional_packages = [
        ("onnxruntime", "ONNX Runtime (for ONNX inference)"),
        ("tensorboard", "TensorBoard (for logging)"),
        ("trueskill", "TrueSkill (for league ratings)"),
    ]
    
    for package, name in optional_packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"⚠️ {name} not found (optional)")
    
    # Проверка CUDA если доступна
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️ CUDA not available, using CPU")
    except:
        print("⚠️ Could not check CUDA status")
    
    print("✅ System requirements check completed")
    return True

def print_usage_examples():
    """Примеры использования"""
    
    print("\n📚 Usage Examples:")
    print("\n1. Quick test:")
    print("   python train_advanced.py --test")
    
    print("\n2. Basic training:")
    print("   python train_advanced.py --iterations 100 --algo gspo")
    
    print("\n3. Hierarchical training:")
    print("   python train_advanced.py --hierarchical --algo ppo --iterations 200")
    
    print("\n4. Training with custom settings:")
    print("   python train_advanced.py \\")
    print("     --iterations 1000 \\")
    print("     --algo grpo \\")
    print("     --max-allies 4 \\")
    print("     --max-enemies 4 \\")
    print("     --export-every 10 \\")
    print("     --recording-freq 2")
    
    print("\n5. Training without ONNX export:")
    print("   python train_advanced.py --no-export-onnx --iterations 500")
    
    print("\n6. Check system requirements:")
    print("   python -c \"from train_advanced import check_system_requirements; check_system_requirements()\"")
    
    print("\n📖 For more options: python train_advanced.py --help")

# Если скрипт запущен с --help или --examples
if __name__ == "__main__" and len(sys.argv) > 1:
    if sys.argv[1] in ["--help", "-h"]:
        parse_arguments().print_help()
        print_usage_examples()
    elif sys.argv[1] == "--examples":
        print_usage_examples()
    elif sys.argv[1] == "--check":
        check_system_requirements()
    elif sys.argv[1] == "--test":
        success = run_quick_test()
        sys.exit(0 if success else 1)
    else:
        sys.exit(main())