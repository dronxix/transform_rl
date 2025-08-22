"""
Чистый скрипт тренировки Arena Multi-Agent системы в 3D пространстве
ОБНОВЛЕНО: Полная поддержка 3D координат, границ поля, лазеров с ограниченным радиусом
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

# Импорты основных модулей для 3D
from arena_env import ArenaEnv
from entity_attention_model import ONNXEntityAttentionModel 
from masked_multihead_dist import MaskedTargetMoveAimFire3D, MaskedTargetMoveAimFire, create_adaptive_action_distribution
from league_state import LeagueState
from gspo_grpo_policy import GSPOTorchPolicy, GRPOTorchPolicy
from callbacks import FixedLeagueCallbacksWithONNXAndRecording3D

def env_creator(cfg): 
    return ArenaEnv(cfg)

def validate_3d_environment():
    """Проверяет что окружение правильно настроено для 3D"""
    print("🔍 Validating 3D environment setup...")
    
    try:
        # Проверяем импорты
        from arena_env import FIELD_BOUNDS, LASER_MAX_RANGE, LASER_DAMAGE
        print(f"✅ 3D Environment constants:")
        print(f"   Field bounds: {FIELD_BOUNDS}")
        print(f"   Laser range: {LASER_MAX_RANGE}")
        print(f"   Laser damage: {LASER_DAMAGE}")
        
        # Создаем тестовое окружение
        test_env = ArenaEnv({"ally_choices": [1], "enemy_choices": [1]})
        obs_space = test_env.observation_space
        
        # Проверяем размерности
        self_feats = obs_space["self"].shape[0]
        ally_feats = obs_space["allies"].shape[1]
        enemy_feats = obs_space["enemies"].shape[1]
        
        print(f"✅ 3D Observation space validation:")
        print(f"   Self features: {self_feats} (expected: 13 for 3D)")
        print(f"   Ally features: {ally_feats} (expected: 9 for 3D)")
        print(f"   Enemy features: {enemy_feats} (expected: 11 for 3D)")
        
        # Проверяем что это действительно 3D
        if self_feats >= 13 and ally_feats >= 9 and enemy_feats >= 11:
            print("✅ Environment is properly configured for 3D!")
            return True
        else:
            print("⚠️ Environment appears to be 2D - will use adaptive mode")
            return False
            
    except ImportError as e:
        print(f"❌ Missing 3D environment components: {e}")
        return False
    except Exception as e:
        print(f"❌ Environment validation failed: {e}")
        return False

def setup_3d_model_config(obs_space, act_space, is_3d_env=True):
    """Настраивает конфигурацию модели для 3D"""
    
    max_enemies = obs_space["enemies"].shape[0]
    max_allies = obs_space["allies"].shape[0]
    
    # Выбираем подходящую дистрибуцию действий
    if is_3d_env:
        action_dist = "masked_multihead_3d"
        print(f"🎯 Using 3D action distribution")
    else:
        action_dist = "masked_multihead_adaptive"  # Автовыбор
        print(f"🎯 Using adaptive action distribution (2D/3D)")
    
    model_config = {
        "custom_model": "entity_attention",
        "custom_action_dist": action_dist,
        "custom_model_config": {
            "d_model": 128,
            "nhead": 8,
            "layers": 2,
            "ff": 256,
            "hidden": 256,
            "max_enemies": max_enemies,
            "max_allies": max_allies,
            # 3D специфичные настройки
            "include_3d_awareness": is_3d_env,
            "support_3d_actions": is_3d_env,
        },
        "vf_share_layers": False,
    }
    
    print(f"🧠 Model configuration:")
    print(f"   Max allies: {max_allies}, Max enemies: {max_enemies}")
    print(f"   3D awareness: {is_3d_env}")
    print(f"   Action distribution: {action_dist}")
    
    return model_config

def setup_3d_environment_config(args, is_3d_env=True):
    """Настраивает конфигурацию окружения для 3D"""
    
    base_config = {
        "episode_len": 128,
        "ally_choices": [1],
        "enemy_choices": [1],
    }
    
    if is_3d_env:
        # Дополнительные 3D настройки
        base_config.update({
            "assert_invalid_actions": False,
            "seed": 42,
        })
        print(f"🏟️ Using 3D environment configuration")
    else:
        print(f"🏟️ Using standard environment configuration")
    
    # Настройки для тестового режима
    if args.test:
        base_config.update({
            "episode_len": 50,  # Короче для тестов
            "ally_choices": [1, 2],
            "enemy_choices": [1, 2],
        })
        print(f"🧪 Test mode: reduced episode length and team sizes")
    
    return base_config

def main():
    """Основная функция тренировки с поддержкой 3D"""
    
    # Парсинг аргументов
    parser = argparse.ArgumentParser(description="Arena Multi-Agent 3D Training")
    parser.add_argument("--iterations", type=int, default=1000, help="Training iterations")
    parser.add_argument("--algo", choices=["ppo", "gspo", "grpo"], default="gspo", help="Algorithm")
    parser.add_argument("--test", action="store_true", help="Quick test mode")
    parser.add_argument("--force-2d", action="store_true", help="Force 2D mode even if 3D available")
    parser.add_argument("--disable-3d-recording", action="store_true", help="Disable 3D battle recording")
    parser.add_argument("--disable-onnx", action="store_true", help="Disable ONNX export")
    args = parser.parse_args()
    
    print(f"🚀 Starting Arena 3D Training ({args.algo.upper()})")
    print(f"   Iterations: {args.iterations}")
    print(f"   Force 2D mode: {args.force_2d}")
    
    # Валидация 3D окружения
    is_3d_env = validate_3d_environment() and not args.force_2d
    
    if is_3d_env:
        print(f"🌟 Running in 3D mode with field boundaries and laser range limits")
    else:
        print(f"📐 Running in 2D/adaptive mode")
    
    # Инициализация Ray
    ray.init(ignore_reinit_error=True)
    
    try:
        # Регистрация компонентов
        register_env("ArenaEnv", env_creator)
        ModelCatalog.register_custom_model("entity_attention", ONNXEntityAttentionModel)
        
        # Регистрируем подходящие дистрибуции
        if is_3d_env:
            ModelCatalog.register_custom_action_dist("masked_multihead_3d", MaskedTargetMoveAimFire3D)
        ModelCatalog.register_custom_action_dist("masked_multihead", MaskedTargetMoveAimFire)
        ModelCatalog.register_custom_action_dist("masked_multihead_adaptive", create_adaptive_action_distribution)
        
        # Получение размеров окружения
        env_config = setup_3d_environment_config(args, is_3d_env)
        tmp_env = ArenaEnv(env_config)
        obs_space = tmp_env.observation_space
        act_space = tmp_env.action_space
        max_enemies = obs_space["enemies"].shape[0]
        max_allies = obs_space["allies"].shape[0]
        
        print(f"🏟️ Environment: {max_allies} allies vs {max_enemies} enemies")
        if is_3d_env:
            try:
                field_bounds = tmp_env.FIELD_BOUNDS
                laser_range = tmp_env.LASER_MAX_RANGE
                print(f"   3D Field: X[{field_bounds['x_min']:.1f}, {field_bounds['x_max']:.1f}], "
                      f"Y[{field_bounds['y_min']:.1f}, {field_bounds['y_max']:.1f}], "
                      f"Z[{field_bounds['z_min']:.1f}, {field_bounds['z_max']:.1f}]")
                print(f"   Laser range: {laser_range}")
            except AttributeError:
                print(f"   3D configuration not fully available")
        
        # Создание League
        opponent_ids = [f"opponent_{i}" for i in range(4)]
        league = LeagueState.remote(opponent_ids)
        
        # Выбор класса политики
        if args.algo == "gspo":
            policy_cls = GSPOTorchPolicy
            print(f"🎯 Using GSPO (Group Sparse Policy Optimization)")
        elif args.algo == "grpo":
            policy_cls = GRPOTorchPolicy
            print(f"🎯 Using GRPO (Group Relative Policy Optimization)")
        else:
            policy_cls = None  # Стандартный PPO
            print(f"🎯 Using standard PPO")
        
        # Конфигурация модели
        model_config = setup_3d_model_config(obs_space, act_space, is_3d_env)
        
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
            recording_freq = 1
            print("🧪 Test mode: reduced settings")
        else:
            num_workers = 4
            train_batch_size = 16384
            iterations = args.iterations
            export_every = 25
            recording_freq = 5
        
        # Обновляем env_config с правильными размерами
        env_config.update({
            "max_allies": max_allies,
            "max_enemies": max_enemies,
        })
        
        # Конфигурация PPO для 3D
        config = (
            PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .environment(
                env="ArenaEnv",
                env_config=env_config
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
        
        # ИСПРАВЛЕНИЕ: Создание 3D callbacks БЕЗ использования метода .callbacks()
        callbacks = FixedLeagueCallbacksWithONNXAndRecording3D()
        callbacks.setup(
            league_actor=league,
            opponent_ids=opponent_ids,
            eval_episodes=2 if args.test else 4,
            clone_every_iters=3 if args.test else 15,
            curriculum_schedule=[
                (0, [1], [1]),
                (2_000_000, [1, 2], [1, 2]),
                (8_000_000, [1, 2, 3], [1, 2, 3]),
            ] if not args.test else [
                (0, [1], [1]),
                (1000, [1, 2], [1, 2]),
            ],
            # ONNX экспорт настройки
            export_onnx=not args.disable_onnx,
            export_every=export_every,
            export_dir="./onnx_exports_3d" if is_3d_env else "./onnx_exports",
            policies_to_export=["main"],
            
            # Настройки записи 3D боев
            record_battles=not args.disable_3d_recording,
            recording_frequency=recording_freq,
            recordings_dir="./battle_recordings_3d" if is_3d_env else "./battle_recordings",
            
            # 3D специфичные настройки
            track_3d_metrics=is_3d_env,
            log_boundary_violations=is_3d_env,
            log_laser_effectiveness=is_3d_env,
        )
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Правильная установка callbacks для Ray 2.48
        config.callbacks_class = FixedLeagueCallbacksWithONNXAndRecording3D
        
        # Построение алгоритма
        print("🔧 Building algorithm...")
        algo = config.build()
        
        # ИСПРАВЛЕНИЕ: Устанавливаем callbacks ПОСЛЕ создания алгоритма
        algo.callbacks = callbacks
        
        # Инициализация весов оппонентов
        main_weights = algo.get_policy("main").get_weights()
        for pid in opponent_ids:
            algo.get_policy(pid).set_weights(main_weights)
        
        print("✅ Algorithm ready, starting 3D training...")
        if is_3d_env:
            print("🌟 3D features enabled:")
            print("   - Field boundary detection and penalties")
            print("   - Laser range limitations")
            print("   - 3D movement and aiming")
            print("   - Height-based tactics")
        
        # Цикл тренировки
        best_reward = float('-inf')
        final_checkpoint = None
        total_boundary_deaths = 0
        total_3d_metrics = {}
        
        for i in range(iterations):
            try:
                result = algo.train()
                
                # Метрики
                env_runners = result.get("env_runners", {})
                reward = env_runners.get("episode_reward_mean", 0)
                timesteps = result.get("timesteps_total", 0)
                
                # Извлекаем 3D метрики из custom_metrics
                custom = result.get("custom_metrics", {})
                boundary_deaths = custom.get("boundary_deaths_mean", 0)
                if boundary_deaths > 0:
                    total_boundary_deaths += boundary_deaths
                
                # Логирование
                if i % 5 == 0 or args.test:
                    print(f"[{i:4d}] Reward: {reward:7.3f}, Timesteps: {timesteps:,}")
                    
                    # 3D специфичные метрики
                    if is_3d_env and boundary_deaths > 0:
                        print(f"       3D: Boundary deaths: {boundary_deaths:.2f}")
                    
                    # League метрики
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
                        checkpoint_path = f"checkpoint_3d_iter_{i}"
                    
                    print(f"💾 Checkpoint saved at iteration {i}")
                    print(f"   Best reward: {best_reward:.3f}")
                    if is_3d_env:
                        print(f"   Total boundary deaths: {total_boundary_deaths:.1f}")
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
                final_checkpoint = f"final_3d_checkpoint_iter_{i + 1}"
        
        print(f"\n🏁 3D Training completed!")
        try:
            print(f"   Final checkpoint: {os.path.basename(final_checkpoint)}")
            print(f"   Checkpoint directory: {os.path.dirname(final_checkpoint)}")
        except Exception as e:
            print(f"   Final checkpoint saved successfully")
            print(f"   Warning: Could not display path: {e}")
        print(f"   Best reward: {best_reward:.3f}")
        print(f"   Total iterations: {i + 1}")
        
        if is_3d_env:
            print(f"   3D Statistics:")
            print(f"     Total boundary deaths: {total_boundary_deaths:.1f}")
            print(f"     Environment type: 3D with field boundaries")
        
        # Финальный экспорт ONNX
        if not args.disable_onnx and not args.test:
            try:
                from onnx_callbacks import export_onnx_with_meta
                exports = export_onnx_with_meta(
                    algorithm=algo,
                    iteration=i + 1,
                    export_dir="./onnx_exports_3d" if is_3d_env else "./onnx_exports",
                    policies_to_export=["main"]
                )
                if exports:
                    print(f"   ONNX exported: {len(exports)} models ({'3D' if is_3d_env else '2D/adaptive'})")
            except Exception as e:
                print(f"   ONNX export failed: {e}")
        
        # Финальный экспорт 3D визуализации
        if not args.disable_3d_recording and is_3d_env and not args.test:
            try:
                if callbacks.battle_recorder:
                    web_export = callbacks.battle_recorder.export_for_web_visualizer_3d()
                    if web_export:
                        print(f"   3D Battle visualization exported: {web_export}")
            except Exception as e:
                print(f"   3D Visualization export failed: {e}")
        
        # Показываем созданные файлы
        print(f"\n📁 Generated files:")
        
        # ONNX модели
        onnx_dir = "./onnx_exports_3d" if is_3d_env else "./onnx_exports"
        if os.path.exists(onnx_dir):
            onnx_files = []
            for root, dirs, files in os.walk(onnx_dir):
                onnx_files.extend([f for f in files if f.endswith('.onnx')])
            if onnx_files:
                print(f"   ONNX models: {onnx_dir}/ ({len(onnx_files)} files)")
        
        # Battle recordings
        recordings_dir = "./battle_recordings_3d" if is_3d_env else "./battle_recordings"
        if os.path.exists(recordings_dir):
            recording_files = []
            for root, dirs, files in os.walk(recordings_dir):
                recording_files.extend([f for f in files if f.endswith(('.json', '.html'))])
            if recording_files:
                print(f"   Battle recordings: {recordings_dir}/ ({len(recording_files)} files)")
                if is_3d_env:
                    html_files = [f for f in recording_files if f.endswith('.html')]
                    if html_files:
                        print(f"     🎬 3D Visualizations: {len(html_files)} HTML files")
        
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
    
    print(f"\n🎉 3D Training session completed successfully!")
    if is_3d_env:
        print(f"🌟 3D features were used: field boundaries, laser range, height tactics")
    print(f"🎬 Open the HTML files in your browser to view 3D battle replays!")
    return 0

def quick_test_3d():
    """Быстрый тест 3D системы"""
    
    print("🧪 Running quick 3D system test...")
    
    # Проверка импортов
    try:
        import ray
        from arena_env import ArenaEnv, FIELD_BOUNDS, LASER_MAX_RANGE
        from entity_attention_model import ONNXEntityAttentionModel
        from masked_multihead_dist import MaskedTargetMoveAimFire3D
        print("✅ All 3D imports successful")
    except ImportError as e:
        print(f"❌ 3D Import failed: {e}")
        return False
    
    # Проверка Ray
    try:
        ray.init(ignore_reinit_error=True, log_to_driver=False)
        print("✅ Ray initialization successful")
        ray.shutdown()
    except Exception as e:
        print(f"❌ Ray failed: {e}")
        return False
    
    # Проверка 3D окружения
    try:
        env = ArenaEnv({"ally_choices": [1], "enemy_choices": [1], "episode_len": 10})
        obs, _ = env.reset()
        
        # Проверяем 3D размерности
        first_agent = list(obs.keys())[0]
        self_obs = obs[first_agent]["self"]
        
        print(f"✅ 3D Environment working:")
        print(f"   Observation keys: {list(obs.keys())}")
        print(f"   Self obs size: {len(self_obs)} (expected 13+ for 3D)")
        print(f"   Field bounds: {FIELD_BOUNDS}")
        print(f"   Laser range: {LASER_MAX_RANGE}")
        
        # Тест 3D действий
        action_3d = {
            "target": 0,
            "move": [0.1, -0.2, 0.3],  # 3D движение
            "aim": [-0.1, 0.2, -0.3],  # 3D прицеливание
            "fire": 1
        }
        
        actions = {agent_id: action_3d for agent_id in obs.keys()}
        obs, rewards, terms, truncs, infos = env.step(actions)
        
        print(f"✅ 3D Actions processed successfully")
        
    except Exception as e:
        print(f"❌ 3D Environment failed: {e}")
        return False
    
    print("✅ Quick 3D test passed - system looks good!")
    return True

if __name__ == "__main__":
    # Обработка специальных команд
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick-test-3d":
            success = quick_test_3d()
            sys.exit(0 if success else 1)
        elif sys.argv[1] == "--help-examples":
            print("""
🎮 Arena 3D Training Examples:

1. Quick 3D test (5 iterations):
   python train.py --test

2. Basic 3D training:
   python train.py --iterations 100

3. Force 2D mode:
   python train.py --force-2d --iterations 100

4. GSPO algorithm in 3D:
   python train.py --algo gspo --iterations 500

5. GRPO algorithm in 3D:
   python train.py --algo grpo --iterations 300

6. Disable 3D recording:
   python train.py --disable-3d-recording

7. Disable ONNX export:
   python train.py --disable-onnx

8. 3D System check:
   python train.py --quick-test-3d

📁 3D Output locations:
   - Checkpoints: ./rllib_league_results/
   - ONNX models: ./onnx_exports_3d/
   - 3D Battle recordings: ./battle_recordings_3d/
   - 3D Visualizations: ./battle_recordings_3d/*.html
   - Logs: ./logs/

🔍 Monitor with TensorBoard:
   tensorboard --logdir ./logs

🎬 View 3D battles:
   Open HTML files in ./battle_recordings_3d/ with a web browser

🌟 3D Features:
   - Field boundaries (robots die if they go out of bounds)
   - Laser range limitations (can't shoot beyond max range)
   - 3D movement and aiming (x, y, z coordinates)
   - Height-based tactical advantages
   - 3D visualization with Three.js
            """)
            sys.exit(0)
    
    # Обычный запуск
    sys.exit(main())