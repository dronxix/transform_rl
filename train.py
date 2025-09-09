"""
Универсальный скрипт тренировки Multi-Agent системы
Автоматически адаптируется к любым форматам actions/obs
"""

import os
import sys
import argparse
import ray
import torch
import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from typing import Dict, Any, Optional

# Импорты универсальных модулей
from entity_attention_model import ONNXEntityAttentionModel, DynamicActionConfig, DynamicObservationProcessor
from masked_multihead_dist import UniversalActionDistribution, create_adaptive_action_distribution
from league_state import LeagueState
from gspo_grpo_policy import GSPOTorchPolicy, GRPOTorchPolicy
from callbacks import UniversalLeagueCallbacks, analyze_environment_compatibility

class UniversalEnvironmentManager:
    """Менеджер для работы с различными типами окружений"""
    
    def __init__(self):
        self.registered_envs = {}
        self.env_configs = {}
        self.detected_capabilities = {}
    
    def register_environment(self, env_name: str, env_creator_fn, default_config: Dict = None):
        """Регистрирует окружение в системе"""
        register_env(env_name, env_creator_fn)
        self.registered_envs[env_name] = env_creator_fn
        if default_config:
            self.env_configs[env_name] = default_config
        print(f"✅ Registered environment: {env_name}")
    
    def auto_detect_environment(self, env_name: str, env_config: Dict) -> Dict[str, Any]:
        """Автоматически определяет возможности окружения"""
        try:
            if env_name in self.registered_envs:
                # Создаем временное окружение для анализа
                temp_env = self.registered_envs[env_name](env_config)
                obs, _ = temp_env.reset()
                
                # Генерируем образец действий
                sample_actions = {}
                for agent_id in obs.keys():
                    if hasattr(temp_env, 'action_space'):
                        sample_actions[agent_id] = temp_env.action_space.sample()
                    else:
                        # Fallback действие
                        sample_actions[agent_id] = {"target": 0, "move": [0.0, 0.0], "aim": [0.0, 0.0], "fire": 0}
                
                # Анализируем возможности
                capabilities, recommendations = analyze_environment_compatibility(
                    obs, sample_actions, env_config
                )
                
                self.detected_capabilities[env_name] = capabilities
                
                print(f"🔍 Environment '{env_name}' analysis:")
                print(f"   Type: {capabilities.get('observation_format', 'unknown')}")
                print(f"   3D Support: {capabilities.get('is_3d', False)}")
                print(f"   Teams: {capabilities.get('has_teams', False)}")
                print(f"   Recording: {capabilities.get('supports_recording', False)}")
                
                return capabilities
                
        except Exception as e:
            print(f"⚠️ Could not analyze environment '{env_name}': {e}")
            return {"supports_recording": False, "is_3d": False, "has_teams": False}
    
    def get_optimal_config(self, env_name: str, capabilities: Dict) -> Dict[str, Any]:
        """Возвращает оптимальную конфигурацию для окружения"""
        config = {
            "model_type": "universal",
            "action_distribution": "universal_action_dist",
            "callbacks_type": "universal",
            "recording_enabled": capabilities.get("supports_recording", False),
            "3d_features": capabilities.get("is_3d", False)
        }
        
        # Настройки специфичные для типа окружения
        if capabilities.get("is_3d"):
            config.update({
                "export_dir": "./onnx_exports_3d",
                "recordings_dir": "./battle_recordings_3d",
                "track_3d_metrics": True
            })
        else:
            config.update({
                "export_dir": "./onnx_exports_2d",
                "recordings_dir": "./battle_recordings_2d",
                "track_3d_metrics": False
            })
        
        return config

def auto_detect_and_import_environment():
    """Автоматически обнаруживает и импортирует доступные окружения"""
    env_manager = UniversalEnvironmentManager()
    
    # Пытаемся импортировать известные окружения
    known_environments = [
        ("ArenaEnv", "arena_env", "ArenaEnv"),
        ("PettingZooEnv", "pettingzoo", None),  # Если используется PettingZoo
        ("CustomEnv", "custom_env", "CustomEnv"),  # Пользовательские
    ]
    
    for env_name, module_name, class_name in known_environments:
        try:
            module = __import__(module_name)
            if class_name and hasattr(module, class_name):
                env_class = getattr(module, class_name)
                
                def env_creator(cfg):
                    return env_class(cfg)
                
                # Определяем дефолтный конфиг
                default_config = {}
                if env_name == "ArenaEnv":
                    default_config = {
                        "episode_len": 128,
                        "ally_choices": [1],
                        "enemy_choices": [1],
                    }
                
                env_manager.register_environment(env_name, env_creator, default_config)
                
        except ImportError as e:
            print(f"ℹ️ Environment {env_name} not available: {e}")
        except Exception as e:
            print(f"⚠️ Error importing {env_name}: {e}")
    
    return env_manager

def setup_universal_model_config(obs_space, action_space, capabilities: Dict) -> Dict[str, Any]:
    """Настраивает универсальную конфигурацию модели"""
    
    # Анализируем пространства
    action_config = DynamicActionConfig(action_space)
    obs_processor = DynamicObservationProcessor(obs_space)
    
    model_config = {
        "custom_model": "universal_entity_attention",
        "custom_action_dist": "universal_action_dist",
        "custom_model_config": {
            "d_model": 128,
            "nhead": 8,
            "layers": 2,
            "ff": 256,
            "hidden": 256,
            # Передаем спецификации действий и наблюдений
            "action_spec": action_config.action_spec,
            "obs_spec": obs_processor.obs_spec,
            # Возможности окружения
            "environment_capabilities": capabilities,
            "support_3d": capabilities.get("is_3d", False),
            "support_teams": capabilities.get("has_teams", False),
        },
        "vf_share_layers": False,
    }
    
    print(f"🧠 Universal Model Configuration:")
    print(f"   Action spec: {action_config.action_spec}")
    print(f"   Obs spec: {obs_processor.obs_spec}")
    print(f"   Capabilities: {capabilities}")
    
    return model_config

def create_universal_policy_mapping(capabilities: Dict):
    """Создает универсальную функцию маппинга политик"""
    
    if capabilities.get("has_teams", False):
        # Командная система
        def policy_mapping_fn(agent_id: str, episode=None, **kwargs):
            if agent_id.startswith("red_") or "team1" in agent_id.lower() or "ally" in agent_id.lower():
                return "main"
            elif agent_id.startswith("blue_") or "team2" in agent_id.lower() or "enemy" in agent_id.lower():
                # Ротация оппонентов
                import hashlib
                hash_val = int(hashlib.md5(str(episode).encode()).hexdigest()[:8], 16) if episode else 0
                opponent_ids = ["opponent_0", "opponent_1", "opponent_2", "opponent_3"]
                return opponent_ids[hash_val % len(opponent_ids)]
            else:
                return "main"  # Fallback
    else:
        # Общая система для non-team окружений
        def policy_mapping_fn(agent_id: str, episode=None, **kwargs):
            # Простая система: первый агент - main, остальные - оппоненты
            if agent_id == "agent_0" or "main" in agent_id.lower():
                return "main"
            else:
                return "opponent_0"
    
    return policy_mapping_fn

def main():
    """Основная функция универсальной тренировки"""
    
    # Парсинг аргументов
    parser = argparse.ArgumentParser(description="Universal Multi-Agent Training")
    parser.add_argument("--env", type=str, default="ArenaEnv", help="Environment name")
    parser.add_argument("--iterations", type=int, default=1000, help="Training iterations")
    parser.add_argument("--algo", choices=["ppo", "gspo", "grpo"], default="ppo", help="Algorithm")
    parser.add_argument("--test", action="store_true", help="Quick test mode")
    parser.add_argument("--config-file", type=str, help="Custom config file")
    parser.add_argument("--disable-recording", action="store_true", help="Disable battle recording")
    parser.add_argument("--disable-onnx", action="store_true", help="Disable ONNX export")
    parser.add_argument("--force-config", type=str, help="Force specific config (2d/3d/universal)")
    args = parser.parse_args()
    
    print(f"🚀 Starting Universal Multi-Agent Training")
    print(f"   Environment: {args.env}")
    print(f"   Algorithm: {args.algo.upper()}")
    print(f"   Iterations: {args.iterations}")
    
    # Инициализация Ray
    ray.init(ignore_reinit_error=True)
    
    try:
        # Автоматическое обнаружение окружений
        env_manager = auto_detect_and_import_environment()
        
        if args.env not in env_manager.registered_envs:
            print(f"❌ Environment '{args.env}' not found.")
            print(f"Available environments: {list(env_manager.registered_envs.keys())}")
            return 1
        
        # Регистрируем универсальные модули
        ModelCatalog.register_custom_model("universal_entity_attention", ONNXEntityAttentionModel)
        ModelCatalog.register_custom_action_dist("universal_action_dist", UniversalActionDistribution)
        ModelCatalog.register_custom_action_dist("adaptive_action_dist", create_adaptive_action_distribution)
        
        # Получение конфигурации окружения
        env_config = env_manager.env_configs.get(args.env, {})
        
        # Настройки для тестового режима
        if args.test:
            env_config.update({
                "episode_len": 50,
                "ally_choices": [1, 2] if "ally_choices" in env_config else None,
                "enemy_choices": [1, 2] if "enemy_choices" in env_config else None,
            })
            # Удаляем None значения
            env_config = {k: v for k, v in env_config.items() if v is not None}
            print("🧪 Test mode: reduced settings")
        
        # Анализ возможностей окружения
        capabilities = env_manager.auto_detect_environment(args.env, env_config)
        optimal_config = env_manager.get_optimal_config(args.env, capabilities)
        
        # Принудительная конфигурация если указана
        if args.force_config:
            if args.force_config == "3d":
                capabilities.update({"is_3d": True, "has_boundaries": True, "supports_recording": True})
            elif args.force_config == "2d":
                capabilities.update({"is_3d": False})
            print(f"🔧 Forced configuration: {args.force_config}")
        
        # Создание тестового окружения для получения пространств
        temp_env = env_manager.registered_envs[args.env](env_config)
        obs_space = temp_env.observation_space
        act_space = temp_env.action_space
        
        # Получаем размеры для League
        if hasattr(obs_space, 'spaces') and 'enemies' in obs_space.spaces:
            max_enemies = obs_space['enemies'].shape[0]
            max_allies = obs_space['allies'].shape[0] if 'allies' in obs_space.spaces else 1
        else:
            max_enemies = 6  # Fallback
            max_allies = 6
        
        print(f"🏟️ Environment: {max_allies} allies vs {max_enemies} enemies")
        
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
        
        # Универсальная конфигурация модели
        model_config = setup_universal_model_config(obs_space, act_space, capabilities)
        
        # Policy mapping
        policy_mapping_fn = create_universal_policy_mapping(capabilities)
        
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
            iterations = min(5, args.iterations)
            export_every = 2
            recording_freq = 1
            print("🧪 Test mode: reduced computational settings")
        else:
            num_workers = min(4, os.cpu_count() - 1)
            train_batch_size = 8192
            iterations = args.iterations
            export_every = 25
            recording_freq = 5
        
        # Конфигурация PPO
        config = (
            PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .environment(
                env=args.env,
                env_config=env_config
            )
            .framework("torch")
            .env_runners(
                num_env_runners=num_workers,
                num_envs_per_env_runner=1,
                rollout_fragment_length=256,
            )
            .resources(
                num_gpus=1 if torch.cuda.is_available() and not args.test else 0,
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
        
        # Создание универсальных callbacks
        callbacks = UniversalLeagueCallbacks()
        callbacks.setup(
            league_actor=league,
            opponent_ids=opponent_ids,
            eval_episodes=2 if args.test else 4,
            clone_every_iters=3 if args.test else 15,
            curriculum_schedule=[
                (0, [1], [1]),
                (2_000_000, [1, 2], [1, 2]),
                (8_000_000, [1, 2, 3], [1, 2, 3]),
            ] if not args.test and capabilities.get("has_teams") else [],
            
            # ONNX экспорт
            export_onnx=not args.disable_onnx,
            export_every=export_every,
            export_dir=optimal_config["export_dir"],
            policies_to_export=["main"],
            
            # Запись боев
            record_battles=not args.disable_recording and capabilities.get("supports_recording", False),
            recording_frequency=recording_freq,
            recordings_dir=optimal_config["recordings_dir"],
            
            # Универсальные настройки
            track_universal_metrics=True,
        )
        
        # Установка callbacks
        config.callbacks_class = UniversalLeagueCallbacks
        
        # Построение алгоритма
        print("🔧 Building universal algorithm...")
        algo = config.build()
        
        # Установка callbacks после создания
        algo.callbacks = callbacks
        
        # Инициализация весов оппонентов
        try:
            main_weights = algo.get_policy("main").get_weights()
            for pid in opponent_ids:
                algo.get_policy(pid).set_weights(main_weights)
            print("✅ Algorithm ready, starting universal training...")
        except Exception as e:
            print(f"⚠️ Could not initialize opponent weights: {e}")
        
        # Показываем активированные возможности
        print(f"🌟 Activated features:")
        for feature, enabled in capabilities.items():
            status = "✅" if enabled else "❌"
            print(f"   {status} {feature}")
        
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
                
                # Универсальные метрики
                custom = result.get("custom_metrics", {})
                
                # Логирование
                if i % 5 == 0 or args.test:
                    print(f"[{i:4d}] Reward: {reward:7.3f}, Timesteps: {timesteps:,}")
                    
                    # Показываем обнаруженные метрики
                    universal_metrics = {k: v for k, v in custom.items() if "universal" in k.lower()}
                    if universal_metrics:
                        print(f"       Universal metrics: {len(universal_metrics)} tracked")
                    
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
                    # Обработка Checkpoint объекта в Ray 2.48+
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
                        checkpoint_path = f"checkpoint_universal_iter_{i}"
                    
                    print(f"💾 Universal checkpoint saved at iteration {i}")
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
                final_checkpoint = f"final_universal_checkpoint_iter_{i + 1}"
        
        print(f"\n🏁 Universal Training completed!")
        try:
            print(f"   Final checkpoint: {os.path.basename(final_checkpoint)}")
            print(f"   Checkpoint directory: {os.path.dirname(final_checkpoint)}")
        except Exception as e:
            print(f"   Final checkpoint saved successfully")
            print(f"   Warning: Could not display path: {e}")
        print(f"   Best reward: {best_reward:.3f}")
        print(f"   Total iterations: {i + 1}")
        print(f"   Environment type: {env_manager.env_detector.environment_type}")
        
        # Показываем использованные возможности
        print(f"   Universal Features Used:")
        for feature, enabled in capabilities.items():
            if enabled:
                print(f"     ✅ {feature}")
        
        # Финальный экспорт ONNX
        if not args.disable_onnx and not args.test:
            try:
                from onnx_callbacks import export_onnx_with_meta
                exports = export_onnx_with_meta(
                    algorithm=algo,
                    iteration=i + 1,
                    export_dir=optimal_config["export_dir"],
                    policies_to_export=["main"]
                )
                if exports:
                    print(f"   ONNX exported: {len(exports)} universal models")
            except Exception as e:
                print(f"   ONNX export failed: {e}")
        
        # Финальный экспорт записей
        if not args.disable_recording and callbacks.battle_recorder and not args.test:
            try:
                web_export = callbacks.battle_recorder.export_for_web_visualizer_3d()
                if web_export:
                    print(f"   Battle visualization exported: {web_export}")
            except Exception as e:
                print(f"   Visualization export failed: {e}")
        
        # Сводка по созданным файлам
        print(f"\n📁 Generated files:")
        
        # ONNX модели
        if os.path.exists(optimal_config["export_dir"]):
            onnx_files = []
            for root, dirs, files in os.walk(optimal_config["export_dir"]):
                onnx_files.extend([f for f in files if f.endswith('.onnx')])
            if onnx_files:
                print(f"   ONNX models: {optimal_config['export_dir']}/ ({len(onnx_files)} files)")
        
        # Battle recordings
        if os.path.exists(optimal_config["recordings_dir"]):
            recording_files = []
            for root, dirs, files in os.walk(optimal_config["recordings_dir"]):
                recording_files.extend([f for f in files if f.endswith(('.json', '.html'))])
            if recording_files:
                print(f"   Battle recordings: {optimal_config['recordings_dir']}/ ({len(recording_files)} files)")
        
        try:
            print(f"   Checkpoints: {os.path.dirname(final_checkpoint)}")
        except:
            print(f"   Checkpoints: saved successfully")
        
        # Очистка
        algo.stop()
        
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        ray.shutdown()
    
    print(f"\n🎉 Universal training session completed successfully!")
    print(f"🌟 The system automatically adapted to your environment format")
    return 0

def create_custom_environment_example():
    """Создает пример кастомного окружения для демонстрации универсальности"""
    
    import gymnasium as gym
    import numpy as np
    from gymnasium import spaces
    
    class CustomMultiAgentEnv:
        """Пример кастомного многоагентного окружения с произвольными actions/obs"""
        
        def __init__(self, config):
            self.config = config
            self.num_agents = config.get("num_agents", 3)
            self.max_steps = config.get("max_steps", 100)
            self.current_step = 0
            
            # Кастомные пространства действий и наблюдений
            self.observation_space = spaces.Dict({
                "sensor_data": spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32),
                "nearby_agents": spaces.Box(low=-5, high=5, shape=(4, 6), dtype=np.float32),
                "agent_mask": spaces.MultiBinary(4),
                "global_info": spaces.Box(low=0, high=10, shape=(12,), dtype=np.float32),
            })
            
            self.action_space = spaces.Dict({
                "movement": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                "tool_selection": spaces.Discrete(5),
                "activate": spaces.Discrete(2),
                "communication": spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32),
            })
            
            self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        
        def reset(self, **kwargs):
            self.current_step = 0
            obs = {}
            for agent_id in self.agents:
                obs[agent_id] = {
                    "sensor_data": np.random.randn(8).astype(np.float32),
                    "nearby_agents": np.random.randn(4, 6).astype(np.float32),
                    "agent_mask": np.random.randint(0, 2, 4),
                    "global_info": np.random.uniform(0, 10, 12).astype(np.float32),
                }
            return obs, {}
        
        def step(self, action_dict):
            self.current_step += 1
            
            obs = {}
            rewards = {}
            terms = {}
            truncs = {}
            infos = {}
            
            for agent_id in self.agents:
                # Генерируем новые наблюдения
                obs[agent_id] = {
                    "sensor_data": np.random.randn(8).astype(np.float32),
                    "nearby_agents": np.random.randn(4, 6).astype(np.float32),
                    "agent_mask": np.random.randint(0, 2, 4),
                    "global_info": np.random.uniform(0, 10, 12).astype(np.float32),
                }
                
                # Простые награды на основе действий
                action = action_dict.get(agent_id, {})
                reward = 0.0
                
                if isinstance(action, dict):
                    # Награда за движение
                    if "movement" in action:
                        movement = np.array(action["movement"])
                        reward += -0.1 * np.linalg.norm(movement)  # Штраф за движение
                    
                    # Награда за использование инструментов
                    if "tool_selection" in action and action["tool_selection"] == 2:
                        reward += 0.5  # Бонус за правильный инструмент
                    
                    # Награда за коммуникацию
                    if "communication" in action:
                        comm = np.array(action["communication"])
                        reward += 0.1 * np.sum(comm)
                
                rewards[agent_id] = reward
                terms[agent_id] = False
                truncs[agent_id] = False
                
                # Кастомные метрики
                infos[agent_id] = {
                    "custom_efficiency": np.random.uniform(0.5, 1.0),
                    "tool_usage_count": np.random.randint(0, 3),
                    "communication_volume": np.random.uniform(0, 1),
                    "position": [np.random.uniform(-5, 5), np.random.uniform(-5, 5)],
                }
            
            # Условие окончания
            done = self.current_step >= self.max_steps
            terms["__all__"] = done
            truncs["__all__"] = done
            
            return obs, rewards, terms, truncs, infos
    
    return CustomMultiAgentEnv

def quick_test_universal_system():
    """Быстрый тест универсальной системы"""
    
    print("🧪 Running quick universal system test...")
    
    # Тест 1: Создание кастомного окружения
    try:
        CustomEnv = create_custom_environment_example()
        custom_env = CustomEnv({"num_agents": 2, "max_steps": 5})
        obs, _ = custom_env.reset()
        
        print("✅ Custom environment creation successful")
        print(f"   Agents: {list(obs.keys())}")
        print(f"   Obs structure: {list(obs[list(obs.keys())[0]].keys())}")
        
        # Тест действий
        actions = {}
        for agent_id in obs.keys():
            actions[agent_id] = custom_env.action_space.sample()
        
        obs, rewards, terms, truncs, infos = custom_env.step(actions)
        print(f"   Step successful, rewards: {list(rewards.values())}")
        
    except Exception as e:
        print(f"❌ Custom environment test failed: {e}")
        return False
    
    # Тест 2: Анализ совместимости
    try:
        sample_actions = actions
        capabilities, recommendations = analyze_environment_compatibility(
            obs, sample_actions, {"num_agents": 2}
        )
        
        print("✅ Environment analysis successful")
        print(f"   Detected capabilities: {capabilities}")
        print(f"   Recommendations: {len(recommendations)}")
        
    except Exception as e:
        print(f"❌ Environment analysis failed: {e}")
        return False
    
    # Тест 3: Конфигурация модели
    try:
        action_config = DynamicActionConfig(custom_env.action_space)
        obs_processor = DynamicObservationProcessor(custom_env.observation_space)
        
        print("✅ Dynamic configuration successful")
        print(f"   Action spec: {action_config.action_spec}")
        print(f"   Obs spec keys: {list(obs_processor.obs_spec.keys())}")
        
    except Exception as e:
        print(f"❌ Dynamic configuration failed: {e}")
        return False
    
    print("✅ Quick universal system test passed - ready for any environment!")
    return True

def demonstrate_flexibility():
    """Демонстрирует гибкость системы с различными форматами"""
    
    print("🎨 Demonstrating Universal System Flexibility...")
    
    # Пример 1: 2D Tactical Environment
    print("\n1️⃣ 2D Tactical Environment:")
    tactical_2d_obs = {
        "red_0": {
            "self": np.array([1.0, 2.0, 0.8, 0.0, 0.0]),  # x, y, hp, ammo, shield
            "allies": np.zeros((2, 4)),  # 2 allies, 4 features each
            "allies_mask": np.array([1, 0]),
            "enemies": np.random.randn(3, 5),  # 3 enemies, 5 features each
            "enemies_mask": np.array([1, 1, 0]),
            "enemy_action_mask": np.array([1, 1, 0]),
        }
    }
    
    tactical_2d_actions = {
        "red_0": {
            "target": 0,
            "move": [0.1, -0.2],  # 2D movement
            "aim": [-0.1, 0.3],   # 2D aiming
            "fire": 1
        }
    }
    
    caps_2d, _ = analyze_environment_compatibility(tactical_2d_obs, tactical_2d_actions)
    print(f"   Detected: {caps_2d['observation_format']}, 3D: {caps_2d['is_3d']}, Teams: {caps_2d['has_teams']}")
    
    # Пример 2: 3D Tactical Environment
    print("\n2️⃣ 3D Tactical Environment:")
    tactical_3d_obs = {
        "blue_1": {
            "self": np.array([1.0, 2.0, 1.5, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # 3D + extras
            "allies": np.zeros((3, 9)),  # 3D features
            "allies_mask": np.array([1, 1, 0]),
            "enemies": np.random.randn(4, 11),  # 3D + combat features
            "enemies_mask": np.array([1, 1, 1, 0]),
            "global_state": np.random.randn(64),
        }
    }
    
    tactical_3d_actions = {
        "blue_1": {
            "target": 1,
            "move": [0.1, -0.2, 0.3],   # 3D movement
            "aim": [-0.1, 0.3, -0.1],   # 3D aiming
            "fire": 0
        }
    }
    
    caps_3d, _ = analyze_environment_compatibility(tactical_3d_obs, tactical_3d_actions)
    print(f"   Detected: {caps_3d['observation_format']}, 3D: {caps_3d['is_3d']}, Teams: {caps_3d['has_teams']}")
    
    # Пример 3: Произвольное окружение
    print("\n3️⃣ Custom Robotics Environment:")
    robotics_obs = {
        "robot_alpha": {
            "joint_positions": np.random.randn(7),
            "camera_feed": np.random.randn(64, 64, 3).flatten()[:100],  # Урезанно
            "force_sensors": np.random.randn(6),
            "task_progress": np.array([0.3, 0.7, 0.1]),
        }
    }
    
    robotics_actions = {
        "robot_alpha": {
            "joint_velocities": np.random.uniform(-1, 1, 7),
            "gripper": 1,
            "mode_switch": 2,
        }
    }
    
    caps_robotics, _ = analyze_environment_compatibility(robotics_obs, robotics_actions)
    print(f"   Detected: {caps_robotics['observation_format']}, 3D: {caps_robotics['is_3d']}, Teams: {caps_robotics['has_teams']}")
    
    print("\n✨ All environment formats analyzed successfully!")
    print("🚀 Universal system can adapt to any obs/action structure!")

if __name__ == "__main__":
    # Обработка специальных команд
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick-test":
            success = quick_test_universal_system()
            sys.exit(0 if success else 1)
        elif sys.argv[1] == "--demo-flexibility":
            demonstrate_flexibility()
            sys.exit(0)
        elif sys.argv[1] == "--help-examples":
            print("""
🎮 Universal Multi-Agent Training Examples:

1. Quick test with auto-detection:
   python train.py --test

2. Custom environment training:
   python train.py --env CustomEnv --iterations 100

3. Force 3D mode:
   python train.py --force-config 3d --iterations 200

4. GSPO algorithm with custom config:
   python train.py --algo gspo --config-file custom.json

5. Training with minimal logging:
   python train.py --disable-recording --disable-onnx

6. Test system flexibility:
   python train.py --quick-test

7. Demonstrate adaptability:
   python train.py --demo-flexibility

📁 Universal Output locations:
   - Checkpoints: ./rllib_universal_results/
   - ONNX models: ./onnx_exports_[2d/3d]/
   - Battle recordings: ./battle_recordings_[2d/3d]/
   - Logs: ./logs/

🔍 Monitor with TensorBoard:
   tensorboard --logdir ./logs

🌟 Universal Features:
   - Automatic adaptation to any obs/action format
   - Dynamic model configuration
   - Environment capability detection
   - Flexible ONNX export
   - Universal battle recording
   - Cross-environment compatibility

🎯 Supported Environment Types:
   - Arena-style tactical combat (2D/3D)
   - Robotics simulations
   - Custom multi-agent systems
   - PettingZoo environments
   - Any dict-based obs/action format
            """)
            sys.exit(0)
    
    # Обычный запуск
    sys.exit(main())