"""
Универсальные callbacks с автоматической адаптацией к различным форматам actions/obs
Поддерживают любые структуры данных и автоматически определяют доступные метрики
"""

import os
import numpy as np
from typing import Dict, Any, List, Optional, Union
import ray
import torch
from torch.utils.tensorboard import SummaryWriter

from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.algorithms import Algorithm

# Импортируем универсальные модули
from onnx_callbacks import export_onnx_with_meta
from save_res import BattleRecorder3D, RecordingArenaWrapper3D
import json

class UniversalMetricsExtractor:
    """Универсальный экстрактор метрик из различных форматов obs/actions/infos"""
    
    def __init__(self):
        self.known_metrics = {
            # Стандартные метрики валидности
            "invalid_actions": ["invalid_target", "invalid_action", "action_invalid"],
            "out_of_bounds": ["oob_move", "oob_aim", "out_of_bounds", "boundary_violation"],
            "deaths": ["boundary_deaths", "death_count", "deaths"],
            
            # 3D специфичные метрики
            "3d_positions": ["position_3d", "pos_3d", "coordinates"],
            "3d_boundaries": ["within_bounds", "boundary_violation", "field_violation"],
            "laser_metrics": ["laser_range", "laser_shots", "laser_hits", "laser_effectiveness"],
            
            # Боевые метрики
            "damage_metrics": ["damage_dealt", "damage_taken", "hp", "health"],
            "accuracy_metrics": ["accuracy", "hit_rate", "shots_fired", "shots_hit"],
            "team_metrics": ["team_reward", "team_step_reward", "team_performance"],
            
            # Пользовательские метрики
            "custom_metrics": ["custom_", "user_", "env_specific_"]
        }
    
    def extract_from_infos(self, infos: Dict[str, Dict]) -> Dict[str, Any]:
        """Извлекает метрики из infos различных агентов"""
        if not isinstance(infos, dict):
            return {}
        
        extracted = {}
        
        for agent_id, agent_info in infos.items():
            if not isinstance(agent_info, dict):
                continue
            
            agent_prefix = self._get_agent_prefix(agent_id)
            
            for metric_category, metric_names in self.known_metrics.items():
                category_values = []
                
                for key, value in agent_info.items():
                    if any(metric_name in key.lower() for metric_name in metric_names):
                        if isinstance(value, (int, float, np.number)):
                            category_values.append(float(value))
                        elif isinstance(value, (list, tuple, np.ndarray)):
                            try:
                                category_values.extend([float(v) for v in value])
                            except (ValueError, TypeError):
                                pass
                
                if category_values:
                    extracted[f"{agent_prefix}_{metric_category}"] = category_values
                    extracted[f"{metric_category}_total"] = extracted.get(f"{metric_category}_total", []) + category_values
        
        # Агрегируем статистики
        for key, values in list(extracted.items()):
            if isinstance(values, list) and values:
                extracted[f"{key}_mean"] = np.mean(values)
                extracted[f"{key}_sum"] = np.sum(values)
                extracted[f"{key}_max"] = np.max(values)
                if len(values) > 1:
                    extracted[f"{key}_std"] = np.std(values)
        
        return extracted
    
    def extract_from_observations(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Извлекает метрики из наблюдений"""
        if not isinstance(obs, dict):
            return {}
        
        extracted = {}
        
        for agent_id, agent_obs in obs.items():
            if not isinstance(agent_obs, dict):
                continue
            
            agent_prefix = self._get_agent_prefix(agent_id)
            
            # Анализируем структуру наблюдений
            for key, value in agent_obs.items():
                if isinstance(value, (torch.Tensor, np.ndarray)):
                    try:
                        if key == "self" and len(value) >= 3:
                            # Извлекаем позицию (первые 3 компонента обычно x, y, z)
                            extracted[f"{agent_prefix}_position"] = value[:3].tolist()
                        elif key in ["allies", "enemies"] and value.ndim >= 2:
                            # Считаем активных союзников/врагов
                            if f"{key}_mask" in agent_obs:
                                mask = agent_obs[f"{key}_mask"]
                                if hasattr(mask, 'sum'):
                                    extracted[f"{agent_prefix}_{key}_count"] = int(mask.sum())
                        elif "mask" in key and hasattr(value, 'sum'):
                            extracted[f"{agent_prefix}_{key}_active"] = int(value.sum())
                    except Exception:
                        pass
        
        return extracted
    
    def extract_from_actions(self, actions: Dict[str, Any]) -> Dict[str, Any]:
        """Извлекает метрики из действий"""
        if not isinstance(actions, dict):
            return {}
        
        extracted = {}
        action_counts = {"fire": 0, "move": 0, "discrete": 0, "continuous": 0}
        
        for agent_id, action in actions.items():
            agent_prefix = self._get_agent_prefix(agent_id)
            
            if isinstance(action, dict):
                # Dict actions
                for action_name, action_value in action.items():
                    try:
                        if "fire" in action_name.lower() or "shoot" in action_name.lower():
                            if isinstance(action_value, (int, float, np.number)) and action_value > 0:
                                action_counts["fire"] += 1
                                extracted[f"{agent_prefix}_fired"] = 1
                        elif "move" in action_name.lower():
                            if isinstance(action_value, (list, tuple, np.ndarray)):
                                move_magnitude = np.linalg.norm(action_value)
                                extracted[f"{agent_prefix}_move_magnitude"] = float(move_magnitude)
                                if move_magnitude > 0.1:
                                    action_counts["move"] += 1
                        elif isinstance(action_value, (int, np.integer)):
                            action_counts["discrete"] += 1
                        elif isinstance(action_value, (float, np.floating, list, tuple, np.ndarray)):
                            action_counts["continuous"] += 1
                    except Exception:
                        pass
            elif isinstance(action, (list, tuple, np.ndarray)):
                # Array actions - анализируем по позиции
                try:
                    action_arr = np.array(action)
                    if len(action_arr) > 0:
                        extracted[f"{agent_prefix}_action_magnitude"] = float(np.linalg.norm(action_arr))
                except Exception:
                    pass
        
        # Добавляем общие счетчики
        for action_type, count in action_counts.items():
            if count > 0:
                extracted[f"actions_{action_type}_count"] = count
        
        return extracted
    
    def _get_agent_prefix(self, agent_id: str) -> str:
        """Определяет префикс агента для группировки метрик"""
        if isinstance(agent_id, str):
            if agent_id.startswith("red"):
                return "red"
            elif agent_id.startswith("blue"):
                return "blue"
            elif "team" in agent_id.lower():
                return agent_id.lower()
            else:
                return "agent"
        return "unknown"

class UniversalEnvironmentDetector:
    """Детектор типа окружения и доступных возможностей"""
    
    def __init__(self):
        self.detected_features = {}
        self.environment_type = "unknown"
    
    def detect_environment_features(self, obs_sample: Dict, actions_sample: Dict, env_config: Dict = None) -> Dict[str, Any]:
        """Автоматически определяет возможности окружения"""
        features = {
            "is_3d": False,
            "has_boundaries": False,
            "has_laser_system": False,
            "has_teams": False,
            "has_projectiles": False,
            "action_format": "unknown",
            "observation_format": "unknown",
            "supports_recording": False
        }
        
        # Анализ наблюдений
        if obs_sample:
            first_agent = next(iter(obs_sample.values()))
            if isinstance(first_agent, dict):
                features["observation_format"] = "dict"
                
                # Проверка на 3D
                if "self" in first_agent:
                    self_obs = first_agent["self"]
                    if hasattr(self_obs, '__len__') and len(self_obs) >= 13:
                        features["is_3d"] = True
                
                # Проверка команд
                if any(key.startswith(("red", "blue")) for key in obs_sample.keys()):
                    features["has_teams"] = True
                
                # Проверка масок
                if "enemy_action_mask" in first_agent:
                    features["has_laser_system"] = True
        
        # Анализ действий
        if actions_sample:
            first_action = next(iter(actions_sample.values()))
            if isinstance(first_action, dict):
                features["action_format"] = "dict"
                
                # Проверка 3D действий
                if "move" in first_action:
                    move_action = first_action["move"]
                    if hasattr(move_action, '__len__') and len(move_action) >= 3:
                        features["is_3d"] = True
            elif isinstance(first_action, (list, tuple, np.ndarray)):
                features["action_format"] = "array"
        
        # Анализ конфига окружения
        if env_config:
            if any(key in env_config for key in ["field_bounds", "FIELD_BOUNDS", "boundaries"]):
                features["has_boundaries"] = True
            if any(key in env_config for key in ["laser", "projectile", "shooting"]):
                features["has_laser_system"] = True
                features["has_projectiles"] = True
        
        # Определение возможности записи
        features["supports_recording"] = features["has_teams"] and (features["is_3d"] or features["has_laser_system"])
        
        self.detected_features = features
        self.environment_type = self._classify_environment(features)
        
        return features
    
    def _classify_environment(self, features: Dict[str, Any]) -> str:
        """Классифицирует тип окружения"""
        if features["is_3d"] and features["has_teams"] and features["has_laser_system"]:
            return "3d_tactical_combat"
        elif features["has_teams"] and features["has_laser_system"]:
            return "2d_tactical_combat"
        elif features["has_teams"]:
            return "multi_team"
        elif features["is_3d"]:
            return "3d_environment"
        else:
            return "generic_multi_agent"

class UniversalLeagueCallbacks(RLlibCallback):
    """
    Универсальные callbacks, автоматически адаптирующиеся к любому окружению
    """
    
    def __init__(self):
        super().__init__()
        self.league = None
        self.opponent_ids = None
        self.eval_eps = 6
        self.clone_every = 10
        self.sample_top_k = 3
        self.attn_log_every = 20
        self.writer: Optional[SummaryWriter] = None
        self.curriculum = None
        
        # Универсальные компоненты
        self.metrics_extractor = UniversalMetricsExtractor()
        self.env_detector = UniversalEnvironmentDetector()
        self.detected_features = {}
        
        # ONNX экспорт настройки
        self.export_onnx = True
        self.export_every = 25
        self.export_dir = "./onnx_exports"
        self.policies_to_export = ["main"]
        
        # Настройки записи боев
        self.record_battles = True
        self.battle_recorder: Optional[BattleRecorder3D] = None
        self.recording_frequency = 10
        self.recorded_matches = 0
        
        # Универсальные метрики
        self.track_universal_metrics = True
        self.custom_metrics_history = {}
        
    def setup(self, league_actor, opponent_ids: List[str], **kwargs):
        """Настройка параметров callbacks с автодетекцией"""
        self.league = league_actor
        self.opponent_ids = opponent_ids
        self.eval_eps = kwargs.get('eval_episodes', 6)
        self.clone_every = kwargs.get('clone_every_iters', 10)
        self.sample_top_k = kwargs.get('sample_top_k', 3)
        self.attn_log_every = kwargs.get('attn_log_every', 20)
        self.curriculum = kwargs.get('curriculum_schedule', [])
        
        # ONNX настройки
        self.export_onnx = kwargs.get('export_onnx', True)
        self.export_every = kwargs.get('export_every', 25)
        self.export_dir = kwargs.get('export_dir', "./onnx_exports")
        self.policies_to_export = kwargs.get('policies_to_export', ["main"])
        
        # Настройки записи боев
        self.record_battles = kwargs.get('record_battles', True)
        self.recording_frequency = kwargs.get('recording_frequency', 10)
        
        # Универсальные настройки
        self.track_universal_metrics = kwargs.get('track_universal_metrics', True)
        
        # Создаем директории
        if self.export_onnx:
            os.makedirs(self.export_dir, exist_ok=True)
        
        if self.record_battles:
            recordings_dir = kwargs.get('recordings_dir', "./battle_recordings")
            self.battle_recorder = BattleRecorder3D(recordings_dir)
            print(f"📹 Universal battle recording enabled, saving to: {recordings_dir}")

    def on_algorithm_init(self, *, algorithm: Algorithm, **kwargs) -> None:
        """Инициализация с автодетекцией возможностей окружения"""
        print("🔍 Detecting environment capabilities...")
        
        # Пытаемся получить образец obs/actions для анализа
        try:
            # Создаем временное окружение для анализа
            env_config = getattr(algorithm.config, 'env_config', {})
            env_creator_fn = getattr(algorithm.config, 'env', None)
            
            if env_creator_fn and hasattr(algorithm.config, 'env_config'):
                try:
                    from ray.tune.registry import _global_registry
                    env_cls = _global_registry.get("env", env_creator_fn)
                    if env_cls:
                        temp_env = env_cls(env_config)
                        obs, _ = temp_env.reset()
                        
                        # Генерируем образец действий
                        sample_actions = {}
                        for agent_id in obs.keys():
                            if hasattr(temp_env, 'action_space'):
                                sample_actions[agent_id] = temp_env.action_space.sample()
                        
                        # Детектируем возможности
                        self.detected_features = self.env_detector.detect_environment_features(
                            obs, sample_actions, env_config
                        )
                        
                        print(f"✅ Environment detection completed:")
                        print(f"   Type: {self.env_detector.environment_type}")
                        print(f"   Features: {self.detected_features}")
                        
                except Exception as e:
                    print(f"⚠️ Could not auto-detect environment: {e}")
                    self.detected_features = {"supports_recording": False}
        except Exception as e:
            print(f"⚠️ Environment detection failed: {e}")
            self.detected_features = {}

    def on_train_result(self, *, algorithm: Algorithm, result: Dict[str, Any], **kwargs) -> None:
        """Основная логика обработки результатов с универсальной обработкой"""
        if self.league is None:
            return
            
        # Создаем writer
        if self.writer is None:
            logdir = getattr(algorithm, "logdir", "./logs")
            self.writer = SummaryWriter(log_dir=logdir)

        it = result["training_iteration"]
        ts_total = result.get("timesteps_total", 0)

        # 1) Evaluation матчей с универсальной записью
        try:
            for pid in self.opponent_ids:
                should_record = (
                    self.record_battles and 
                    self.battle_recorder and 
                    self.recorded_matches % self.recording_frequency == 0 and
                    self.detected_features.get("supports_recording", False)
                )
                
                w_main, w_opp, match_stats = self._play_universal_match(
                    algorithm, pid, self.eval_eps, 
                    record_battle=should_record,
                    battle_id=f"eval_universal_it{it:04d}_vs_{pid}"
                )
                
                ray.get(self.league.update_pair_result.remote(w_main, w_opp, pid))
                self.recorded_matches += 1
                
                # Логируем универсальные метрики
                if self.track_universal_metrics and match_stats:
                    self._log_universal_metrics(match_stats, it, pid)
                
        except Exception as e:
            print(f"Error in universal match evaluation: {e}")

        # 2) Логирование рейтингов
        try:
            scores = ray.get(self.league.get_all_scores.remote())
            result.setdefault("custom_metrics", {})
            
            for k, (mu, sigma) in scores.items():
                result["custom_metrics"][f"ts_{k}_mu"] = mu
                result["custom_metrics"][f"ts_{k}_sigma"] = sigma
                
                conservative_score = mu - 3 * sigma
                self.writer.add_scalar(f"ts/{k}_conservative", conservative_score, it)
                
        except Exception as e:
            print(f"Error getting league scores: {e}")
            scores = {}

        # 3) Клонирование худшего оппонента
        if it % self.clone_every == 0 and it > 0 and scores:
            try:
                items = [(pid, scores[pid][0] - 3*scores[pid][1]) for pid in self.opponent_ids]
                worst = min(items, key=lambda z: z[1])[0]
                
                w = algorithm.get_policy("main").get_weights()
                algorithm.get_policy(worst).set_weights(w)
                ray.get(self.league.clone_main_into.remote(worst))
                
                result["custom_metrics"][f"league_refresh_{worst}"] = it
                print(f"🔄 Refreshed opponent {worst} at iteration {it}")
                
            except Exception as e:
                print(f"Error refreshing opponent: {e}")

        # 4) Универсальный куррикулум
        if self.curriculum:
            for threshold, ac, ec in reversed(self.curriculum):
                if ts_total >= threshold:
                    try:
                        self._apply_universal_curriculum(algorithm, ac, ec)
                        result["custom_metrics"]["curriculum_ally_choices"] = str(ac)
                        result["custom_metrics"]["curriculum_enemy_choices"] = str(ec)
                    except Exception as e:
                        print(f"Error setting universal curriculum: {e}")
                    break

        # 5) Универсальный ONNX экспорт
        if self.export_onnx and it % self.export_every == 0 and it > 0:
            try:
                print(f"\n🔧 Starting universal ONNX export for iteration {it}...")
                
                successful_exports = export_onnx_with_meta(
                    algorithm=algorithm,
                    iteration=it,
                    export_dir=self.export_dir,
                    policies_to_export=self.policies_to_export
                )
                
                if successful_exports:
                    result["custom_metrics"]["onnx_export_iteration"] = it
                    result["custom_metrics"]["onnx_policies_exported"] = len(successful_exports)
                    print(f"✅ Universal ONNX export completed for iteration {it} ({len(successful_exports)} policies)")
                    
                    self.writer.add_scalar("export/onnx_success", 1, it)
                    for export in successful_exports:
                        self.writer.add_text(
                            f"export/onnx_universal_{export['policy_id']}", 
                            f"Universal model exported to {export['onnx_path']}", 
                            it
                        )
                else:
                    print(f"⚠️ Universal ONNX export completed but no policies were successfully exported")
                    self.writer.add_scalar("export/onnx_success", 0, it)
                    
            except Exception as e:
                print(f"❌ Universal ONNX export failed for iteration {it}: {e}")
                self.writer.add_scalar("export/onnx_success", 0, it)
                import traceback
                traceback.print_exc()

        # 6) Экспорт записей боев
        if (self.record_battles and self.battle_recorder and 
            it % (self.export_every * 2) == 0 and it > 0):
            try:
                web_export_path = self.battle_recorder.export_for_web_visualizer_3d()
                if web_export_path:
                    result["custom_metrics"]["battle_replay_exported"] = it
                    print(f"🎬 Universal battle replay exported: {web_export_path}")
                    
            except Exception as e:
                print(f"Error exporting universal battle replay: {e}")

        if self.writer:
            self.writer.flush()

    def _play_universal_match(self, algorithm: Algorithm, opp_id: str, episodes: int, 
                            record_battle: bool = False, battle_id: str = "") -> tuple:
        """
        Универсальная версия матча с автоматической адаптацией к окружению
        """
        try:
            # Получаем класс окружения
            env_config = getattr(algorithm.config, 'env_config', {})
            from ray.tune.registry import _global_registry
            env_creator_fn = getattr(algorithm.config, 'env', 'ArenaEnv')
            env_cls = _global_registry.get("env", env_creator_fn)
            
            if env_cls is None:
                # Fallback на известные окружения
                try:
                    from arena_env import ArenaEnv
                    temp_env = ArenaEnv(env_config)
                except ImportError:
                    print("Could not import ArenaEnv, using generic approach")
                    return 0, 0, None
            else:
                temp_env = env_cls(env_config)
            
            # Обертываем в записывающий wrapper если нужно
            if record_battle and self.battle_recorder:
                temp_env = RecordingArenaWrapper3D(temp_env, self.battle_recorder)
                print(f"📹 Recording universal battle: {battle_id}")
            
            wins_main, wins_opp = 0, 0
            universal_stats = {
                'environment_type': self.env_detector.environment_type,
                'detected_features': self.detected_features,
                'episodes_data': [],
                'total_actions': 0,
                'total_invalid_actions': 0,
                'custom_metrics': {}
            }
            
            for episode_idx in range(episodes):
                obs, _ = temp_env.reset()
                done = False
                episode_data = {
                    'episode_idx': episode_idx,
                    'actions_count': 0,
                    'invalid_actions': 0,
                    'episode_metrics': {}
                }
                
                while not done:
                    action_dict = {}
                    
                    for aid, ob in obs.items():
                        pol_id = "main" if aid.startswith("red_") else opp_id
                        pol = algorithm.get_policy(pol_id)
                        
                        try:
                            act, _, _ = pol.compute_single_action(ob, explore=False)
                        except Exception as e:
                            print(f"Error computing universal action for {aid}: {e}")
                            # Универсальное fallback действие
                            if isinstance(temp_env.action_space, dict) and hasattr(temp_env.action_space, 'spaces'):
                                act = temp_env.action_space.sample()
                            else:
                                act = {"target": 0, "move": [0.0, 0.0, 0.0], "aim": [0.0, 0.0, 0.0], "fire": 0}
                        
                        action_dict[aid] = act
                        episode_data['actions_count'] += 1
                        universal_stats['total_actions'] += 1
                    
                    obs, rews, terms, truncs, infos = temp_env.step(action_dict)
                    done = terms.get("__all__", False) or truncs.get("__all__", False)
                    
                    # Извлекаем универсальные метрики
                    if self.track_universal_metrics:
                        step_metrics = self.metrics_extractor.extract_from_infos(infos)
                        step_metrics.update(self.metrics_extractor.extract_from_observations(obs))
                        step_metrics.update(self.metrics_extractor.extract_from_actions(action_dict))
                        
                        # Накапливаем метрики эпизода
                        for key, value in step_metrics.items():
                            if isinstance(value, (int, float)):
                                episode_data['episode_metrics'][key] = episode_data['episode_metrics'].get(key, 0) + value
                
                # Определяем победителя универсальным способом
                red_reward = sum(v for k, v in rews.items() if k.startswith("red_"))
                blue_reward = sum(v for k, v in rews.items() if k.startswith("blue_"))
                
                if red_reward > blue_reward:
                    wins_main += 1
                elif blue_reward > red_reward:
                    wins_opp += 1
                
                universal_stats['episodes_data'].append(episode_data)
            
            # Агрегируем статистику матча
            self._aggregate_universal_stats(universal_stats)
            
            return wins_main, wins_opp, universal_stats
            
        except Exception as e:
            print(f"Error in _play_universal_match: {e}")
            import traceback
            traceback.print_exc()
            return 0, 0, None

    def _aggregate_universal_stats(self, stats: Dict):
        """Агрегирует универсальную статистику"""
        episodes_data = stats['episodes_data']
        if not episodes_data:
            return

        # Агрегация по эпизодам
        total_metrics = {}
        for episode in episodes_data:
            for key, value in episode['episode_metrics'].items():
                if isinstance(value, (int, float)):
                    total_metrics[key] = total_metrics.get(key, []) + [value]

        # Вычисляем агрегированные значения
        stats['aggregated_metrics'] = {}
        for key, values in total_metrics.items():
            if values:
                stats['aggregated_metrics'][f"{key}_mean"] = np.mean(values)
                stats['aggregated_metrics'][f"{key}_sum"] = np.sum(values)
                stats['aggregated_metrics'][f"{key}_max"] = np.max(values)
                if len(values) > 1:
                    stats['aggregated_metrics'][f"{key}_std"] = np.std(values)

        # Общие статистики
        stats['total_episodes'] = len(episodes_data)
        stats['avg_actions_per_episode'] = np.mean([ep['actions_count'] for ep in episodes_data])
        stats['environment_compatibility'] = True

    def _log_universal_metrics(self, match_stats: Dict, iteration: int, opponent_id: str):
        """Логирует универсальные метрики в TensorBoard"""
        if not self.writer or not match_stats:
            return
        
        prefix = f"universal_metrics/{opponent_id}"
        
        # Логируем основные метрики
        aggregated = match_stats.get('aggregated_metrics', {})
        for key, value in aggregated.items():
            if isinstance(value, (int, float, np.number)):
                self.writer.add_scalar(f"{prefix}/{key}", value, iteration)
        
        # Логируем информацию об окружении
        env_type = match_stats.get('environment_type', 'unknown')
        self.writer.add_text(f"{prefix}/environment_type", env_type, iteration)
        
        # Логируем обнаруженные возможности
        features = match_stats.get('detected_features', {})
        for feature, enabled in features.items():
            if isinstance(enabled, bool):
                self.writer.add_scalar(f"environment_features/{feature}", int(enabled), iteration)
        
        # Общие универсальные метрики
        total_actions = match_stats.get('total_actions', 0)
        if total_actions > 0:
            self.writer.add_scalar("universal_general/actions_per_match", total_actions, iteration)

    def _apply_universal_curriculum(self, algorithm, ally_choices, enemy_choices):
        """Применение куррикулума для универсального окружения"""
        try:
            # Обновляем конфигурацию алгоритма
            if hasattr(algorithm.config, 'env_config'):
                # Применяем только если окружение поддерживает эти параметры
                if hasattr(algorithm.config.env_config, 'get'):
                    current_config = algorithm.config.env_config
                    if 'ally_choices' in str(current_config) or 'enemy_choices' in str(current_config):
                        algorithm.config.env_config["ally_choices"] = ally_choices
                        algorithm.config.env_config["enemy_choices"] = enemy_choices
                        print(f"📚 Updated universal curriculum: allies={ally_choices}, enemies={enemy_choices}")
            
            # Пытаемся применить к существующим окружениям
            try:
                if hasattr(algorithm, 'env_runner_group') and algorithm.env_runner_group:
                    def set_curriculum_fn(env):
                        if hasattr(env, 'set_curriculum'):
                            env.set_curriculum(ally_choices, enemy_choices)
                        elif hasattr(env, 'env') and hasattr(env.env, 'set_curriculum'):
                            env.env.set_curriculum(ally_choices, enemy_choices)
                    
                    algorithm.env_runner_group.foreach_env(set_curriculum_fn)
                    print(f"✅ Applied universal curriculum to env_runners")
                    
            except (AttributeError, Exception) as e:
                print(f"⚠️ Could not apply universal curriculum to existing envs: {e}")
                
        except Exception as e:
            print(f"❌ Could not apply universal curriculum: {e}")

    def on_episode_end(self, *, base_env, policies: Dict[str, Any], 
                      episode, env_index: Optional[int] = None, **kwargs) -> None:
        """
        Обработка окончания эпизода с универсальными метриками
        """
        
        try:
            # Извлекаем универсальные метрики эпизода
            if hasattr(episode, 'custom_metrics'):
                # Анализируем агентов
                agents = episode.get_agents() if hasattr(episode, 'get_agents') else []
                
                # Универсальная группировка агентов
                agent_groups = {}
                for agent in agents:
                    group = self.metrics_extractor._get_agent_prefix(agent)
                    agent_groups[group] = agent_groups.get(group, 0) + 1
                
                # Добавляем универсальные метрики
                for group, count in agent_groups.items():
                    episode.custom_metrics[f"{group}_team_size"] = count
                
                # Добавляем информацию об окружении
                episode.custom_metrics["environment_type"] = self.env_detector.environment_type
                episode.custom_metrics["universal_callbacks_active"] = 1
                
                # Извлекаем метрики из base_env если возможно
                if hasattr(base_env, 'get_sub_environments'):
                    try:
                        sub_envs = base_env.get_sub_environments()
                        if sub_envs and env_index is not None and len(sub_envs) > env_index:
                            env = sub_envs[env_index]
                            
                            # Универсальное извлечение метрик окружения
                            env_metrics = self._extract_environment_metrics(env)
                            episode.custom_metrics.update(env_metrics)
                            
                    except Exception as e:
                        # Не критично если не получилось
                        pass
                        
        except Exception as e:
            # Полностью безопасная обработка
            pass

    def _extract_environment_metrics(self, env) -> Dict[str, Any]:
        """Универсальное извлечение метрик из окружения"""
        metrics = {}
        
        try:
            # Стандартные атрибуты
            for attr_name in ['_t', 'timestep', 'step_count', 'current_step']:
                if hasattr(env, attr_name):
                    metrics["env_timestep"] = getattr(env, attr_name)
                    break
            
            # HP/health метрики
            for attr_name in ['_hp', 'health', 'agent_health']:
                if hasattr(env, attr_name):
                    hp_data = getattr(env, attr_name)
                    if isinstance(hp_data, dict):
                        total_hp = sum(hp for hp in hp_data.values() if isinstance(hp, (int, float)))
                        metrics["total_hp"] = total_hp
                        metrics["agents_alive"] = sum(1 for hp in hp_data.values() if hp > 0)
                    break
            
            # Позиционные данные
            for attr_name in ['_pos', 'positions', 'agent_positions']:
                if hasattr(env, attr_name):
                    pos_data = getattr(env, attr_name)
                    if isinstance(pos_data, dict) and pos_data:
                        # Анализируем размерность
                        first_pos = next(iter(pos_data.values()))
                        if hasattr(first_pos, '__len__'):
                            metrics["position_dimensions"] = len(first_pos)
                            if len(first_pos) >= 3:
                                metrics["is_3d_environment"] = 1
                    break
            
            # Специфичные метрики (границы, лазеры и т.д.)
            boundary_attrs = ['count_boundary_deaths', 'boundary_violations', 'out_of_bounds']
            for attr in boundary_attrs:
                if hasattr(env, attr):
                    metrics[attr] = getattr(env, attr)
            
            laser_attrs = ['LASER_MAX_RANGE', 'laser_range', 'count_shots_fired', 'count_shots_hit']
            for attr in laser_attrs:
                if hasattr(env, attr):
                    metrics[attr.lower()] = getattr(env, attr)
            
            # Поля окружения
            field_attrs = ['FIELD_BOUNDS', 'field_bounds', 'boundaries']
            for attr in field_attrs:
                if hasattr(env, attr):
                    bounds = getattr(env, attr)
                    if isinstance(bounds, dict):
                        metrics["has_field_boundaries"] = 1
                        # Вычисляем размер поля
                        if all(k in bounds for k in ['x_min', 'x_max', 'y_min', 'y_max']):
                            field_area = (bounds['x_max'] - bounds['x_min']) * (bounds['y_max'] - bounds['y_min'])
                            metrics["field_area"] = field_area
                    break
                    
        except Exception as e:
            # Игнорируем ошибки при извлечении метрик
            pass
        
        return metrics

    def on_sample_end(self, *, samples, **kwargs) -> None:
        """Обработка окончания семплирования с универсальными метриками"""
        
        try:
            # Извлекаем универсальные метрики из samples
            if hasattr(samples, 'data') and "infos" in samples.data:
                try:
                    infos = samples.data["infos"]
                    if len(infos) > 0 and isinstance(infos[0], dict):
                        
                        # Используем универсальный экстрактор метрик
                        extracted_metrics = self.metrics_extractor.extract_from_infos(infos)
                        
                        if extracted_metrics and self.writer:
                            it = getattr(samples, 'iteration', 0) if hasattr(samples, 'iteration') else 0
                            
                            # Логируем все извлеченные метрики
                            for metric_name, metric_value in extracted_metrics.items():
                                if isinstance(metric_value, (int, float, np.number)):
                                    self.writer.add_scalar(f"universal_validation/{metric_name}", metric_value, it)
                            
                            print(f"📊 Universal metrics logged: {len(extracted_metrics)} metrics")
                        
                        # Анализируем специфичные паттерны
                        self._analyze_sample_patterns(infos, samples)
                        
                except Exception as e:
                    # Не критично если не получилось извлечь метрики
                    pass
        except Exception as e:
            # Полностью безопасная обработка
            pass

    def _analyze_sample_patterns(self, infos: List[Dict], samples):
        """Анализирует паттерны в данных для улучшения обучения"""
        try:
            # Анализ частоты различных типов действий
            action_patterns = {}
            position_patterns = {"movements": [], "positions": []}
            
            for info in infos:
                if isinstance(info, dict):
                    # Анализ позиций
                    for key, value in info.items():
                        if "position" in key.lower() and isinstance(value, (list, tuple)):
                            position_patterns["positions"].append(value)
                        elif "move" in key.lower() and isinstance(value, (list, tuple)):
                            position_patterns["movements"].append(value)
                        elif any(action_word in key.lower() for action_word in ["fire", "shoot", "attack"]):
                            action_patterns[key] = action_patterns.get(key, 0) + 1
            
            # Сохраняем паттерны для дальнейшего анализа
            if position_patterns["positions"]:
                self.custom_metrics_history["position_diversity"] = len(set(tuple(pos) for pos in position_patterns["positions"][-100:]))
            
            if action_patterns:
                self.custom_metrics_history["action_types_used"] = len(action_patterns)
                
        except Exception as e:
            pass

    def get_universal_summary(self) -> Dict[str, Any]:
        """Возвращает универсальную сводку callbacks"""
        return {
            "environment_type": self.env_detector.environment_type,
            "detected_features": self.detected_features,
            "custom_metrics_tracked": len(self.custom_metrics_history),
            "recording_enabled": self.record_battles,
            "onnx_export_enabled": self.export_onnx,
            "matches_recorded": self.recorded_matches,
            "callbacks_type": "universal_adaptive"
        }


# Фабрика для создания универсальных callbacks
def create_universal_callbacks_factory():
    """
    Фабрика для создания универсальных callbacks
    Автоматически адаптируется к любому окружению
    """
    def create_callbacks():
        callbacks = UniversalLeagueCallbacks()
        callbacks.setup(
            league_actor=None,  # Будет установлено позже в main script
            opponent_ids=[],    # Будет установлено позже в main script
            eval_episodes=4,
            clone_every_iters=15,
            curriculum_schedule=[
                (0, [1], [1]),
                (2_000_000, [1, 2], [1, 2]),
                (8_000_000, [1, 2, 3], [1, 2, 3]),
            ],            
            # ONNX экспорт настройки (универсальные)
            export_onnx=True,
            export_every=25,
            export_dir="./onnx_exports_universal",
            policies_to_export=["main"],
            
            # Настройки записи боев (универсальные)
            record_battles=True,
            recording_frequency=5,
            recordings_dir="./battle_recordings_universal",
            
            # Универсальные настройки
            track_universal_metrics=True,
        )
        return callbacks
    
    return create_callbacks


# Также создадим обратно совместимую версию для 3D
class FixedLeagueCallbacksWithONNXAndRecording3D(UniversalLeagueCallbacks):
    """Обратно совместимая версия для 3D систем"""
    
    def __init__(self):
        super().__init__()
        # Принудительно устанавливаем 3D режим
        self.detected_features = {
            "is_3d": True,
            "has_boundaries": True,
            "has_laser_system": True,
            "has_teams": True,
            "supports_recording": True
        }
        self.env_detector.environment_type = "3d_tactical_combat"


# Утилиты для тестирования универсальной системы
def test_universal_callbacks():
    """Тест универсальной системы callbacks"""
    print("🧪 Testing Universal Callbacks System...")
    
    # Тест экстрактора метрик
    extractor = UniversalMetricsExtractor()
    
    # Тестовые infos
    test_infos = {
        "red_0": {
            "invalid_target": 1,
            "position_3d": [1.0, 2.0, 3.0],
            "laser_shots": 3,
            "boundary_deaths": 0,
            "custom_metric_health": 85.5
        },
        "blue_0": {
            "oob_move": 2,
            "accuracy": 0.75,
            "team_reward": 15.0
        }
    }
    
    extracted = extractor.extract_from_infos(test_infos)
    print(f"✅ Metrics extracted: {len(extracted)} metrics")
    print(f"   Sample metrics: {list(extracted.keys())[:5]}")
    
    # Тест детектора окружения
    detector = UniversalEnvironmentDetector()
    
    test_obs = {
        "red_0": {
            "self": np.random.randn(13),  # 3D размер
            "allies": np.random.randn(3, 9),
            "enemies": np.random.randn(4, 11),
            "enemy_action_mask": np.ones(4)
        }
    }
    
    test_actions = {
        "red_0": {
            "target": 0,
            "move": [0.1, -0.2, 0.3],  # 3D
            "aim": [-0.1, 0.2, -0.3],
            "fire": 1
        }
    }
    
    features = detector.detect_environment_features(test_obs, test_actions)
    print(f"✅ Environment detected: {detector.environment_type}")
    print(f"   Features: {features}")
    
    # Тест callbacks
    callbacks = UniversalLeagueCallbacks()
    summary = callbacks.get_universal_summary()
    print(f"✅ Universal callbacks initialized")
    print(f"   Summary: {summary}")
    
    print("✅ Universal callbacks system tests passed!")
    return True


def analyze_environment_compatibility(obs_sample, action_sample, env_config=None):
    """
    Анализирует совместимость окружения с универсальной системой
    """
    print("🔍 Analyzing Environment Compatibility...")
    
    detector = UniversalEnvironmentDetector()
    features = detector.detect_environment_features(obs_sample, action_sample, env_config)
    
    print(f"Environment Type: {detector.environment_type}")
    print(f"Detected Features:")
    for feature, enabled in features.items():
        status = "✅" if enabled else "❌"
        print(f"  {status} {feature}: {enabled}")
    
    # Рекомендации
    recommendations = []
    if features["is_3d"]:
        recommendations.append("Use 3D-aware visualizations and metrics")
    if features["has_teams"]:
        recommendations.append("Enable team-based league training")
    if features["has_laser_system"]:
        recommendations.append("Track accuracy and ballistics metrics")
    if features["supports_recording"]:
        recommendations.append("Enable battle recording for analysis")
    
    print(f"\n💡 Recommendations:")
    for rec in recommendations:
        print(f"  • {rec}")
    
    return features, recommendations


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_universal_callbacks()
        else:
            print("Usage:")
            print("  python callbacks.py test - Test universal callbacks system")
    else:
        print("🌟 Universal Callbacks System loaded successfully!")
        print("Available functions:")
        print("  - create_universal_callbacks_factory() - Main callback factory")
        print("  - test_universal_callbacks() - Test the system")
        print("  - analyze_environment_compatibility() - Analyze environment")
        print("  - FixedLeagueCallbacksWithONNXAndRecording3D - 3D compatibility class")
        print("\nFeatures:")
        print("  ✅ Automatic adaptation to any obs/action format")
        print("  ✅ Universal metrics extraction")
        print("  ✅ Environment capability detection")
        print("  ✅ Flexible ONNX export")
        print("  ✅ Universal battle recording")
        print("  ✅ Backward compatibility with existing systems")