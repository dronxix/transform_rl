"""
Обновленные callbacks с исправленным ONNX экспортом и записью 3D боев
ИСПРАВЛЕНИЕ: Добавлен обязательный параметр algorithm в on_episode_end
ОБНОВЛЕНО: Поддержка 3D координат, границ поля, лазеров с ограниченным радиусом
"""

import os
import numpy as np
from typing import Dict, Any, List, Optional
import ray
from torch.utils.tensorboard import SummaryWriter

from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.algorithms import Algorithm

# Импортируем наши исправленные модули
from onnx_callbacks import export_onnx_with_meta
from save_res import BattleRecorder3D, RecordingArenaWrapper3D
import json

class FixedLeagueCallbacksWithONNXAndRecording3D(RLlibCallback):
    """
    ИСПРАВЛЕННЫЕ callbacks с поддержкой 3D:
    1. Правильным ONNX экспортом с meta.json файлами
    2. Записью 3D боев для визуализации
    3. Улучшенной обработкой ошибок
    4. ИСПРАВЛЕНА сигнатура on_episode_end для Ray 2.48+
    5. ДОБАВЛЕНА поддержка 3D координат и метрик
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
        
        # ONNX экспорт настройки
        self.export_onnx = True
        self.export_every = 25
        self.export_dir = "./onnx_exports"
        self.policies_to_export = ["main"]
        
        # Настройки записи 3D боев
        self.record_battles = True
        self.battle_recorder: Optional[BattleRecorder3D] = None
        self.recording_frequency = 10  # Записывать каждый N-й evaluation матч
        self.recorded_matches = 0
        
        # 3D специфичные настройки
        self.track_3d_metrics = True
        self.log_boundary_violations = True
        self.log_laser_effectiveness = True
        
    def setup(self, league_actor, opponent_ids: List[str], **kwargs):
        """Настройка параметров callbacks"""
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
        
        # Настройки записи 3D боев
        self.record_battles = kwargs.get('record_battles', True)
        self.recording_frequency = kwargs.get('recording_frequency', 10)
        
        # 3D специфичные настройки
        self.track_3d_metrics = kwargs.get('track_3d_metrics', True)
        self.log_boundary_violations = kwargs.get('log_boundary_violations', True)
        self.log_laser_effectiveness = kwargs.get('log_laser_effectiveness', True)
        
        # Создаем директории
        if self.export_onnx:
            os.makedirs(self.export_dir, exist_ok=True)
        
        if self.record_battles:
            recordings_dir = kwargs.get('recordings_dir', "./battle_recordings_3d")
            self.battle_recorder = BattleRecorder3D(recordings_dir)
            print(f"📹 3D Battle recording enabled, saving to: {recordings_dir}")

    def on_algorithm_init(self, *, algorithm: Algorithm, **kwargs) -> None:
        """Инициализация algorithm"""
        pass

    def on_train_result(self, *, algorithm: Algorithm, result: Dict[str, Any], **kwargs) -> None:
        """Основная логика обработки результатов тренировки с 3D поддержкой"""
        if self.league is None:
            return
            
        # Создаем writer
        if self.writer is None:
            logdir = getattr(algorithm, "logdir", "./logs")
            self.writer = SummaryWriter(log_dir=logdir)

        it = result["training_iteration"]
        ts_total = result.get("timesteps_total", 0)

        # 1) Evaluation матчей с возможной записью 3D боев
        try:
            for pid in self.opponent_ids:
                # Определяем нужно ли записывать этот матч
                should_record = (
                    self.record_battles and 
                    self.battle_recorder and 
                    self.recorded_matches % self.recording_frequency == 0
                )
                
                w_main, w_opp, match_3d_stats = self._play_match_3d(
                    algorithm, pid, self.eval_eps, 
                    record_battle=should_record,
                    battle_id=f"eval_3d_it{it:04d}_vs_{pid}"
                )
                
                ray.get(self.league.update_pair_result.remote(w_main, w_opp, pid))
                self.recorded_matches += 1
                
                # Логируем 3D метрики
                if self.track_3d_metrics and match_3d_stats:
                    self._log_3d_metrics(match_3d_stats, it, pid)
                
        except Exception as e:
            print(f"Error in 3D match evaluation: {e}")

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

        # 4) Куррикулум для 3D окружения
        if self.curriculum:
            for threshold, ac, ec in reversed(self.curriculum):
                if ts_total >= threshold:
                    try:
                        self._apply_curriculum_3d(algorithm, ac, ec)
                        result["custom_metrics"]["curriculum_ally_choices"] = str(ac)
                        result["custom_metrics"]["curriculum_enemy_choices"] = str(ec)
                    except Exception as e:
                        print(f"Error setting 3D curriculum: {e}")
                    break

        # 5) ИСПРАВЛЕННЫЙ ONNX экспорт для 3D моделей
        if self.export_onnx and it % self.export_every == 0 and it > 0:
            try:
                print(f"\n🔧 Starting 3D ONNX export for iteration {it}...")
                
                successful_exports = export_onnx_with_meta(
                    algorithm=algorithm,
                    iteration=it,
                    export_dir=self.export_dir,
                    policies_to_export=self.policies_to_export
                )
                
                if successful_exports:
                    result["custom_metrics"]["onnx_export_iteration"] = it
                    result["custom_metrics"]["onnx_policies_exported"] = len(successful_exports)
                    print(f"✅ 3D ONNX export completed for iteration {it} ({len(successful_exports)} policies)")
                    
                    # Логируем в TensorBoard
                    self.writer.add_scalar("export/onnx_success", 1, it)
                    for export in successful_exports:
                        self.writer.add_text(
                            f"export/onnx_3d_{export['policy_id']}", 
                            f"3D Model exported to {export['onnx_path']}", 
                            it
                        )
                else:
                    print(f"⚠️ 3D ONNX export completed but no policies were successfully exported")
                    self.writer.add_scalar("export/onnx_success", 0, it)
                    
            except Exception as e:
                print(f"❌ 3D ONNX export failed for iteration {it}: {e}")
                self.writer.add_scalar("export/onnx_success", 0, it)
                import traceback
                traceback.print_exc()

        # 6) Экспорт записей 3D боев для визуализации
        if (self.record_battles and self.battle_recorder and 
            it % (self.export_every * 2) == 0 and it > 0):
            try:
                web_export_path = self.battle_recorder.export_for_web_visualizer_3d()
                if web_export_path:
                    result["custom_metrics"]["battle_3d_replay_exported"] = it
                    print(f"🎬 3D Battle replay exported: {web_export_path}")
                    
            except Exception as e:
                print(f"Error exporting 3D battle replay: {e}")

        if self.writer:
            self.writer.flush()

    def _play_match_3d(self, algorithm: Algorithm, opp_id: str, episodes: int, 
                      record_battle: bool = False, battle_id: str = "") -> tuple:
        """
        Улучшенная версия матча с поддержкой 3D и возможностью записи для Ray 2.48
        """
        try:
            from arena_env import ArenaEnv
            env_config = algorithm.config.env_config.copy() if hasattr(algorithm.config, 'env_config') else {}
            temp_env = ArenaEnv(env_config)
            
            # Обертываем в записывающий wrapper если нужно
            if record_battle and self.battle_recorder:
                temp_env = RecordingArenaWrapper3D(temp_env, self.battle_recorder)
                print(f"📹 Recording 3D battle: {battle_id}")
            
            wins_main, wins_opp = 0, 0
            match_3d_stats = {
                'total_boundary_violations': 0,
                'total_laser_shots': 0,
                'total_laser_hits': 0,
                'total_out_of_range_attempts': 0,
                'average_battle_height': 0.0,
                'field_usage_3d': 0.0,
                'episodes_stats': []
            }
            
            for episode_idx in range(episodes):
                obs, _ = temp_env.reset()
                done = False
                episode_stats = {
                    'boundary_violations': 0,
                    'laser_shots': 0,
                    'laser_hits': 0,
                    'max_height_used': 0.0,
                    'min_height_used': 6.0,
                    'avg_height': 0.0,
                    'out_of_range_attempts': 0
                }
                
                step_count = 0
                total_height = 0.0
                height_samples = 0
                
                while not done:
                    action_dict = {}
                    
                    for aid, ob in obs.items():
                        pol_id = "main" if aid.startswith("red_") else opp_id
                        pol = algorithm.get_policy(pol_id)
                        
                        # Безопасное получение действия
                        try:
                            act, _, _ = pol.compute_single_action(ob, explore=False)
                        except Exception as e:
                            print(f"Error computing action for {aid}: {e}")
                            # Fallback действие для 3D
                            act = [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0]
                        
                        # Конвертируем действие в правильный формат для 3D
                        if isinstance(act, dict):
                            action_dict[aid] = act
                        else:
                            # Безопасная конвертация массива в словарь для 3D
                            act_array = np.array(act).flatten()
                            action_dict[aid] = {
                                "target": int(act_array[0]) if len(act_array) > 0 else 0,
                                "move": act_array[1:4].tolist() if len(act_array) > 3 else [0.0, 0.0, 0.0],  # 3D движение
                                "aim": act_array[4:7].tolist() if len(act_array) > 6 else [0.0, 0.0, 0.0],   # 3D прицеливание
                                "fire": int(round(float(act_array[7]))) if len(act_array) > 7 else 0,
                            }
                    
                    obs, rews, terms, truncs, infos = temp_env.step(action_dict)
                    done = terms.get("__all__", False) or truncs.get("__all__", False)
                    
                    # Собираем 3D статистику
                    if self.track_3d_metrics:
                        for aid, info in infos.items():
                            if aid.startswith(("red_", "blue_")):
                                # Boundary violations
                                if info.get("boundary_deaths", 0) > getattr(temp_env, '_prev_boundary_deaths', 0):
                                    episode_stats['boundary_violations'] += 1
                                    temp_env._prev_boundary_deaths = info.get("boundary_deaths", 0)
                                
                                # 3D позиции для статистики высоты
                                if "position_3d" in info:
                                    pos_3d = info["position_3d"]
                                    if len(pos_3d) >= 3:
                                        z_pos = pos_3d[2]
                                        total_height += z_pos
                                        height_samples += 1
                                        episode_stats['max_height_used'] = max(episode_stats['max_height_used'], z_pos)
                                        episode_stats['min_height_used'] = min(episode_stats['min_height_used'], z_pos)
                                
                                # Лазерная эффективность (из action)
                                action = action_dict.get(aid, {})
                                if action.get("fire", 0) > 0:
                                    episode_stats['laser_shots'] += 1
                                    
                                    # Проверяем есть ли попадание в этом шаге
                                    reward = rews.get(aid, 0)
                                    if reward > 0.1:  # Положительная награда может указывать на попадание
                                        episode_stats['laser_hits'] += 1
                    
                    step_count += 1
                
                # Финализируем статистику эпизода
                if height_samples > 0:
                    episode_stats['avg_height'] = total_height / height_samples
                
                match_3d_stats['episodes_stats'].append(episode_stats)
                
                # Определяем победителя
                red_sum = sum(v for k, v in rews.items() if k.startswith("red_"))
                blue_sum = sum(v for k, v in rews.items() if k.startswith("blue_"))
                
                if red_sum > blue_sum:
                    wins_main += 1
                elif blue_sum > red_sum:
                    wins_opp += 1
            
            # Агрегируем статистику матча
            self._aggregate_match_3d_stats(match_3d_stats)
            
            # Если записывали бой, показываем статистику
            if record_battle and hasattr(temp_env, 'recorder'):
                stats = temp_env.recorder.get_summary_statistics()
                print(f"📊 3D Battle {battle_id} stats: {stats}")
                    
            return wins_main, wins_opp, match_3d_stats
            
        except Exception as e:
            print(f"Error in _play_match_3d: {e}")
            import traceback
            traceback.print_exc()
            return 0, 0, None

    def _aggregate_match_3d_stats(self, match_stats: Dict):
        """Агрегирует 3D статистику по всем эпизодам матча (устранены переполнения)."""
        episodes = match_stats['episodes_stats']
        if not episodes:
            return

        # Суммарные метрики
        match_stats['total_boundary_violations'] = int(sum(int(ep['boundary_violations']) for ep in episodes))
        match_stats['total_laser_shots'] = int(sum(int(ep['laser_shots']) for ep in episodes))
        match_stats['total_laser_hits'] = int(sum(int(ep['laser_hits']) for ep in episodes))
        match_stats['total_out_of_range_attempts'] = int(sum(int(ep.get('out_of_range_attempts', 0)) for ep in episodes))

        # Средние метрики (в float64)
        heights = [float(ep['avg_height']) for ep in episodes if float(ep['avg_height']) > 0.0]
        if heights:
            match_stats['average_battle_height'] = float(np.mean(np.asarray(heights, dtype=np.float64)))

        # Использование 3D пространства
        max_heights = [float(ep['max_height_used']) for ep in episodes]
        min_heights = [float(ep['min_height_used']) for ep in episodes if float(ep['min_height_used']) < 6.0]

        if max_heights and min_heights:
            height_range_used = float(np.mean(np.asarray(max_heights, dtype=np.float64))) - \
                                float(np.mean(np.asarray(min_heights, dtype=np.float64)))
            total_height_available = 6.0  # Z от 0 до 6
            match_stats['field_usage_3d'] = float(np.clip(height_range_used / total_height_available, 0.0, 1.0))

        # Эффективность лазера
        shots = match_stats['total_laser_shots']
        hits  = match_stats['total_laser_hits']
        match_stats['laser_effectiveness'] = float(hits / shots) if shots > 0 else 0.0

    def _log_3d_metrics(self, match_stats: Dict, iteration: int, opponent_id: str):
        """Логирует 3D метрики в TensorBoard"""
        if not self.writer or not match_stats:
            return
        
        prefix = f"3d_metrics/{opponent_id}"
        
        # Boundary violations
        if self.log_boundary_violations:
            self.writer.add_scalar(f"{prefix}/boundary_violations", 
                                 match_stats['total_boundary_violations'], iteration)
        
        # Лазерная эффективность
        if self.log_laser_effectiveness:
            self.writer.add_scalar(f"{prefix}/laser_shots", 
                                 match_stats['total_laser_shots'], iteration)
            self.writer.add_scalar(f"{prefix}/laser_hits", 
                                 match_stats['total_laser_hits'], iteration)
            self.writer.add_scalar(f"{prefix}/laser_effectiveness", 
                                 match_stats['laser_effectiveness'], iteration)
        
        # 3D использование пространства
        self.writer.add_scalar(f"{prefix}/average_height", 
                             match_stats['average_battle_height'], iteration)
        self.writer.add_scalar(f"{prefix}/field_usage_3d", 
                             match_stats['field_usage_3d'], iteration)
        
        # Общие 3D метрики
        self.writer.add_scalar("3d_general/avg_boundary_violations", 
                             match_stats['total_boundary_violations'], iteration)
        self.writer.add_scalar("3d_general/avg_laser_effectiveness", 
                             match_stats['laser_effectiveness'], iteration)
        self.writer.add_scalar("3d_general/avg_height_usage", 
                             match_stats['average_battle_height'], iteration)

    def _apply_curriculum_3d(self, algorithm, ally_choices, enemy_choices):
        """Применение куррикулума для 3D окружения Ray 2.48"""
        try:
            # Обновляем конфигурацию алгоритма
            if hasattr(algorithm.config, 'env_config'):
                algorithm.config.env_config["ally_choices"] = ally_choices
                algorithm.config.env_config["enemy_choices"] = enemy_choices
                print(f"📚 Updated 3D curriculum: allies={ally_choices}, enemies={enemy_choices}")
            
            # Пытаемся применить к существующим 3D окружениям
            try:
                if hasattr(algorithm, 'env_runner_group') and algorithm.env_runner_group:
                    def set_curriculum_3d_fn(env):
                        if hasattr(env, 'set_curriculum'):
                            env.set_curriculum(ally_choices, enemy_choices)
                        # Если это наш 3D wrapper, обновляем базовое окружение
                        elif hasattr(env, 'env') and hasattr(env.env, 'set_curriculum'):
                            env.env.set_curriculum(ally_choices, enemy_choices)
                    
                    algorithm.env_runner_group.foreach_env(set_curriculum_3d_fn)
                    print(f"✅ Applied 3D curriculum to env_runners")
                    
            except (AttributeError, Exception) as e:
                print(f"⚠️ Could not apply 3D curriculum to existing envs: {e}")
                
        except Exception as e:
            print(f"❌ Could not apply 3D curriculum: {e}")

    def on_episode_end(self, *, base_env, policies: Dict[str, Any], 
                      episode, env_index: Optional[int] = None, **kwargs) -> None:
        """
        ИСПРАВЛЕНО: Убран обязательный параметр algorithm для совместимости с Ray 2.48+
        Обработка окончания эпизода с 3D метриками
        """
        
        try:
            # Записываем дополнительную 3D статистику эпизода
            if hasattr(episode, 'custom_metrics'):
                # Добавляем метрики о командах (если используется иерархическая система)
                red_agents = [aid for aid in episode.get_agents() if aid.startswith("red_")]
                blue_agents = [aid for aid in episode.get_agents() if aid.startswith("blue_")]
                
                episode.custom_metrics["red_team_size"] = len(red_agents)
                episode.custom_metrics["blue_team_size"] = len(blue_agents)
                
                # 3D специфичные метрики
                if hasattr(base_env, 'get_sub_environments'):
                    try:
                        sub_envs = base_env.get_sub_environments()
                        if sub_envs and env_index is not None and len(sub_envs) > env_index:
                            env = sub_envs[env_index]
                            
                            # Основные метрики выживаемости
                            if hasattr(env, '_alive_red') and hasattr(env, '_alive_blue'):
                                red_alive = sum(env._alive_red.values()) if env._alive_red else 0
                                blue_alive = sum(env._alive_blue.values()) if env._alive_blue else 0
                                
                                episode.custom_metrics["red_survivors"] = red_alive
                                episode.custom_metrics["blue_survivors"] = blue_alive
                                
                                total_agents = len(red_agents) + len(blue_agents)
                                if total_agents > 0:
                                    episode.custom_metrics["survival_rate"] = (red_alive + blue_alive) / total_agents
                            
                            # 3D специфичные метрики
                            if self.track_3d_metrics:
                                # Boundary violations
                                boundary_deaths = getattr(env, 'count_boundary_deaths', 0)
                                episode.custom_metrics["boundary_deaths"] = boundary_deaths
                                
                                # Laser metrics
                                if hasattr(env, '_agents_red') and hasattr(env, '_agents_blue'):
                                    # Подсчитываем средние высоты команд
                                    if hasattr(env, '_pos'):
                                        red_heights = []
                                        blue_heights = []
                                        
                                        for aid in env._agents_red:
                                            if aid in env._pos and env._is_alive(aid):
                                                red_heights.append(env._pos[aid][2])  # Z координата
                                        
                                        for aid in env._agents_blue:
                                            if aid in env._pos and env._is_alive(aid):
                                                blue_heights.append(env._pos[aid][2])  # Z координата
                                        
                                        if red_heights:
                                            episode.custom_metrics["red_avg_height"] = np.mean(red_heights)
                                            episode.custom_metrics["red_height_variance"] = np.var(red_heights)
                                        
                                        if blue_heights:
                                            episode.custom_metrics["blue_avg_height"] = np.mean(blue_heights)
                                            episode.custom_metrics["blue_height_variance"] = np.var(blue_heights)
                                        
                                        # Общая высота боя
                                        all_heights = red_heights + blue_heights
                                        if all_heights:
                                            episode.custom_metrics["battle_avg_height"] = np.mean(all_heights)
                                            episode.custom_metrics["height_usage_ratio"] = np.mean(all_heights) / 6.0  # Нормализуем к максимуму
                                
                                # Laser range metrics
                                episode.custom_metrics["laser_max_range"] = getattr(env, 'LASER_MAX_RANGE', 8.0)
                                
                                # Field bounds utilization
                                if hasattr(env, 'FIELD_BOUNDS'):
                                    bounds = env.FIELD_BOUNDS
                                    field_volume = ((bounds['x_max'] - bounds['x_min']) * 
                                                  (bounds['y_max'] - bounds['y_min']) * 
                                                  (bounds['z_max'] - bounds['z_min']))
                                    episode.custom_metrics["field_total_volume"] = field_volume
                                
                    except Exception as e:
                        # Не критично если не получилось получить 3D статистику окружения
                        pass
        except Exception as e:
            # Полностью безопасная обработка - если что-то пошло не так, просто пропускаем
            pass

    def on_sample_end(self, *, samples, **kwargs) -> None:
        """Обработка окончания семплирования с 3D метриками"""
        
        try:
            # Логируем статистику invalid actions если есть
            if hasattr(samples, 'data') and "infos" in samples.data:
                try:
                    infos = samples.data["infos"]
                    if len(infos) > 0 and isinstance(infos[0], dict):
                        
                        # Стандартные метрики валидности
                        total_invalid_target = sum(info.get("invalid_target", 0) for info in infos)
                        total_oob_move = sum(info.get("oob_move", 0) for info in infos)
                        total_oob_aim = sum(info.get("oob_aim", 0) for info in infos)
                        
                        # 3D специфичные метрики
                        total_boundary_deaths = sum(info.get("boundary_deaths", 0) for info in infos)
                        
                        if (total_invalid_target > 0 or total_oob_move > 0 or 
                            total_oob_aim > 0 or total_boundary_deaths > 0):
                            
                            if self.writer:
                                it = getattr(samples, 'iteration', 0) if hasattr(samples, 'iteration') else 0
                                self.writer.add_scalar("validation/invalid_targets", total_invalid_target, it)
                                self.writer.add_scalar("validation/oob_moves", total_oob_move, it)
                                self.writer.add_scalar("validation/oob_aims", total_oob_aim, it)
                                
                                # 3D специфичные метрики
                                if self.log_boundary_violations:
                                    self.writer.add_scalar("validation_3d/boundary_deaths", total_boundary_deaths, it)
                                
                            print(f"⚠️ Invalid actions: targets={total_invalid_target}, "
                                  f"oob_moves={total_oob_move}, oob_aims={total_oob_aim}, "
                                  f"boundary_deaths={total_boundary_deaths}")
                        
                        # Дополнительные 3D метрики из infos
                        if self.track_3d_metrics:
                            # Собираем 3D позиции для анализа
                            positions_3d = []
                            laser_ranges = []
                            
                            for info in infos:
                                if isinstance(info, dict):
                                    # 3D позиции
                                    if "position_3d" in info:
                                        pos_3d = info["position_3d"]
                                        if len(pos_3d) >= 3:
                                            positions_3d.append(pos_3d)
                                    
                                    # Laser ranges
                                    if "laser_range" in info:
                                        laser_ranges.append(info["laser_range"])
                            
                            # Анализ высотного распределения
                            if positions_3d and self.writer:
                                heights = [pos[2] for pos in positions_3d]
                                it = getattr(samples, 'iteration', 0) if hasattr(samples, 'iteration') else 0
                                
                                self.writer.add_scalar("3d_distribution/avg_height", np.mean(heights), it)
                                self.writer.add_scalar("3d_distribution/height_variance", np.var(heights), it)
                                self.writer.add_scalar("3d_distribution/max_height", np.max(heights), it)
                                self.writer.add_scalar("3d_distribution/min_height", np.min(heights), it)
                                
                                # Использование пространства по высоте
                                height_usage = (np.max(heights) - np.min(heights)) / 6.0  # Нормализуем к полной высоте
                                self.writer.add_scalar("3d_distribution/height_usage_ratio", height_usage, it)
                            
                            # Laser range distribution
                            if laser_ranges and self.writer:
                                it = getattr(samples, 'iteration', 0) if hasattr(samples, 'iteration') else 0
                                self.writer.add_scalar("3d_laser/avg_range", np.mean(laser_ranges), it)
                                
                except Exception as e:
                    # Не критично если не получилось собрать 3D статистику
                    pass
        except Exception as e:
            # Полностью безопасная обработка
            pass


def create_fixed_callbacks_3d_factory():
    """
    Фабрика для создания исправленных 3D callbacks
    Решает проблемы с ONNX экспортом и добавляет запись 3D боев
    """
    def create_callbacks():
        callbacks = FixedLeagueCallbacksWithONNXAndRecording3D()
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
            # ONNX экспорт настройки (ИСПРАВЛЕНО для 3D)
            export_onnx=True,
            export_every=25,  # Каждые 25 итераций
            export_dir="./onnx_exports_3d",
            policies_to_export=["main"],
            
            # Настройки записи 3D боев
            record_battles=True,
            recording_frequency=5,  # Записывать каждый 5-й evaluation матч
            recordings_dir="./battle_recordings_3d",
            
            # 3D специфичные настройки
            track_3d_metrics=True,
            log_boundary_violations=True,
            log_laser_effectiveness=True,
        )
        return callbacks
    
    return create_callbacks


# Также создадим простую утилиту для тестирования 3D ONNX экспорта
def test_3d_onnx_export_standalone():
    """
    Отдельная функция для тестирования 3D ONNX экспорта
    Можно запустить независимо от тренировки
    """
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env
    from arena_env import ArenaEnv
    from entity_attention_model import ONNXEntityAttentionModel
    from masked_multihead_dist import MaskedTargetMoveAimFire
    from ray.rllib.models import ModelCatalog
    
    def env_creator(cfg): 
        return ArenaEnv(cfg)
    
    print("🧪 Testing standalone 3D ONNX export...")
    
    # Инициализация Ray
    ray.init(ignore_reinit_error=True)
    
    try:
        register_env("ArenaEnv", env_creator)
        ModelCatalog.register_custom_model("entity_attention", ONNXEntityAttentionModel)
        ModelCatalog.register_custom_action_dist("masked_multihead", MaskedTargetMoveAimFire)
        
        # Получаем размеры 3D окружения
        tmp_env = ArenaEnv({"ally_choices": [1], "enemy_choices": [1]})
        obs_space = tmp_env.observation_space
        act_space = tmp_env.action_space
        max_enemies = obs_space["enemies"].shape[0]
        max_allies = obs_space["allies"].shape[0]
        
        print(f"🏟️ 3D Environment detected:")
        print(f"   Max allies: {max_allies}, Max enemies: {max_enemies}")
        print(f"   Self features: {obs_space['self'].shape[0]} (should be 13 for 3D)")
        print(f"   Ally features: {obs_space['allies'].shape[1]} (should be 9 for 3D)")
        print(f"   Enemy features: {obs_space['enemies'].shape[1]} (should be 11 for 3D)")
        
        # Создаем минимальную конфигурацию для 3D
        config = (
            PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .environment(
                env="ArenaEnv",
                env_config={
                    "episode_len": 50,
                    "ally_choices": [1],
                    "enemy_choices": [1],
                }
            )
            .framework("torch")
            .env_runners(num_env_runners=0)  # Только локальный worker
            .training(train_batch_size=512, minibatch_size=256)
            .multi_agent(
                policies={
                    "main": (None, obs_space, act_space, {
                        "model": {
                            "custom_model": "entity_attention",
                            "custom_action_dist": "masked_multihead",
                            "custom_model_config": {
                                "max_enemies": max_enemies,
                                "max_allies": max_allies,
                                "d_model": 128,
                                "nhead": 8,
                                "layers": 2,
                            },
                            "vf_share_layers": False,
                        }
                    }),
                },
                policy_mapping_fn=lambda aid, *args, **kwargs: "main",
                policies_to_train=["main"],
            )
        )
        
        # Создаем алгоритм
        algo = config.build()
        
        # Делаем один шаг тренировки для инициализации
        print("🏋️ Training one step to initialize 3D model...")
        result = algo.train()
        print(f"✅ Training step completed, episode_reward_mean: {result.get('env_runners', {}).get('episode_reward_mean', 'N/A')}")
        
        # Тестируем 3D ONNX экспорт
        print("🔧 Testing 3D ONNX export...")
        export_dir = "./test_3d_onnx_export"
        os.makedirs(export_dir, exist_ok=True)
        
        successful_exports = export_onnx_with_meta(
            algorithm=algo,
            iteration=1,
            export_dir=export_dir,
            policies_to_export=["main"]
        )
        
        if successful_exports:
            print(f"✅ 3D ONNX export test PASSED! Exported {len(successful_exports)} policies")
            
            # Тестируем 3D инференс
            from onnx_callbacks import run_inference_test
            
            for export in successful_exports:
                onnx_path = export["onnx_path"]
                print(f"🔬 Testing 3D inference for {onnx_path}...")
                
                try:
                    run_inference_test(onnx_path, batch_size=2, verbose=False)
                    print(f"✅ 3D Inference test PASSED for {export['policy_id']}")
                except Exception as e:
                    print(f"❌ 3D Inference test FAILED for {export['policy_id']}: {e}")
        else:
            print("❌ 3D ONNX export test FAILED - no policies exported")
            
        # Очистка
        algo.stop()
        
    except Exception as e:
        print(f"❌ 3D Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ray.shutdown()
    
    print("🏁 3D ONNX export test completed")


def test_3d_battle_recording():
    """
    Тест системы записи 3D боев
    """
    print("🎮 Testing 3D battle recording system...")
    
    try:
        from arena_env import ArenaEnv
        from save_res import BattleRecorder3D, RecordingArenaWrapper3D
        
        # Создаем 3D рекордер и окружение
        recorder = BattleRecorder3D("./test_3d_recordings")
        env = ArenaEnv({
            "ally_choices": [2], 
            "enemy_choices": [2], 
            "episode_len": 20
        })
        wrapped_env = RecordingArenaWrapper3D(env, recorder)
        
        print("✅ 3D recording system initialized")
        
        # Проверяем что окружение действительно 3D
        obs, _ = wrapped_env.reset()
        first_agent = list(obs.keys())[0]
        self_obs = obs[first_agent]["self"]
        
        print(f"📊 Environment check:")
        print(f"   Self observation size: {len(self_obs)} (should be 13 for 3D)")
        print(f"   Field bounds: {getattr(env, 'FIELD_BOUNDS', 'Not found')}")
        print(f"   Laser range: {getattr(env, 'LASER_MAX_RANGE', 'Not found')}")
        
        # Запускаем короткий бой
        for step in range(10):
            actions = {}
            for agent_id in obs.keys():
                actions[agent_id] = {
                    "target": np.random.randint(0, env.max_enemies),
                    "move": np.random.uniform(-0.3, 0.3, 3),  # 3D движение
                    "aim": np.random.uniform(-0.5, 0.5, 3),   # 3D прицеливание  
                    "fire": np.random.randint(0, 2),
                }
            
            obs, rewards, terms, truncs, infos = wrapped_env.step(actions)
            
            if terms.get("__all__") or truncs.get("__all__"):
                break
        
        # Завершаем запись
        wrapped_env.reset()
        
        # Проверяем что файлы созданы
        recording_files = os.listdir("./test_3d_recordings")
        json_files = [f for f in recording_files if f.endswith('.json')]
        
        if json_files:
            print(f"✅ 3D Recording test PASSED! Created {len(json_files)} recording files")
            
            # Проверяем структуру файла
            with open(os.path.join("./test_3d_recordings", json_files[0]), 'r') as f:
                data = json.load(f)
            
            required_3d_fields = ['field_bounds', 'laser_config', 'boundary_deaths']
            missing_fields = [field for field in required_3d_fields if field not in data]
            
            if missing_fields:
                print(f"⚠️ Missing 3D fields in recording: {missing_fields}")
            else:
                print(f"✅ 3D Recording format validation PASSED")
                print(f"   Field bounds: {data['field_bounds']}")
                print(f"   Laser config: {data['laser_config']}")
                print(f"   Boundary deaths: {data.get('boundary_deaths', 0)}")
        else:
            print("❌ 3D Recording test FAILED - no files created")
            
        # Тестируем экспорт для веб-визуализатора
        try:
            web_export = recorder.export_for_web_visualizer_3d()
            if web_export:
                print(f"✅ 3D Web export test PASSED: {web_export}")
            else:
                print("❌ 3D Web export test FAILED")
        except Exception as e:
            print(f"❌ 3D Web export test FAILED: {e}")
        
    except Exception as e:
        print(f"❌ 3D Recording test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("🏁 3D Battle recording test completed")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test_onnx":
            test_3d_onnx_export_standalone()
        elif sys.argv[1] == "test_recording":
            test_3d_battle_recording()
        else:
            print("Usage:")
            print("  python callbacks.py test_onnx - Test 3D ONNX export")
            print("  python callbacks.py test_recording - Test 3D battle recording")
    else:
        print("🚀 3D Callbacks module loaded successfully!")
        print("Available functions:")
        print("  - create_fixed_callbacks_3d_factory() - Main callback factory")
        print("  - test_3d_onnx_export_standalone() - Test ONNX export")
        print("  - test_3d_battle_recording() - Test battle recording")
        print("\nRun with 'test_onnx' or 'test_recording' arguments to run tests.")