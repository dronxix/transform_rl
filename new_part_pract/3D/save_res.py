"""
Система записи данных боев для последующей 3D визуализации
ОБНОВЛЕНО: Поддержка 3D координат, границ поля, лазеров с радиусом действия
ИСПРАВЛЕНИЕ: Правильная сериализация в JSON (bool -> int)
"""

import json
import numpy as np
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import time
from collections import defaultdict

@dataclass
class RobotState3D:
    """Состояние робота в конкретный момент времени в 3D пространстве"""
    id: str
    team: str
    x: float
    y: float
    z: float  # Новая Z координата
    hp: float
    alive: bool
    target_enemy: int
    move_action: List[float]  # Теперь [x, y, z] движение
    aim_action: List[float]   # Теперь [x, y, z] прицеливание
    fire_action: bool
    # Дополнительные 3D поля
    within_bounds: bool = True  # Находится ли робот в границах поля
    laser_range: float = 8.0    # Радиус действия лазера
    can_shoot_targets: List[str] = None  # Список врагов в радиусе стрельбы
    command_type: Optional[str] = None
    command_target: Optional[str] = None
    command_priority: Optional[int] = None

@dataclass
class BattleEvent3D:
    """События боя в 3D пространстве"""
    type: str  # "fire", "hit", "death", "boundary_violation", "laser_miss"
    timestamp: float
    robot_id: Optional[str] = None
    target_id: Optional[str] = None
    position: Optional[List[float]] = None  # [x, y, z] позиция события
    damage: Optional[float] = None
    distance: Optional[float] = None  # 3D расстояние для выстрелов
    success: Optional[bool] = None   # Успешность действия

@dataclass
class BattleFrame3D:
    """Кадр боя с состояниями всех роботов в 3D"""
    timestamp: float
    step: int
    robots: List[RobotState3D]
    field_bounds: Dict[str, float]  # Границы 3D поля
    laser_config: Dict[str, float]  # Конфигурация лазеров
    events: List[BattleEvent3D]     # События в этом кадре
    global_state: Dict[str, Any]

@dataclass
class BattleRecord3D:
    """Полная запись боя в 3D"""
    battle_id: str
    start_time: float
    end_time: float
    red_team_size: int
    blue_team_size: int
    winner: str
    frames: List[BattleFrame3D]
    final_stats: Dict[str, Any]
    # 3D специфичные поля
    field_bounds: Dict[str, float]
    laser_config: Dict[str, float]
    boundary_deaths: int = 0  # Количество смертей от выхода за границы

class BattleRecorder3D:
    """Класс для записи данных 3D боев во время тренировки"""
    
    def __init__(self, output_dir: str = "./battle_recordings_3d"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.current_battle: Optional[BattleRecord3D] = None
        self.current_frame_events: List[BattleEvent3D] = []
        self.frame_counter = 0
        
        # Статистика для накопления
        self.battle_stats = defaultdict(list)
        
        # 3D конфигурация по умолчанию
        self.default_field_bounds = {
            'x_min': -10.0, 'x_max': 10.0,
            'y_min': -8.0,  'y_max': 8.0,
            'z_min': 0.0,   'z_max': 6.0
        }
        self.default_laser_config = {
            'max_range': 8.0,
            'damage': 15.0,
            'accuracy_falloff': 0.1
        }
        
    def start_battle(self, battle_id: str, red_size: int, blue_size: int, 
                    field_bounds: Optional[Dict] = None, 
                    laser_config: Optional[Dict] = None):
        """Начинает запись нового 3D боя"""
        
        field_bounds = field_bounds or self.default_field_bounds
        laser_config = laser_config or self.default_laser_config
        
        self.current_battle = BattleRecord3D(
            battle_id=battle_id,
            start_time=time.time(),
            end_time=0,
            red_team_size=red_size,
            blue_team_size=blue_size,
            winner="",
            frames=[],
            final_stats={},
            field_bounds=field_bounds.copy(),
            laser_config=laser_config.copy(),
            boundary_deaths=0
        )
        self.current_frame_events = []
        self.frame_counter = 0
        print(f"🎬 Started recording 3D battle: {battle_id}")
        print(f"   Field bounds: {field_bounds}")
        print(f"   Laser config: {laser_config}")
    
    def record_frame(self, 
                    observations: Dict[str, Dict],
                    actions: Dict[str, Dict],
                    rewards: Dict[str, float],
                    infos: Dict[str, Dict],
                    global_state: Optional[Dict] = None):
        """Записывает один кадр 3D боя (строгий dataclass BattleFrame3D)."""

        if not self.current_battle:
            return

        field_bounds = self.current_battle.field_bounds
        laser_config = self.current_battle.laser_config

        robots: List[RobotState3D] = []
        self.current_frame_events = getattr(self, "current_frame_events", [])
        if self.current_frame_events is None:
            self.current_frame_events = []

        # Предсоберём позиции (для дальностей/валидации)
        robot_positions: Dict[str, np.ndarray] = {}

        # --- проход 1: позиции и hp ---
        for agent_id, agent_obs in observations.items():
            self_obs = np.asarray(agent_obs.get("self", np.zeros(13, dtype=np.float32)), dtype=np.float32)

            # Позиция: предпочтительно из infos["position_3d"], иначе из self_obs[0:3]
            info = infos.get(agent_id, {}) if isinstance(infos, dict) else {}
            pos3d = info.get("position_3d", None)

            if isinstance(pos3d, (list, tuple, np.ndarray)) and len(pos3d) >= 3:
                x, y, z = float(pos3d[0]), float(pos3d[1]), float(pos3d[2])
            else:
                if self_obs.shape[0] >= 3:
                    x, y, z = float(self_obs[0]), float(self_obs[1]), float(self_obs[2])
                elif self_obs.shape[0] >= 2:
                    x, y = float(self_obs[0]), float(self_obs[1])
                    z = 2.0
                else:
                    x = y = 0.0
                    z = 2.0

            x, y, z = [float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) for v in (x, y, z)]
            robot_positions[agent_id] = np.array([x, y, z], dtype=np.float32)

        # --- проход 2: конструируем RobotState3D + события ---
        for agent_id, agent_obs in observations.items():
            team = "red" if agent_id.startswith("red_") else "blue"
            pos = robot_positions.get(agent_id, np.array([0.0, 0.0, 2.0], dtype=np.float32))
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])

            self_obs = np.asarray(agent_obs.get("self", np.zeros(13, dtype=np.float32)), dtype=np.float32)
            hp_norm = float(self_obs[3]) if self_obs.shape[0] > 3 and np.isfinite(self_obs[3]) else 1.0
            hp_norm = float(np.clip(hp_norm, 0.0, 1.0))
            hp = 100.0 * hp_norm

            within_bounds = (
                field_bounds['x_min'] <= x <= field_bounds['x_max'] and
                field_bounds['y_min'] <= y <= field_bounds['y_max'] and
                field_bounds['z_min'] <= z <= field_bounds['z_max']
            )

            action = actions.get(agent_id, {}) if isinstance(actions, dict) else {}
            target_enemy = int(action.get("target", 0))
            move_action = action.get("move", [0.0, 0.0, 0.0])
            aim_action  = action.get("aim",  [0.0, 0.0, 0.0])
            fire_action = int(action.get("fire", 0)) > 0

            def _vec3(v):
                arr = np.zeros(3, dtype=np.float32)
                v = np.asarray(v, dtype=np.float32).flatten()
                arr[:min(3, v.shape[0])] = v[:3]
                return [float(arr[0]), float(arr[1]), float(arr[2])]

            move_action = _vec3(move_action)
            aim_action  = _vec3(aim_action)

            info = infos.get(agent_id, {}) if isinstance(infos, dict) else {}
            command_type     = info.get("command_type")
            command_target   = info.get("command_target")
            command_priority = info.get("command_priority")

            robot_state = RobotState3D(
                id=agent_id,
                team=team,
                x=x, y=y, z=z,
                hp=hp,
                alive=hp > 0.0,
                target_enemy=target_enemy,
                move_action=move_action,
                aim_action=aim_action,
                fire_action=bool(fire_action),
                within_bounds=within_bounds,
                laser_range=float(laser_config['max_range']),
                can_shoot_targets=None,
                command_type=command_type,
                command_target=command_target,
                command_priority=int(command_priority) if isinstance(command_priority, (int, np.integer)) else None,
            )
            robots.append(robot_state)

            # событие выстрела (уточним дальше цель/дистанцию)
            if fire_action:
                self.current_frame_events.append(BattleEvent3D(
                    type="fire",
                    timestamp=float(time.time()),
                    robot_id=agent_id,
                    target_id=None,
                    position=[x, y, z],
                    damage=None,
                    distance=None,
                    success=None,
                ))

        # Дополняем can_shoot_targets из enemy_action_mask (если есть)
        for agent_id, agent_obs in observations.items():
            mask = agent_obs.get("enemy_action_mask", None)
            if mask is None:
                continue
            mask = np.asarray(mask).astype(np.int32).flatten()
            robot = next((r for r in robots if r.id == agent_id), None)
            if robot is None:
                continue
            can_ids: List[str] = []
            # приклеим индексы в имена врагов по командам:
            if agent_id.startswith("red_"):
                # враги — blue_*
                for j, m in enumerate(mask):
                    if m > 0:
                        can_ids.append(f"blue_{j}")
            else:
                for j, m in enumerate(mask):
                    if m > 0:
                        can_ids.append(f"red_{j}")
            robot.can_shoot_targets = can_ids

        # Уточняем метаданные выстрелов (ближайшая цель + дистанция + success)
        for event in list(self.current_frame_events):
            if event.type != "fire" or not event.robot_id:
                continue
            shooter = next((r for r in robots if r.id == event.robot_id), None)
            if not shooter or not shooter.can_shoot_targets:
                continue
            shooter_pos = np.array([shooter.x, shooter.y, shooter.z], dtype=np.float32)
            min_d, best_id = float("inf"), None
            for tgt_id in shooter.can_shoot_targets:
                tgt = next((r for r in robots if r.id == tgt_id), None)
                if tgt is None:
                    continue
                d = float(np.linalg.norm(shooter_pos - np.array([tgt.x, tgt.y, tgt.z], dtype=np.float32)))
                if d < min_d:
                    min_d, best_id = d, tgt_id
            if best_id is not None:
                event.target_id = best_id
                event.distance = float(min_d)
                event.success = bool(min_d <= float(laser_config['max_range']))
                if event.success:
                    # добавим "hit"
                    self.current_frame_events.append(BattleEvent3D(
                        type="hit",
                        timestamp=event.timestamp + 1e-3,
                        robot_id=event.robot_id,
                        target_id=best_id,
                        position=None,
                        damage=None,
                        distance=float(min_d),
                        success=True,
                    ))
            else:
                # отметим промах
                self.current_frame_events.append(BattleEvent3D(
                    type="laser_miss",
                    timestamp=event.timestamp + 1e-3,
                    robot_id=event.robot_id,
                    target_id=None,
                    position=None,
                    damage=None,
                    distance=None,
                    success=False,
                ))

        # Собираем dataclass кадра
        frame = BattleFrame3D(
            timestamp=float(time.time()),
            step=int(getattr(self, "frame_counter", 0)),
            robots=robots,
            field_bounds=field_bounds.copy(),
            laser_config=laser_config.copy(),
            events=list(self.current_frame_events),
            global_state=global_state or {},
        )
        self.current_battle.frames.append(frame)
        self.current_frame_events = []
        self.frame_counter = int(getattr(self, "frame_counter", 0)) + 1
    
    def end_battle(self, winner: str, final_stats: Optional[Dict] = None):
        """Завершает запись боя и сохраняет"""
        if not self.current_battle:
            return
        
        self.current_battle.end_time = time.time()
        self.current_battle.winner = winner
        self.current_battle.final_stats = final_stats or {}
        
        # Рассчитываем дополнительную 3D статистику
        self._calculate_3d_battle_stats()
        
        # Сохраняем
        self._save_battle()
        
        print(f"🏁 Finished recording 3D battle: {self.current_battle.battle_id}")
        print(f"   Duration: {self.current_battle.end_time - self.current_battle.start_time:.1f}s")
        print(f"   Frames: {len(self.current_battle.frames)}")
        print(f"   Winner: {winner}")
        print(f"   Boundary deaths: {self.current_battle.boundary_deaths}")
        
        # Добавляем в общую статистику
        self.battle_stats["durations"].append(
            self.current_battle.end_time - self.current_battle.start_time
        )
        self.battle_stats["winners"].append(winner)
        self.battle_stats["frame_counts"].append(len(self.current_battle.frames))
        self.battle_stats["boundary_deaths"].append(self.current_battle.boundary_deaths)
        
        self.current_battle = None
    
    def _calculate_3d_battle_stats(self):
        """Рассчитывает дополнительную 3D статистику боя (устойчиво к dict-кадрам)."""
        if not self.current_battle or not self.current_battle.frames:
            return

        stats = {
            "total_shots": 0,
            "total_hits": 0,
            "total_deaths": 0,
            "boundary_deaths": self.current_battle.boundary_deaths,
            "laser_misses": 0,
            "team_stats": {"red": {}, "blue": {}},
            "average_distance_3d": 0.0,
            "average_height": 0.0,
            "field_usage": {},
            "laser_effectiveness": 0.0
        }

        all_shots = all_hits = all_misses = 0
        all_heights: List[float] = []
        all_distances: List[float] = []

        for frame in self.current_battle.frames:
            # поддержка dict или dataclass
            events = frame.get("events", []) if isinstance(frame, dict) else frame.events
            robots = frame.get("robots", []) if isinstance(frame, dict) else frame.robots

            # события
            for ev in events:
                etype = ev.get("type") if isinstance(ev, dict) else ev.type
                if etype == "fire":
                    all_shots += 1
                elif etype == "hit":
                    all_hits += 1
                    dist = ev.get("distance") if isinstance(ev, dict) else ev.distance
                    if dist is not None:
                        all_distances.append(float(dist))
                elif etype == "laser_miss":
                    all_misses += 1
                elif etype == "death":
                    stats["total_deaths"] += 1

            # высоты живых роботов
            for r in robots:
                alive = (r.get("alive") if isinstance(r, dict) else r.alive)
                if alive:
                    z = r.get("z") if isinstance(r, dict) else r.z
                    if z is not None and np.isfinite(z):
                        all_heights.append(float(z))

        stats["total_shots"] = int(all_shots)
        stats["total_hits"] = int(all_hits)
        stats["laser_misses"] = int(all_misses)
        stats["laser_effectiveness"] = float(all_hits / all_shots) if all_shots > 0 else 0.0
        if all_distances:
            stats["average_distance_3d"] = float(np.mean(np.asarray(all_distances, dtype=np.float64)))
        if all_heights:
            stats["average_height"] = float(np.mean(np.asarray(all_heights, dtype=np.float64)))

        # использование поля по высоте
        fb = self.current_battle.field_bounds
        field_volume = ((fb['x_max'] - fb['x_min']) *
                        (fb['y_max'] - fb['y_min']) *
                        (fb['z_max'] - fb['z_min']))
        stats["field_usage"] = {
            "total_volume": float(field_volume),
            "average_robot_height": stats["average_height"],
            "height_utilization": (stats["average_height"] / fb['z_max']) * 100.0 if fb['z_max'] > 0 else 0.0
        }

        self.current_battle.final_stats.update(stats)
    
    def _convert_for_json(self, obj):
        """Конвертирует объект для JSON сериализации"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, bool):
            return obj
        elif isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        else:
            return obj
    
    def _save_battle(self):
        """Сохраняет запись боя в JSON файл"""
        if not self.current_battle:
            return
        
        filename = f"battle_3d_{self.current_battle.battle_id}_{int(self.current_battle.start_time)}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            # Конвертируем в словарь для JSON
            battle_dict = asdict(self.current_battle)
            
            # Конвертируем все для JSON совместимости
            battle_dict_clean = self._convert_for_json(battle_dict)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(battle_dict_clean, f, ensure_ascii=False, indent=2)
            
            print(f"💾 Saved 3D battle recording: {filepath}")
            
        except Exception as e:
            print(f"❌ Error saving 3D battle: {e}")
            import traceback
            traceback.print_exc()
    
    def export_for_web_visualizer_3d(self, battle_files: Optional[List[str]] = None) -> str:
        """Экспортирует данные для 3D веб-визуализатора"""
        
        if battle_files is None:
            # Берем все 3D файлы из директории
            battle_files = [
                f for f in os.listdir(self.output_dir) 
                if f.startswith("battle_3d_") and f.endswith(".json")
            ]
        
        if not battle_files:
            print("No 3D battle files found for export")
            return ""
        
        # Берем последний файл по умолчанию
        latest_file = max(battle_files, key=lambda f: os.path.getmtime(os.path.join(self.output_dir, f)))
        filepath = os.path.join(self.output_dir, latest_file)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                battle_data = json.load(f)
        except Exception as e:
            print(f"Error loading 3D battle file {filepath}: {e}")
            return ""
        
        # Конвертируем в формат для 3D веб-визуализатора
        web_data = {
            "battle_info": {
                "id": battle_data["battle_id"],
                "duration": battle_data["end_time"] - battle_data["start_time"],
                "winner": battle_data["winner"],
                "red_team_size": battle_data["red_team_size"],
                "blue_team_size": battle_data["blue_team_size"],
                "is_3d": True,
                "field_bounds": battle_data["field_bounds"],
                "laser_config": battle_data["laser_config"],
                "boundary_deaths": battle_data.get("boundary_deaths", 0)
            },
            "frames": [],
            "statistics": battle_data.get("final_stats", {}),
            "config": {
                "field_bounds": battle_data["field_bounds"],
                "laser_config": battle_data["laser_config"]
            }
        }
        
        # Конвертируем кадры (можем взять каждый N-й для производительности)
        frame_step = max(1, len(battle_data["frames"]) // 1000)  # Максимум 1000 кадров
        
        for i, frame in enumerate(battle_data["frames"][::frame_step]):
            web_frame = {
                "timestamp": frame["timestamp"],
                "step": frame["step"],
                "robots": {},
                "events": frame["events"],
                "field_bounds": frame["field_bounds"],
                "laser_config": frame["laser_config"]
            }
            
            for robot in frame["robots"]:
                web_frame["robots"][robot["id"]] = {
                    "team": robot["team"],
                    "x": robot["x"],
                    "y": robot["y"],
                    "z": robot["z"],  # 3D позиция
                    "hp": robot["hp"],
                    "alive": robot["alive"],
                    "within_bounds": robot["within_bounds"],
                    "target": robot["target_enemy"],
                    "move": robot["move_action"],    # [x, y, z]
                    "aim": robot["aim_action"],      # [x, y, z]
                    "fire": robot["fire_action"],
                    "laser_range": robot["laser_range"],
                    "can_shoot_targets": robot.get("can_shoot_targets", []),
                    "command": {
                        "type": robot.get("command_type"),
                        "target": robot.get("command_target"),
                        "priority": robot.get("command_priority", 1)
                    }
                }
            
            web_data["frames"].append(web_frame)
        
        # Сохраняем в формате для 3D веб-визуализатора
        web_export_path = os.path.join(self.output_dir, "latest_battle_3d_web.json")
        try:
            with open(web_export_path, 'w', encoding='utf-8') as f:
                json.dump(web_data, f, ensure_ascii=False, indent=2)
            
            print(f"🌐 Exported for 3D web visualizer: {web_export_path}")
            return web_export_path
            
        except Exception as e:
            print(f"Error exporting 3D web visualizer: {e}")
            return ""
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Возвращает сводную статистику всех записанных 3D боев"""
        if not self.battle_stats["durations"]:
            return {"message": "No 3D battles recorded yet"}
        
        durations = np.array(self.battle_stats["durations"])
        winners = self.battle_stats["winners"]
        boundary_deaths = self.battle_stats.get("boundary_deaths", [])
        
        win_counts = {}
        for winner in winners:
            win_counts[winner] = win_counts.get(winner, 0) + 1
        
        return {
            "total_battles": len(durations),
            "average_duration": float(np.mean(durations)),
            "min_duration": float(np.min(durations)),
            "max_duration": float(np.max(durations)),
            "win_rate_by_team": win_counts,
            "average_frames": float(np.mean(self.battle_stats["frame_counts"])),
            "total_boundary_deaths": int(np.sum(boundary_deaths)) if boundary_deaths else 0,
            "average_boundary_deaths_per_battle": float(np.mean(boundary_deaths)) if boundary_deaths else 0.0,
            "battle_format": "3D with field boundaries and laser range limits"
        }


# Обновленная интеграция с Arena Environment для 3D
class RecordingArenaWrapper3D:
    """Обертка для ArenaEnv, которая автоматически записывает 3D бои"""
    
    def __init__(self, base_env, recorder: BattleRecorder3D):
        self.env = base_env
        self.recorder = recorder
        self.battle_counter = 0
        self.current_observations = {}
        self.current_actions = {}
        
    def reset(self, **kwargs):
        # Завершаем предыдущий бой если был
        if self.recorder.current_battle is not None:
            try:
                # Определяем победителя по HP
                red_agents = [aid for aid in self.current_observations.keys() if aid.startswith("red_")]
                blue_agents = [aid for aid in self.current_observations.keys() if aid.startswith("blue_")]
                
                red_hp = sum(self.env._hp.get(aid, 0) for aid in red_agents if self.env._is_alive(aid))
                blue_hp = sum(self.env._hp.get(aid, 0) for aid in blue_agents if self.env._is_alive(aid))
                
                winner = "red" if red_hp > blue_hp else "blue" if blue_hp > red_hp else "draw"
                self.recorder.end_battle(winner)
            except Exception as e:
                print(f"Error ending 3D battle: {e}")
                self.recorder.current_battle = None
        
        # Начинаем новый бой
        obs, info = self.env.reset(**kwargs)
        
        self.battle_counter += 1
        red_size = len([aid for aid in obs.keys() if aid.startswith("red_")])
        blue_size = len([aid for aid in obs.keys() if aid.startswith("blue_")])
        
        battle_id = f"battle_3d_{self.battle_counter:04d}"
        
        # Получаем конфигурацию 3D поля из environment
        field_bounds = getattr(self.env, 'FIELD_BOUNDS', None)
        if not field_bounds:
            # Импортируем из arena_env если доступно
            try:
                from arena_env import FIELD_BOUNDS, LASER_MAX_RANGE, LASER_DAMAGE
                field_bounds = FIELD_BOUNDS
                laser_config = {
                    'max_range': LASER_MAX_RANGE,
                    'damage': LASER_DAMAGE,
                    'accuracy_falloff': 0.1
                }
            except ImportError:
                field_bounds = self.recorder.default_field_bounds
                laser_config = self.recorder.default_laser_config
        else:
            laser_config = {
                'max_range': getattr(self.env, 'LASER_MAX_RANGE', 8.0),
                'damage': getattr(self.env, 'LASER_DAMAGE', 15.0),
                'accuracy_falloff': 0.1
            }
        
        self.recorder.start_battle(battle_id, red_size, blue_size, field_bounds, laser_config)
        
        self.current_observations = obs
        self.current_actions = {}
        
        return obs, info
    
    def step(self, action_dict):
        self.current_actions = action_dict
        
        obs, rewards, terms, truncs, infos = self.env.step(action_dict)
        
        # Записываем кадр
        try:
            # Извлекаем дополнительную 3D информацию из окружения
            global_state = {
                "timestep": getattr(self.env, '_t', 0),
                "red_hp": sum(self.env._hp.get(aid, 0) for aid in self.env._agents_red),
                "blue_hp": sum(self.env._hp.get(aid, 0) for aid in self.env._agents_blue),
                "boundary_deaths": getattr(self.env, 'count_boundary_deaths', 0),
                "laser_config": {
                    'max_range': getattr(self.env, 'LASER_MAX_RANGE', 8.0),
                    'damage': getattr(self.env, 'LASER_DAMAGE', 15.0)
                }
            }
            
            # Добавляем 3D позиции в infos если их там нет
            for aid in obs.keys():
                if aid.startswith(("red_", "blue_")) and aid in self.env._pos:
                    if "position_3d" not in infos.get(aid, {}):
                        if aid not in infos:
                            infos[aid] = {}
                        infos[aid]["position_3d"] = self.env._pos[aid].tolist()
            
            self.recorder.record_frame(
                observations=self.current_observations,
                actions=self.current_actions,
                rewards=rewards,
                infos=infos,
                global_state=global_state
            )
        except Exception as e:
            print(f"Error recording 3D frame: {e}")
        
        self.current_observations = obs
        return obs, rewards, terms, truncs, infos
    
    def __getattr__(self, name):
        # Проксируем все остальные атрибуты к base env
        return getattr(self.env, name)


# Пример использования 3D системы
def demo_3d_battle_recording():
    """Демонстрация записи и визуализации 3D боя"""
    
    # Создаем 3D рекордер
    recorder = BattleRecorder3D("./demo_battles_3d")
    
    # Создаем тестовое 3D окружение
    from arena_env import ArenaEnv
    env = ArenaEnv({
        "ally_choices": [2], 
        "enemy_choices": [2], 
        "episode_len": 50
    })
    wrapped_env = RecordingArenaWrapper3D(env, recorder)
    
    print("🎮 Starting demo 3D battle recording...")
    print(f"   Field bounds: {env.FIELD_BOUNDS if hasattr(env, 'FIELD_BOUNDS') else 'Default'}")
    print(f"   Laser range: {env.LASER_MAX_RANGE if hasattr(env, 'LASER_MAX_RANGE') else 'Default'}")
    
    # Запускаем один бой
    obs, _ = wrapped_env.reset()
    
    for step in range(50):
        # Случайные 3D действия для демонстрации
        actions = {}
        for agent_id, agent_obs in obs.items():
            actions[agent_id] = {
                "target": np.random.randint(0, env.max_enemies),
                "move": np.random.uniform(-0.5, 0.5, 3),  # 3D движение
                "aim": np.random.uniform(-0.5, 0.5, 3),   # 3D прицеливание
                "fire": np.random.randint(0, 2),
            }
        
        obs, rewards, terms, truncs, infos = wrapped_env.step(actions)
        
        if terms.get("__all__") or truncs.get("__all__"):
            break
    
    # Завершаем и экспортируем
    obs, _ = wrapped_env.reset()  # Это завершит текущий бой
    
    # Экспортируем для 3D веб-визуализатора
    web_export_path = recorder.export_for_web_visualizer_3d()
    
    # Показываем статистику
    stats = recorder.get_summary_statistics()
    print("\n📊 3D Recording Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    if web_export_path:
        print(f"\n🎬 3D battle data exported: {web_export_path}")
        print("   Use this JSON file with the 3D visualizer!")
    
    return web_export_path


# Утилиты для работы с 3D данными
def convert_2d_to_3d_battle(old_file_path: str, output_path: str):
    """Конвертирует старые 2D записи в 3D формат"""
    
    try:
        with open(old_file_path, 'r', encoding='utf-8') as f:
            old_data = json.load(f)
    except Exception as e:
        print(f"Error loading 2D battle file: {e}")
        return False
    
    # Конвертируем в 3D формат
    new_data = {
        "battle_id": old_data["battle_id"] + "_converted_3d",
        "start_time": old_data["start_time"],
        "end_time": old_data["end_time"],
        "red_team_size": old_data["red_team_size"],
        "blue_team_size": old_data["blue_team_size"],
        "winner": old_data["winner"],
        "field_bounds": {
            'x_min': -10.0, 'x_max': 10.0,
            'y_min': -8.0,  'y_max': 8.0,
            'z_min': 0.0,   'z_max': 6.0
        },
        "laser_config": {
            'max_range': 8.0,
            'damage': 15.0,
            'accuracy_falloff': 0.1
        },
        "boundary_deaths": 0,
        "frames": [],
        "final_stats": old_data.get("final_stats", {})
    }
    
    # Конвертируем кадры
    for old_frame in old_data["frames"]:
        new_frame = {
            "timestamp": old_frame["timestamp"],
            "step": old_frame["step"],
            "field_bounds": new_data["field_bounds"],
            "laser_config": new_data["laser_config"],
            "events": [],
            "global_state": old_frame.get("global_state", {}),
            "robots": []
        }
        
        # Конвертируем роботов
        for old_robot in old_frame["robots"]:
            # Добавляем Z координату
            z = 2.0 + np.random.uniform(-1.0, 1.0)  # Случайная высота
            
            new_robot = {
                "id": old_robot["id"],
                "team": old_robot["team"],
                "x": old_robot["x"],
                "y": old_robot["y"],
                "z": z,
                "hp": old_robot["hp"],
                "alive": old_robot["alive"],
                "within_bounds": True,  # Предполагаем что в 2D все были в границах
                "target_enemy": old_robot["target_enemy"],
                "move_action": old_robot["move_action"] + [0.0],  # Добавляем Z компонент
                "aim_action": old_robot["aim_action"] + [0.0],    # Добавляем Z компонент
                "fire_action": old_robot["fire_action"],
                "laser_range": 8.0,
                "can_shoot_targets": [],
                "command_type": old_robot.get("command_type"),
                "command_target": old_robot.get("command_target"),
                "command_priority": old_robot.get("command_priority")
            }
            
            new_frame["robots"].append(new_robot)
        
        # Конвертируем события
        for old_event in old_frame.get("events", []):
            new_event = {
                "type": old_event["type"],
                "timestamp": old_event["timestamp"],
                "robot_id": old_event.get("shooter" if old_event["type"] == "fire" else "robot"),
                "target_id": old_event.get("target_index"),
                "position": [0.0, 0.0, 2.0],  # Дефолтная позиция
                "distance": None,
                "damage": None,
                "success": None
            }
            new_frame["events"].append(new_event)
        
        new_data["frames"].append(new_frame)
    
    # Сохраняем конвертированный файл
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Converted 2D battle to 3D format: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving converted 3D battle: {e}")
        return False


def validate_3d_battle_file(file_path: str) -> bool:
    """Валидирует корректность 3D файла боя"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return False
    
    # Проверяем обязательные поля
    required_fields = [
        "battle_id", "start_time", "end_time", "red_team_size", 
        "blue_team_size", "winner", "frames", "field_bounds", "laser_config"
    ]
    
    for field in required_fields:
        if field not in data:
            print(f"❌ Missing required field: {field}")
            return False
    
    # Проверяем структуру field_bounds
    bounds_required = ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max']
    for bound in bounds_required:
        if bound not in data["field_bounds"]:
            print(f"❌ Missing field bound: {bound}")
            return False
    
    # Проверяем структуру laser_config
    laser_required = ['max_range', 'damage']
    for config in laser_required:
        if config not in data["laser_config"]:
            print(f"❌ Missing laser config: {config}")
            return False
    
    # Проверяем кадры
    if not data["frames"]:
        print(f"❌ No frames in battle")
        return False
    
    # Проверяем первый кадр
    first_frame = data["frames"][0]
    frame_required = ["timestamp", "step", "robots", "events"]
    for field in frame_required:
        if field not in first_frame:
            print(f"❌ Missing frame field: {field}")
            return False
    
    # Проверяем первого робота
    if first_frame["robots"]:
        first_robot = first_frame["robots"][0]
        robot_required = ["id", "team", "x", "y", "z", "hp", "alive"]
        for field in robot_required:
            if field not in first_robot:
                print(f"❌ Missing robot field: {field}")
                return False
    
    print(f"✅ 3D battle file is valid")
    print(f"   Battle: {data['battle_id']}")
    print(f"   Duration: {data['end_time'] - data['start_time']:.1f}s")
    print(f"   Frames: {len(data['frames'])}")
    print(f"   Teams: {data['red_team_size']}v{data['blue_team_size']}")
    print(f"   Winner: {data['winner']}")
    print(f"   Boundary deaths: {data.get('boundary_deaths', 0)}")
    
    return True


if __name__ == "__main__":
    # Тест 3D системы записи боев
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            demo_3d_battle_recording()
        elif sys.argv[1] == "validate" and len(sys.argv) > 2:
            validate_3d_battle_file(sys.argv[2])
        elif sys.argv[1] == "convert" and len(sys.argv) > 3:
            convert_2d_to_3d_battle(sys.argv[2], sys.argv[3])
        else:
            print("Usage:")
            print("  python save_res.py demo - Run 3D demo")
            print("  python save_res.py validate <file> - Validate 3D battle file")
            print("  python save_res.py convert <2d_file> <3d_file> - Convert 2D to 3D")
    else:
        demo_3d_battle_recording()