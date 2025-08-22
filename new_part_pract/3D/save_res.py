"""
Система записи данных боев для последующей 3D визуализации
ОБНОВЛЕНО: Улучшенная система записи лазерных выстрелов и снарядов
ИСПРАВЛЕНИЕ: Правильная сериализация в JSON (bool -> int)
ДОБАВЛЕНО: Поддержка новой системы лазерных снарядов с баллистикой
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
    z: float
    hp: float
    alive: bool
    target_enemy: int
    move_action: List[float]  # [x, y, z] движение
    aim_action: List[float]   # [x, y, z] прицеливание
    fire_action: bool
    # Дополнительные 3D поля
    within_bounds: bool = True
    laser_range: float = 8.0
    can_shoot_targets: List[str] = None
    command_type: Optional[str] = None
    command_target: Optional[str] = None
    command_priority: Optional[int] = None
    # Новые поля для системы снарядов
    accuracy: Optional[float] = None  # Расчетная точность стрельбы
    last_shot_timestamp: Optional[float] = None

@dataclass
class LaserShot3D:
    """Информация о лазерном выстреле/снаряде"""
    id: str  # Уникальный ID выстрела
    shooter_id: str
    target_id: Optional[str]
    start_position: List[float]  # [x, y, z] начальная позиция
    target_position: List[float]  # [x, y, z] целевая позиция
    actual_target: List[float]   # [x, y, z] фактическая цель с учетом разброса
    timestamp: float
    speed: float = 25.0
    max_range: float = 8.0
    accuracy: float = 0.8
    predicted_hit: bool = False
    travel_time: float = 0.0  # Время полета до цели

@dataclass
class BattleEvent3D:
    """События боя в 3D пространстве"""
    type: str  # "fire", "projectile_launch", "hit", "miss", "death", "boundary_violation"
    timestamp: float
    robot_id: Optional[str] = None
    target_id: Optional[str] = None
    position: Optional[List[float]] = None  # [x, y, z] позиция события
    damage: Optional[float] = None
    distance: Optional[float] = None
    success: Optional[bool] = None
    # Новые поля для снарядов
    projectile_id: Optional[str] = None
    accuracy: Optional[float] = None
    travel_time: Optional[float] = None
    impact_position: Optional[List[float]] = None  # Где фактически попал снаряд

@dataclass
class BattleFrame3D:
    """Кадр боя с состояниями всех роботов в 3D"""
    timestamp: float
    step: int
    robots: List[RobotState3D]
    field_bounds: Dict[str, float]
    laser_config: Dict[str, float]
    events: List[BattleEvent3D]
    global_state: Dict[str, Any]
    # Новые поля для снарядов
    active_projectiles: List[LaserShot3D] = None  # Активные снаряды в этом кадре

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
    field_bounds: Dict[str, float]
    laser_config: Dict[str, float]
    boundary_deaths: int = 0
    # Новые поля для снарядов
    total_shots_fired: int = 0
    total_shots_hit: int = 0
    average_accuracy: float = 0.0
    projectile_analytics: Dict[str, Any] = None

class BattleRecorder3D:
    """Класс для записи данных 3D боев во время тренировки с поддержкой снарядов"""
    
    def __init__(self, output_dir: str = "./battle_recordings_3d"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.current_battle: Optional[BattleRecord3D] = None
        self.current_frame_events: List[BattleEvent3D] = []
        self.frame_counter = 0
        
        # Снарядная система
        self.active_projectiles: Dict[str, LaserShot3D] = {}
        self.projectile_counter = 0
        self.shot_accuracy_history = []
        
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
            'accuracy_falloff': 0.1,
            'projectile_speed': 25.0
        }
        
    def start_battle(self, battle_id: str, red_size: int, blue_size: int, 
                    field_bounds: Optional[Dict] = None, 
                    laser_config: Optional[Dict] = None):
        """Начинает запись нового 3D боя с системой снарядов"""
        
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
            boundary_deaths=0,
            total_shots_fired=0,
            total_shots_hit=0,
            average_accuracy=0.0,
            projectile_analytics={}
        )
        
        # Сброс снарядной системы
        self.active_projectiles.clear()
        self.projectile_counter = 0
        self.shot_accuracy_history.clear()
        self.current_frame_events = []
        self.frame_counter = 0
        
        print(f"🎬 Started recording 3D battle with projectile system: {battle_id}")
        print(f"   Field bounds: {field_bounds}")
        print(f"   Laser config: {laser_config}")

    def calculate_shot_accuracy(self, robot_data: Dict) -> float:
        """Рассчитывает точность выстрела на основе данных робота"""
        aim_action = robot_data.get("aim_action")
        
        # Безопасная проверка aim_action
        if aim_action is None:
            return 0.8
        
        # Конвертируем в numpy array если это не так
        if not isinstance(aim_action, np.ndarray):
            aim_action = np.array(aim_action, dtype=np.float32)
        
        # Проверяем что массив не пустой и имеет валидные значения
        if aim_action.size == 0 or not np.any(np.isfinite(aim_action)):
            return 0.8
        
        # Дополняем до 3D если нужно
        if len(aim_action) < 3:
            aim_action = np.pad(aim_action, (0, 3 - len(aim_action)), mode='constant')
        aim_action = aim_action[:3]  # Берем только первые 3 компонента
        
        # Вычисляем разброс прицеливания
        aim_magnitude = float(np.linalg.norm(aim_action))
        base_accuracy = 0.9
        aim_penalty = min(0.5, aim_magnitude * 0.3)
        
        # Дополнительные факторы
        hp_factor = (robot_data.get("hp", 100) / 100.0) * 0.1  # Раненые роботы менее точны
        
        accuracy = max(0.2, base_accuracy - aim_penalty - (1 - hp_factor))
        return float(accuracy)

    def create_projectile(self, shooter_data: Dict, target_data: Optional[Dict] = None) -> LaserShot3D:
        """Создает новый снаряд"""
        self.projectile_counter += 1
        projectile_id = f"proj_{self.projectile_counter:04d}"
        
        # Безопасное извлечение позиции стрелка
        start_pos = [
            float(shooter_data.get("x", 0)), 
            float(shooter_data.get("y", 0)), 
            float(shooter_data.get("z", 2))
        ]
        
        if target_data:
            target_pos = [
                float(target_data.get("x", 0)), 
                float(target_data.get("y", 0)), 
                float(target_data.get("z", 2))
            ]
        else:
            # Выстрел в пустоту - используем направление прицеливания
            aim = shooter_data.get("aim_action", [1, 0, 0])
            
            # Безопасная обработка aim_action
            if isinstance(aim, np.ndarray):
                aim = aim.tolist()
            elif not isinstance(aim, (list, tuple)):
                aim = [1, 0, 0]
            
            # Дополняем до 3D если нужно
            if len(aim) < 3:
                aim = list(aim) + [0] * (3 - len(aim))
            aim = aim[:3]  # Берем только первые 3
            
            max_range = self.current_battle.laser_config.get("max_range", 8.0)
            target_pos = [
                start_pos[0] + float(aim[0]) * max_range,
                start_pos[1] + float(aim[1]) * max_range, 
                start_pos[2] + float(aim[2]) * max_range
            ]
        
        accuracy = self.calculate_shot_accuracy(shooter_data)
        
        # Рассчитываем фактическую цель с разбросом
        spread = (1 - accuracy) * 2.0
        actual_target = [
            target_pos[0] + (np.random.random() - 0.5) * spread,
            target_pos[1] + (np.random.random() - 0.5) * spread,
            target_pos[2] + (np.random.random() - 0.5) * spread * 0.5
        ]
        
        # Ограничиваем дальность
        distance = np.sqrt(sum((actual_target[i] - start_pos[i])**2 for i in range(3)))
        max_range = float(shooter_data.get("laser_range", 8.0))
        if distance > max_range:
            scale = max_range / distance
            actual_target = [
                start_pos[i] + (actual_target[i] - start_pos[i]) * scale 
                for i in range(3)
            ]
        
        speed = self.current_battle.laser_config.get("projectile_speed", 25.0)
        travel_time = distance / speed if speed > 0 else 0.0
        
        projectile = LaserShot3D(
            id=projectile_id,
            shooter_id=str(shooter_data["id"]),
            target_id=str(target_data["id"]) if target_data else None,
            start_position=start_pos,
            target_position=target_pos,
            actual_target=actual_target,
            timestamp=time.time(),
            speed=float(speed),
            max_range=max_range,
            accuracy=float(accuracy),
            predicted_hit=distance <= 1.0 and target_data is not None,
            travel_time=float(travel_time)
        )
        
        return projectile

    def record_frame(self, 
                    observations: Dict[str, Dict],
                    actions: Dict[str, Dict],
                    rewards: Dict[str, float],
                    infos: Dict[str, Dict],
                    global_state: Optional[Dict] = None):
        """Записывает один кадр 3D боя с поддержкой снарядов"""

        if not self.current_battle:
            return

        field_bounds = self.current_battle.field_bounds
        laser_config = self.current_battle.laser_config

        robots: List[RobotState3D] = []
        self.current_frame_events = getattr(self, "current_frame_events", [])
        if self.current_frame_events is None:
            self.current_frame_events = []

        # Собираем позиции роботов
        robot_positions: Dict[str, np.ndarray] = {}
        robot_data_map: Dict[str, Dict] = {}

        # Проход 1: позиции и основные данные
        for agent_id, agent_obs in observations.items():
            self_obs = np.asarray(agent_obs.get("self", np.zeros(13, dtype=np.float32)), dtype=np.float32)

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
            
            # Сохраняем данные робота для снарядов
            action = actions.get(agent_id, {}) if isinstance(actions, dict) else {}
            robot_data_map[agent_id] = {
                "id": agent_id,
                "x": x, "y": y, "z": z,
                "hp": float(self_obs[3]) * 100.0 if self_obs.shape[0] > 3 and np.isfinite(self_obs[3]) else 100.0,
                "team": "red" if agent_id.startswith("red_") else "blue",
                "alive": True,  # Будет обновлено ниже
                "aim_action": action.get("aim", [0.0, 0.0, 0.0]),
                "fire_action": int(action.get("fire", 0)) > 0,
                "laser_range": laser_config.get("max_range", 8.0)
            }

        # Проход 2: создание RobotState3D и обработка выстрелов
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
            aim_action = action.get("aim", [0.0, 0.0, 0.0])
            fire_action = int(action.get("fire", 0)) > 0

            def _vec3(v):
                arr = np.zeros(3, dtype=np.float32)
                v = np.asarray(v, dtype=np.float32).flatten()
                arr[:min(3, v.shape[0])] = v[:3]
                return [float(arr[0]), float(arr[1]), float(arr[2])]

            move_action = _vec3(move_action)
            aim_action = _vec3(aim_action)

            # Рассчитываем точность для этого робота
            accuracy = self.calculate_shot_accuracy(robot_data_map[agent_id]) if fire_action else None

            info = infos.get(agent_id, {}) if isinstance(infos, dict) else {}
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
                command_type=info.get("command_type"),
                command_target=info.get("command_target"),
                command_priority=int(info["command_priority"]) if isinstance(info.get("command_priority"), (int, np.integer)) else None,
                accuracy=accuracy,
                last_shot_timestamp=time.time() if fire_action else None
            )
            robots.append(robot_state)

            # Обработка выстрелов с созданием снарядов
            if fire_action and hp > 0.0:
                try:
                    self.current_battle.total_shots_fired += 1
                    
                    # Ищем ближайшего врага как цель
                    enemy_robots = [r for r in robot_data_map.values() 
                                  if r["team"] != team and r.get("hp", 0) > 0]
                    
                    target_robot = None
                    min_dist = float('inf')
                    if enemy_robots:
                        for enemy in enemy_robots:
                            try:
                                dist = np.sqrt((x - enemy["x"])**2 + (y - enemy["y"])**2 + (z - enemy["z"])**2)
                                if dist < min_dist:
                                    min_dist = dist
                                    target_robot = enemy
                            except (KeyError, TypeError, ValueError):
                                continue

                    # Создаем снаряд
                    projectile = self.create_projectile(robot_data_map[agent_id], target_robot)
                    self.active_projectiles[projectile.id] = projectile
                    
                    # Записываем событие запуска снаряда
                    self.current_frame_events.append(BattleEvent3D(
                        type="projectile_launch",
                        timestamp=float(time.time()),
                        robot_id=agent_id,
                        target_id=target_robot["id"] if target_robot else None,
                        position=[x, y, z],
                        projectile_id=projectile.id,
                        accuracy=accuracy,
                        travel_time=projectile.travel_time,
                        distance=min_dist if target_robot and min_dist != float('inf') else None
                    ))
                    
                    # Сохраняем точность для статистики
                    if accuracy and np.isfinite(accuracy):
                        self.shot_accuracy_history.append(accuracy)
                        
                except Exception as e:
                    print(f"Warning: Error processing shot for {agent_id}: {e}")
                    # Продолжаем выполнение, не прерывая запись кадра

        # Обновляем можем ли стрелять по врагам
        for agent_id, agent_obs in observations.items():
            mask = agent_obs.get("enemy_action_mask", None)
            if mask is None:
                continue
            mask = np.asarray(mask).astype(np.int32).flatten()
            robot = next((r for r in robots if r.id == agent_id), None)
            if robot is None:
                continue
            can_ids: List[str] = []
            if agent_id.startswith("red_"):
                for j, m in enumerate(mask):
                    if m > 0:
                        can_ids.append(f"blue_{j}")
            else:
                for j, m in enumerate(mask):
                    if m > 0:
                        can_ids.append(f"red_{j}")
            robot.can_shoot_targets = can_ids

        # Создаем кадр с активными снарядами
        frame = BattleFrame3D(
            timestamp=float(time.time()),
            step=int(getattr(self, "frame_counter", 0)),
            robots=robots,
            field_bounds=field_bounds.copy(),
            laser_config=laser_config.copy(),
            events=list(self.current_frame_events),
            global_state=global_state or {},
            active_projectiles=list(self.active_projectiles.values())
        )
        
        self.current_battle.frames.append(frame)
        self.current_frame_events = []
        self.frame_counter = int(getattr(self, "frame_counter", 0)) + 1
    
    def end_battle(self, winner: str, final_stats: Optional[Dict] = None):
        """Завершает запись боя и сохраняет с аналитикой снарядов"""
        if not self.current_battle:
            return
        
        self.current_battle.end_time = time.time()
        self.current_battle.winner = winner
        self.current_battle.final_stats = final_stats or {}
        
        # Рассчитываем статистику снарядов
        if self.shot_accuracy_history:
            self.current_battle.average_accuracy = float(np.mean(self.shot_accuracy_history))
        
        # Аналитика снарядов
        self.current_battle.projectile_analytics = {
            "total_projectiles_created": self.projectile_counter,
            "average_accuracy": self.current_battle.average_accuracy,
            "accuracy_distribution": {
                "high": sum(1 for a in self.shot_accuracy_history if a > 0.8),
                "medium": sum(1 for a in self.shot_accuracy_history if 0.5 < a <= 0.8),
                "low": sum(1 for a in self.shot_accuracy_history if a <= 0.5)
            } if self.shot_accuracy_history else {},
            "projectile_range_stats": {
                "max_range_used": self.current_battle.laser_config.get("max_range", 8.0),
                "average_shot_distance": 0.0  # Можно добавить если нужно
            }
        }
        
        # Рассчитываем дополнительную 3D статистику
        self._calculate_3d_battle_stats()
        
        # Сохраняем
        self._save_battle()
        
        print(f"🏁 Finished recording 3D battle with projectiles: {self.current_battle.battle_id}")
        print(f"   Duration: {self.current_battle.end_time - self.current_battle.start_time:.1f}s")
        print(f"   Frames: {len(self.current_battle.frames)}")
        print(f"   Winner: {winner}")
        print(f"   Shots fired: {self.current_battle.total_shots_fired}")
        print(f"   Average accuracy: {self.current_battle.average_accuracy:.2%}")
        print(f"   Boundary deaths: {self.current_battle.boundary_deaths}")
        
        # Добавляем в общую статистику
        self.battle_stats["durations"].append(
            self.current_battle.end_time - self.current_battle.start_time
        )
        self.battle_stats["winners"].append(winner)
        self.battle_stats["frame_counts"].append(len(self.current_battle.frames))
        self.battle_stats["boundary_deaths"].append(self.current_battle.boundary_deaths)
        self.battle_stats["shots_fired"].append(self.current_battle.total_shots_fired)
        self.battle_stats["average_accuracy"].append(self.current_battle.average_accuracy)
        
        self.current_battle = None

    def _calculate_3d_battle_stats(self):
        """Рассчитывает дополнительную 3D статистику боя с снарядами"""
        if not self.current_battle or not self.current_battle.frames:
            return

        stats = {
            "total_shots": self.current_battle.total_shots_fired,
            "total_hits": self.current_battle.total_shots_hit,
            "total_deaths": 0,
            "boundary_deaths": self.current_battle.boundary_deaths,
            "laser_misses": 0,
            "team_stats": {"red": {}, "blue": {}},
            "average_distance_3d": 0.0,
            "average_height": 0.0,
            "field_usage": {},
            "laser_effectiveness": self.current_battle.average_accuracy,
            # Новые снарядные метрики
            "projectile_stats": self.current_battle.projectile_analytics or {},
            "accuracy_by_team": {"red": [], "blue": []},
            "shots_by_team": {"red": 0, "blue": 0}
        }

        all_heights: List[float] = []
        all_distances: List[float] = []
        projectile_launches = 0
        hits_detected = 0

        for frame in self.current_battle.frames:
            events = frame.get("events", []) if isinstance(frame, dict) else frame.events
            robots = frame.get("robots", []) if isinstance(frame, dict) else frame.robots

            # Подсчет событий
            for ev in events:
                etype = ev.get("type") if isinstance(ev, dict) else ev.type
                if etype == "projectile_launch":
                    projectile_launches += 1
                    robot_id = ev.get("robot_id") if isinstance(ev, dict) else ev.robot_id
                    if robot_id:
                        team = "red" if robot_id.startswith("red_") else "blue"
                        stats["shots_by_team"][team] += 1
                        
                        accuracy = ev.get("accuracy") if isinstance(ev, dict) else ev.accuracy
                        if accuracy:
                            stats["accuracy_by_team"][team].append(accuracy)
                elif etype == "hit":
                    hits_detected += 1
                elif etype == "death":
                    stats["total_deaths"] += 1

            # Высоты роботов
            for r in robots:
                alive = r.get("alive") if isinstance(r, dict) else r.alive
                if alive:
                    z = r.get("z") if isinstance(r, dict) else r.z
                    if z is not None and np.isfinite(z):
                        all_heights.append(float(z))

        # Финализируем статистику
        stats["total_shots"] = max(stats["total_shots"], projectile_launches)
        stats["total_hits"] = max(stats["total_hits"], hits_detected)
        self.current_battle.total_shots_hit = stats["total_hits"]
        
        if stats["total_shots"] > 0:
            stats["hit_rate"] = stats["total_hits"] / stats["total_shots"]
        else:
            stats["hit_rate"] = 0.0

        if all_heights:
            stats["average_height"] = float(np.mean(np.asarray(all_heights, dtype=np.float64)))

        # Статистика по командам
        for team in ["red", "blue"]:
            if stats["accuracy_by_team"][team]:
                stats["team_stats"][team] = {
                    "shots_fired": stats["shots_by_team"][team],
                    "average_accuracy": float(np.mean(stats["accuracy_by_team"][team])),
                    "accuracy_std": float(np.std(stats["accuracy_by_team"][team])),
                    "best_accuracy": float(np.max(stats["accuracy_by_team"][team])),
                    "worst_accuracy": float(np.min(stats["accuracy_by_team"][team]))
                }

        # Использование поля
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
        """Конвертирует объект для JSON сериализации включая новые типы"""
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
        elif hasattr(obj, '__dict__'):  # Для dataclass объектов
            return self._convert_for_json(asdict(obj))
        else:
            return obj
    
    def _save_battle(self):
        """Сохраняет запись боя в JSON файл с поддержкой снарядов"""
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
            
            print(f"💾 Saved 3D battle recording with projectiles: {filepath}")
            
        except Exception as e:
            print(f"❌ Error saving 3D battle: {e}")
            import traceback
            traceback.print_exc()
    
    def export_for_web_visualizer_3d(self, battle_files: Optional[List[str]] = None) -> str:
        """Экспортирует данные для 3D веб-визуализатора с поддержкой снарядов"""
        
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
        
        # Конвертируем в формат для 3D веб-визуализатора с снарядами
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
                "boundary_deaths": battle_data.get("boundary_deaths", 0),
                # Новые поля для снарядов
                "total_shots_fired": battle_data.get("total_shots_fired", 0),
                "total_shots_hit": battle_data.get("total_shots_hit", 0),
                "average_accuracy": battle_data.get("average_accuracy", 0.0),
                "projectile_system": True
            },
            "frames": [],
            "statistics": battle_data.get("final_stats", {}),
            "config": {
                "field_bounds": battle_data["field_bounds"],
                "laser_config": battle_data["laser_config"],
                "projectile_analytics": battle_data.get("projectile_analytics", {})
            }
        }
        
        # Конвертируем кадры с поддержкой снарядов
        frame_step = max(1, len(battle_data["frames"]) // 1000)  # Максимум 1000 кадров
        
        for i, frame in enumerate(battle_data["frames"][::frame_step]):
            web_frame = {
                "timestamp": frame["timestamp"],
                "step": frame["step"],
                "robots": {},
                "events": frame["events"],
                "field_bounds": frame["field_bounds"],
                "laser_config": frame["laser_config"],
                # Новые поля для снарядов
                "active_projectiles": frame.get("active_projectiles", [])
            }
            
            for robot in frame["robots"]:
                web_frame["robots"][robot["id"]] = {
                    "team": robot["team"],
                    "x": robot["x"],
                    "y": robot["y"],
                    "z": robot["z"],
                    "hp": robot["hp"],
                    "alive": robot["alive"],
                    "within_bounds": robot["within_bounds"],
                    "target": robot["target_enemy"],
                    "move": robot["move_action"],
                    "aim": robot["aim_action"],
                    "fire": robot["fire_action"],
                    "laser_range": robot["laser_range"],
                    "can_shoot_targets": robot.get("can_shoot_targets", []),
                    # Новые поля для снарядов
                    "accuracy": robot.get("accuracy"),
                    "last_shot_timestamp": robot.get("last_shot_timestamp"),
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
            
            print(f"🌐 Exported for 3D web visualizer with projectiles: {web_export_path}")
            return web_export_path
            
        except Exception as e:
            print(f"Error exporting 3D web visualizer: {e}")
            return ""
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Возвращает сводную статистику всех записанных 3D боев с снарядами"""
        if not self.battle_stats["durations"]:
            return {"message": "No 3D battles recorded yet"}
        
        durations = np.array(self.battle_stats["durations"])
        winners = self.battle_stats["winners"]
        boundary_deaths = self.battle_stats.get("boundary_deaths", [])
        shots_fired = self.battle_stats.get("shots_fired", [])
        accuracy_data = self.battle_stats.get("average_accuracy", [])
        
        win_counts = {}
        for winner in winners:
            win_counts[winner] = win_counts.get(winner, 0) + 1
        
        stats = {
            "total_battles": len(durations),
            "average_duration": float(np.mean(durations)),
            "min_duration": float(np.min(durations)),
            "max_duration": float(np.max(durations)),
            "win_rate_by_team": win_counts,
            "average_frames": float(np.mean(self.battle_stats["frame_counts"])),
            "total_boundary_deaths": int(np.sum(boundary_deaths)) if boundary_deaths else 0,
            "average_boundary_deaths_per_battle": float(np.mean(boundary_deaths)) if boundary_deaths else 0.0,
            "battle_format": "3D with projectile system, field boundaries and laser range limits"
        }
        
        # Добавляем снарядную статистику
        if shots_fired:
            stats.update({
                "projectile_statistics": {
                    "total_shots_fired": int(np.sum(shots_fired)),
                    "average_shots_per_battle": float(np.mean(shots_fired)),
                    "max_shots_in_battle": int(np.max(shots_fired)),
                    "min_shots_in_battle": int(np.min(shots_fired))
                }
            })
        
        if accuracy_data:
            accuracy_array = np.array(accuracy_data)
            stats.update({
                "accuracy_statistics": {
                    "overall_average_accuracy": float(np.mean(accuracy_array)),
                    "accuracy_std": float(np.std(accuracy_array)),
                    "best_battle_accuracy": float(np.max(accuracy_array)),
                    "worst_battle_accuracy": float(np.min(accuracy_array)),
                    "battles_with_high_accuracy": int(np.sum(accuracy_array > 0.8)),
                    "battles_with_low_accuracy": int(np.sum(accuracy_array < 0.5))
                }
            })
        
        return stats


# Обновленная интеграция с Arena Environment для 3D с снарядами
class RecordingArenaWrapper3D:
    """Обертка для ArenaEnv, которая автоматически записывает 3D бои с снарядами"""
    
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
        
        battle_id = f"battle_3d_proj_{self.battle_counter:04d}"
        
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
                    'accuracy_falloff': 0.1,
                    'projectile_speed': 25.0
                }
            except ImportError:
                field_bounds = self.recorder.default_field_bounds
                laser_config = self.recorder.default_laser_config
        else:
            laser_config = {
                'max_range': getattr(self.env, 'LASER_MAX_RANGE', 8.0),
                'damage': getattr(self.env, 'LASER_DAMAGE', 15.0),
                'accuracy_falloff': 0.1,
                'projectile_speed': 25.0
            }
        
        self.recorder.start_battle(battle_id, red_size, blue_size, field_bounds, laser_config)
        
        self.current_observations = obs
        self.current_actions = {}
        
        return obs, info
    
    def step(self, action_dict):
        self.current_actions = action_dict
        
        obs, rewards, terms, truncs, infos = self.env.step(action_dict)
        
        # Записываем кадр с расширенной информацией о снарядах
        try:
            # Извлекаем дополнительную 3D информацию из окружения
            global_state = {
                "timestep": getattr(self.env, '_t', 0),
                "red_hp": sum(self.env._hp.get(aid, 0) for aid in self.env._agents_red),
                "blue_hp": sum(self.env._hp.get(aid, 0) for aid in self.env._agents_blue),
                "boundary_deaths": getattr(self.env, 'count_boundary_deaths', 0),
                "laser_config": {
                    'max_range': getattr(self.env, 'LASER_MAX_RANGE', 8.0),
                    'damage': getattr(self.env, 'LASER_DAMAGE', 15.0),
                    'projectile_speed': 25.0
                },
                # Добавляем снарядную информацию
                "active_projectiles_count": len(self.recorder.active_projectiles),
                "total_shots_this_battle": self.recorder.current_battle.total_shots_fired if self.recorder.current_battle else 0
            }
            
            # Добавляем 3D позиции в infos если их там нет
            for aid in obs.keys():
                if aid.startswith(("red_", "blue_")) and aid in self.env._pos:
                    if "position_3d" not in infos.get(aid, {}):
                        if aid not in infos:
                            infos[aid] = {}
                        infos[aid]["position_3d"] = self.env._pos[aid].tolist()
                    
                    # Добавляем информацию о последнем выстреле
                    if self.current_actions.get(aid, {}).get("fire", 0):
                        infos[aid]["last_fire_action"] = True
                        infos[aid]["fire_timestamp"] = time.time()
            
            self.recorder.record_frame(
                observations=self.current_observations,
                actions=self.current_actions,
                rewards=rewards,
                infos=infos,
                global_state=global_state
            )
        except Exception as e:
            print(f"Error recording 3D frame with projectiles: {e}")
            import traceback
            traceback.print_exc()
        
        self.current_observations = obs
        return obs, rewards, terms, truncs, infos
    
    def __getattr__(self, name):
        # Проксируем все остальные атрибуты к base env
        return getattr(self.env, name)


# Пример использования улучшенной 3D системы с снарядами
def demo_3d_battle_recording_with_projectiles():
    """Демонстрация записи и визуализации 3D боя с снарядами"""
    
    # Создаем 3D рекордер с поддержкой снарядов
    recorder = BattleRecorder3D("./demo_battles_3d_projectiles")
    
    # Создаем тестовое 3D окружение
    from arena_env import ArenaEnv
    env = ArenaEnv({
        "ally_choices": [2], 
        "enemy_choices": [2], 
        "episode_len": 60
    })
    wrapped_env = RecordingArenaWrapper3D(env, recorder)
    
    print("🎮 Starting demo 3D battle recording with projectile system...")
    print(f"   Field bounds: {env.FIELD_BOUNDS if hasattr(env, 'FIELD_BOUNDS') else 'Default'}")
    print(f"   Laser range: {env.LASER_MAX_RANGE if hasattr(env, 'LASER_MAX_RANGE') else 'Default'}")
    print("   🚀 Projectile system: ENABLED")
    
    # Запускаем один бой
    obs, _ = wrapped_env.reset()
    
    for step in range(60):
        # Случайные 3D действия с повышенной вероятностью стрельбы для демонстрации
        actions = {}
        for agent_id, agent_obs in obs.items():
            actions[agent_id] = {
                "target": np.random.randint(0, env.max_enemies),
                "move": np.random.uniform(-0.3, 0.3, 3),  # 3D движение
                "aim": np.random.uniform(-0.4, 0.4, 3),   # 3D прицеливание
                "fire": 1 if np.random.random() < 0.15 else 0,  # 15% шанс выстрела
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
    print("\n📊 3D Recording Statistics with Projectiles:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    if web_export_path:
        print(f"\n🎬 3D battle data with projectiles exported: {web_export_path}")
        print("   Use this JSON file with the enhanced 3D visualizer!")
        print("   🚀 Features: Realistic laser projectiles, accuracy system, ballistics")
    
    return web_export_path


# Утилиты для анализа снарядной системы
def analyze_projectile_effectiveness(battle_file_path: str):
    """Анализирует эффективность снарядной системы в записанном бою"""
    
    try:
        with open(battle_file_path, 'r', encoding='utf-8') as f:
            battle_data = json.load(f)
    except Exception as e:
        print(f"Error loading battle file: {e}")
        return None
    
    if not battle_data.get("projectile_analytics"):
        print("No projectile analytics found in battle file")
        return None
    
    analytics = battle_data["projectile_analytics"]
    stats = battle_data.get("final_stats", {})
    
    print(f"\n🎯 Projectile System Analysis: {battle_data['battle_id']}")
    print(f"=" * 50)
    
    # Основная статистика
    print(f"Total projectiles fired: {analytics.get('total_projectiles_created', 0)}")
    print(f"Average accuracy: {analytics.get('average_accuracy', 0):.2%}")
    print(f"Hit rate: {stats.get('hit_rate', 0):.2%}")
    print(f"Total hits: {stats.get('total_hits', 0)}")
    
    # Распределение точности
    accuracy_dist = analytics.get('accuracy_distribution', {})
    if accuracy_dist:
        print(f"\nAccuracy Distribution:")
        print(f"  High accuracy (>80%): {accuracy_dist.get('high', 0)} shots")
        print(f"  Medium accuracy (50-80%): {accuracy_dist.get('medium', 0)} shots")
        print(f"  Low accuracy (<50%): {accuracy_dist.get('low', 0)} shots")
    
    # Статистика по командам
    team_stats = stats.get('team_stats', {})
    for team, data in team_stats.items():
        if data:
            print(f"\n{team.upper()} Team Performance:")
            print(f"  Shots fired: {data.get('shots_fired', 0)}")
            print(f"  Average accuracy: {data.get('average_accuracy', 0):.2%}")
            print(f"  Best shot accuracy: {data.get('best_accuracy', 0):.2%}")
            print(f"  Worst shot accuracy: {data.get('worst_accuracy', 0):.2%}")
    
    return analytics


def validate_3d_battle_file_with_projectiles(file_path: str) -> bool:
    """Валидирует корректность 3D файла боя с системой снарядов"""
    
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
    
    # Проверяем поля снарядной системы
    projectile_fields = ["total_shots_fired", "average_accuracy"]
    projectile_support = all(field in data for field in projectile_fields)
    
    # Проверяем структуру кадров
    if not data["frames"]:
        print(f"❌ No frames in battle")
        return False
    
    # Проверяем первый кадр на поддержку снарядов
    first_frame = data["frames"][0]
    has_projectile_events = any(
        event.get("type") == "projectile_launch" 
        for event in first_frame.get("events", [])
    )
    
    print(f"✅ 3D battle file is valid")
    print(f"   Battle: {data['battle_id']}")
    print(f"   Duration: {data['end_time'] - data['start_time']:.1f}s")
    print(f"   Frames: {len(data['frames'])}")
    print(f"   Teams: {data['red_team_size']}v{data['blue_team_size']}")
    print(f"   Winner: {data['winner']}")
    print(f"   🚀 Projectile system: {'✅' if projectile_support else '❌'}")
    if projectile_support:
        print(f"     Shots fired: {data.get('total_shots_fired', 0)}")
        print(f"     Average accuracy: {data.get('average_accuracy', 0):.2%}")
        print(f"     Projectile analytics: {'✅' if data.get('projectile_analytics') else '❌'}")
    
    return True


if __name__ == "__main__":
    # Тест улучшенной 3D системы записи боев с снарядами
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            demo_3d_battle_recording_with_projectiles()
        elif sys.argv[1] == "validate" and len(sys.argv) > 2:
            validate_3d_battle_file_with_projectiles(sys.argv[2])
        elif sys.argv[1] == "analyze" and len(sys.argv) > 2:
            analyze_projectile_effectiveness(sys.argv[2])
        else:
            print("Usage:")
            print("  python save_res.py demo - Run 3D demo with projectiles")
            print("  python save_res.py validate <file> - Validate 3D battle file")
            print("  python save_res.py analyze <file> - Analyze projectile effectiveness")
    else:
        demo_3d_battle_recording_with_projectiles()