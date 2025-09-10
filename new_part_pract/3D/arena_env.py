"""
ArenaEnv — 3D версия для Ray 2.48 с поддержкой системы снарядов
Главные изменения: 
- Трехмерное пространство (x, y, z)
- Ограниченное поле боя с смертью при выходе за границы
- Лазеры с ограниченным радиусом действия
- ДОБАВЛЕНО: Улучшенная система лазерных выстрелов для визуализации снарядов
- ДОБАВЛЕНО: Детальная информация о точности и баллистике
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from gymnasium import spaces
from ray.rllib.env import MultiAgentEnv

# Максимальные формы (реальное число агентов <= этих максимумов)
MAX_ALLIES = 6
MAX_ENEMIES = 6
ALLY_FEATS = 9  # Увеличено на 1 для Z координаты
ENEMY_FEATS = 11  # Увеличено на 1 для Z координаты
SELF_FEATS = 13  # Увеличено на 1 для Z координаты
GLOBAL_FEATS = 64

CONT_ACTION_MOVE = 3  # Теперь 3D движение (x, y, z)
CONT_ACTION_AIM = 3   # Теперь 3D прицеливание (x, y, z)

TEAM_RED = "red"
TEAM_BLUE = "blue"

# 3D Параметры поля боя
FIELD_BOUNDS = {
    'x_min': -10.0, 'x_max': 10.0,
    'y_min': -8.0,  'y_max': 8.0,
    'z_min': 0.0,   'z_max': 6.0
}

# Параметры лазера и снарядов
LASER_MAX_RANGE = 8.0      # Максимальная дальность лазера
LASER_DAMAGE = 15.0        # Урон лазера
PROJECTILE_SPEED = 25.0    # Скорость снаряда (units/second)
ACCURACY_BASE = 0.9        # Базовая точность
ACCURACY_FALLOFF = 0.1     # Снижение точности с расстоянием

def _box(lo, hi, shape):
    return spaces.Box(low=lo, high=hi, shape=shape, dtype=np.float32)

class LaserShot:
    """Класс для отслеживания лазерного выстрела"""
    def __init__(self, shooter_id: str, shooter_pos: np.ndarray, target_pos: np.ndarray, 
                 aim_vector: np.ndarray, timestamp: float):
        self.shooter_id = shooter_id
        self.shooter_pos = shooter_pos.copy()
        self.target_pos = target_pos.copy()
        self.aim_vector = aim_vector.copy()
        self.timestamp = timestamp
        self.accuracy = self._calculate_accuracy()
        self.actual_impact = self._calculate_impact_point()
        self.hit_probability = self._calculate_hit_probability()
        
    def _calculate_accuracy(self) -> float:
        """Рассчитывает точность выстрела на основе прицеливания"""
        # Безопасная проверка aim_vector
        if self.aim_vector is None or self.aim_vector.size == 0:
            return 0.8
            
        aim_magnitude = float(np.linalg.norm(self.aim_vector))
        distance = float(np.linalg.norm(self.target_pos - self.shooter_pos))
        
        # Базовая точность снижается с увеличением разброса прицеливания
        aim_penalty = min(0.5, aim_magnitude * 0.3)
        
        # Точность снижается с расстоянием
        distance_penalty = min(0.4, (distance / LASER_MAX_RANGE) * 0.3)
        
        accuracy = max(0.2, ACCURACY_BASE - aim_penalty - distance_penalty)
        return float(accuracy)
    
    def _calculate_impact_point(self) -> np.ndarray:
        """Рассчитывает фактическую точку попадания с учетом разброса"""
        spread = (1 - self.accuracy) * 2.0  # Максимальный разброс 2 единицы
        
        # Добавляем случайное отклонение
        spread_x = (np.random.random() - 0.5) * spread
        spread_y = (np.random.random() - 0.5) * spread  
        spread_z = (np.random.random() - 0.5) * spread * 0.5  # Меньший разброс по высоте
        
        impact_point = self.target_pos + np.array([spread_x, spread_y, spread_z])
        
        # Ограничиваем дальность
        distance = np.linalg.norm(impact_point - self.shooter_pos)
        if distance > LASER_MAX_RANGE:
            direction = impact_point - self.shooter_pos
            direction = direction / np.linalg.norm(direction)
            impact_point = self.shooter_pos + direction * LASER_MAX_RANGE
            
        return impact_point
    
    def _calculate_hit_probability(self) -> float:
        """Рассчитывает вероятность попадания"""
        distance_to_target = np.linalg.norm(self.actual_impact - self.target_pos)
        
        # Попадание если в радиусе 1.0 единицы от цели
        hit_radius = 1.0
        if distance_to_target <= hit_radius:
            return 1.0
        else:
            # Вероятность попадания падает с расстоянием
            return max(0.0, 1.0 - (distance_to_target - hit_radius) / 2.0)
    
    def will_hit(self, target_pos: np.ndarray) -> bool:
        """Определяет попадет ли выстрел в цель"""
        distance_to_target = np.linalg.norm(self.actual_impact - target_pos)
        return distance_to_target <= 1.0  # Радиус попадания
    
    def get_shot_info(self) -> Dict[str, Any]:
        """Возвращает информацию о выстреле для записи"""
        return {
            "shooter_id": self.shooter_id,
            "shooter_pos": self.shooter_pos.tolist(),
            "target_pos": self.target_pos.tolist(),
            "actual_impact": self.actual_impact.tolist(),
            "accuracy": self.accuracy,
            "hit_probability": self.hit_probability,
            "aim_vector": self.aim_vector.tolist(),
            "timestamp": self.timestamp,
            "max_range": LASER_MAX_RANGE,
            "distance": float(np.linalg.norm(self.target_pos - self.shooter_pos))
        }

class ArenaEnv(MultiAgentEnv):
    def __init__(self, env_config: Optional[Dict[str, Any]] = None):
        self.cfg = env_config or {}
        self.rng = np.random.default_rng(self.cfg.get("seed", 0))
        self.max_len = int(self.cfg.get("episode_len", 128))
        self.assert_invalid = bool(self.cfg.get("assert_invalid_actions", True))

        # Какими размерами команд будем «тасовать» каждый эпизод
        self.ally_choices  = list(self.cfg.get("ally_choices",  [1, 2, 3]))
        self.enemy_choices = list(self.cfg.get("enemy_choices", [1, 2, 3]))

        self.max_allies  = int(self.cfg.get("max_allies",  MAX_ALLIES))
        self.max_enemies = int(self.cfg.get("max_enemies", MAX_ENEMIES))

        # Entity-obs с масками (обновлено для 3D)
        self.single_obs_space = spaces.Dict({
            "self": _box(-15, 15, (SELF_FEATS,)),
            "allies": _box(-15, 15, (self.max_allies, ALLY_FEATS)),
            "allies_mask": spaces.MultiBinary(self.max_allies),
            "enemies": _box(-15, 15, (self.max_enemies, ENEMY_FEATS)),
            "enemies_mask": spaces.MultiBinary(self.max_enemies),
            "global_state": _box(-15, 15, (GLOBAL_FEATS,)),
            "enemy_action_mask": spaces.MultiBinary(self.max_enemies),
        })
        
        # Составное действие по четырём головам (обновлено для 3D)
        self.single_act_space = spaces.Dict({
            "target": spaces.Discrete(self.max_enemies),
            "move":   _box(-1, 1, (CONT_ACTION_MOVE,)),  # 3D движение
            "aim":    _box(-1, 1, (CONT_ACTION_AIM,)),   # 3D прицеливание
            "fire":   spaces.Discrete(2),
        })

        # Текущее состояние боя
        self._agents_red: List[str] = []
        self._agents_blue: List[str] = []
        self._alive_red: Dict[str, bool] = {}
        self._alive_blue: Dict[str, bool] = {}
        self._hp: Dict[str, float] = {}
        self._pos: Dict[str, np.ndarray] = {}  # Теперь 3D позиции
        self._team: Dict[str, str] = {}
        self._t = 0

        # Метрики валидности
        self.count_invalid_target = 0
        self.count_oob_move = 0
        self.count_oob_aim = 0
        self.count_boundary_deaths = 0
        
        # Новые метрики для снарядной системы
        self.count_shots_fired = 0
        self.count_shots_hit = 0
        self.count_shots_missed = 0
        self.shot_accuracy_history = []
        self.recent_laser_shots: List[LaserShot] = []

        # Константы для внешнего доступа
        self.FIELD_BOUNDS = FIELD_BOUNDS
        self.LASER_MAX_RANGE = LASER_MAX_RANGE
        self.LASER_DAMAGE = LASER_DAMAGE
        self.PROJECTILE_SPEED = PROJECTILE_SPEED

    @property
    def observation_space(self): return self.single_obs_space

    @property
    def action_space(self): return self.single_act_space

    # ==== API куррикулума: менять списки размеров команд на лету ====
    def set_curriculum(self, ally_choices: List[int], enemy_choices: List[int]):
        self.ally_choices = [int(x) for x in ally_choices]
        self.enemy_choices = [int(x) for x in enemy_choices]

    # ==== служебки ====
    def _make_team(self, prefix: str, n: int) -> List[str]:
        return [f"{prefix}_{i}" for i in range(n)]

    def _spawn(self):
        # Вытащили размер боя на эпизод
        n_red  = int(self.rng.choice(self.ally_choices))
        n_blue = int(self.rng.choice(self.enemy_choices))
        n_red  = max(1, min(n_red,  self.max_allies))
        n_blue = max(1, min(n_blue, self.max_enemies))

        self._agents_red  = self._make_team("red",  n_red)
        self._agents_blue = self._make_team("blue", n_blue)

        self._alive_red  = {aid: True for aid in self._agents_red}
        self._alive_blue = {aid: True for aid in self._agents_blue}
        self._hp.clear(); self._pos.clear(); self._team.clear()

        # 3D спавн команд
        for i, aid in enumerate(self._agents_red):
            self._hp[aid] = 100.0
            # Красная команда слева (-8 to -6 по X)
            self._pos[aid] = np.array([
                -7.0 + self.rng.uniform(-1.0, 1.0),  # x: левая сторона
                float(i - n_red/2) * 2.0 + self.rng.uniform(-0.5, 0.5),  # y: распределены по линии
                2.0 + self.rng.uniform(-1.0, 1.0)   # z: средняя высота
            ], dtype=np.float32)
            self._team[aid] = TEAM_RED
            
        for j, aid in enumerate(self._agents_blue):
            self._hp[aid] = 100.0
            # Синяя команда справа (6 to 8 по X)
            self._pos[aid] = np.array([
                7.0 + self.rng.uniform(-1.0, 1.0),   # x: правая сторона
                float(j - n_blue/2) * 2.0 + self.rng.uniform(-0.5, 0.5), # y: распределены по линии
                2.0 + self.rng.uniform(-1.0, 1.0)   # z: средняя высота
            ], dtype=np.float32)
            self._team[aid] = TEAM_BLUE

    def _resolve_target(self, shooter_id: str, tgt_idx: int) -> tuple[int, bool, bool]:
        """
        Возвращает (скорректированный_target_idx, was_corrected, can_fire_now).
        - Если выбран невалидный индекс, переносим на ближайшего живого врага.
        - Если живых врагов нет — возвращаем 0 и can_fire_now=False.
        - Стрельбу разрешаем только если цель в радиусе лазера.
        """
        enemy_ids = self._enemy_ids(shooter_id)
        alive_indices = [i for i, eid in enumerate(enemy_ids) if self._is_alive(eid)]
        if not alive_indices:
            # Никого живого — стрелять не по кому
            return 0, True, False

        # Приводим индекс к int и в допустимый диапазон
        try:
            tgt_idx = int(tgt_idx)
        except Exception:
            tgt_idx = 0
        if tgt_idx < 0 or tgt_idx >= len(enemy_ids) or not self._is_alive(enemy_ids[tgt_idx]):
            # Выбираем ближайшего живого к стрелку
            shooter_pos = self._pos[shooter_id]
            tgt_idx = min(
                alive_indices,
                key=lambda j: self._distance_3d(shooter_pos, self._pos[enemy_ids[j]])
            )
            was_corrected = True
        else:
            was_corrected = False

        # Разрешим стрельбу только если цель в радиусе
        can_fire_now = self._can_laser_hit(self._pos[shooter_id], self._pos[enemy_ids[tgt_idx]])
        return tgt_idx, was_corrected, can_fire_now

    def _vec(self, size):  # шумовые признаки для простоты
        return self.rng.normal(0, 0.1, size=size).astype(np.float32)

    def _check_boundaries(self, pos: np.ndarray) -> bool:
        """Проверяет, находится ли позиция в границах поля"""
        return (FIELD_BOUNDS['x_min'] <= pos[0] <= FIELD_BOUNDS['x_max'] and
                FIELD_BOUNDS['y_min'] <= pos[1] <= FIELD_BOUNDS['y_max'] and
                FIELD_BOUNDS['z_min'] <= pos[2] <= FIELD_BOUNDS['z_max'])

    def _clamp_to_boundaries(self, pos: np.ndarray) -> np.ndarray:
        """Ограничивает позицию границами поля"""
        return np.array([
            np.clip(pos[0], FIELD_BOUNDS['x_min'], FIELD_BOUNDS['x_max']),
            np.clip(pos[1], FIELD_BOUNDS['y_min'], FIELD_BOUNDS['y_max']),
            np.clip(pos[2], FIELD_BOUNDS['z_min'], FIELD_BOUNDS['z_max'])
        ], dtype=np.float32)

    def _distance_3d(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Вычисляет 3D расстояние между двумя позициями"""
        return float(np.linalg.norm(pos1 - pos2))

    def _can_laser_hit(self, shooter_pos: np.ndarray, target_pos: np.ndarray) -> bool:
        """Проверяет, может ли лазер достичь цели"""
        distance = self._distance_3d(shooter_pos, target_pos)
        return distance <= LASER_MAX_RANGE

    def _create_laser_shot(self, shooter_id: str, target_id: str, aim_vector: np.ndarray) -> LaserShot:
        """Создает объект лазерного выстрела для отслеживания"""
        shooter_pos = self._pos[shooter_id]
        target_pos = self._pos[target_id]
        timestamp = float(self._t)  # Используем текущий шаг как timestamp
        
        shot = LaserShot(shooter_id, shooter_pos, target_pos, aim_vector, timestamp)
        self.recent_laser_shots.append(shot)
        
        # Ограничиваем историю выстрелов
        if len(self.recent_laser_shots) > 100:
            self.recent_laser_shots.pop(0)
            
        return shot

    def _process_laser_shot(self, shot: LaserShot, target_id: str) -> Tuple[bool, float]:
        """
        Обрабатывает лазерный выстрел и определяет попадание
        Возвращает (hit, damage)
        """
        target_pos = self._pos[target_id]
        
        # Проверяем попадание
        hit = shot.will_hit(target_pos)
        
        if hit:
            # Рассчитываем урон с учетом точности
            base_damage = LASER_DAMAGE
            accuracy_bonus = shot.accuracy * 0.3  # До 30% бонуса за точность
            damage_variance = self.rng.uniform(-0.2, 0.2)  # ±20% разброс урона
            
            damage = base_damage * (1 + accuracy_bonus + damage_variance)
            damage = max(5.0, damage)  # Минимальный урон
            
            self.count_shots_hit += 1
            return True, float(damage)
        else:
            self.count_shots_missed += 1
            return False, 0.0

    def get_shooting_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику стрельбы для анализа"""
        total_shots = self.count_shots_fired
        hit_rate = (self.count_shots_hit / total_shots) if total_shots > 0 else 0.0
        
        avg_accuracy = (sum(self.shot_accuracy_history) / len(self.shot_accuracy_history)) \
                      if self.shot_accuracy_history else 0.0
        
        return {
            "total_shots": total_shots,
            "shots_hit": self.count_shots_hit,
            "shots_missed": self.count_shots_missed,
            "hit_rate": hit_rate,
            "average_accuracy": avg_accuracy,
            "recent_shots": len(self.recent_laser_shots),
            "projectile_system_enabled": True
        }

    # ==== стандартный API Gym ====
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._t = 0
        self._spawn()
        
        # Сброс метрик
        self.count_invalid_target = 0
        self.count_oob_move = 0
        self.count_oob_aim = 0
        self.count_boundary_deaths = 0
        self.count_shots_fired = 0
        self.count_shots_hit = 0
        self.count_shots_missed = 0
        self.shot_accuracy_history.clear()
        self.recent_laser_shots.clear()

        obs, infos = {}, {}
        battle_type = (len(self._agents_red), len(self._agents_blue))
        for aid in self._agents_red + self._agents_blue:
            obs[aid] = self._build_obs(aid)
            infos[aid] = {
                "battle_type": battle_type,
                "field_bounds": FIELD_BOUNDS,
                "laser_range": LASER_MAX_RANGE,
                "projectile_speed": PROJECTILE_SPEED,
                "accuracy_system": True
            }
        return obs, infos

    def step(self, action_dict: Dict[str, Any]):
        # CRITICAL FIX: Handle both Dict[str, Dict] and Dict[str, np.ndarray] formats
        processed_actions = {}
        
        for aid, act in action_dict.items():
            if isinstance(act, dict):
                processed_actions[aid] = act
            elif isinstance(act, (np.ndarray, list)):
                act_array = np.array(act, dtype=np.float32).flatten()
                processed_actions[aid] = {
                    "target": int(act_array[0]) if len(act_array) > 0 else 0,
                    "move": act_array[1:4] if len(act_array) > 3 else np.zeros(3, dtype=np.float32),
                    "aim": act_array[4:7] if len(act_array) > 6 else np.zeros(3, dtype=np.float32),
                    "fire": int(act_array[7]) if len(act_array) > 7 else 0,
                }
            else:
                continue
        
        # 1) Применяем действия (с валидацией)
        shots_this_step: List[LaserShot] = []
        
        for aid, act in processed_actions.items():
            if not self._is_alive(aid):
                continue
                
            # Правильная обработка 3D действий
            if isinstance(act["move"], np.ndarray):
                mv = act["move"].astype(np.float32)
            else:
                mv = np.array(act["move"], dtype=np.float32)
                
            if isinstance(act["aim"], np.ndarray):
                am = act["aim"].astype(np.float32)
            else:
                am = np.array(act["aim"], dtype=np.float32)
            
            # Обеспечиваем 3D размерность
            if len(mv) < 3:
                mv = np.pad(mv, (0, 3 - len(mv)), mode='constant')
            mv = mv[:3]
            
            if len(am) < 3:
                am = np.pad(am, (0, 3 - len(am)), mode='constant')
            am = am[:3]
            
            # Извлекаем скалярные значения правильно
            fire = int(act["fire"]) if np.isscalar(act["fire"]) else int(act["fire"].item() if hasattr(act["fire"], 'item') else act["fire"])
            tgt_idx = int(act["target"]) if np.isscalar(act["target"]) else int(act["target"].item() if hasattr(act["target"], 'item') else act["target"])
            
            # Валидация bounds для 3D
            if np.any(mv < -1.0) or np.any(mv > 1.0):
                self.count_oob_move += 1
                if self.assert_invalid:
                    raise AssertionError(f"move out of bounds: {mv}")
            if np.any(am < -1.0) or np.any(am > 1.0):
                self.count_oob_aim += 1
                if self.assert_invalid:
                    raise AssertionError(f"aim out of bounds: {am}")

            mv = np.clip(mv, -1, 1)
            am = np.clip(am, -1, 1)

            enemy_ids = self._enemy_ids(aid)
            tgt_idx, was_corrected, can_fire_now = self._resolve_target(aid, tgt_idx)
            if was_corrected:
                self.count_invalid_target += 1

            # 3D кинематика
            old_pos = self._pos[aid].copy()
            movement_3d = np.array([mv[0], mv[1], mv[2]], dtype=np.float32) * 0.3
            new_pos = old_pos + movement_3d
            
            # Проверяем границы и обновляем позицию
            if self._check_boundaries(new_pos):
                self._pos[aid] = new_pos
            else:
                # Робот попытался выйти за границы - убиваем его
                self._hp[aid] = 0.0
                self.count_boundary_deaths += 1
                print(f"Robot {aid} died by going out of bounds: {new_pos}")
            
            # Улучшенная 3D система стрельбы с снарядами
            if fire == 1 and len(enemy_ids) > tgt_idx and can_fire_now:
                tgt = enemy_ids[tgt_idx]
                if self._is_alive(tgt):
                    self.count_shots_fired += 1
                    
                    # Создаем объект выстрела
                    shot = self._create_laser_shot(aid, tgt, am)
                    shots_this_step.append(shot)
                    
                    # Сохраняем точность для статистики
                    self.shot_accuracy_history.append(shot.accuracy)
                    
                    # Обрабатываем выстрел
                    hit, damage = self._process_laser_shot(shot, tgt)
                    
                    if hit:
                        self._hp[tgt] -= damage
                        print(f"Laser hit! {aid} -> {tgt}, damage: {damage:.1f}, "
                              f"accuracy: {shot.accuracy:.2%}, distance: {shot.get_shot_info()['distance']:.1f}")
                    else:
                        print(f"Laser missed! {aid} -> {tgt}, "
                              f"accuracy: {shot.accuracy:.2%}, distance: {shot.get_shot_info()['distance']:.1f}")

        # 2) Смерти
        for aid in list(self._agents_red + self._agents_blue):
            if self._hp[aid] <= 0 and self._is_alive(aid):
                if self._team[aid] == TEAM_RED:  
                    self._alive_red[aid] = False
                else:                             
                    self._alive_blue[aid] = False

        self._t += 1

        # 3) Вознаграждения и завершение
        obs, rews, terms, truncs, infos = {}, {}, {}, {}, {}
        red_alive  = any(self._alive_red.values())  if self._alive_red  else False
        blue_alive = any(self._alive_blue.values()) if self._alive_blue else False
        done = (not red_alive) or (not blue_alive) or (self._t >= self.max_len)

        red_hp  = sum(max(0.0, self._hp[a]) for a in self._agents_red)
        blue_hp = sum(max(0.0, self._hp[a]) for a in self._agents_blue)
        red_score, blue_score = red_hp - blue_hp, blue_hp - red_hp

        # Собираем obs только для живых агентов
        alive_agents = []
        for aid in self._agents_red + self._agents_blue:
            if self._is_alive(aid): 
                obs[aid] = self._build_obs(aid)
                alive_agents.append(aid)
                rews[aid] = 0.0

        # Плотный shaping по разнице HP
        for aid in self._agents_red:  
            if aid in obs: 
                score_val = red_score * 0.001
                if isinstance(score_val, np.ndarray):
                    score_val = float(score_val.item())
                elif not isinstance(score_val, float):
                    score_val = float(score_val)
                rews[aid] = float(rews[aid] + score_val)
                
        for aid in self._agents_blue: 
            if aid in obs: 
                score_val = blue_score * 0.001
                if isinstance(score_val, np.ndarray):
                    score_val = float(score_val.item())
                elif not isinstance(score_val, float):
                    score_val = float(score_val)
                rews[aid] = float(rews[aid] + score_val)

        # Бонус за победу/проигрыш
        if done:
            if red_alive and not blue_alive:
                for aid in self._agents_red:  
                    if aid in obs: 
                        rews[aid] = float(rews[aid] + 5.0)
                for aid in self._agents_blue: 
                    if aid in obs: 
                        rews[aid] = float(rews[aid] - 5.0)
            elif blue_alive and not red_alive:
                for aid in self._agents_blue: 
                    if aid in obs: 
                        rews[aid] = float(rews[aid] + 5.0)
                for aid in self._agents_red:  
                    if aid in obs: 
                        rews[aid] = float(rews[aid] - 5.0)
        
        # Финальная проверка что все rewards - это Python float
        for aid in rews:
            if not isinstance(rews[aid], float):
                if hasattr(rews[aid], 'item'):
                    rews[aid] = float(rews[aid].item())
                else:
                    rews[aid] = float(rews[aid])

        for aid in alive_agents:
            terms[aid] = False
            truncs[aid] = False
            
        terms["__all__"] = False
        truncs["__all__"] = done

        # 4) Infos только для агентов в obs с дополнительной информацией о снарядах
        red_step_sum = float(np.clip(sum(rews.get(a, 0.0) for a in self._agents_red), -100.0, 100.0))
        blue_step_sum = float(np.clip(sum(rews.get(a, 0.0) for a in self._agents_blue), -100.0, 100.0))

        # Статистика выстрелов для этого шага
        shooting_stats = self.get_shooting_statistics()

        for aid in alive_agents:
            infos[aid] = {
                "invalid_target": self.count_invalid_target,
                "oob_move": self.count_oob_move,
                "oob_aim": self.count_oob_aim,
                "boundary_deaths": self.count_boundary_deaths,
                "team_step_reward": red_step_sum if aid.startswith("red_") else blue_step_sum,
                "position_3d": self._pos[aid].tolist(),
                "laser_range": LASER_MAX_RANGE,
                "field_bounds": FIELD_BOUNDS,
                "projectile_speed": PROJECTILE_SPEED,
                # Новая информация о снарядах
                "shots_fired_total": shooting_stats["total_shots"],
                "shots_hit_total": shooting_stats["shots_hit"],
                "current_hit_rate": shooting_stats["hit_rate"],
                "average_accuracy": shooting_stats["average_accuracy"],
                "shots_this_step": len(shots_this_step),
                "projectile_system": True
            }
            
            # Добавляем информацию о выстрелах этого робота в этом шаге
            robot_shots = [shot for shot in shots_this_step if shot.shooter_id == aid]
            if robot_shots:
                infos[aid]["laser_shots"] = [shot.get_shot_info() for shot in robot_shots]

        return obs, rews, terms, truncs, infos

    # ==== помощники ====
    def _is_alive(self, aid: str) -> bool:
        if aid.startswith("red_"):  
            return self._alive_red.get(aid, False)
        else:                       
            return self._alive_blue.get(aid, False)

    def _enemy_ids(self, aid: str) -> List[str]:
        return self._agents_blue if self._team[aid] == TEAM_RED else self._agents_red

    def _ally_ids(self, aid: str) -> List[str]:
        ids = self._agents_red if self._team[aid] == TEAM_RED else self._agents_blue
        return [a for a in ids if a != aid]

    def _build_obs(self, aid: str) -> Dict[str, np.ndarray]:
        # Свои признаки (теперь с 3D позицией)
        self_vec = np.concatenate([
            self._pos[aid],  # 3D позиция (x, y, z)
            np.array([self._hp[aid] / 100.0], dtype=np.float32),
            self._vec(SELF_FEATS - 4)  # -4 потому что позиция 3D + HP
        ], axis=0).astype(np.float32)

        # Союзники (с относительными 3D позициями)
        allies = self._ally_ids(aid)[:self.max_allies]
        allies_arr = np.zeros((self.max_allies, ALLY_FEATS), dtype=np.float32)
        allies_mask = np.zeros((self.max_allies,), dtype=np.int32)
        for i, al in enumerate(allies):
            alive = int(self._is_alive(al))
            allies_mask[i] = alive
            if alive:
                allies_arr[i, :3] = self._pos[al] - self._pos[aid]  # Относительная 3D позиция
                allies_arr[i, 3] = self._hp[al] / 100.0
                allies_arr[i, 4:] = self._vec(ALLY_FEATS - 4)

        # Противники + action_mask (с относительными 3D позициями)
        enemies = self._enemy_ids(aid)[:self.max_enemies]
        enemies_arr = np.zeros((self.max_enemies, ENEMY_FEATS), dtype=np.float32)
        enemies_mask = np.zeros((self.max_enemies,), dtype=np.int32)
        enemy_action_mask = np.zeros((self.max_enemies,), dtype=np.int32)
        for j, en in enumerate(enemies):
            alive = int(self._is_alive(en))
            enemies_mask[j] = alive
            # Проверяем, находится ли враг в радиусе действия лазера
            if alive and self._can_laser_hit(self._pos[aid], self._pos[en]):
                enemy_action_mask[j] = alive
            else:
                enemy_action_mask[j] = 0  # Не можем атаковать - слишком далеко
                
            if alive:
                enemies_arr[j, :3] = self._pos[en] - self._pos[aid]  # Относительная 3D позиция
                enemies_arr[j, 3] = self._hp[en] / 100.0
                enemies_arr[j, 4] = self._distance_3d(self._pos[aid], self._pos[en]) / LASER_MAX_RANGE  # Нормализованная дистанция
                enemies_arr[j, 5:] = self._vec(ENEMY_FEATS - 5)

        # Глобальное состояние (для централизованного V) с 3D информацией
        global_state = np.zeros((GLOBAL_FEATS,), dtype=np.float32)
        red_hp  = sum(max(0.0, self._hp[a]) for a in self._agents_red)
        blue_hp = sum(max(0.0, self._hp[a]) for a in self._agents_blue)
        
        # 3D центры команд
        red_positions = [self._pos[a] for a in self._agents_red if self._is_alive(a)]
        blue_positions = [self._pos[a] for a in self._agents_blue if self._is_alive(a)]
        
        red_center = np.mean(red_positions, axis=0) if red_positions else np.zeros(3, np.float32)
        blue_center = np.mean(blue_positions, axis=0) if blue_positions else np.zeros(3, np.float32)

        global_state[0] = red_hp  / (100.0 * max(1, len(self._agents_red)))
        global_state[1] = blue_hp / (100.0 * max(1, len(self._agents_blue)))
        global_state[2:5] = red_center    # 3D центр красной команды
        global_state[5:8] = blue_center   # 3D центр синей команды
        
        # Информация о поле боя
        global_state[8] = (FIELD_BOUNDS['x_max'] - FIELD_BOUNDS['x_min']) / 20.0  # Нормализованный размер поля
        global_state[9] = (FIELD_BOUNDS['y_max'] - FIELD_BOUNDS['y_min']) / 20.0
        global_state[10] = (FIELD_BOUNDS['z_max'] - FIELD_BOUNDS['z_min']) / 20.0
        global_state[11] = LASER_MAX_RANGE / 20.0  # Нормализованная дальность лазера
        
        # Добавляем информацию о снарядной системе
        total_shots = max(1, self.count_shots_fired)  # Избегаем деления на ноль
        global_state[12] = self.count_shots_fired / 100.0  # Нормализованное количество выстрелов
        global_state[13] = self.count_shots_hit / total_shots  # Текущий hit rate
        global_state[14] = (sum(self.shot_accuracy_history) / len(self.shot_accuracy_history)) \
                          if self.shot_accuracy_history else 0.0  # Средняя точность
        global_state[15] = len(self.recent_laser_shots) / 10.0  # Нормализованное количество недавних выстрелов
        
        global_state[16:] = self._vec(GLOBAL_FEATS - 16)

        # Клипинг всех значений к bounds
        return {
            "self": np.clip(self_vec, -15.0, 15.0),
            "allies": np.clip(allies_arr, -15.0, 15.0),
            "allies_mask": allies_mask,
            "enemies": np.clip(enemies_arr, -15.0, 15.0),
            "enemies_mask": enemies_mask,
            "global_state": np.clip(global_state, -15.0, 15.0),
            "enemy_action_mask": enemy_action_mask,
        }

    def get_battle_summary(self) -> Dict[str, Any]:
        """Возвращает сводку текущего боя для анализа"""
        return {
            "step": self._t,
            "red_agents": len(self._agents_red),
            "blue_agents": len(self._agents_blue),
            "red_alive": sum(self._alive_red.values()),
            "blue_alive": sum(self._alive_blue.values()),
            "red_hp": sum(max(0.0, self._hp[a]) for a in self._agents_red),
            "blue_hp": sum(max(0.0, self._hp[a]) for a in self._agents_blue),
            "boundary_deaths": self.count_boundary_deaths,
            "shooting_stats": self.get_shooting_statistics(),
            "field_bounds": FIELD_BOUNDS,
            "laser_config": {
                "max_range": LASER_MAX_RANGE,
                "damage": LASER_DAMAGE,
                "projectile_speed": PROJECTILE_SPEED
            }
        }

    def export_battle_data_for_visualization(self) -> Dict[str, Any]:
        """Экспортирует данные боя в формате для 3D визуализатора"""
        robot_states = []
        
        for aid in self._agents_red + self._agents_blue:
            if aid in self._pos:  # Робот существует
                robot_states.append({
                    "id": aid,
                    "team": self._team[aid],
                    "x": float(self._pos[aid][0]),
                    "y": float(self._pos[aid][1]), 
                    "z": float(self._pos[aid][2]),
                    "hp": float(self._hp[aid]),
                    "alive": self._is_alive(aid),
                    "within_bounds": self._check_boundaries(self._pos[aid]),
                    "laser_range": LASER_MAX_RANGE
                })
        
        # Информация о недавних выстрелах
        recent_shots = []
        for shot in self.recent_laser_shots[-10:]:  # Последние 10 выстрелов
            recent_shots.append(shot.get_shot_info())
        
        return {
            "timestamp": float(self._t),
            "step": self._t,
            "robots": robot_states,
            "field_bounds": FIELD_BOUNDS,
            "laser_config": {
                "max_range": LASER_MAX_RANGE,
                "damage": LASER_DAMAGE,
                "projectile_speed": PROJECTILE_SPEED,
                "accuracy_base": ACCURACY_BASE
            },
            "recent_shots": recent_shots,
            "shooting_statistics": self.get_shooting_statistics(),
            "battle_summary": self.get_battle_summary()
        }


# Вспомогательные функции для создания демо-данных с снарядами
def create_demo_battle_with_projectiles(steps: int = 50) -> List[Dict[str, Any]]:
    """Создает демонстрационные данные боя с системой снарядов"""
    
    env = ArenaEnv({
        "ally_choices": [2],
        "enemy_choices": [2], 
        "episode_len": steps
    })
    
    obs, _ = env.reset()
    frames = []
    
    for step in range(steps):
        # Генерируем случайные действия с упором на стрельбу
        actions = {}
        for agent_id in obs.keys():
            actions[agent_id] = {
                "target": np.random.randint(0, env.max_enemies),
                "move": np.random.uniform(-0.3, 0.3, 3),  # 3D движение
                "aim": np.random.uniform(-0.4, 0.4, 3),   # 3D прицеливание  
                "fire": 1 if np.random.random() < 0.2 else 0,  # 20% шанс выстрела
            }
        
        obs, rewards, terms, truncs, infos = env.step(actions)
        
        # Экспортируем кадр для визуализатора
        frame_data = env.export_battle_data_for_visualization()
        frame_data["step"] = step
        frames.append(frame_data)
        
        if terms.get("__all__") or truncs.get("__all__"):
            break
    
    return frames


def test_projectile_system():
    """Тестирует новую систему снарядов"""
    print("🚀 Testing Enhanced Projectile System")
    print("=" * 40)
    
    # Создаем окружение
    env = ArenaEnv({
        "ally_choices": [2],
        "enemy_choices": [2],
        "episode_len": 20
    })
    
    obs, _ = env.reset()
    print(f"✅ Environment initialized with {len(obs)} agents")
    print(f"   Field bounds: {env.FIELD_BOUNDS}")
    print(f"   Laser range: {env.LASER_MAX_RANGE}")
    print(f"   Projectile speed: {env.PROJECTILE_SPEED}")
    
    # Симулируем несколько шагов с выстрелами
    for step in range(10):
        actions = {}
        for agent_id in obs.keys():
            # Повышенная вероятность стрельбы для тестирования
            actions[agent_id] = {
                "target": 0,  # Всегда стреляем в первого врага
                "move": np.random.uniform(-0.2, 0.2, 3),
                "aim": np.random.uniform(-0.3, 0.3, 3),
                "fire": 1 if np.random.random() < 0.4 else 0,  # 40% шанс
            }
        
        obs, rewards, terms, truncs, infos = env.step(actions)
        
        # Показываем статистику каждые несколько шагов
        if step % 3 == 0:
            stats = env.get_shooting_statistics()
            print(f"\nStep {step}: Shooting Statistics")
            print(f"  Total shots: {stats['total_shots']}")
            print(f"  Hits: {stats['shots_hit']}")
            print(f"  Hit rate: {stats['hit_rate']:.2%}")
            print(f"  Avg accuracy: {stats['average_accuracy']:.2%}")
            print(f"  Recent shots tracked: {stats['recent_shots']}")
        
        if terms.get("__all__") or truncs.get("__all__"):
            print(f"\n🏁 Battle ended at step {step}")
            break
    
    # Финальная статистика
    final_stats = env.get_shooting_statistics()
    battle_summary = env.get_battle_summary()
    
    print(f"\n📊 Final Battle Results:")
    print(f"  Duration: {battle_summary['step']} steps")
    print(f"  Red team: {battle_summary['red_alive']}/{battle_summary['red_agents']} alive")
    print(f"  Blue team: {battle_summary['blue_alive']}/{battle_summary['blue_agents']} alive") 
    print(f"  Boundary deaths: {battle_summary['boundary_deaths']}")
    
    print(f"\n🎯 Projectile System Performance:")
    print(f"  Total projectiles fired: {final_stats['total_shots']}")
    print(f"  Successful hits: {final_stats['shots_hit']}")
    print(f"  Misses: {final_stats['shots_missed']}")
    print(f"  Overall hit rate: {final_stats['hit_rate']:.2%}")
    print(f"  Average accuracy: {final_stats['average_accuracy']:.2%}")
    print(f"  Projectile system: {final_stats['projectile_system_enabled']}")
    
    # Тест экспорта данных
    export_data = env.export_battle_data_for_visualization()
    print(f"\n📤 Export test:")
    print(f"  Robots in export: {len(export_data['robots'])}")
    print(f"  Recent shots: {len(export_data['recent_shots'])}")
    print(f"  Laser config included: {'laser_config' in export_data}")
    
    print(f"\n✅ Projectile system test completed successfully!")
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_projectile_system()
        elif sys.argv[1] == "demo":
            print("🎮 Creating demo battle data with projectiles...")
            frames = create_demo_battle_with_projectiles(30)
            print(f"✅ Generated {len(frames)} frames with projectile system")
            
            # Показываем статистику по выстрелам
            total_shots = sum(len(frame.get("recent_shots", [])) for frame in frames)
            print(f"📊 Demo statistics:")
            print(f"  Total shots tracked: {total_shots}")
            print(f"  Frames with shooting: {sum(1 for f in frames if f.get('recent_shots'))}")
            print(f"  Enhanced 3D features: ✅")
            
        else:
            print("Usage:")
            print("  python arena_env.py test - Test projectile system")
            print("  python arena_env.py demo - Create demo data")
    else:
        print("🌟 Enhanced Arena Environment with Projectile System loaded!")
        print("   Features:")
        print("   - 3D movement and positioning")
        print("   - Field boundaries with death penalties") 
        print("   - Laser range limitations")
        print("   - Realistic projectile ballistics")
        print("   - Accuracy-based hit system")
        print("   - Detailed shooting statistics")
        print("   - Export support for 3D visualizer")
        print("\n   Use 'test' or 'demo' arguments to run examples.")