"""
ArenaEnv — исправленная версия для Ray 2.48
Главное исправление: правильная обработка действий и rewards
"""

import numpy as np
from typing import Dict, Any, Optional, List
from gymnasium import spaces
from ray.rllib.env import MultiAgentEnv

# Максимальные формы (реальное число агентов <= этих максимумов)
MAX_ALLIES = 6
MAX_ENEMIES = 6
ALLY_FEATS = 8
ENEMY_FEATS = 10
SELF_FEATS = 12
GLOBAL_FEATS = 64

CONT_ACTION_MOVE = 2
CONT_ACTION_AIM = 2

TEAM_RED = "red"
TEAM_BLUE = "blue"

def _box(lo, hi, shape):
    return spaces.Box(low=lo, high=hi, shape=shape, dtype=np.float32)

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

        # Entity-obs с масками
        self.single_obs_space = spaces.Dict({
            "self": _box(-10, 10, (SELF_FEATS,)),
            "allies": _box(-10, 10, (self.max_allies, ALLY_FEATS)),
            "allies_mask": spaces.MultiBinary(self.max_allies),
            "enemies": _box(-10, 10, (self.max_enemies, ENEMY_FEATS)),
            "enemies_mask": spaces.MultiBinary(self.max_enemies),
            "global_state": _box(-10, 10, (GLOBAL_FEATS,)),
            "enemy_action_mask": spaces.MultiBinary(self.max_enemies),
        })
        # Составное действие по четырём головам
        self.single_act_space = spaces.Dict({
            "target": spaces.Discrete(self.max_enemies),
            "move":   _box(-1, 1, (CONT_ACTION_MOVE,)),
            "aim":    _box(-1, 1, (CONT_ACTION_AIM,)),
            "fire":   spaces.Discrete(2),
        })

        # Текущее состояние боя
        self._agents_red: List[str] = []
        self._agents_blue: List[str] = []
        self._alive_red: Dict[str, bool] = {}
        self._alive_blue: Dict[str, bool] = {}
        self._hp: Dict[str, float] = {}
        self._pos: Dict[str, np.ndarray] = {}
        self._team: Dict[str, str] = {}
        self._t = 0

        # Метрики валидности
        self.count_invalid_target = 0
        self.count_oob_move = 0
        self.count_oob_aim = 0

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

        for i, aid in enumerate(self._agents_red):
            self._hp[aid] = 100.0
            self._pos[aid] = np.array([-5.0, float(i)*2.0], dtype=np.float32)
            self._team[aid] = TEAM_RED
        for j, aid in enumerate(self._agents_blue):
            self._hp[aid] = 100.0
            self._pos[aid] = np.array([+5.0, float(j)*2.0], dtype=np.float32)
            self._team[aid] = TEAM_BLUE

    def _vec(self, size):  # шумовые признаки для простоты
        return self.rng.normal(0, 0.1, size=size).astype(np.float32)

    # ==== стандартный API Gym ====
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._t = 0
        self._spawn()
        self.count_invalid_target = 0
        self.count_oob_move = 0
        self.count_oob_aim = 0

        obs, infos = {}, {}
        battle_type = (len(self._agents_red), len(self._agents_blue))
        for aid in self._agents_red + self._agents_blue:
            obs[aid] = self._build_obs(aid)
            infos[aid] = {"battle_type": battle_type}
        return obs, infos

    def step(self, action_dict: Dict[str, Any]):
        # CRITICAL FIX: Handle both Dict[str, Dict] and Dict[str, np.ndarray] formats
        # Ray может передавать действия в разных форматах
        processed_actions = {}
        
        for aid, act in action_dict.items():
            if isinstance(act, dict):
                # Действие уже в формате словаря
                processed_actions[aid] = act
            elif isinstance(act, (np.ndarray, list)):
                # Действие в формате массива - преобразуем в словарь
                act_array = np.array(act, dtype=np.float32).flatten()
                processed_actions[aid] = {
                    "target": int(act_array[0]) if len(act_array) > 0 else 0,
                    "move": act_array[1:3] if len(act_array) > 2 else np.zeros(2, dtype=np.float32),
                    "aim": act_array[3:5] if len(act_array) > 4 else np.zeros(2, dtype=np.float32),
                    "fire": int(act_array[5]) if len(act_array) > 5 else 0,
                }
            else:
                # Неизвестный формат - пропускаем
                continue
        
        # 1) Применяем действия (с валидацией)
        for aid, act in processed_actions.items():
            if not self._is_alive(aid):
                continue
                
            # Правильная обработка действий
            if isinstance(act["move"], np.ndarray):
                mv = act["move"].astype(np.float32)
            else:
                mv = np.array(act["move"], dtype=np.float32)
                
            if isinstance(act["aim"], np.ndarray):
                am = act["aim"].astype(np.float32)
            else:
                am = np.array(act["aim"], dtype=np.float32)
            
            # Извлекаем скалярные значения правильно
            fire = int(act["fire"]) if np.isscalar(act["fire"]) else int(act["fire"].item() if hasattr(act["fire"], 'item') else act["fire"])
            tgt_idx = int(act["target"]) if np.isscalar(act["target"]) else int(act["target"].item() if hasattr(act["target"], 'item') else act["target"])
            
            # Валидация bounds
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
            valid = (0 <= tgt_idx < len(enemy_ids)) and (len(enemy_ids) > 0 and self._is_alive(enemy_ids[tgt_idx]))
            if not valid and tgt_idx != 0:  # tgt_idx=0 может быть валидным если нет врагов
                self.count_invalid_target += 1
                if self.assert_invalid:
                    raise AssertionError(f"invalid target: idx={tgt_idx}, alive_enemies={[e for e in enemy_ids if self._is_alive(e)]}")

            # Простая кинематика
            self._pos[aid] += np.array([mv[0], mv[1]], dtype=np.float32) * 0.3
            
            # Попадание по цели с примитивной вероятностью
            if valid and fire == 1 and len(enemy_ids) > tgt_idx:
                tgt = enemy_ids[tgt_idx]
                dist = np.linalg.norm(self._pos[aid] - self._pos[tgt])
                hit_p = np.exp(-0.1 * dist) * (0.5 + 0.5 * (1 - np.linalg.norm(am)))
                if self.rng.random() < hit_p:
                    self._hp[tgt] -= 10.0

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
                # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Всегда возвращаем float, а не np.float32
                rews[aid] = 0.0

        # Плотный shaping по разнице HP
        for aid in self._agents_red:  
            if aid in obs: 
                # КРИТИЧНО: Убеждаемся что это Python float, а не numpy
                score_val = red_score * 0.001
                if isinstance(score_val, np.ndarray):
                    score_val = float(score_val.item())
                elif not isinstance(score_val, float):
                    score_val = float(score_val)
                rews[aid] = float(rews[aid] + score_val)
                
        for aid in self._agents_blue: 
            if aid in obs: 
                # КРИТИЧНО: Убеждаемся что это Python float, а не numpy
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
        
        # КРИТИЧНО: Финальная проверка что все rewards - это Python float
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

        # 4) Infos только для агентов в obs
        red_step_sum  = float(sum(rews.get(a, 0.0) for a in self._agents_red))  # Python float
        blue_step_sum = float(sum(rews.get(a, 0.0) for a in self._agents_blue)) # Python float

        for aid in alive_agents:
            infos[aid] = {
                "invalid_target": self.count_invalid_target,
                "oob_move": self.count_oob_move,
                "oob_aim": self.count_oob_aim,
                "team_step_reward": red_step_sum if aid.startswith("red_") else blue_step_sum,
            }

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
        # Свои признаки
        self_vec = np.concatenate([
            self._pos[aid], 
            np.array([self._hp[aid] / 100.0], dtype=np.float32),
            self._vec(SELF_FEATS - 3)
        ], axis=0).astype(np.float32)

        # Союзники
        allies = self._ally_ids(aid)[:self.max_allies]
        allies_arr = np.zeros((self.max_allies, ALLY_FEATS), dtype=np.float32)
        allies_mask = np.zeros((self.max_allies,), dtype=np.int32)
        for i, al in enumerate(allies):
            alive = int(self._is_alive(al))
            allies_mask[i] = alive
            if alive:
                allies_arr[i, :2] = self._pos[al] - self._pos[aid]
                allies_arr[i, 2] = self._hp[al] / 100.0
                allies_arr[i, 3:] = self._vec(ALLY_FEATS - 3)

        # Противники + action_mask (куда можно целиться)
        enemies = self._enemy_ids(aid)[:self.max_enemies]
        enemies_arr = np.zeros((self.max_enemies, ENEMY_FEATS), dtype=np.float32)
        enemies_mask = np.zeros((self.max_enemies,), dtype=np.int32)
        enemy_action_mask = np.zeros((self.max_enemies,), dtype=np.int32)
        for j, en in enumerate(enemies):
            alive = int(self._is_alive(en))
            enemies_mask[j] = alive
            enemy_action_mask[j] = alive
            if alive:
                enemies_arr[j, :2] = self._pos[en] - self._pos[aid]
                enemies_arr[j, 2] = self._hp[en] / 100.0
                enemies_arr[j, 3:] = self._vec(ENEMY_FEATS - 3)

        # Глобальное состояние (для централизованного V)
        global_state = np.zeros((GLOBAL_FEATS,), dtype=np.float32)
        red_hp  = sum(max(0.0, self._hp[a]) for a in self._agents_red)
        blue_hp = sum(max(0.0, self._hp[a]) for a in self._agents_blue)
        red_center  = np.mean([self._pos[a] for a in self._agents_red], axis=0) if self._agents_red else np.zeros(2, np.float32)
        blue_center = np.mean([self._pos[a] for a in self._agents_blue], axis=0) if self._agents_blue else np.zeros(2, np.float32)

        global_state[0] = red_hp  / (100.0 * max(1, len(self._agents_red)))
        global_state[1] = blue_hp / (100.0 * max(1, len(self._agents_blue)))
        global_state[2:4] = red_center
        global_state[4:6] = blue_center
        global_state[6:] = self._vec(GLOBAL_FEATS - 6)

        # Клипинг всех значений к bounds [-10, 10]
        return {
            "self": np.clip(self_vec, -10.0, 10.0),
            "allies": np.clip(allies_arr, -10.0, 10.0),
            "allies_mask": allies_mask,
            "enemies": np.clip(enemies_arr, -10.0, 10.0),
            "enemies_mask": enemies_mask,
            "global_state": np.clip(global_state, -10.0, 10.0),
            "enemy_action_mask": enemy_action_mask,
        }