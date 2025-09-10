# 🛠️ Руководство по модификации Arena системы

Полное руководство по изменению архитектуры, размеров действий, наблюдений и использованию системы обучения.

## 📋 Содержание

1. [Архитектура системы](#архитектура-системы)
2. [Изменение действий (Actions)](#изменение-действий-actions)
3. [Изменение наблюдений (Observations)](#изменение-наблюдений-observations)
4. [Модификация моделей](#модификация-моделей)
5. [Процесс обучения](#процесс-обучения)
6. [Использование системы](#использование-системы)

---

## 🏗️ Архитектура системы

### Компоненты системы:

```
Arena System
├── 🏟️ Environment (arena_env.py)
│   ├── ArenaEnv - основное окружение
│   └── Константы размеров и типов
├── 🧠 Models
│   ├── entity_attention_model.py - основная модель
│   ├── hierarchical_command_policy.py - командная система
│   └── masked_multihead_dist.py - дистрибуция действий
├── 🎯 Training (train_rllib_league.py)
│   ├── Конфигурация алгоритмов
│   ├── League система
│   └── Callbacks и ONNX экспорт
└── 🔮 Inference (universal_inference.py)
    ├── ONNX инференс
    ├── Ray инференс
    └── PyTorch инференс
```

### Связи между компонентами:

```
Environment → Models → Training → Export → Inference
     ↓           ↓         ↓         ↓        ↓
   obs_space   model    algorithm  onnx   prediction
  action_space arch     config     files   results
```

---

## 🎯 Изменение действий (Actions)

### Текущая структура действий:

```python
# В arena_env.py и masked_multihead_dist.py
action = {
    "target": Discrete(max_enemies),    # ID врага (0-5)
    "move": Box(-1, 1, (2,)),          # Движение [x, y]
    "aim": Box(-1, 1, (2,)),           # Прицел [x, y]  
    "fire": Discrete(2)                 # Стрелять (0/1)
}
```

### 🔧 Как изменить действия:

#### 1. **Добавить новое действие (например, "shield")**

**Файл: `arena_env.py`**
```python
# В ArenaEnv.__init__
self.single_act_space = spaces.Dict({
    "target": spaces.Discrete(self.max_enemies),
    "move": _box(-1, 1, (CONT_ACTION_MOVE,)),
    "aim": _box(-1, 1, (CONT_ACTION_AIM,)),
    "fire": spaces.Discrete(2),
    "shield": spaces.Discrete(2),  # ← НОВОЕ ДЕЙСТВИЕ
})

# В ArenaEnv.step() обработка нового действия
shield = int(act["shield"]) if np.isscalar(act["shield"]) else int(act["shield"].item())
# Логика щита...
```

**Файл: `masked_multihead_dist.py`**
```python
class MaskedTargetMoveAimFireShield(TorchDistributionWrapper):  # Новое имя!
    def __init__(self, inputs, model):
        super().__init__(inputs, model)
        
        # Добавляем размер для shield
        idx = 0
        logits_t = inputs[..., idx:idx+self.ne]; idx += self.ne
        mu_move  = inputs[..., idx:idx+2];        idx += 2
        logstd_mv= inputs[..., idx:idx+2];        idx += 2
        mu_aim   = inputs[..., idx:idx+2];        idx += 2
        logstd_am= inputs[..., idx:idx+2];        idx += 2
        logit_fr = inputs[..., idx:idx+1];        idx += 1
        logit_sh = inputs[..., idx:idx+1];        idx += 1  # ← НОВОЕ

        self.cat = Categorical(logits=logits_t)
        self.mv  = Normal(mu_move, logstd_mv.exp())
        self.am  = Normal(mu_aim,  logstd_am.exp())
        self.fr  = Bernoulli(logits=logit_fr)
        self.sh  = Bernoulli(logits=logit_sh)  # ← НОВОЕ
    
    @staticmethod
    def required_model_output_shape(action_space, model_config):
        max_enemies = model_config.get("custom_model_config", {}).get("max_enemies", 6)
        return max_enemies + 2 + 2 + 2 + 2 + 1 + 1  # +1 для shield
    
    def sample(self):
        t = self.cat.sample().unsqueeze(-1).float()
        mv = torch.tanh(self.mv.rsample())
        am = torch.tanh(self.am.rsample())
        fr = self.fr.sample().float()
        sh = self.sh.sample().float()  # ← НОВОЕ
        
        flat_action = torch.cat([t, mv, am, fr, sh], dim=-1)  # Добавили sh
        return self._convert_to_numpy_dict(flat_action)
    
    def _convert_to_numpy_dict(self, tensor_action):
        # Обновляем извлечение компонентов
        target = tensor_action[..., 0].long()
        move = tensor_action[..., 1:3]
        aim = tensor_action[..., 3:5]
        fire = tensor_action[..., 5].long()
        shield = tensor_action[..., 6].long()  # ← НОВОЕ
        
        result = {
            "target": target.numpy(),
            "move": move.numpy(),
            "aim": aim.numpy(),
            "fire": fire.numpy(),
            "shield": shield.numpy(),  # ← НОВОЕ
        }
        return result

# Регистрируем новую дистрибуцию
ModelCatalog.register_custom_action_dist("masked_multihead_shield", MaskedTargetMoveAimFireShield)
```

**Файл: `entity_attention_model.py`**
```python
class EntityAttentionModel(TorchModelV2, nn.Module):
    def __init__(self, ...):
        # Добавляем голову для щита
        self.head_shield_logit = MLP([d_model, hidden, 1])
    
    def forward(self, input_dict, state, seq_lens):
        # ... существующий код ...
        
        # Добавляем shield к выходу
        logit_shield = self.head_shield_logit(h)
        
        # Обновляем выход
        out = torch.cat([
            masked_logits, mu_move, log_std_move, 
            mu_aim, log_std_aim, logit_fire, 
            logit_shield  # ← НОВОЕ
        ], dim=-1)
        
        return out, state
```

**Файл: `train_rllib_league.py`**
```python
# Обновляем регистрацию дистрибуции
base_model_config = {
    "custom_model": "entity_attention",
    "custom_action_dist": "masked_multihead_shield",  # Новое имя
    # ... остальное
}
```

#### 2. **Изменить размер существующего действия**

Например, изменить движение с 2D на 3D:

**В `arena_env.py`:**
```python
CONT_ACTION_MOVE = 3  # Было 2

self.single_act_space = spaces.Dict({
    "move": _box(-1, 1, (CONT_ACTION_MOVE,)),  # Теперь 3D
    # ... остальное
})
```

**В `entity_attention_model.py`:**
```python
self.head_move_mu = MLP([d_model, hidden, 3])  # Было 2
self.log_std_move = nn.Parameter(torch.full((3,), -0.5))  # Было (2,)
```

**В `masked_multihead_dist.py`:**
```python
mu_move = inputs[..., idx:idx+3]     # Было idx+2
log_std_move = inputs[..., idx:idx+3] # Было idx+2
```

#### 3. **Полностью новая структура действий**

Например, RTS-стиль с построением юнитов:

```python
# Новая структура в arena_env.py
self.single_act_space = spaces.Dict({
    "unit_action": spaces.Discrete(5),  # MOVE, ATTACK, BUILD, GATHER, IDLE
    "target_position": _box(-10, 10, (2,)),
    "target_unit": spaces.Discrete(self.max_enemies + self.max_allies),
    "build_type": spaces.Discrete(4),   # WARRIOR, ARCHER, BUILDER, NONE
    "resource_allocation": _box(0, 1, (3,)),  # ATTACK, DEFENSE, ECONOMY
})
```

---

## 👁️ Изменение наблюдений (Observations)

### Текущая структура наблюдений:

```python
# В arena_env.py
obs = {
    "self": Box(-10, 10, (SELF_FEATS,)),           # 12 элементов
    "allies": Box(-10, 10, (MAX_ALLIES, 8)),       # 6x8 союзники  
    "enemies": Box(-10, 10, (MAX_ENEMIES, 10)),    # 6x10 враги
    "allies_mask": MultiBinary(MAX_ALLIES),        # 6 маска союзников
    "enemies_mask": MultiBinary(MAX_ENEMIES),      # 6 маска врагов
    "global_state": Box(-10, 10, (GLOBAL_FEATS,)), # 64 глобальное
    "enemy_action_mask": MultiBinary(MAX_ENEMIES)   # 6 маска действий
}
```

### 🔧 Как изменить наблюдения:

#### 1. **Добавить новый тип наблюдения**

Например, карту местности:

**В `arena_env.py`:**
```python
# Добавляем константы
MAP_SIZE = 32
TERRAIN_CHANNELS = 3  # высота, препятствия, ресурсы

# В ArenaEnv.__init__
self.single_obs_space = spaces.Dict({
    # ... существующие наблюдения ...
    "terrain_map": _box(0, 1, (TERRAIN_CHANNELS, MAP_SIZE, MAP_SIZE)),
    "minimap": _box(0, 1, (4, 16, 16)),  # 4 канала: союзники, враги, нейтралы, препятствия
})

# В _build_obs
def _build_obs(self, aid: str) -> Dict[str, np.ndarray]:
    # ... существующий код ...
    
    # Создаем карту местности
    terrain_map = np.zeros((TERRAIN_CHANNELS, MAP_SIZE, MAP_SIZE), dtype=np.float32)
    terrain_map[0] = self._generate_height_map()
    terrain_map[1] = self._generate_obstacles()
    terrain_map[2] = self._generate_resources()
    
    # Создаем миникарту
    minimap = self._create_minimap(aid)
    
    return {
        # ... существующие поля ...
        "terrain_map": terrain_map,
        "minimap": minimap,
    }
```

**В `entity_attention_model.py`:**
```python
class EntityAttentionModel(TorchModelV2, nn.Module):
    def __init__(self, ...):
        # Добавляем CNN для обработки карт
        self.terrain_cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 16, d_model // 4)
        )
        
        self.minimap_cnn = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(16 * 16, d_model // 4)
        )
        
        # Обновляем размер интегратора
        integrated_size = d_model//2 + d_model//4 + d_model//4 + d_model//4 + d_model//4
    
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        
        # ... существующий код для обработки агентов ...
        
        # Обрабатываем карты
        terrain_features = self.terrain_cnn(obs["terrain_map"])
        minimap_features = self.minimap_cnn(obs["minimap"])
        
        # Интегрируем все признаки
        integrated = torch.cat([
            global_encoded, allies_aggregated, enemies_aggregated,
            terrain_features, minimap_features
        ], dim=-1)
        
        # ... остальной код ...
```

#### 2. **Изменить размер существующих наблюдений**

Например, увеличить информацию о союзниках:

**В `arena_env.py`:**
```python
ALLY_FEATS = 12  # Было 8
ENEMY_FEATS = 15  # Было 10

# В _build_obs обновляем заполнение
def _build_obs(self, aid: str) -> Dict[str, np.ndarray]:
    # Для союзников добавляем больше информации
    allies_arr[i, :2] = self._pos[al] - self._pos[aid]  # относительная позиция
    allies_arr[i, 2] = self._hp[al] / 100.0             # HP
    allies_arr[i, 3] = self._get_energy(al) / 100.0     # энергия
    allies_arr[i, 4] = self._get_ammo(al) / 50.0        # патроны
    allies_arr[i, 5:7] = self._get_velocity(al)         # скорость
    allies_arr[i, 7:9] = self._get_aim_direction(al)    # направление прицела
    allies_arr[i, 9] = self._get_reload_time(al)        # время перезарядки
    allies_arr[i, 10] = self._get_shield_strength(al)   # прочность щита
    allies_arr[i, 11] = self._get_experience(al)        # опыт
```

#### 3. **Полностью новая структура наблюдений**

RTS-стиль с экономикой:

```python
# В arena_env.py
self.single_obs_space = spaces.Dict({
    "player_resources": _box(0, 1000, (4,)),           # золото, дерево, камень, еда
    "population": _box(0, 200, (2,)),                  # текущее, максимальное
    "buildings": _box(0, 1, (10, 6)),                  # 10 зданий x 6 параметров
    "buildings_mask": spaces.MultiBinary(10),
    "units": _box(-10, 10, (50, 8)),                   # 50 юнитов x 8 параметров
    "units_mask": spaces.MultiBinary(50),
    "research_progress": _box(0, 1, (20,)),            # прогресс исследований
    "map_control": _box(0, 1, (8, 8)),                 # контроль территории
})
```

---

## 🧠 Модификация моделей

### Изменение архитектуры модели:

#### 1. **Добавить новые слои**

```python
# В entity_attention_model.py
class EntityAttentionModel(TorchModelV2, nn.Module):
    def __init__(self, ...):
        # Добавляем дополнительные слои
        self.tactical_analyzer = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(0.1)
        )
        
        # Добавляем механизм внимания между командами
        self.team_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            batch_first=True
        )
    
    def forward(self, input_dict, state, seq_lens):
        # ... существующий код ...
        
        # Тактический анализ
        tactical_features = self.tactical_analyzer(h)
        
        # Внимание между командами
        team_features, _ = self.team_attention(
            tactical_features.unsqueeze(1),
            tactical_features.unsqueeze(1), 
            tactical_features.unsqueeze(1)
        )
        
        # Объединяем признаки
        h = h + tactical_features + team_features.squeeze(1)
        
        # ... остальной код ...
```

#### 2. **Изменить размеры модели**

```python
# В train_rllib_league.py
base_model_config = {
    "custom_model": "entity_attention",
    "custom_model_config": {
        "d_model": 256,        # Было 128
        "nhead": 16,           # Было 8  
        "layers": 4,           # Было 2
        "ff": 1024,            # Было 256
        "hidden": 512,         # Было 256
        # ... остальное
    }
}
```

#### 3. **Новая архитектура модели**

Например, CNN + RNN модель:

```python
# Новый файл: cnn_rnn_model.py
class CNNRNNModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        
        cfg = model_config.get("custom_model_config", {})
        
        # CNN для пространственных данных
        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten()
        )
        
        # RNN для временной последовательности
        self.temporal_rnn = nn.LSTM(
            input_size=64 * 64 + 32,  # CNN features + vector features
            hidden_size=256,
            num_layers=2,
            batch_first=True
        )
        
        # Выходные головы
        self.action_heads = nn.ModuleDict({
            "target": nn.Linear(256, 6),
            "move": nn.Linear(256, 4),  # mu_x, std_x, mu_y, std_y
            "fire": nn.Linear(256, 1)
        })
        
        self.value_head = nn.Linear(256, 1)
    
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        
        # Пространственные признаки через CNN
        spatial_features = self.spatial_cnn(obs["minimap"])
        
        # Векторные признаки
        vector_features = torch.cat([
            obs["self"], 
            obs["allies"].flatten(1),
            obs["enemies"].flatten(1)
        ], dim=1)
        
        # Объединяем для RNN
        rnn_input = torch.cat([spatial_features, vector_features], dim=1)
        rnn_input = rnn_input.unsqueeze(1)  # Добавляем временное измерение
        
        # RNN обработка
        rnn_out, new_state = self.temporal_rnn(rnn_input, state)
        features = rnn_out.squeeze(1)
        
        # Выходы
        target_logits = self.action_heads["target"](features)
        move_params = self.action_heads["move"](features)
        fire_logits = self.action_heads["fire"](features)
        
        # Применяем маски
        mask = obs["enemy_action_mask"].float()
        target_logits = target_logits + (1.0 - mask) * (-1e9)
        
        output = torch.cat([target_logits, move_params, fire_logits], dim=-1)
        
        self._value_out = self.value_head(features).squeeze(-1)
        
        return output, new_state
    
    def get_initial_state(self):
        # Начальное состояние для LSTM
        return [
            torch.zeros(2, 1, 256),  # hidden states
            torch.zeros(2, 1, 256)   # cell states
        ]

# Регистрируем новую модель
ModelCatalog.register_custom_model("cnn_rnn", CNNRNNModel)
```

---

## 🎓 Процесс обучения

### Полный цикл обучения:

```
1. Подготовка окружения
   ↓
2. Конфигурация модели
   ↓  
3. Настройка алгоритма
   ↓
4. Тренировка с League
   ↓
5. Экспорт ONNX моделей
   ↓
6. Валидация и тестирование
```

### 🚀 Пошаговое использование:

#### 1. **Быстрый старт (5 минут)**

```bash
# Клонируйте код и установите зависимости
pip install torch ray[rllib] onnxruntime gymnasium numpy

# Быстрый тест системы
python updated_training_script.py --test

# Если тест прошел успешно, запустите обучение
python updated_training_script.py --iterations 100 --algo gspo
```

#### 2. **Стандартное обучение**

```bash
# Обычная тренировка с ONNX экспортом
python updated_training_script.py \
    --iterations 1000 \
    --algo gspo \
    --export-onnx \
    --record-battles \
    --export-every 50

# Иерархическая тренировка
python updated_training_script.py \
    --hierarchical \
    --iterations 500 \
    --algo ppo
```

#### 3. **Продвинутая настройка**

```python
# Создайте custom_config.py
from updated_training_script import *

# Кастомная конфигурация
args = argparse.Namespace(
    iterations=2000,
    algo="grpo",
    hierarchical=True,
    max_allies=8,           # Больше союзников
    max_enemies=8,          # Больше врагов  
    episode_len=256,        # Длиннее эпизоды
    export_onnx=True,
    export_every=25,
    record_battles=True,
    num_workers=8,          # Больше воркеров
    train_batch_size=32768, # Больший батч
)

# Запуск с кастомными настройками
main_with_args(args)
```

#### 4. **Мониторинг обучения**

```bash
# В отдельном терминале запустите TensorBoard
tensorboard --logdir ./logs

# Откройте http://localhost:6006 для просмотра метрик:
# - Episode reward mean  
# - League TrueSkill рейтинги
# - Invalid actions статистика
# - Curriculum прогресс
```

#### 5. **Тестирование результатов**

```bash
# Автоматический поиск и тест моделей
python quick_inference.py interactive

# Сравнение чекпоинтов
python quick_inference.py compare checkpoints/checkpoint_*

# Визуализация боев
# Откройте ./battle_recordings/replay_*.html в браузере
```

### 🔧 Настройка параметров обучения:

#### Алгоритмы:
- **PPO**: Стандартный, стабильный
- **GSPO**: Групповые advantages (рекомендуется)
- **GRPO**: Групповые returns, более агрессивный

#### Curriculum (автоматическое усложнение):
```python
curriculum_schedule=[
    (0, [1], [1]),                    # Начинаем с 1v1
    (2_000_000, [1, 2], [1, 2]),      # Переходим к 1v1, 2v2
    (8_000_000, [1, 2, 3], [1, 2, 3]), # Финальные размеры команд
]
```

#### League (оппоненты):
```python
opponents=6,              # Количество оппонентов
eval_episodes=4,          # Эпизодов для оценки
clone_every=15,           # Клонировать каждые N итераций
```

---

## 🎮 Использование системы

### Структура проекта после обучения:

```
your_project/
├── 📁 checkpoints/           # Ray чекпоинты
│   ├── checkpoint_000100/
│   └── checkpoint_000500/
├── 📁 onnx_exports/          # ONNX модели
│   ├── latest/               # Последние модели
│   │   ├── policy_main.onnx
│   │   └── policy_main_meta.json
│   └── iter_000500/
├── 📁 battle_recordings/     # Записи боев
│   ├── battle_0001.json
│   └── replay_battle_0001.html
├── 📁 logs/                  # TensorBoard логи
└── 📁 models/               # Исходный код моделей
```

### 🔮 Инференс в продакшене:

#### 1. **Простой случай**
```python
from universal_inference import load_latest_models, create_sample_input

# Автозагрузка последних моделей
manager = load_latest_models("./onnx_exports")

# В игровом цикле
def get_action_for_robot(robot):
    input_data = InferenceInput(
        self_features=robot.get_observation(),
        allies=[ally.get_observation() for ally in robot.allies],
        enemies=[enemy.get_observation() for enemy in robot.visible_enemies],
        global_state=game.get_global_state()
    )
    
    result = manager.predict("main", input_data)
    return result.action_dict
```

#### 2. **Иерархическая система**
```python
# 1. Командир анализирует ситуацию
commander_input = InferenceInput(
    self_features=np.zeros(12),  # Командир не имеет self
    allies=[robot.get_observation() for robot in team.robots],
    enemies=[enemy.get_observation() for enemy in visible_enemies],
    global_state=game.get_global_state()
)

commander_result = manager.predict("commander", commander_input)

# 2. Раздаем команды исполнителям
for i, robot in enumerate(team.robots):
    command = create_command_for_follower(commander_result.action_dict, i)
    
    follower_input = InferenceInput(
        self_features=robot.get_observation(),
        allies=[other.get_observation() for other in team.robots if other != robot],
        enemies=[enemy.get_observation() for enemy in robot.visible_enemies],
        global_state=game.get_global_state(),
        command=command  # Команда от командира
    )
    
    action = manager.predict("follower", follower_input)
    robot.execute_action(action.action_dict)
```

#### 3. **Интеграция с Unity/Unreal**
```csharp
// C# wrapper для Unity
using System;
using System.Diagnostics;

public class ArenaAIController : MonoBehaviour 
{
    private Process pythonProcess;
    
    void Start() 
    {
        // Запускаем Python процесс с моделью
        pythonProcess = new Process();
        pythonProcess.StartInfo.FileName = "python";
        pythonProcess.StartInfo.Arguments = "unity_inference_server.py";
        pythonProcess.StartInfo.UseShellExecute = false;
        pythonProcess.StartInfo.RedirectStandardInput = true;
        pythonProcess.StartInfo.RedirectStandardOutput = true;
        pythonProcess.Start();
    }
    
    public RobotAction GetAction(RobotState state)
    {
        // Сериализуем состояние
        string jsonState = JsonUtility.ToJson(state);
        
        // Отправляем в Python
        pythonProcess.StandardInput.WriteLine(jsonState);
        
        // Получаем ответ
        string jsonAction = pythonProcess.StandardOutput.ReadLine();
        
        // Десериализуем действие
        return JsonUtility.FromJson<RobotAction>(jsonAction);
    }
}
```

### 📊 Мониторинг и отладка:

#### 1. **Проверка качества моделей**
```python
# Скрипт для анализа качества: analyze_models.py
from universal_inference import *
import matplotlib.pyplot as plt

def analyze_model_performance(model_path):
    manager = UniversalInferenceManager()
    model_id = manager.load_model(model_path)
    
    # Тестируем на разных сценариях
    scenarios = {
        "easy": create_sample_input("simple"),
        "medium": create_sample_input("complex"), 
        "hard": create_custom_scenario(allies=1, enemies=4)
    }
    
    results = {}
    for name, input_data in scenarios.items():
        # Множественные прогоны для стабильности
        actions = []
        times = []
        
        for _ in range(100):
            start = time.time()
            result = manager.predict(model_id, input_data)
            times.append((time.time() - start) * 1000)
            actions.append(result.action_dict)
        
        results[name] = {
            "avg_time": np.mean(times),
            "std_time": np.std(times),
            "action_diversity": calculate_action_diversity(actions),
            "fire_rate": np.mean([a["fire"] for a in actions]),
            "avg_target": np.mean([a["target"] for a in actions])
        }
    
    return results

def calculate_action_diversity(actions):
    """Измеряет разнообразие действий (хорошо для избежания застревания)"""
    targets = [a["target"] for a in actions]
    moves = [tuple(a["move"]) for a in actions]
    
    target_entropy = -sum(p * np.log(p) for p in np.bincount(targets) / len(targets) if p > 0)
    move_diversity = len(set(moves)) / len(moves)
    
    return {"target_entropy": target_entropy, "move_diversity": move_diversity}
```

#### 2. **Визуализация поведения**
```python
# Создание heatmap действий
def visualize_model_behavior(model_path, output_dir="./analysis"):
    os.makedirs(output_dir, exist_ok=True)
    
    manager = UniversalInferenceManager()
    model_id = manager.load_model(model_path)
    
    # Создаем grid ситуаций
    positions = np.linspace(-5, 5, 20)
    heatmap = np.zeros((20, 20))
    
    for i, x in enumerate(positions):
        for j, y in enumerate(positions):
            # Враг в позиции (x, y)
            input_data = InferenceInput(
                self_features=np.array([0, 0, 1] + [0]*9, dtype=np.float32),
                allies=[],
                enemies=[np.array([x, y, 1] + [0]*7, dtype=np.float32)],
                global_state=np.zeros(64, dtype=np.float32)
            )
            
            result = manager.predict(model_id, input_data)
            heatmap[i, j] = 1.0 if result.fire else 0.0
    
    # Сохраняем heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap, extent=[-5, 5, -5, 5], origin='lower', cmap='hot')
    plt.colorbar(label='Fire Probability')
    plt.xlabel('Enemy X Position')
    plt.ylabel('Enemy Y Position')
    plt.title(f'Fire Decision Heatmap - {os.path.basename(model_path)}')
    plt.savefig(f"{output_dir}/fire_heatmap.png", dpi=300, bbox_inches='tight')
    
    print(f"Heatmap saved to {output_dir}/fire_heatmap.png")
```

#### 3. **A/B тестирование моделей**
```python
def ab_test_models(model_a_path, model_b_path, num_battles=100):
    """Сравнивает две модели в прямых боях"""
    
    # Создаем временное окружение для тестирования
    from arena_env import ArenaEnv
    env = ArenaEnv({"ally_choices": [2], "enemy_choices": [2], "episode_len": 128})
    
    manager = UniversalInferenceManager()
    model_a = manager.load_model(model_a_path, "model_a")
    model_b = manager.load_model(model_b_path, "model_b")
    
    wins_a, wins_b, draws = 0, 0, 0
    
    for battle in range(num_battles):
        obs, _ = env.reset()
        done = False
        
        while not done:
            actions = {}
            
            for agent_id, agent_obs in obs.items():
                # Определяем какую модель использовать
                if agent_id.startswith("red_"):
                    model_id = "model_a"
                else:
                    model_id = "model_b"
                
                # Конвертируем obs в InferenceInput
                input_data = obs_to_inference_input(agent_obs, env)
                result = manager.predict(model_id, input_data)
                actions[agent_id] = result.action_dict
            
            obs, rewards, terms, truncs, infos = env.step(actions)
            done = terms.get("__all__") or truncs.get("__all__")
        
        # Определяем победителя
        red_total = sum(r for aid, r in rewards.items() if aid.startswith("red_"))
        blue_total = sum(r for aid, r in rewards.items() if aid.startswith("blue_"))
        
        if red_total > blue_total:
            wins_a += 1
        elif blue_total > red_total:
            wins_b += 1
        else:
            draws += 1
        
        if battle % 10 == 0:
            print(f"Battle {battle}: A={wins_a}, B={wins_b}, Draws={draws}")
    
    print(f"\nFinal Results:")
    print(f"Model A wins: {wins_a} ({wins_a/num_battles*100:.1f}%)")
    print(f"Model B wins: {wins_b} ({wins_b/num_battles*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_battles*100:.1f}%)")
    
    return {"wins_a": wins_a, "wins_b": wins_b, "draws": draws}

def obs_to_inference_input(obs, env):
    """Конвертирует Ray observation в InferenceInput"""
    return InferenceInput(
        self_features=obs["self"],
        allies=[obs["allies"][i] for i in range(len(obs["allies"])) if obs["allies_mask"][i]],
        enemies=[obs["enemies"][i] for i in range(len(obs["enemies"])) if obs["enemies_mask"][i]],
        global_state=obs["global_state"]
    )
```

---

## 🔄 Типичные workflow'ы

### 1. **Разработка новой игровой механики**

```bash
# 1. Изменяем окружение
edit arena_env.py  # Добавляем новые действия/наблюдения

# 2. Обновляем модель
edit entity_attention_model.py  # Новые головы/слои

# 3. Обновляем дистрибуцию действий  
edit masked_multihead_dist.py  # Новые компоненты действий

# 4. Тестируем изменения
python updated_training_script.py --test

# 5. Короткая тренировка для проверки
python updated_training_script.py --iterations 50

# 6. Полная тренировка если все работает
python updated_training_script.py --iterations 1000
```

### 2. **Оптимизация производительности**

```bash
# 1. Профилирование текущей модели
python -m cProfile -o profile.stats updated_training_script.py --iterations 10

# 2. Анализ узких мест
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
"

# 3. Оптимизация (уменьшение модели, ONNX экспорт)
python updated_training_script.py --export-onnx --export-every 5

# 4. Бенчмарк ONNX vs Ray
python quick_inference.py batch

# 5. Выбор оптимальной конфигурации
```

### 3. **Добавление новых алгоритмов**

```python
# Файл: my_custom_algorithm.py
from gspo_grpo_policy import GSPOTorchPolicy
from ray.rllib.policy.sample_batch import SampleBatch

class MyCustomPolicy(GSPOTorchPolicy):
    """Кастомная политика с новой логикой"""
    
    def postprocess_trajectory(self, sample_batch: SampleBatch, 
                             other_agent_batches=None, episode=None):
        
        # Сначала базовый postprocessing
        sample_batch = super().postprocess_trajectory(
            sample_batch, other_agent_batches, episode)
        
        # Добавляем свою логику
        rewards = sample_batch["rewards"]
        
        # Пример: бонус за координацию
        if len(other_agent_batches) > 0:
            coordination_bonus = self.calculate_coordination_bonus(
                sample_batch, other_agent_batches)
            rewards = rewards + coordination_bonus
            sample_batch["rewards"] = rewards
        
        return sample_batch
    
    def calculate_coordination_bonus(self, sample_batch, other_batches):
        # Ваша логика подсчета бонуса за координацию
        return np.zeros_like(sample_batch["rewards"])

# В train_rllib_league.py добавляем:
def get_policy_class(algo_name):
    if algo_name == "my_custom":
        return MyCustomPolicy
    elif algo_name == "gspo":
        return GSPOTorchPolicy
    # ... остальное
```

### 4. **Создание турнира моделей**

```python
# tournament.py
def run_tournament(model_directory="./checkpoints"):
    import glob
    from itertools import combinations
    
    # Находим все чекпоинты
    checkpoints = sorted(glob.glob(f"{model_directory}/checkpoint_*"))
    
    # Создаем турнирную таблицу
    results = {}
    
    for model_a, model_b in combinations(checkpoints, 2):
        print(f"Battle: {os.path.basename(model_a)} vs {os.path.basename(model_b)}")
        
        result = ab_test_models(model_a, model_b, num_battles=20)
        
        results[(model_a, model_b)] = result
    
    # Подсчитываем рейтинги
    elo_ratings = calculate_elo_ratings(results)
    
    # Сохраняем результаты
    with open("tournament_results.json", "w") as f:
        json.dump({
            "matches": {f"{a}_vs_{b}": result for (a,b), result in results.items()},
            "elo_ratings": elo_ratings
        }, f, indent=2)
    
    print("Tournament completed! See tournament_results.json")
    
    return elo_ratings

def calculate_elo_ratings(match_results):
    """Простой расчет ELO рейтингов"""
    models = set()
    for (a, b) in match_results.keys():
        models.add(a)
        models.add(b)
    
    ratings = {model: 1000.0 for model in models}
    
    for (model_a, model_b), result in match_results.items():
        wins_a = result["wins_a"]
        wins_b = result["wins_b"]
        total = wins_a + wins_b + result["draws"]
        
        if total > 0:
            score_a = (wins_a + 0.5 * result["draws"]) / total
            
            # ELO update
            expected_a = 1 / (1 + 10 ** ((ratings[model_b] - ratings[model_a]) / 400))
            ratings[model_a] += 32 * (score_a - expected_a)
            ratings[model_b] += 32 * ((1 - score_a) - (1 - expected_a))
    
    return sorted(ratings.items(), key=lambda x: x[1], reverse=True)
```

---

## ⚠️ Важные особенности и подводные камни

### 1. **Совместимость версий**
```python
# Проверка совместимости в начале скрипта
def check_compatibility():
    import ray
    import torch
    
    if ray.__version__.startswith("2.4"):
        print("✅ Ray 2.4.x detected - compatible")
    else:
        print(f"⚠️ Ray {ray.__version__} - may have compatibility issues")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("ℹ️ Using CPU training (slower)")
```

### 2. **Размеры батчей и память**
```python
# Автоматический расчет оптимальных размеров
def calculate_optimal_batch_size():
    available_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 8e9
    
    # Примерный расход памяти на один sample
    memory_per_sample = 1024  # bytes
    
    optimal_batch = int(available_memory * 0.7 / memory_per_sample / 8)  # 70% памяти, запас x8
    optimal_batch = max(512, min(optimal_batch, 32768))  # Ограничиваем разумными пределами
    
    return {
        "train_batch_size": optimal_batch,
        "minibatch_size": optimal_batch // 8
    }
```

### 3. **Отладка проблем обучения**
```python
# Скрипт диагностики: debug_training.py
def diagnose_training_issues(checkpoint_dir="./checkpoints"):
    """Анализирует проблемы обучения"""
    
    import glob
    
    checkpoints = sorted(glob.glob(f"{checkpoint_dir}/checkpoint_*"))
    
    if len(checkpoints) < 2:
        print("❌ Need at least 2 checkpoints for analysis")
        return
    
    print("🔍 Analyzing training progression...")
    
    # Загружаем несколько чекпоинтов
    performance_trend = []
    
    for i, checkpoint in enumerate(checkpoints[-5:]):  # Последние 5
        try:
            # Тестируем производительность
            manager = UniversalInferenceManager()
            model_id = manager.load_model(checkpoint, f"model_{i}", "ray")
            
            # Простой тест
            input_data = create_sample_input("simple")
            result = manager.predict(model_id, input_data)
            
            # Оцениваем "разумность" действий
            reasonableness = evaluate_action_reasonableness(result)
            performance_trend.append(reasonableness)
            
        except Exception as e:
            print(f"⚠️ Failed to load {checkpoint}: {e}")
    
    # Анализируем тренд
    if len(performance_trend) >= 3:
        if performance_trend[-1] < performance_trend[0]:
            print("📉 Performance degrading - possible overfitting")
        elif performance_trend[-1] > performance_trend[-3]:
            print("📈 Performance improving - training healthy")
        else:
            print("📊 Performance stable - consider adjusting learning rate")
    
    return performance_trend

def evaluate_action_reasonableness(result):
    """Простая эвристика разумности действий"""
    score = 0.0
    
    # Проверяем что цель в разумном диапазоне
    if 0 <= result.target <= 5:
        score += 1.0
    
    # Проверяем что движение не слишком экстремальное  
    move_magnitude = np.linalg.norm(result.move)
    if move_magnitude < 1.5:  # Разумное движение
        score += 1.0
    
    # Проверяем что прицел тоже разумный
    aim_magnitude = np.linalg.norm(result.aim)
    if aim_magnitude < 1.5:
        score += 1.0
    
    # Бонус за не слишком агрессивное поведение
    if not result.fire or move_magnitude > 0.1:  # Двигается ИЛИ не стреляет
        score += 0.5
    
    return score / 3.5  # Нормализуем к [0, 1]
```

---

## 🎯 Заключение

Эта система предоставляет **полную гибкость** для:

- ✅ **Изменения действий**: Добавление/удаление/модификация любых компонентов
- ✅ **Изменения наблюдений**: Поддержка любых типов входных данных  
- ✅ **Модификации моделей**: От простых изменений до полностью новых архитектур
- ✅ **Экспериментов с алгоритмами**: Легкое добавление новых подходов
- ✅ **Продакшен деплоя**: ONNX экспорт для максимальной производительности

### 🚀 Следующие шаги:

1. **Начните с простого**: Запустите `--test` режим
2. **Поэкспериментируйте**: Измените одно действие/наблюдение
3. **Обучите модель**: Короткая тренировка для проверки
4. **Оптимизируйте**: Экспорт в ONNX и бенчмарк
5. **Масштабируйте**: Полная тренировка и деплой

**Вся система спроектирована модульно** - изменения в одном компоненте минимально влияют на остальные. Используйте готовые примеры как отправную точку и адаптируйте под ваши нужды! 🤖⚔️