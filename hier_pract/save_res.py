"""
Система записи данных боев для последующей визуализации
Интегрируется с тренировкой и записывает траектории, действия, команды
"""

import json
import numpy as np
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import time
from collections import defaultdict

@dataclass
class RobotState:
    """Состояние робота в конкретный момент времени"""
    id: str
    team: str
    x: float
    y: float
    hp: float
    alive: bool
    target_enemy: int
    move_action: List[float]
    aim_action: List[float]
    fire_action: bool
    command_type: Optional[str] = None
    command_target: Optional[str] = None
    command_priority: Optional[int] = None

@dataclass
class BattleFrame:
    """Кадр боя с состояниями всех роботов"""
    timestamp: float
    step: int
    robots: List[RobotState]
    global_state: Dict[str, Any]
    events: List[Dict[str, Any]]  # Выстрелы, попадания, смерти

@dataclass
class BattleRecord:
    """Полная запись боя"""
    battle_id: str
    start_time: float
    end_time: float
    red_team_size: int
    blue_team_size: int
    winner: str
    frames: List[BattleFrame]
    final_stats: Dict[str, Any]

class BattleRecorder:
    """Класс для записи данных боев во время тренировки"""
    
    def __init__(self, output_dir: str = "./battle_recordings"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.current_battle: Optional[BattleRecord] = None
        self.current_frame_events: List[Dict[str, Any]] = []
        self.frame_counter = 0
        
        # Статистика для накопления
        self.battle_stats = defaultdict(list)
        
    def start_battle(self, battle_id: str, red_size: int, blue_size: int):
        """Начинает запись нового боя"""
        self.current_battle = BattleRecord(
            battle_id=battle_id,
            start_time=time.time(),
            end_time=0,
            red_team_size=red_size,
            blue_team_size=blue_size,
            winner="",
            frames=[],
            final_stats={}
        )
        self.current_frame_events = []
        self.frame_counter = 0
        print(f"🎬 Started recording battle: {battle_id}")
    
    def record_frame(self, 
                    observations: Dict[str, Dict],
                    actions: Dict[str, Dict],
                    rewards: Dict[str, float],
                    infos: Dict[str, Dict],
                    global_state: Optional[Dict] = None):
        """Записывает один кадр боя"""
        
        if not self.current_battle:
            return
        
        robots = []
        
        for agent_id, obs in observations.items():
            if agent_id.startswith(("red_", "blue_")):
                team = "red" if agent_id.startswith("red_") else "blue"
                
                # Извлекаем позицию из наблюдений (предполагаем что она в self[0:2])
                position = obs.get("self", np.zeros(12))[:2]
                hp_normalized = obs.get("self", np.zeros(12))[2] if len(obs.get("self", [])) > 2 else 1.0
                hp = hp_normalized * 100.0  # Денормализуем HP
                
                # Получаем действие
                action = actions.get(agent_id, {})
                target_enemy = action.get("target", 0)
                move_action = action.get("move", [0.0, 0.0])
                aim_action = action.get("aim", [0.0, 0.0])
                fire_action = action.get("fire", 0) > 0.5
                
                # Получаем команду если есть (для иерархической системы)
                info = infos.get(agent_id, {})
                command_type = info.get("command_type")
                command_target = info.get("command_target") 
                command_priority = info.get("command_priority")
                
                robot_state = RobotState(
                    id=agent_id,
                    team=team,
                    x=float(position[0]) * 30 + 300,  # Масштабируем для визуализации
                    y=float(position[1]) * 30 + 200,
                    hp=float(hp),
                    alive=hp > 0,
                    target_enemy=int(target_enemy),
                    move_action=list(move_action),
                    aim_action=list(aim_action),
                    fire_action=fire_action,
                    command_type=command_type,
                    command_target=command_target,
                    command_priority=command_priority
                )
                
                robots.append(robot_state)
                
                # Записываем события
                if fire_action:
                    self.current_frame_events.append({
                        "type": "fire",
                        "shooter": agent_id,
                        "target_index": target_enemy,
                        "timestamp": time.time()
                    })
                
                # Проверяем смерть (HP упало до 0 в этом кадре)
                if hp <= 0 and len(self.current_battle.frames) > 0:
                    # Проверяем был ли жив в предыдущем кадре
                    prev_frame = self.current_battle.frames[-1]
                    prev_robot = next((r for r in prev_frame.robots if r.id == agent_id), None)
                    if prev_robot and prev_robot.alive:
                        self.current_frame_events.append({
                            "type": "death",
                            "robot": agent_id,
                            "timestamp": time.time()
                        })
        
        # Создаем кадр
        frame = BattleFrame(
            timestamp=time.time(),
            step=self.frame_counter,
            robots=robots,
            global_state=global_state or {},
            events=self.current_frame_events.copy()
        )
        
        self.current_battle.frames.append(frame)
        self.current_frame_events = []
        self.frame_counter += 1
    
    def end_battle(self, winner: str, final_stats: Optional[Dict] = None):
        """Завершает запись боя и сохраняет"""
        if not self.current_battle:
            return
        
        self.current_battle.end_time = time.time()
        self.current_battle.winner = winner
        self.current_battle.final_stats = final_stats or {}
        
        # Рассчитываем дополнительную статистику
        self._calculate_battle_stats()
        
        # Сохраняем
        self._save_battle()
        
        print(f"🏁 Finished recording battle: {self.current_battle.battle_id}")
        print(f"   Duration: {self.current_battle.end_time - self.current_battle.start_time:.1f}s")
        print(f"   Frames: {len(self.current_battle.frames)}")
        print(f"   Winner: {winner}")
        
        # Добавляем в общую статистику
        self.battle_stats["durations"].append(
            self.current_battle.end_time - self.current_battle.start_time
        )
        self.battle_stats["winners"].append(winner)
        self.battle_stats["frame_counts"].append(len(self.current_battle.frames))
        
        self.current_battle = None
    
    def _calculate_battle_stats(self):
        """Рассчитывает дополнительную статистику боя"""
        if not self.current_battle or not self.current_battle.frames:
            return
        
        stats = {
            "total_shots": 0,
            "total_deaths": 0,
            "team_stats": {"red": {}, "blue": {}},
            "average_distance": 0,
            "action_distribution": defaultdict(int)
        }
        
        # Подсчитываем события
        for frame in self.current_battle.frames:
            for event in frame.events:
                if event["type"] == "fire":
                    stats["total_shots"] += 1
                elif event["type"] == "death":
                    stats["total_deaths"] += 1
        
        # Статистика по командам
        for team in ["red", "blue"]:
            team_robots = [r for r in self.current_battle.frames[-1].robots if r.team == team]
            alive_count = sum(1 for r in team_robots if r.alive)
            total_hp = sum(r.hp for r in team_robots if r.alive)
            
            stats["team_stats"][team] = {
                "alive_count": alive_count,
                "total_hp": total_hp,
                "initial_count": self.current_battle.red_team_size if team == "red" else self.current_battle.blue_team_size
            }
        
        # Средняя дистанция между командами
        if len(self.current_battle.frames) > 0:
            distances = []
            for frame in self.current_battle.frames[::5]:  # Каждый 5-й кадр для экономии
                red_robots = [r for r in frame.robots if r.team == "red" and r.alive]
                blue_robots = [r for r in frame.robots if r.team == "blue" and r.alive]
                
                for red in red_robots:
                    for blue in blue_robots:
                        dist = np.sqrt((red.x - blue.x)**2 + (red.y - blue.y)**2)
                        distances.append(dist)
            
            if distances:
                stats["average_distance"] = float(np.mean(distances))
        
        self.current_battle.final_stats.update(stats)
    
    def _save_battle(self):
        """Сохраняет запись боя в JSON файл"""
        if not self.current_battle:
            return
        
        filename = f"battle_{self.current_battle.battle_id}_{int(self.current_battle.start_time)}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # Конвертируем в словарь для JSON
        battle_dict = asdict(self.current_battle)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(battle_dict, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Saved battle recording: {filepath}")
    
    def export_for_web_visualizer(self, battle_files: Optional[List[str]] = None) -> str:
        """Экспортирует данные для веб-визуализатора"""
        
        if battle_files is None:
            # Берем все файлы из директории
            battle_files = [
                f for f in os.listdir(self.output_dir) 
                if f.startswith("battle_") and f.endswith(".json")
            ]
        
        if not battle_files:
            print("No battle files found for export")
            return ""
        
        # Берем последний файл по умолчанию
        latest_file = max(battle_files, key=lambda f: os.path.getmtime(os.path.join(self.output_dir, f)))
        filepath = os.path.join(self.output_dir, latest_file)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            battle_data = json.load(f)
        
        # Конвертируем в формат для веб-визуализатора
        web_data = {
            "battle_info": {
                "id": battle_data["battle_id"],
                "duration": battle_data["end_time"] - battle_data["start_time"],
                "winner": battle_data["winner"],
                "red_team_size": battle_data["red_team_size"],
                "blue_team_size": battle_data["blue_team_size"]
            },
            "frames": [],
            "statistics": battle_data.get("final_stats", {})
        }
        
        # Конвертируем кадры (можем взять каждый N-й для производительности)
        frame_step = max(1, len(battle_data["frames"]) // 1000)  # Максимум 1000 кадров
        
        for i, frame in enumerate(battle_data["frames"][::frame_step]):
            web_frame = {
                "timestamp": frame["timestamp"],
                "step": frame["step"],
                "robots": {},
                "events": frame["events"]
            }
            
            for robot in frame["robots"]:
                web_frame["robots"][robot["id"]] = {
                    "team": robot["team"],
                    "x": robot["x"],
                    "y": robot["y"],
                    "hp": robot["hp"],
                    "alive": robot["alive"],
                    "target": robot["target_enemy"],
                    "move": robot["move_action"],
                    "aim": robot["aim_action"],
                    "fire": robot["fire_action"],
                    "command": {
                        "type": robot.get("command_type"),
                        "target": robot.get("command_target"),
                        "priority": robot.get("command_priority", 1)
                    }
                }
            
            web_data["frames"].append(web_frame)
        
        # Сохраняем в формате для веб-визуализатора
        web_export_path = os.path.join(self.output_dir, "latest_battle_web.json")
        with open(web_export_path, 'w', encoding='utf-8') as f:
            json.dump(web_data, f, ensure_ascii=False, indent=2)
        
        print(f"🌐 Exported for web visualizer: {web_export_path}")
        
        # Также создаем HTML файл с встроенными данными
        html_path = self._create_standalone_html(web_data)
        return html_path
    
    def _create_standalone_html(self, battle_data: Dict) -> str:
        """Создает автономный HTML файл с данными боя"""
        
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Battle Replay: {battle_id}</title>
    <style>
        body {{ margin: 0; padding: 20px; font-family: Arial, sans-serif; background: #1a1a2e; color: white; }}
        .header {{ text-align: center; margin-bottom: 20px; }}
        .controls {{ text-align: center; margin-bottom: 20px; }}
        .arena {{ width: 600px; height: 400px; border: 2px solid #fff; margin: 0 auto; position: relative; background: #16213e; }}
        .robot {{ position: absolute; width: 16px; height: 16px; border-radius: 50%; border: 2px solid; transition: all 0.1s; }}
        .red-robot {{ background: #ff4444; border-color: #ff6666; }}
        .blue-robot {{ background: #4444ff; border-color: #6666ff; }}
        .robot.dead {{ opacity: 0.3; background: #666 !important; }}
        .stats {{ margin-top: 20px; text-align: center; }}
        button {{ padding: 10px 20px; margin: 5px; border: none; border-radius: 5px; background: #4CAF50; color: white; cursor: pointer; }}
        button:hover {{ background: #45a049; }}
        .timeline {{ width: 80%; margin: 20px auto; }}
        .timeline input {{ width: 100%; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎬 Battle Replay: {battle_id}</h1>
        <p>Winner: {winner} | Duration: {duration:.1f}s | Teams: {red_size}v{blue_size}</p>
    </div>
    
    <div class="controls">
        <button onclick="playPause()">⏯ Play/Pause</button>
        <button onclick="restart()">⏮ Restart</button>
        <button onclick="stepForward()">⏭ Step</button>
        <span>Speed: </span>
        <input type="range" id="speedSlider" min="0.1" max="3" value="1" step="0.1" onchange="updateSpeed()">
        <span id="speedDisplay">1.0x</span>
    </div>
    
    <div class="arena" id="arena"></div>
    
    <div class="timeline">
        <input type="range" id="frameSlider" min="0" max="{max_frame}" value="0" onchange="jumpToFrame()">
        <div>Frame: <span id="frameDisplay">0</span> / {max_frame}</div>
    </div>
    
    <div class="stats" id="stats">
        <div>Red HP: <span id="redHP">100</span> | Blue HP: <span id="blueHP">100</span></div>
    </div>

    <script>
        const battleData = {battle_data_json};
        
        let currentFrame = 0;
        let isPlaying = false;
        let playSpeed = 1.0;
        let playInterval = null;
        
        function initializeArena() {{
            const arena = document.getElementById('arena');
            arena.innerHTML = '';
            
            // Create robot elements based on first frame
            if (battleData.frames.length > 0) {{
                const firstFrame = battleData.frames[0];
                for (const robotId in firstFrame.robots) {{
                    const robot = firstFrame.robots[robotId];
                    const element = document.createElement('div');
                    element.id = robotId;
                    element.className = `robot ${{robot.team}}-robot`;
                    element.title = robotId;
                    arena.appendChild(element);
                }}
            }}
            
            updateFrame(0);
        }}
        
        function updateFrame(frameIndex) {{
            frameIndex = Math.max(0, Math.min(frameIndex, battleData.frames.length - 1));
            currentFrame = frameIndex;
            
            const frame = battleData.frames[frameIndex];
            
            // Update robot positions and states
            for (const robotId in frame.robots) {{
                const robot = frame.robots[robotId];
                const element = document.getElementById(robotId);
                if (element) {{
                    element.style.left = robot.x + 'px';
                    element.style.top = robot.y + 'px';
                    element.className = `robot ${{robot.team}}-robot${{!robot.alive ? ' dead' : ''}}`;
                    element.title = `${{robotId}}: HP=${{robot.hp.toFixed(1)}} ${{robot.command.type || ''}}`;
                }}
            }}
            
            // Update stats
            const redHP = Object.values(frame.robots)
                .filter(r => r.team === 'red' && r.alive)
                .reduce((sum, r) => sum + r.hp, 0);
            const blueHP = Object.values(frame.robots)
                .filter(r => r.team === 'blue' && r.alive)
                .reduce((sum, r) => sum + r.hp, 0);
            
            document.getElementById('redHP').textContent = redHP.toFixed(1);
            document.getElementById('blueHP').textContent = blueHP.toFixed(1);
            document.getElementById('frameDisplay').textContent = frameIndex;
            document.getElementById('frameSlider').value = frameIndex;
        }}
        
        function playPause() {{
            isPlaying = !isPlaying;
            if (isPlaying) {{
                playInterval = setInterval(() => {{
                    if (currentFrame >= battleData.frames.length - 1) {{
                        isPlaying = false;
                        clearInterval(playInterval);
                        return;
                    }}
                    updateFrame(currentFrame + 1);
                }}, 100 / playSpeed);
            }} else {{
                clearInterval(playInterval);
            }}
        }}
        
        function restart() {{
            isPlaying = false;
            clearInterval(playInterval);
            updateFrame(0);
        }}
        
        function stepForward() {{
            if (currentFrame < battleData.frames.length - 1) {{
                updateFrame(currentFrame + 1);
            }}
        }}
        
        function jumpToFrame() {{
            const frameIndex = parseInt(document.getElementById('frameSlider').value);
            updateFrame(frameIndex);
        }}
        
        function updateSpeed() {{
            playSpeed = parseFloat(document.getElementById('speedSlider').value);
            document.getElementById('speedDisplay').textContent = playSpeed.toFixed(1) + 'x';
        }}
        
        // Initialize on load
        window.addEventListener('load', initializeArena);
    </script>
</body>
</html>
        """.format(
            battle_id=battle_data["battle_info"]["id"],
            winner=battle_data["battle_info"]["winner"],
            duration=battle_data["battle_info"]["duration"],
            red_size=battle_data["battle_info"]["red_team_size"],
            blue_size=battle_data["battle_info"]["blue_team_size"],
            max_frame=len(battle_data["frames"]) - 1,
            battle_data_json=json.dumps(battle_data)
        )
        
        html_path = os.path.join(self.output_dir, f"replay_{battle_data['battle_info']['id']}.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        print(f"🎬 Created battle replay: {html_path}")
        return html_path
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Возвращает сводную статистику всех записанных боев"""
        if not self.battle_stats["durations"]:
            return {"message": "No battles recorded yet"}
        
        durations = np.array(self.battle_stats["durations"])
        winners = self.battle_stats["winners"]
        
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
        }


# Интеграция с Arena Environment
class RecordingArenaWrapper:
    """Обертка для ArenaEnv, которая автоматически записывает бои"""
    
    def __init__(self, base_env, recorder: BattleRecorder):
        self.env = base_env
        self.recorder = recorder
        self.battle_counter = 0
        self.current_observations = {}
        self.current_actions = {}
        
    def reset(self, **kwargs):
        # Завершаем предыдущий бой если был
        if self.recorder.current_battle is not None:
            # Определяем победителя по HP
            red_agents = [aid for aid in self.current_observations.keys() if aid.startswith("red_")]
            blue_agents = [aid for aid in self.current_observations.keys() if aid.startswith("blue_")]
            
            red_hp = sum(self.env._hp.get(aid, 0) for aid in red_agents if self.env._is_alive(aid))
            blue_hp = sum(self.env._hp.get(aid, 0) for aid in blue_agents if self.env._is_alive(aid))
            
            winner = "red" if red_hp > blue_hp else "blue" if blue_hp > red_hp else "draw"
            self.recorder.end_battle(winner)
        
        # Начинаем новый бой
        obs, info = self.env.reset(**kwargs)
        
        self.battle_counter += 1
        red_size = len([aid for aid in obs.keys() if aid.startswith("red_")])
        blue_size = len([aid for aid in obs.keys() if aid.startswith("blue_")])
        
        battle_id = f"battle_{self.battle_counter:04d}"
        self.recorder.start_battle(battle_id, red_size, blue_size)
        
        self.current_observations = obs
        self.current_actions = {}
        
        return obs, info
    
    def step(self, action_dict):
        self.current_actions = action_dict
        
        obs, rewards, terms, truncs, infos = self.env.step(action_dict)
        
        # Записываем кадр
        self.recorder.record_frame(
            observations=self.current_observations,
            actions=self.current_actions,
            rewards=rewards,
            infos=infos,
            global_state={
                "timestep": getattr(self.env, '_t', 0),
                "red_hp": sum(self.env._hp.get(aid, 0) for aid in self.env._agents_red),
                "blue_hp": sum(self.env._hp.get(aid, 0) for aid in self.env._agents_blue),
            }
        )
        
        self.current_observations = obs
        return obs, rewards, terms, truncs, infos
    
    def __getattr__(self, name):
        # Проксируем все остальные атрибуты к base env
        return getattr(self.env, name)


# Пример использования
def demo_battle_recording():
    """Демонстрация записи и визуализации боя"""
    
    # Создаем рекордер
    recorder = BattleRecorder("./demo_battles")
    
    # Создаем тестовое окружение
    from arena_env import ArenaEnv
    env = ArenaEnv({"ally_choices": [2], "enemy_choices": [2], "episode_len": 50})
    wrapped_env = RecordingArenaWrapper(env, recorder)
    
    print("🎮 Starting demo battle recording...")
    
    # Запускаем один бой
    obs, _ = wrapped_env.reset()
    
    for step in range(50):
        # Случайные действия для демонстрации
        actions = {}
        for agent_id, agent_obs in obs.items():
            actions[agent_id] = {
                "target": np.random.randint(0, env.max_enemies),
                "move": np.random.uniform(-0.5, 0.5, 2),
                "aim": np.random.uniform(-0.5, 0.5, 2),
                "fire": np.random.randint(0, 2),
            }
        
        obs, rewards, terms, truncs, infos = wrapped_env.step(actions)
        
        if terms.get("__all__") or truncs.get("__all__"):
            break
    
    # Завершаем и экспортируем
    obs, _ = wrapped_env.reset()  # Это завершит текущий бой
    
    # Экспортируем для веб-визуализатора
    html_path = recorder.export_for_web_visualizer()
    
    # Показываем статистику
    stats = recorder.get_summary_statistics()
    print("\n📊 Recording Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n🎬 Open {html_path} in your browser to view the battle replay!")
    
    return html_path


if __name__ == "__main__":
    demo_battle_recording()