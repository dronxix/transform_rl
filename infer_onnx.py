"""
Исправленный ONNX инференс с правильным чтением мета-файлов
"""

import numpy as np
import onnxruntime as ort
import json
import os
import glob
from typing import Dict, List, Optional, Tuple

class ONNXInferenceEngine:
    """Движок инференса для ONNX политик"""
    
    def __init__(self, onnx_path: str, meta_path: Optional[str] = None):
        self.onnx_path = onnx_path
        self.meta_path = meta_path or self._find_meta_file(onnx_path)
        
        # Загружаем метаданные
        self.meta = self._load_meta()
        
        # Создаем ONNX сессию
        self.session = ort.InferenceSession(
            onnx_path, 
            providers=["CPUExecutionProvider"]
        )
        
        # Получаем информацию о входах и выходах
        self.input_info = {inp.name: inp for inp in self.session.get_inputs()}
        self.output_info = {out.name: out for out in self.session.get_outputs()}
        
        print(f"✓ Loaded ONNX model: {os.path.basename(onnx_path)}")
        print(f"  Policy: {self.meta.get('policy_id', 'unknown')}")
        print(f"  Max enemies: {self.meta['max_enemies']}, Max allies: {self.meta['max_allies']}")
        print(f"  Inputs: {list(self.input_info.keys())}")
        print(f"  Outputs: {list(self.output_info.keys())}")
    
    def _find_meta_file(self, onnx_path: str) -> Optional[str]:
        """Ищет мета-файл рядом с ONNX файлом"""
        onnx_dir = os.path.dirname(onnx_path)
        onnx_name = os.path.basename(onnx_path)
        
        # Пытаемся найти соответствующий мета-файл
        possible_names = [
            onnx_name.replace('.onnx', '_meta.json'),
            onnx_name.replace('.onnx', '.json'),
            'meta.json',
            'export_summary.json'
        ]
        
        for name in possible_names:
            meta_path = os.path.join(onnx_dir, name)
            if os.path.exists(meta_path):
                return meta_path
        
        print(f"Warning: No meta file found for {onnx_path}")
        return None
    
    def _load_meta(self) -> Dict:
        """Загружает метаданные или создает значения по умолчанию"""
        if self.meta_path and os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                
                # Если это export_summary.json, извлекаем первую политику
                if 'exported_policies' in meta and meta['exported_policies']:
                    policy_meta = meta['exported_policies'][0]
                    env_info = meta.get('environment_info', {})
                    
                    return {
                        "policy_id": policy_meta.get('policy_id', 'unknown'),
                        "max_allies": env_info.get('max_allies', 6),
                        "max_enemies": env_info.get('max_enemies', 6),
                        "obs_dims": env_info.get('obs_dims', {
                            "self": 12, "ally": 8, "enemy": 10, "global_state": 64
                        }),
                        "action_spec": {
                            "target_discrete_n": env_info.get('max_enemies', 6),
                            "move_box": 2,
                            "aim_box": 2,
                            "fire_binary": 1,
                            "total_action_dim": 6
                        }
                    }
                else:
                    return meta
                    
            except Exception as e:
                print(f"Error loading meta file {self.meta_path}: {e}")
        
        # Значения по умолчанию
        print("Using default meta values")
        return {
            "policy_id": "unknown",
            "max_allies": 6,
            "max_enemies": 6,
            "obs_dims": {
                "self": 12,
                "ally": 8,
                "enemy": 10,
                "global_state": 64,
            },
            "action_spec": {
                "target_discrete_n": 6,
                "move_box": 2,
                "aim_box": 2,
                "fire_binary": 1,
                "total_action_dim": 6
            }
        }
    
    def prepare_observation(self, 
                          self_vec: np.ndarray,
                          allies: List[np.ndarray], 
                          enemies: List[np.ndarray],
                          batch_size: int = 1) -> Dict[str, np.ndarray]:
        """Подготавливает наблюдения для инференса"""
        
        max_allies = self.meta["max_allies"]
        max_enemies = self.meta["max_enemies"]
        obs_dims = self.meta["obs_dims"]
        
        # Подготавливаем массивы
        self_arr = np.zeros((batch_size, obs_dims["self"]), dtype=np.float32)
        allies_arr = np.zeros((batch_size, max_allies, obs_dims["ally"]), dtype=np.float32)
        allies_mask = np.zeros((batch_size, max_allies), dtype=np.int32)
        enemies_arr = np.zeros((batch_size, max_enemies, obs_dims["enemy"]), dtype=np.float32)
        enemies_mask = np.zeros((batch_size, max_enemies), dtype=np.int32)
        enemy_action_mask = np.zeros((batch_size, max_enemies), dtype=np.int32)
        
        for b in range(batch_size):
            # Self
            if len(self_vec.shape) == 1:
                self_arr[b] = self_vec
            else:
                self_arr[b] = self_vec[b]
            
            # Allies
            n_allies = min(len(allies), max_allies)
            if n_allies > 0:
                for i in range(n_allies):
                    ally_vec = allies[i] if len(allies[i].shape) == 1 else allies[i][b]
                    allies_arr[b, i, :len(ally_vec)] = ally_vec
                    allies_mask[b, i] = 1
            
            # Enemies
            n_enemies = min(len(enemies), max_enemies)
            if n_enemies > 0:
                for j in range(n_enemies):
                    enemy_vec = enemies[j] if len(enemies[j].shape) == 1 else enemies[j][b]
                    enemies_arr[b, j, :len(enemy_vec)] = enemy_vec
                    enemies_mask[b, j] = 1
                    enemy_action_mask[b, j] = 1  # Все враги доступны для атаки
        
        # Возвращаем в формате, ожидаемом ONNX моделью
        return {
            "self_vec": self_arr,
            "allies": allies_arr,
            "allies_mask": allies_mask,
            "enemies": enemies_arr,
            "enemies_mask": enemies_mask,
            "enemy_action_mask": enemy_action_mask,
        }
    
    def predict(self, observations: Dict[str, np.ndarray]) -> np.ndarray:
        """Выполняет предсказание"""
        
        # Подготавливаем входы для ONNX
        onnx_inputs = {}
        for input_name in self.input_info.keys():
            if input_name in observations:
                onnx_inputs[input_name] = observations[input_name]
            else:
                # Попытка найти соответствующий ключ
                alt_names = {
                    "self_vec": ["self", "self_features"],
                    "allies": ["allies", "ally_features"],
                    "allies_mask": ["allies_mask", "ally_mask"],
                    "enemies": ["enemies", "enemy_features"],
                    "enemies_mask": ["enemies_mask", "enemy_mask"],
                    "enemy_action_mask": ["enemy_action_mask", "action_mask"],
                }
                
                found = False
                for alt_name in alt_names.get(input_name, []):
                    if alt_name in observations:
                        onnx_inputs[input_name] = observations[alt_name]
                        found = True
                        break
                
                if not found:
                    raise ValueError(f"Required input '{input_name}' not found in observations")
        
        # Выполняем предсказание
        try:
            outputs = self.session.run(["action"], onnx_inputs)
            return outputs[0]
        except Exception as e:
            print(f"ONNX inference error: {e}")
            print(f"Input shapes: {[(k, v.shape) for k, v in onnx_inputs.items()]}")
            raise
    
    def decode_action(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        """Декодирует выход модели в понятные действия"""
        if action.ndim == 1:
            action = action.reshape(1, -1)
        
        batch_size = action.shape[0]
        decoded = {
            "target": action[:, 0].astype(int),
            "move": action[:, 1:3],
            "aim": action[:, 3:5],
            "fire": action[:, 5].astype(int),
        }
        
        if batch_size == 1:
            # Убираем batch dimension для single sample
            decoded = {k: v[0] for k, v in decoded.items()}
        
        return decoded


def run_inference_test(onnx_path: str, batch_size: int = 3, verbose: bool = True):
    """
    Тестирование инференса с автоматическим чтением мета-файлов
    """
    print(f"=== ONNX Inference Test ===")
    print(f"Model: {onnx_path}")
    print(f"Batch size: {batch_size}")
    
    # Создаем движок инференса
    engine = ONNXInferenceEngine(onnx_path)
    
    # Генерируем случайные тестовые данные
    rng = np.random.default_rng(42)
    obs_dims = engine.meta["obs_dims"]
    max_allies = engine.meta["max_allies"]
    max_enemies = engine.meta["max_enemies"]
    
    # Создаем наблюдения
    self_vec = rng.normal(0, 0.5, size=(obs_dims["self"],)).astype(np.float32)
    
    # Случайное количество союзников и врагов
    n_allies = rng.integers(0, max_allies + 1)
    n_enemies = rng.integers(1, max_enemies + 1)  # Минимум 1 враг
    
    allies = []
    for i in range(n_allies):
        ally_vec = rng.normal(0, 0.3, size=(obs_dims["ally"],)).astype(np.float32)
        allies.append(ally_vec)
    
    enemies = []
    for j in range(n_enemies):
        enemy_vec = rng.normal(0, 0.3, size=(obs_dims["enemy"],)).astype(np.float32)
        enemies.append(enemy_vec)
    
    if verbose:
        print(f"\nTest scenario:")
        print(f"  Self features: {self_vec.shape}")
        print(f"  Allies: {n_allies}/{max_allies}")
        print(f"  Enemies: {n_enemies}/{max_enemies}")
    
    # Подготавливаем наблюдения
    observations = engine.prepare_observation(
        self_vec=self_vec,
        allies=allies,
        enemies=enemies,
        batch_size=batch_size
    )
    
    if verbose:
        print(f"\nPrepared observations:")
        for name, arr in observations.items():
            print(f"  {name}: {arr.shape}")
    
    # Выполняем предсказание
    try:
        actions = engine.predict(observations)
        
        if verbose:
            print(f"\nPrediction successful!")
            print(f"  Raw action shape: {actions.shape}")
            print(f"  Raw actions:\n{actions}")
        
        # Декодируем действия
        for b in range(batch_size):
            decoded = engine.decode_action(actions[b])
            
            if verbose:
                print(f"\nBatch {b} decoded:")
                print(f"  Target enemy: {decoded['target']}")
                print(f"  Move: [{decoded['move'][0]:.3f}, {decoded['move'][1]:.3f}]")
                print(f"  Aim: [{decoded['aim'][0]:.3f}, {decoded['aim'][1]:.3f}]")
                print(f"  Fire: {decoded['fire']}")
        
        return actions
        
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        raise


def run_performance_benchmark(onnx_path: str, num_iterations: int = 1000):
    """Тест производительности инференса"""
    import time
    
    print(f"\n=== Performance Benchmark ===")
    print(f"Iterations: {num_iterations}")
    
    engine = ONNXInferenceEngine(onnx_path)
    obs_dims = engine.meta["obs_dims"]
    
    # Подготавливаем статические данные для бенчмарка
    rng = np.random.default_rng(42)
    self_vec = rng.normal(0, 0.5, size=(obs_dims["self"],)).astype(np.float32)
    allies = [rng.normal(0, 0.3, size=(obs_dims["ally"],)).astype(np.float32) for _ in range(2)]
    enemies = [rng.normal(0, 0.3, size=(obs_dims["enemy"],)).astype(np.float32) for _ in range(3)]
    
    observations = engine.prepare_observation(
        self_vec=self_vec,
        allies=allies,
        enemies=enemies,
        batch_size=1
    )
    
    # Разогрев
    print("Warming up...")
    for _ in range(10):
        engine.predict(observations)
    
    # Бенчмарк
    print("Running benchmark...")
    start_time = time.time()
    
    for _ in range(num_iterations):
        engine.predict(observations)
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    
    print(f"\nPerformance results:")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Average time per inference: {avg_time*1000:.3f}ms")
    print(f"  Inferences per second: {1/avg_time:.1f}")


def find_latest_onnx_models(export_dir: str = "./onnx_exports") -> List[str]:
    """Находит последние экспортированные ONNX модели"""
    
    if not os.path.exists(export_dir):
        print(f"Export directory not found: {export_dir}")
        return []
    
    # Ищем latest символическую ссылку
    latest_dir = os.path.join(export_dir, "latest")
    if os.path.exists(latest_dir):
        onnx_files = glob.glob(os.path.join(latest_dir, "*.onnx"))
        if onnx_files:
            print(f"Found {len(onnx_files)} ONNX files in latest export:")
            for f in onnx_files:
                print(f"  {os.path.basename(f)}")
            return onnx_files
    
    # Если latest нет, ищем самую новую итерацию
    iter_dirs = glob.glob(os.path.join(export_dir, "iter_*"))
    if iter_dirs:
        latest_iter = max(iter_dirs, key=os.path.getmtime)
        onnx_files = glob.glob(os.path.join(latest_iter, "*.onnx"))
        if onnx_files:
            print(f"Found {len(onnx_files)} ONNX files in {os.path.basename(latest_iter)}:")
            for f in onnx_files:
                print(f"  {os.path.basename(f)}")
            return onnx_files
    
    print("No ONNX files found")
    return []


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        onnx_path = sys.argv[1]
    else:
        # Автоматически находим последнюю модель
        models = find_latest_onnx_models()
        if not models:
            print("No ONNX models found. Please run training with ONNX export first.")
            sys.exit(1)
        onnx_path = models[0]  # Берем первую найденную модель
    
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX file not found: {onnx_path}")
        sys.exit(1)
    
    print(f"Testing ONNX inference with: {onnx_path}")
    
    try:
        # Основной тест
        run_inference_test(onnx_path, batch_size=3, verbose=True)
        
        # Тест производительности (раскомментируйте если нужно)
        # run_performance_benchmark(onnx_path, num_iterations=100)
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)