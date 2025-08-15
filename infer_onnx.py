"""
Пример онлайнового инференса ONNX (CPU) с переменным числом объектов через маски.
ИСПРАВЛЕНО: автоматическое чтение размерностей из meta.json
"""

import numpy as np
import onnxruntime as ort
import json
import os

def load_meta(onnx_dir: str):
    """Загружает метаданные модели"""
    meta_path = os.path.join(onnx_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # Fallback на значения по умолчанию
        print(f"Warning: meta.json not found at {meta_path}, using defaults")
        return {
            "max_allies": 6,
            "max_enemies": 6,
            "obs_dims": {
                "self": 12,
                "ally": 8,
                "enemy": 10,
                "global_state": 64,
            }
        }

def run_inference(onnx_path: str, batch: int = 3, meta_data: dict = None):
    """
    Запуск инференса с автоматическим определением размерностей
    """
    # Получаем директорию с ONNX файлом
    onnx_dir = os.path.dirname(onnx_path)
    
    # Загружаем метаданные если не переданы
    if meta_data is None:
        meta_data = load_meta(onnx_dir)
    
    # Извлекаем размерности
    na_max = meta_data["max_allies"]
    ne_max = meta_data["max_enemies"]
    obs_dims = meta_data["obs_dims"]
    self_dim = obs_dims["self"]
    ally_dim = obs_dims["ally"]
    enemy_dim = obs_dims["enemy"]
    gs_dim = obs_dims["global_state"]
    
    print(f"Using dimensions from meta:")
    print(f"  max_allies={na_max}, max_enemies={ne_max}")
    print(f"  self_dim={self_dim}, ally_dim={ally_dim}, enemy_dim={enemy_dim}, gs_dim={gs_dim}")
    
    # Создаем ONNX сессию
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    
    # Инициализируем массивы
    self_arr  = np.zeros((batch, self_dim), dtype=np.float32)
    allies    = np.zeros((batch, na_max, ally_dim), dtype=np.float32)
    a_mask    = np.zeros((batch, na_max), dtype=np.int32)
    enemies   = np.zeros((batch, ne_max, enemy_dim), dtype=np.float32)
    e_mask    = np.zeros((batch, ne_max), dtype=np.int32)
    gstate    = np.zeros((batch, gs_dim), dtype=np.float32)
    e_actmask = np.zeros((batch, ne_max), dtype=np.int32)

    # Генерируем случайные данные для тестирования
    rng = np.random.default_rng(42)  # Фиксированный seed для воспроизводимости
    
    for b in range(batch):
        # Случайное количество союзников и врагов
        nA = rng.integers(low=0, high=na_max+1)
        nE = rng.integers(low=1, high=ne_max+1)  # Минимум 1 враг
        
        # Заполняем союзников
        if nA > 0:
            allies[b, :nA, :] = rng.normal(0, 0.5, size=(nA, ally_dim)).astype(np.float32)
            a_mask[b, :nA] = 1
        
        # Заполняем врагов
        enemies[b, :nE, :] = rng.normal(0, 0.5, size=(nE, enemy_dim)).astype(np.float32)
        e_mask[b, :nE] = 1
        e_actmask[b, :nE] = 1  # Все враги доступны для атаки
        
        # Заполняем остальные данные
        self_arr[b, :] = rng.normal(0, 0.5, size=(self_dim,)).astype(np.float32)
        gstate[b, :] = rng.normal(0, 0.2, size=(gs_dim,)).astype(np.float32)
        
        print(f"Batch {b}: {nA} allies, {nE} enemies")

    # Запускаем инференс
    try:
        action = sess.run(["action"], {
            "self": self_arr,
            "allies": allies,
            "allies_mask": a_mask,
            "enemies": enemies,
            "enemies_mask": e_mask,
            "global_state": gstate,
            "enemy_action_mask": e_actmask,
        })[0]
        
        print(f"\nInference successful!")
        print(f"Action shape: {action.shape}")  # [B, 1+2+2+1] = [B, 6]
        print(f"Action sample:\n{action}")
        
        # Декодируем действия для понимания
        print(f"\nDecoded actions:")
        for b in range(batch):
            target = int(action[b, 0])
            move = action[b, 1:3]
            aim = action[b, 3:5]
            fire = int(action[b, 5])
            print(f"  Batch {b}: target={target}, move={move}, aim={aim}, fire={fire}")
            
        return action
        
    except Exception as e:
        print(f"Inference failed: {e}")
        print(f"Input shapes:")
        print(f"  self: {self_arr.shape}")
        print(f"  allies: {allies.shape}")
        print(f"  allies_mask: {a_mask.shape}")
        print(f"  enemies: {enemies.shape}")
        print(f"  enemies_mask: {e_mask.shape}")
        print(f"  global_state: {gstate.shape}")
        print(f"  enemy_action_mask: {e_actmask.shape}")
        raise

def run_performance_test(onnx_path: str, num_iterations: int = 1000):
    """Тест производительности инференса"""
    import time
    
    meta_data = load_meta(os.path.dirname(onnx_path))
    
    # Разогрев
    print("Warming up...")
    for _ in range(10):
        run_inference(onnx_path, batch=1, meta_data=meta_data)
    
    # Тест производительности
    print(f"Running performance test ({num_iterations} iterations)...")
    start_time = time.time()
    
    for _ in range(num_iterations):
        run_inference(onnx_path, batch=1, meta_data=meta_data)
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    
    print(f"Performance results:")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Average time per inference: {avg_time*1000:.3f}ms")
    print(f"  Inferences per second: {1/avg_time:.1f}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        onnx_path = sys.argv[1]
    else:
        onnx_path = "./export_onnx/policy_main_multihead.onnx"
    
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX file not found: {onnx_path}")
        print("Please run export_onnx.py first or provide correct path")
        sys.exit(1)
    
    print(f"Running inference test with: {onnx_path}")
    run_inference(onnx_path, batch=3)
    
    # Раскомментируйте для теста производительности
    # run_performance_test(onnx_path, num_iterations=100)