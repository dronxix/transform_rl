#!/usr/bin/env python3
"""
Быстрый инференс для Arena моделей
Простой интерфейс для тестирования моделей без сложной настройки
"""

import os
import sys
import argparse
import json
from pathlib import Path

def find_models():
    """Автоматически находит модели в стандартных местах"""
    
    search_locations = [
        "./onnx_exports/latest/*.onnx",
        "./onnx_exports/*/*.onnx",
        "./checkpoints/checkpoint_*",
        "./*.onnx",
        "./models/*.onnx",
        "./saved_models/*.pt",
    ]
    
    import glob
    found = {}
    
    for pattern in search_locations:
        files = glob.glob(pattern)
        for file in files:
            name = Path(file).stem
            if "checkpoint" in file:
                # Для чекпоинтов берем номер
                try:
                    num = file.split("checkpoint_")[-1]
                    name = f"checkpoint_{num}"
                except:
                    name = Path(file).name
            
            found[name] = {
                "path": file,
                "type": "onnx" if file.endswith(".onnx") else 
                        "torch" if file.endswith((".pt", ".pth")) else "ray",
                "size": os.path.getsize(file)
            }
    
    return found

def quick_test(model_path: str, model_type: str = "auto"):
    """Быстрое тестирование модели"""
    
    try:
        # Динамический импорт чтобы не падать если нет зависимостей
        from infer import UniversalInferenceManager, create_sample_input
        
        manager = UniversalInferenceManager()
        model_id = manager.load_model(model_path, model_type=model_type)
        
        # Тестируем разные сценарии
        scenarios = ["simple", "complex", "minimal"]
        
        print(f"🎯 Testing {model_id}:")
        
        for scenario in scenarios:
            try:
                input_data = create_sample_input(scenario)
                result = manager.predict(model_id, input_data)
                
                print(f"  {scenario:8}: ✅ {result}")
                
            except Exception as e:
                print(f"  {scenario:8}: ❌ {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Install with: pip install torch ray onnxruntime")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def interactive_menu():
    """Интерактивное меню для выбора и тестирования моделей"""
    
    print("🔍 Searching for models...")
    models = find_models()
    
    if not models:
        print("❌ No models found!")
        print("Expected locations:")
        print("  ./onnx_exports/latest/*.onnx")
        print("  ./checkpoints/checkpoint_*")
        print("  ./*.onnx")
        return
    
    print(f"\n📋 Found {len(models)} models:")
    model_list = list(models.items())
    
    for i, (name, info) in enumerate(model_list, 1):
        size_mb = info["size"] / (1024 * 1024)
        print(f"  {i:2d}. {name:25} ({info['type']:4}) {size_mb:6.1f}MB")
    
    while True:
        try:
            choice = input(f"\nSelect model (1-{len(models)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                break
            
            if choice.lower() == 'all':
                # Тестируем все модели
                print("\n🔄 Testing all models...")
                for name, info in models.items():
                    print(f"\n--- {name} ---")
                    quick_test(info["path"], info["type"])
                break
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(model_list):
                    name, info = model_list[idx]
                    print(f"\n🧪 Testing {name}...")
                    
                    success = quick_test(info["path"], info["type"])
                    
                    if success:
                        print(f"✅ {name} working correctly!")
                    else:
                        print(f"❌ {name} test failed")
                    
                    continue_choice = input("\nTest another? (y/n): ").strip().lower()
                    if continue_choice != 'y':
                        break
                else:
                    print(f"Invalid choice. Enter 1-{len(models)}")
            except ValueError:
                print("Invalid input. Enter a number or 'q'")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break

def batch_test(output_file: str = None):
    """Пакетное тестирование всех найденных моделей"""
    
    models = find_models()
    if not models:
        print("No models found for batch testing")
        return
    
    results = {}
    
    print(f"🔄 Batch testing {len(models)} models...")
    
    for name, info in models.items():
        print(f"\nTesting {name}...")
        
        try:
            from universal_inference import UniversalInferenceManager, create_sample_input
            import time
            
            manager = UniversalInferenceManager()
            model_id = manager.load_model(info["path"], model_type=info["type"])
            
            # Тестируем простой сценарий
            input_data = create_sample_input("simple")
            
            # Измеряем время
            start_time = time.time()
            result = manager.predict(model_id, input_data)
            inference_time = (time.time() - start_time) * 1000
            
            results[name] = {
                "status": "success",
                "inference_time_ms": inference_time,
                "model_type": result.model_type,
                "result": str(result),
                "file_size_mb": info["size"] / (1024 * 1024),
                "file_type": info["type"]
            }
            
            print(f"  ✅ {inference_time:.2f}ms")
            
        except Exception as e:
            results[name] = {
                "status": "failed", 
                "error": str(e),
                "file_size_mb": info["size"] / (1024 * 1024),
                "file_type": info["type"]
            }
            print(f"  ❌ {e}")
    
    # Выводим сводку
    print(f"\n📊 Batch Test Results:")
    print(f"{'Model':<25} {'Status':<8} {'Time':<10} {'Type':<6} {'Size':<8}")
    print("-" * 70)
    
    for name, result in results.items():
        if result["status"] == "success":
            time_str = f"{result['inference_time_ms']:.1f}ms"
            print(f"{name:<25} {'✅':<8} {time_str:<10} {result['file_type']:<6} {result['file_size_mb']:.1f}MB")
        else:
            print(f"{name:<25} {'❌':<8} {'FAILED':<10} {result['file_type']:<6} {result['file_size_mb']:.1f}MB")
    
    # Сохраняем результаты
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Quick Arena Model Inference")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Интерактивный режим
    interactive_parser = subparsers.add_parser('interactive', help='Interactive model selection')
    
    # Быстрый тест
    test_parser = subparsers.add_parser('test', help='Test specific model')
    test_parser.add_argument('model_path', help='Path to model')
    test_parser.add_argument('--type', choices=['onnx', 'torch', 'ray', 'auto'], 
                           default='auto', help='Model type')
    
    # Пакетный тест
    batch_parser = subparsers.add_parser('batch', help='Test all found models')
    batch_parser.add_argument('--output', help='Save results to JSON file')
    
    # Поиск моделей
    find_parser = subparsers.add_parser('find', help='Find available models')
    
    # Сравнение
    compare_parser = subparsers.add_parser('compare', help='Compare models')
    compare_parser.add_argument('models', nargs='+', help='Model paths to compare')
    
    args = parser.parse_args()
    
    if args.command == 'interactive' or not args.command:
        interactive_menu()
    
    elif args.command == 'test':
        if not os.path.exists(args.model_path):
            print(f"❌ File not found: {args.model_path}")
            return 1
        
        success = quick_test(args.model_path, args.type)
        return 0 if success else 1
    
    elif args.command == 'batch':
        batch_test(args.output)
    
    elif args.command == 'find':
        models = find_models()
        if models:
            print(f"Found {len(models)} models:")
            for name, info in models.items():
                size_mb = info["size"] / (1024 * 1024)
                print(f"  {name:25} ({info['type']:4}) {size_mb:6.1f}MB - {info['path']}")
        else:
            print("No models found")
    
    elif args.command == 'compare':
        try:
            from infer import UniversalInferenceManager, create_sample_input
            
            manager = UniversalInferenceManager()
            
            # Загружаем все указанные модели
            for model_path in args.models:
                if os.path.exists(model_path):
                    manager.load_model(model_path)
                else:
                    print(f"⚠️ Not found: {model_path}")
            
            if not manager.engines:
                print("❌ No models loaded successfully")
                return 1
            
            # Сравниваем
            input_data = create_sample_input("simple")
            results = manager.predict_multiple(input_data)
            
            print(f"\n📊 Comparison Results:")
            for model_id, result in results.items():
                if result:
                    print(f"  {model_id}: {result}")
                else:
                    print(f"  {model_id}: ❌ Failed")
        
        except ImportError:
            print("❌ Missing dependencies for comparison")
            return 1
    
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())s