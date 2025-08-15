"""
Экспорт детерминированной политики main в ONNX + сохранение meta.json
ИСПРАВЛЕНО: Прямая загрузка политики из policy_state.pkl без Ray
"""

import os, json, pickle
import torch
import numpy as np

from arena_env import ArenaEnv
from entity_attention_model import EntityAttentionModel
from masked_multihead_dist import MaskedTargetMoveAimFire

class ScriptedPolicy(torch.nn.Module):
    def __init__(self, model, ne: int, na: int):
        super().__init__()
        self.m = model
        self.ne = ne
        self.na = na

    @torch.no_grad()
    def forward(self,
                self_vec,
                allies, allies_mask,
                enemies, enemies_mask,
                global_state,
                enemy_action_mask):
        obs = {
            "self": self_vec,
            "allies": allies,
            "allies_mask": allies_mask,
            "enemies": enemies,
            "enemies_mask": enemies_mask,
            "global_state": global_state,
            "enemy_action_mask": enemy_action_mask,
        }
        logits, _ = self.m({"obs": obs}, [], None)
        idx = 0
        logits_t = logits[:, idx:idx+self.ne]; idx += self.ne
        mu_move  = logits[:, idx:idx+2];        idx += 2
        idx += 2  # log_std_move
        mu_aim   = logits[:, idx:idx+2];        idx += 2
        idx += 2  # log_std_aim
        logit_fr = logits[:, idx:idx+1];        idx += 1

        target = torch.argmax(logits_t, dim=-1).unsqueeze(-1).float()
        move = torch.tanh(mu_move)
        aim  = torch.tanh(mu_aim)
        fire = (logit_fr > 0).float()
        return torch.cat([target, move, aim, fire], dim=-1)

def load_policy_from_checkpoint(checkpoint_path: str, policy_id: str = "main"):
    """
    Загружает политику напрямую из файлов чекпоинта без восстановления алгоритма
    """
    print(f"Loading policy '{policy_id}' from {checkpoint_path}")
    
    # Путь к файлу политики
    policy_file = os.path.join(checkpoint_path, f"policies", policy_id, "policy_state.pkl")
    
    if not os.path.exists(policy_file):
        # Альтернативные пути для разных версий Ray
        alternative_paths = [
            os.path.join(checkpoint_path, f"policy_state_{policy_id}.pkl"),
            os.path.join(checkpoint_path, "policy_state.pkl"),
            os.path.join(checkpoint_path, f"{policy_id}_policy_state.pkl"),
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                policy_file = alt_path
                break
        else:
            raise FileNotFoundError(f"Could not find policy state file. Tried:\n" + 
                                  f"  {policy_file}\n" + 
                                  "\n".join(f"  {p}" for p in alternative_paths))
    
    print(f"Found policy file: {policy_file}")
    
    # Загружаем состояние политики
    with open(policy_file, "rb") as f:
        policy_state = pickle.load(f)
    
    # Извлекаем веса модели
    if "weights" in policy_state:
        model_weights = policy_state["weights"]
    elif "_model_state_dict" in policy_state:
        model_weights = policy_state["_model_state_dict"]
    else:
        # Пытаемся найти веса в других ключах
        for key in policy_state.keys():
            if "model" in key.lower() or "weight" in key.lower():
                model_weights = policy_state[key]
                break
        else:
            print(f"Available keys in policy_state: {list(policy_state.keys())}")
            raise KeyError("Could not find model weights in policy state")
    
    return model_weights, policy_state

def create_model_from_config(obs_space, act_space, model_config):
    """Создает модель из конфигурации"""
    from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
    
    # Получаем размеры
    if hasattr(obs_space, 'spaces'):
        max_enemies = obs_space["enemies"].shape[0]
        max_allies = obs_space["allies"].shape[0]
    else:
        max_enemies = model_config.get("max_enemies", 6)
        max_allies = model_config.get("max_allies", 6)
    
    # Обновляем конфиг модели
    custom_config = model_config.get("custom_model_config", {})
    custom_config.update({
        "max_enemies": max_enemies,
        "max_allies": max_allies,
    })
    
    # Считаем размер выхода для action_dist
    num_outputs = MaskedTargetMoveAimFire.required_model_output_shape(act_space, model_config)
    
    # Создаем модель
    model = EntityAttentionModel(
        obs_space=obs_space,
        action_space=act_space,
        num_outputs=num_outputs,
        model_config=model_config,
        name="policy_main"
    )
    
    return model

def main(checkpoint_path: str, export_dir: str = "./export_onnx", policy_id: str = "main"):
    os.makedirs(export_dir, exist_ok=True)
    
    print("=== ONNX Export without Ray ===")
    
    # Получаем размеры из окружения
    tmp_env = ArenaEnv({"ally_choices": [1], "enemy_choices": [1]})
    obs_space = tmp_env.observation_space
    act_space = tmp_env.action_space
    
    # Извлекаем реальные размеры
    max_enemies = obs_space["enemies"].shape[0]
    max_allies = obs_space["allies"].shape[0]
    self_feats = obs_space["self"].shape[0]
    ally_feats = obs_space["allies"].shape[1]
    enemy_feats = obs_space["enemies"].shape[1]
    global_feats = obs_space["global_state"].shape[0]
    
    print(f"Detected dimensions: max_enemies={max_enemies}, max_allies={max_allies}")
    print(f"Feature sizes: self={self_feats}, ally={ally_feats}, enemy={enemy_feats}, global={global_feats}")

    # Конфигурация модели (должна совпадать с обучением)
    model_config = {
        "custom_model": "entity_attention",
        "custom_action_dist": "masked_multihead",
        "custom_model_config": {
            "d_model": 128,
            "nhead": 8,
            "layers": 2,
            "ff": 256,
            "hidden": 256,
            "max_enemies": max_enemies,
            "max_allies": max_allies,
        },
        "vf_share_layers": False,
    }
    
    # Загружаем веса политики
    try:
        model_weights, policy_state = load_policy_from_checkpoint(checkpoint_path, policy_id)
        print("Successfully loaded policy weights")
    except Exception as e:
        print(f"Error loading policy: {e}")
        raise
    
    # Создаем модель
    model = create_model_from_config(obs_space, act_space, model_config)
    
    # Загружаем веса
    try:
        if isinstance(model_weights, dict):
            model.load_state_dict(model_weights, strict=False)
        else:
            # Если веса в другом формате
            model.set_weights(model_weights)
        
        model.eval()
        print("Model weights loaded successfully")
        
    except Exception as e:
        print(f"Error loading weights into model: {e}")
        print(f"Model state dict keys: {list(model.state_dict().keys())[:5]}...")
        print(f"Loaded weights type: {type(model_weights)}")
        if isinstance(model_weights, dict):
            print(f"Loaded weights keys: {list(model_weights.keys())[:5]}...")
        raise

    # Создаем wrapper для экспорта
    wrapper = ScriptedPolicy(model, ne=max_enemies, na=max_allies)

    # Подготавливаем тестовые входы
    B = 1
    ex_self  = torch.zeros(B, self_feats, dtype=torch.float32)
    ex_allies= torch.zeros(B, max_allies, ally_feats, dtype=torch.float32)
    ex_amask = torch.zeros(B, max_allies, dtype=torch.int32)
    ex_enems = torch.zeros(B, max_enemies, enemy_feats, dtype=torch.float32)
    ex_emask = torch.zeros(B, max_enemies, dtype=torch.int32)
    ex_gs    = torch.zeros(B, global_feats, dtype=torch.float32)
    ex_emact = torch.zeros(B, max_enemies, dtype=torch.int32)

    # Тестовый прогон перед экспортом
    print("Testing model forward pass...")
    try:
        with torch.no_grad():
            test_output = wrapper(ex_self, ex_allies, ex_amask, ex_enems, ex_emask, ex_gs, ex_emact)
        print(f"Test output shape: {test_output.shape}")
        print("Model test passed!")
    except Exception as e:
        print(f"Model test failed: {e}")
        raise

    # Экспорт в ONNX
    onnx_path = os.path.join(export_dir, "policy_main_multihead.onnx")
    
    print("Exporting to ONNX...")
    try:
        torch.onnx.export(
            wrapper,
            (ex_self, ex_allies, ex_amask, ex_enems, ex_emask, ex_gs, ex_emact),
            onnx_path,
            opset_version=17,
            input_names=["self", "allies", "allies_mask", "enemies", "enemies_mask", "global_state", "enemy_action_mask"],
            output_names=["action"],
            dynamic_axes={name: {0: "batch"} for name in
                          ["self","allies","allies_mask","enemies","enemies_mask","global_state","enemy_action_mask","action"]},
            do_constant_folding=True,
            export_params=True,
            verbose=False,
        )
        print(f"Successfully exported to: {onnx_path}")
        
    except Exception as e:
        print(f"ONNX export failed: {e}")
        raise

    # Сохраняем метаданные
    meta = {
        "max_allies": int(max_allies),
        "max_enemies": int(max_enemies),
        "obs_dims": {
            "self": int(self_feats),
            "ally": int(ally_feats),
            "enemy": int(enemy_feats),
            "global_state": int(global_feats),
        },
        "action_spec": {
            "target_discrete_n": int(max_enemies),
            "move_box": 2,
            "aim_box": 2,
            "fire_binary": 1,
            "total_action_dim": 6  # target(1) + move(2) + aim(2) + fire(1)
        },
        "model_config": {
            "d_model": model_config["custom_model_config"]["d_model"],
            "nhead": model_config["custom_model_config"]["nhead"],
            "layers": model_config["custom_model_config"]["layers"],
            "ff": model_config["custom_model_config"]["ff"],
            "hidden": model_config["custom_model_config"]["hidden"],
        },
        "export_info": {
            "checkpoint_path": os.path.abspath(checkpoint_path),
            "policy_id": policy_id,
            "exported_at": str(torch.version.__version__),
        },
        "note": "Use masks to indicate real counts; pad to max_allies/max_enemies."
    }
    
    meta_path = os.path.join(export_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata: {meta_path}")

    # Валидация ONNX
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        
        # Тестовый прогон
        test_result = sess.run(["action"], {
            "self": ex_self.numpy(),
            "allies": ex_allies.numpy(),
            "allies_mask": ex_amask.numpy(),
            "enemies": ex_enems.numpy(),
            "enemies_mask": ex_emask.numpy(),
            "global_state": ex_gs.numpy(),
            "enemy_action_mask": ex_emact.numpy(),
        })
        print(f"ONNX validation passed! Output shape: {test_result[0].shape}")
        
    except ImportError:
        print("onnxruntime not available, skipping validation")
    except Exception as e:
        print(f"ONNX validation failed: {e}")

    print("=== Export completed successfully! ===")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python export_onnx.py <checkpoint_path> [export_dir] [policy_id]")
        print("Example: python export_onnx.py ./checkpoints/checkpoint_000100")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    export_dir = sys.argv[2] if len(sys.argv) > 2 else "./export_onnx"
    policy_id = sys.argv[3] if len(sys.argv) > 3 else "main"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint path does not exist: {checkpoint_path}")
        sys.exit(1)
    
    main(checkpoint_path, export_dir, policy_id)