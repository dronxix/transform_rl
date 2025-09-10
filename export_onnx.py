"""
Экспорт детерминированной политики main в ONNX + сохранение meta.json
ИСПРАВЛЕНО: Использует Policy.from_checkpoint - правильный способ для Ray
"""

import os, json
import torch
import ray
from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env

from arena_env import ArenaEnv
from entity_attention_model import EntityAttentionModel   # noqa: F401
from masked_multihead_dist import MaskedTargetMoveAimFire # noqa: F401

def env_creator(cfg): 
    return ArenaEnv(cfg)

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

def main(checkpoint_path: str, export_dir: str = "./export_onnx", policy_id: str = "main"):
    os.makedirs(export_dir, exist_ok=True)
    
    # ОБЯЗАТЕЛЬНО инициализируем Ray
    ray.init(ignore_reinit_error=True)
    
    try:
        # Регистрируем окружение для правильной загрузки политики
        register_env("ArenaEnv", env_creator)
        
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

        # ПРАВИЛЬНЫЙ способ загрузки политики в Ray
        print(f"Loading policy '{policy_id}' from checkpoint: {checkpoint_path}")
        
        policy = Policy.from_checkpoint(
            checkpoint=checkpoint_path,
            policy_ids=[policy_id]  # Указываем какую политику загружать
        )[policy_id]  # Извлекаем нужную политику из словаря
        
        print(f"Successfully loaded policy: {type(policy)}")
        
        # Получаем модель из политики
        model = policy.model
        model.eval()
        print(f"Model type: {type(model)}")
        
        # Проверяем что модель имеет правильные размеры
        if hasattr(model, 'max_enemies') and hasattr(model, 'max_allies'):
            print(f"Model dimensions: max_enemies={model.max_enemies}, max_allies={model.max_allies}")
            ne = model.max_enemies
            na = model.max_allies
        else:
            print("Using dimensions from environment")
            ne = max_enemies
            na = max_allies

        # Создаем wrapper для экспорта
        wrapper = ScriptedPolicy(model, ne=ne, na=na)

        # Подготавливаем тестовые входы
        B = 1
        ex_self  = torch.zeros(B, self_feats, dtype=torch.float32)
        ex_allies= torch.zeros(B, na, ally_feats, dtype=torch.float32)
        ex_amask = torch.zeros(B, na, dtype=torch.int32)
        ex_enems = torch.zeros(B, ne, enemy_feats, dtype=torch.float32)
        ex_emask = torch.zeros(B, ne, dtype=torch.int32)
        ex_gs    = torch.zeros(B, global_feats, dtype=torch.float32)
        ex_emact = torch.zeros(B, ne, dtype=torch.int32)

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

        # Сохраняем метаданные
        meta = {
            "max_allies": int(na),
            "max_enemies": int(ne),
            "obs_dims": {
                "self": int(self_feats),
                "ally": int(ally_feats),
                "enemy": int(enemy_feats),
                "global_state": int(global_feats),
            },
            "action_spec": {
                "target_discrete_n": int(ne),
                "move_box": 2,
                "aim_box": 2,
                "fire_binary": 1,
                "total_action_dim": 6  # target(1) + move(2) + aim(2) + fire(1)
            },
            "model_config": {
                "d_model": getattr(model, 'd_model', 128),
                "nhead": getattr(model, 'nhead', 8),
                "layers": getattr(model, 'layers', 2),
            },
            "export_info": {
                "checkpoint_path": os.path.abspath(checkpoint_path),
                "policy_id": policy_id,
                "policy_type": str(type(policy)),
                "model_type": str(type(model)),
            },
            "note": "Use masks to indicate real counts; pad to max_allies/max_enemies."
        }
        
        meta_path = os.path.join(export_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"Successfully exported ONNX: {onnx_path}")
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
            print(f"Sample output: {test_result[0][0]}")
            
        except ImportError:
            print("onnxruntime not available, skipping validation")
        except Exception as e:
            print(f"ONNX validation failed: {e}")

        print("=== Export completed successfully! ===")
        
    finally:
        # Обязательно останавливаем Ray
        ray.shutdown()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python export_onnx.py <checkpoint_path> [export_dir] [policy_id]")
        print("Example: python export_onnx.py ./checkpoints/checkpoint_000100")
        print("Example: python export_onnx.py ./checkpoints/checkpoint_000100 ./export opponent_0")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    export_dir = sys.argv[2] if len(sys.argv) > 2 else "./export_onnx"
    policy_id = sys.argv[3] if len(sys.argv) > 3 else "main"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint path does not exist: {checkpoint_path}")
        sys.exit(1)
    
    main(checkpoint_path, export_dir, policy_id)