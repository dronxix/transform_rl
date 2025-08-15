"""
Экспорт детерминированной политики main в ONNX + сохранение meta.json
ИСПРАВЛЕНО для Ray 2.48 и обновленной модели
"""

import os, json
import torch
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from arena_env import ArenaEnv
from entity_attention_model import EntityAttentionModel   # noqa: F401
from masked_multihead_dist import MaskedTargetMoveAimFire # noqa: F401

def env_creator(cfg): return ArenaEnv(cfg)

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

def main(checkpoint_path: str, export_dir: str = "./export_onnx"):
    os.makedirs(export_dir, exist_ok=True)
    ray.init(ignore_reinit_error=True)
    register_env("ArenaEnv", env_creator)

    # ИСПРАВЛЕНИЕ: Создаем конфигурацию совместимую с Ray 2.48
    # Получаем размеры из временного окружения
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

    # Базовая конфигурация модели (должна совпадать с train_rllib_league.py)
    base_model_config = {
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

    # ИСПРАВЛЕНИЕ: Используем обновленный API Ray 2.48
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env="ArenaEnv", 
            env_config={
                "episode_len": 128,
                "ally_choices": [1],
                "enemy_choices": [1],
                "max_allies": max_allies,
                "max_enemies": max_enemies,
            }
        )
        .framework("torch")
        .env_runners(num_env_runners=0)  # Не нужны для экспорта
        .training(model=base_model_config)
        .multi_agent(
            policies={
                "main": (None, obs_space, act_space, {"model": base_model_config}),
            },
            policy_mapping_fn=lambda aid, *args, **kw: "main",
            policies_to_train=["main"],
        )
    )
    
    algo = config.build()
    
    try:
        algo.restore(checkpoint_path)
        print(f"Successfully restored checkpoint: {checkpoint_path}")
    except Exception as e:
        print(f"Error restoring checkpoint: {e}")
        print("Trying alternative restore method...")
        # Альтернативный способ для Ray 2.48
        import pickle
        with open(os.path.join(checkpoint_path, "algorithm_state.pkl"), "rb") as f:
            state = pickle.load(f)
        algo.__setstate__(state)
    
    pol = algo.get_policy("main")
    model = pol.model
    model.eval()

    # ИСПРАВЛЕНИЕ: Используем реальные размеры вместо model.obs_space_struct
    ne = max_enemies
    na = max_allies
    wrapper = ScriptedPolicy(model, ne=ne, na=na)

    # Фиктивные входы для экспорт-трейса
    B = 1
    ex_self  = torch.zeros(B, self_feats, dtype=torch.float32)
    ex_allies= torch.zeros(B, na, ally_feats, dtype=torch.float32)
    ex_amask = torch.zeros(B, na, dtype=torch.int32)
    ex_enems = torch.zeros(B, ne, enemy_feats, dtype=torch.float32)
    ex_emask = torch.zeros(B, ne, dtype=torch.int32)
    ex_gs    = torch.zeros(B, global_feats, dtype=torch.float32)
    ex_emact = torch.zeros(B, ne, dtype=torch.int32)

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
    )

    # meta.json с реальными размерами
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
            "fire_binary": 1
        },
        "model_config": {
            "d_model": base_model_config["custom_model_config"]["d_model"],
            "nhead": base_model_config["custom_model_config"]["nhead"],
            "layers": base_model_config["custom_model_config"]["layers"],
        },
        "note": "Use masks to indicate real counts; pad to max_allies/max_enemies."
    }
    
    meta_path = os.path.join(export_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Successfully exported ONNX: {onnx_path}")
    print(f"Saved metadata: {meta_path}")
    
    # Проверяем что экспорт работает
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        print("ONNX model validation: OK")
        
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
        print(f"Test inference result shape: {test_result[0].shape}")
        
    except Exception as e:
        print(f"ONNX validation failed: {e}")
    
    algo.stop()
    ray.shutdown()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cp = sys.argv[1]
    else:
        cp = "./rllib_league_results/checkpoint_000100"
    main(cp)