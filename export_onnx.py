"""
Экспорт детерминированной политики main в ONNX + сохранение meta.json
(максимальные Na/Ne, размерности obs и спецификация действия).
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

    # Будет достаточно «пустого» PPO для восстановления policy и модели
    algo = PPOConfig().environment(env="ArenaEnv", env_config={"episode_len": 8}).framework("torch").build()
    algo.restore(checkpoint_path)
    pol = algo.get_policy("main")
    model = pol.model
    model.eval()

    ne = model.max_enemies; na = model.max_allies
    wrapper = ScriptedPolicy(model, ne=ne, na=na)

    # Фиктивные входы для экспорт-трейса
    B = 1
    ex_self  = torch.zeros(B, model.obs_space_struct["self"].shape[0])
    ex_allies= torch.zeros(B, na, model.obs_space_struct["allies"].shape[1])
    ex_amask = torch.zeros(B, na, dtype=torch.int32)
    ex_enems = torch.zeros(B, ne, model.obs_space_struct["enemies"].shape[1])
    ex_emask = torch.zeros(B, ne, dtype=torch.int32)
    ex_gs    = torch.zeros(B, model.obs_space_struct["global_state"].shape[0])
    ex_emact = torch.zeros(B, ne, dtype=torch.int32)

    onnx_path = os.path.join(export_dir, "policy_main_multihead.onnx")
    torch.onnx.export(
        wrapper,
        (ex_self, ex_allies, ex_amask, ex_enems, ex_emask, ex_gs, ex_emact),
        onnx_path,
        opset_version=17,
        input_names=["self", "allies", "allies_mask", "enemies", "enemies_mask", "global_state", "enemy_action_mask"],
        output_names=["action"],
        dynamic_axes={name: {0: "batch"} for name in
                      ["self","allies","allies_mask","enemies","enemies_mask","global_state","enemy_action_mask","action"]}
    )

    # meta.json рядом
    meta = {
        "max_allies": int(na),
        "max_enemies": int(ne),
        "obs_dims": {
            "self": int(model.obs_space_struct["self"].shape[0]),
            "ally": int(model.obs_space_struct["allies"].shape[1]),
            "enemy": int(model.obs_space_struct["enemies"].shape[1]),
            "global_state": int(model.obs_space_struct["global_state"].shape[0]),
        },
        "action_spec": {
            "target_discrete_n": int(ne),
            "move_box": 2,
            "aim_box": 2,
            "fire_binary": 1
        },
        "note": "Use masks to indicate real counts; pad to max_allies/max_enemies."
    }
    with open(os.path.join(export_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Saved ONNX:", onnx_path)
    print("Saved meta:", os.path.join(export_dir, "meta.json"))
    algo.stop()
    ray.shutdown()

if __name__ == "__main__":
    cp = "./rllib_league_results/checkpoint_000100"
    main(cp)
