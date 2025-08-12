"""
Пример онлайнового инференса ONNX (CPU) с переменным числом объектов через маски.
"""

import numpy as np
import onnxruntime as ort

def run_inference(onnx_path: str, batch: int = 3, na_max: int = 6, ne_max: int = 6,
                  self_dim: int = 12, ally_dim: int = 8, enemy_dim: int = 10, gs_dim: int = 64):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    self_arr  = np.zeros((batch, self_dim), dtype=np.float32)
    allies    = np.zeros((batch, na_max, ally_dim), dtype=np.float32)
    a_mask    = np.zeros((batch, na_max), dtype=np.int32)
    enemies   = np.zeros((batch, ne_max, enemy_dim), dtype=np.float32)
    e_mask    = np.zeros((batch, ne_max), dtype=np.int32)
    gstate    = np.zeros((batch, gs_dim), dtype=np.float32)
    e_actmask = np.zeros((batch, ne_max), dtype=np.int32)

    rng = np.random.default_rng(0)
    for b in range(batch):
        nA = rng.integers(low=0, high=na_max+1)
        nE = rng.integers(low=1, high=ne_max+1)
        if nA > 0:
            allies[b, :nA, :] = rng.normal(0, 0.5, size=(nA, ally_dim)).astype(np.float32)
            a_mask[b, :nA] = 1
        enemies[b, :nE, :] = rng.normal(0, 0.5, size=(nE, enemy_dim)).astype(np.float32)
        e_mask[b, :nE] = 1
        e_actmask[b, :nE] = 1
        self_arr[b, :] = rng.normal(0, 0.5, size=(self_dim,)).astype(np.float32)
        gstate[b, :] = rng.normal(0, 0.2, size=(gs_dim,)).astype(np.float32)

    action = sess.run(["action"], {
        "self": self_arr,
        "allies": allies,
        "allies_mask": a_mask,
        "enemies": enemies,
        "enemies_mask": e_mask,
        "global_state": gstate,
        "enemy_action_mask": e_actmask,
    })[0]
    print("action shape:", action.shape)  # [B, 1+2+2+1]
    print(action)

if __name__ == "__main__":
    run_inference("./export_onnx/policy_main_multihead.onnx")
