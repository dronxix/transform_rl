"""
GSPO: заменяем стандартный advantage на GAE(λ) по групповому риварду (infos['team_step_reward']).
GRPO: считаем эпизодный дисконтированный групповой возврат, сравниваем с EMA и
      раздаём относительный advantage всем шагам эпизода.
"""

import numpy as np
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.episode_v2 import EpisodeV2

def _gae(rew: np.ndarray, vpred: np.ndarray, gamma: float, lam: float, dones: np.ndarray):
    T = len(rew)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nextnonterminal = 1.0 - float(dones[t])
        nextv = vpred[t+1] if t+1 < T else 0.0
        delta = rew[t] + gamma * nextv * nextnonterminal - vpred[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        adv[t] = lastgaelam
    vtarg = adv + vpred
    return adv, vtarg

# ---------- GSPO ----------
def _postprocess_gspo(policy, sample_batch: SampleBatch, other_agent_batches=None, episode: EpisodeV2 | None=None):
    infos = sample_batch.get(SampleBatch.INFOS)
    if infos is None or len(infos) == 0 or "team_step_reward" not in infos[0]:
        team_rew = sample_batch[SampleBatch.REWARDS].astype(np.float32)
    else:
        team_rew = np.array([info["team_step_reward"] for info in infos], dtype=np.float32)
    vpred = sample_batch[SampleBatch.VF_PREDS].astype(np.float32)
    dones = sample_batch[SampleBatch.DONES].astype(np.float32)
    gamma = float(policy.config.get("gamma", 0.99))
    lam   = float(policy.config.get("lambda", 0.95))
    adv, vtarg = _gae(team_rew, vpred, gamma, lam, dones)
    sample_batch[SampleBatch.ADVANTAGES]    = adv
    sample_batch[SampleBatch.VALUE_TARGETS] = vtarg
    sample_batch["gspo_adv_mean"]           = float(np.mean(adv))
    return sample_batch

GSPOTorchPolicy = PPOTorchPolicy.with_updates(
    name="GSPOTorchPolicy",
    postprocess_fn=_postprocess_gspo,
)

# ---------- GRPO ----------
class _EMA:
    def __init__(self, beta=0.99):
        self.beta = beta
        self.mean = 0.0
        self.var  = 1e-6
        self.inited = False
    def update(self, x: float):
        if not self.inited:
            self.mean = x; self.var = 1e-6; self.inited = True; return
        delta = x - self.mean
        self.mean = self.beta * self.mean + (1 - self.beta) * x
        self.var  = self.beta * self.var  + (1 - self.beta) * (delta * (x - self.mean))

def _episode_discounted_group_return(sample_batch: SampleBatch, gamma: float):
    infos = sample_batch.get(SampleBatch.INFOS)
    if infos is None or len(infos) == 0 or "team_step_reward" not in infos[0]:
        team_rew = sample_batch[SampleBatch.REWARDS].astype(np.float32)
    else:
        team_rew = np.array([info["team_step_reward"] for info in infos], dtype=np.float32)
    g = 0.0
    for r in reversed(team_rew):
        g = r + gamma * g
    return float(g)

def _postprocess_grpo(policy, sample_batch: SampleBatch, other_agent_batches=None, episode: EpisodeV2 | None=None):
    if not hasattr(policy, "grpo_ema"):
        policy.grpo_ema = _EMA(beta=float(policy.config.get("grpo_ema_beta", 0.99)))
    gamma = float(policy.config.get("gamma", 0.99))
    G_disc = _episode_discounted_group_return(sample_batch, gamma)

    policy.grpo_ema.update(G_disc)
    mean = policy.grpo_ema.mean
    std  = float(np.sqrt(max(policy.grpo_ema.var, 1e-8)))
    rel_adv = (G_disc - mean) / (std + 1e-6)

    T = len(sample_batch[SampleBatch.REWARDS])
    adv = np.full((T,), rel_adv, dtype=np.float32)
    vpred = sample_batch[SampleBatch.VF_PREDS].astype(np.float32)
    vtarg = vpred + adv

    sample_batch[SampleBatch.ADVANTAGES]    = adv
    sample_batch[SampleBatch.VALUE_TARGETS] = vtarg
    sample_batch["grpo_rel_adv"]            = float(rel_adv)
    return sample_batch

GRPOTorchPolicy = PPOTorchPolicy.with_updates(
    name="GRPOTorchPolicy",
    postprocess_fn=_postprocess_grpo,
)
