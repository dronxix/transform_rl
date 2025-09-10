"""
GSPO/GRPO: Исправленная версия для Ray 2.48+
Использует правильные ключи SampleBatch
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
class GSPOTorchPolicy(PPOTorchPolicy):
    """GSPO: групповые advantages через GAE по team_step_reward"""
    
    def postprocess_trajectory(self, sample_batch: SampleBatch, 
                             other_agent_batches=None, episode=None):
        # Сначала вызываем базовый postprocessing
        sample_batch = super().postprocess_trajectory(
            sample_batch, other_agent_batches, episode)
        
        # ИСПРАВЛЕНИЕ: Используем строковые ключи вместо констант
        infos = sample_batch.get("infos")
        team_rew = None
        
        # Проверяем что infos корректные и содержат нужные данные
        if (infos is not None and 
            len(infos) > 0 and 
            isinstance(infos[0], dict) and
            "team_step_reward" in infos[0]):
            
            try:
                team_rew = np.array([info["team_step_reward"] for info in infos], dtype=np.float32)
                print(f"DEBUG GSPO: Using team rewards, shape={team_rew.shape}")
            except (KeyError, TypeError, ValueError) as e:
                print(f"DEBUG GSPO: Error extracting team rewards: {e}")
                team_rew = None
        
        # Fallback на обычные rewards
        if team_rew is None:
            team_rew = sample_batch["rewards"].astype(np.float32)
            print(f"DEBUG GSPO: Using individual rewards, shape={team_rew.shape}")
            
        vpred = sample_batch["vf_preds"].astype(np.float32)
        dones = sample_batch["dones"].astype(np.float32)
        gamma = float(self.config.get("gamma", 0.99))
        lam = float(self.config.get("lambda", 0.95))
        
        adv, vtarg = _gae(team_rew, vpred, gamma, lam, dones)
        
        # ИСПРАВЛЕНИЕ: Используем строковые ключи
        sample_batch["advantages"] = adv
        sample_batch["value_targets"] = vtarg
        # sample_batch["gspo_adv_mean"] = float(np.mean(adv))
        mean_adv = np.mean(adv, dtype=np.float32)
        sample_batch["gspo_adv_mean"] = np.full_like(adv, mean_adv, dtype=np.float32)
        
        return sample_batch

# ---------- GRPO ----------
class _EMA:
    def __init__(self, beta=0.99):
        self.beta = beta
        self.mean = 0.0
        self.var = 1e-6
        self.inited = False
        
    def update(self, x: float):
        if not self.inited:
            self.mean = x
            self.var = 1e-6
            self.inited = True
            return
        delta = x - self.mean
        self.mean = self.beta * self.mean + (1 - self.beta) * x
        self.var = self.beta * self.var + (1 - self.beta) * (delta * (x - self.mean))

def _episode_discounted_group_return(sample_batch: SampleBatch, gamma: float):
    infos = sample_batch.get("infos")
    team_rew = None
    
    # Безопасная обработка infos
    if (infos is not None and 
        len(infos) > 0 and 
        isinstance(infos[0], dict) and
        "team_step_reward" in infos[0]):
        
        try:
            team_rew = np.array([info["team_step_reward"] for info in infos], dtype=np.float32)
        except (KeyError, TypeError, ValueError):
            team_rew = None
    
    if team_rew is None:
        team_rew = sample_batch["rewards"].astype(np.float32)
    
    g = 0.0
    for r in reversed(team_rew):
        g = r + gamma * g
    return float(g)

class GRPOTorchPolicy(PPOTorchPolicy):
    """GRPO: эпизодный групповой return vs EMA"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grpo_ema = _EMA(beta=float(self.config.get("grpo_ema_beta", 0.99)))
    
    def postprocess_trajectory(self, sample_batch: SampleBatch, 
                             other_agent_batches=None, episode=None):
        # Сначала базовый postprocessing
        sample_batch = super().postprocess_trajectory(
            sample_batch, other_agent_batches, episode)
        
        gamma = float(self.config.get("gamma", 0.99))
        G_disc = _episode_discounted_group_return(sample_batch, gamma)
        
        self.grpo_ema.update(G_disc)
        mean = self.grpo_ema.mean
        std = float(np.sqrt(max(self.grpo_ema.var, 1e-8)))
        rel_adv = (G_disc - mean) / (std + 1e-6)
        
        T = len(sample_batch["rewards"])
        adv = np.full((T,), rel_adv, dtype=np.float32)
        vpred = sample_batch["vf_preds"].astype(np.float32)
        vtarg = vpred + adv
        
        # ИСПРАВЛЕНИЕ: Используем строковые ключи
        sample_batch["advantages"] = adv
        sample_batch["value_targets"] = vtarg
        # sample_batch["grpo_rel_adv"] = float(rel_adv)
        T = len(sample_batch["rewards"])
        sample_batch["grpo_rel_adv"] = np.full((T,), np.float32(rel_adv), dtype=np.float32)
        
        return sample_batch