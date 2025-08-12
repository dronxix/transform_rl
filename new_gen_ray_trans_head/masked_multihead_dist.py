"""
Кастомный TorchDistributionWrapper:
- target: Categorical по masked_logits,
- move/aim: Normal с tanh-squash,
- fire: Bernoulli по логиту.
"""

import torch
from torch.distributions import Categorical, Normal, Bernoulli
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models import ModelCatalog

class MaskedTargetMoveAimFire(TorchDistributionWrapper):
    def __init__(self, inputs, model):
        super().__init__(inputs, model)
        self.ne = model.max_enemies
        idx = 0
        logits_t = inputs[..., idx:idx+self.ne]; idx += self.ne
        mu_move  = inputs[..., idx:idx+2];        idx += 2
        logstd_mv= inputs[..., idx:idx+2];        idx += 2
        mu_aim   = inputs[..., idx:idx+2];        idx += 2
        logstd_am= inputs[..., idx:idx+2];        idx += 2
        logit_fr = inputs[..., idx:idx+1];        idx += 1

        self.cat = Categorical(logits=logits_t)
        self.mv  = Normal(mu_move, logstd_mv.exp())
        self.am  = Normal(mu_aim,  logstd_am.exp())
        self.fr  = Bernoulli(logits=logit_fr)

    def sample(self):
        t = self.cat.sample().unsqueeze(-1).float()
        mv = torch.tanh(self.mv.rsample())
        am = torch.tanh(self.am.rsample())
        fr = self.fr.sample().float()
        return torch.cat([t, mv, am, fr], dim=-1)

    def deterministic_sample(self):
        t = torch.argmax(self.cat.logits, dim=-1).unsqueeze(-1).float()
        mv = torch.tanh(self.mv.loc)
        am = torch.tanh(self.am.loc)
        fr = (self.fr.logits > 0).float()
        return torch.cat([t, mv, am, fr], dim=-1)

    def logp(self, x):
        eps = 1e-6
        t_idx = x[..., 0].long()
        mv = torch.clamp(x[..., 1:3], -1+eps, 1-eps)
        am = torch.clamp(x[..., 3:5], -1+eps, 1-eps)
        fr = x[..., 5]

        # inverse tanh
        z_mv = 0.5 * (torch.log1p(mv) - torch.log1p(-mv))
        z_am = 0.5 * (torch.log1p(am) - torch.log1p(-am))

        lp_t  = self.cat.log_prob(t_idx)
        lp_mv = self.mv.log_prob(z_mv).sum(-1) - torch.log(1 - mv.pow(2) + eps).sum(-1)
        lp_am = self.am.log_prob(z_am).sum(-1) - torch.log(1 - am.pow(2) + eps).sum(-1)
        p = torch.sigmoid(self.fr.logits.squeeze(-1))
        lp_fr = torch.where(fr > 0.5, torch.log(p + eps), torch.log(1 - p + eps))
        return lp_t + lp_mv + lp_am + lp_fr

    def entropy(self):
        cat_H = self.cat.entropy()
        mv_H  = self.mv.entropy().sum(-1)
        am_H  = self.am.entropy().sum(-1)
        p = torch.sigmoid(self.fr.logits.squeeze(-1))
        bern_H = -(p*torch.log(p+1e-8) + (1-p)*torch.log(1-p+1e-8))
        return cat_H + mv_H + am_H + bern_H

ModelCatalog.register_custom_action_dist("masked_multihead", MaskedTargetMoveAimFire)
