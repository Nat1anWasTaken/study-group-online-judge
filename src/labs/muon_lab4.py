import math

import torch
from torch.optim._muon import _zeropower_via_newtonschulz


class NorMuon(torch.optim.Muon):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        for group in self.param_groups:
            group['beta2'] = 0.95

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(parameter)
                    state['second_momentum_buffer'] = torch.zeros_like(parameter[:1])
                momentum = state['momentum_buffer']
                momentum.lerp_(parameter.grad, 1 - group['momentum'])
                update = (
                    parameter.grad.lerp(momentum, group['momentum'])
                    if group['nesterov'] else momentum
                )
                update = _zeropower_via_newtonschulz(
                    update, group['ns_coefficients'], group['ns_steps'], group['eps']
                ).to(parameter.dtype)
                original_norm = update.norm()
                variance = state['second_momentum_buffer']
                variance.lerp_(
                    update.square().mean(dim=0, keepdim=True), 1 - group['beta2']
                )
                update = update / (variance.sqrt() + 1e-10)
                update = update * (original_norm / (update.norm() + 1e-10))
                adjusted_lr = group['lr'] * 0.2 * math.sqrt(max(parameter.shape))
                parameter.mul_(1 - group['lr'] * group['weight_decay'])
                parameter.add_(update, alpha=-adjusted_lr)
        return loss


class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, matrices, decay, no_decay, lr, weight_decay, betas):
        self.muon = NorMuon(
            matrices, lr=lr, weight_decay=weight_decay, momentum=0.95,
            nesterov=True, ns_steps=5, adjust_lr_fn="match_rms_adamw",
        )
        self.adam = torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=lr, betas=betas, fused=True,
        )
        super().__init__(self.muon.param_groups + self.adam.param_groups, {"lr": lr})

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        self.muon.step()
        self.adam.step()
        return loss

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adam": self.adam.state_dict()}

    def load_state_dict(self, state):
        self.muon.load_state_dict(state["muon"])
        self.adam.load_state_dict(state["adam"])
        self.param_groups = self.muon.param_groups + self.adam.param_groups
