import torch
from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz


class SplitQKVMuon(torch.optim.Muon):
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                momentum = state["momentum_buffer"]
                momentum.lerp_(parameter.grad, 1 - group["momentum"])
                update = (
                    parameter.grad.lerp(momentum, group["momentum"])
                    if group["nesterov"] else momentum
                )
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                for weight, block in zip(
                    parameter.chunk(3, dim=1), update.chunk(3, dim=1), strict=True
                ):
                    direction = _zeropower_via_newtonschulz(
                        block, group["ns_coefficients"], group["ns_steps"], group["eps"]
                    )
                    adjusted_lr = _adjust_lr(
                        group["lr"], group["adjust_lr_fn"], weight.shape
                    )
                    weight.add_(direction, alpha=-adjusted_lr)
        return loss


class MuonAdamW(torch.optim.Optimizer):
    def __init__(
        self, matrices, decay, no_decay, muon_lr, adamw_lr, weight_decay, betas, qkv
    ):
        self.muon = torch.optim.Muon(
            matrices, lr=muon_lr, weight_decay=weight_decay, momentum=0.95,
            nesterov=True, ns_steps=5, adjust_lr_fn="match_rms_adamw",
        )
        self.qkv = SplitQKVMuon(
            qkv, lr=muon_lr, weight_decay=weight_decay, momentum=0.95,
            nesterov=True, ns_steps=5, adjust_lr_fn="match_rms_adamw",
        )
        self.adam = torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=adamw_lr, betas=betas, fused=True,
        )
        super().__init__(
            self.muon.param_groups + self.qkv.param_groups + self.adam.param_groups,
            {"lr": muon_lr},
        )

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        self.muon.step()
        self.qkv.step()
        self.adam.step()
        return loss

    def state_dict(self):
        return {
            "muon": self.muon.state_dict(),
            "qkv": self.qkv.state_dict(),
            "adam": self.adam.state_dict(),
        }

    def load_state_dict(self, state):
        self.muon.load_state_dict(state["muon"])
        self.qkv.load_state_dict(state["qkv"])
        self.adam.load_state_dict(state["adam"])
        self.param_groups = self.muon.param_groups + self.qkv.param_groups + self.adam.param_groups
