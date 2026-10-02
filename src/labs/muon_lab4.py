import math
from collections import defaultdict

import torch
from torch.optim._muon import _zeropower_via_newtonschulz

BATCHED = True
COMPILED = True


def matrix_update(weight, grad, momentum, lr):
    momentum.lerp_(grad, 0.05)
    update = grad.lerp(momentum, 0.95)
    direction = _zeropower_via_newtonschulz(update, (3.4445, -4.7750, 2.0315), 5, 1e-7)
    weight.mul_(1 - lr * 0.1)
    weight.add_(direction.float() * (-lr * (0.2 * math.sqrt(max(weight.shape)))))


def batch_update(weights, grads, momenta, lr):
    momenta.lerp_(grads, 0.05)
    update = grads.lerp(momenta, 0.95).bfloat16()
    transposed = update.shape[-2] > update.shape[-1]
    if transposed:
        update = update.transpose(-2, -1)
    update.div_(update.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-7))
    for _ in range(5):
        gram = update @ update.transpose(-2, -1)
        gram_update = torch.baddbmm(gram, gram, gram, beta=-4.7750, alpha=2.0315)
        update = torch.baddbmm(update, gram_update, update, beta=3.4445)
    if transposed:
        update = update.transpose(-2, -1)
    weights.mul_(1 - lr * 0.1)
    weights.add_(update.float() * (-lr * (0.2 * math.sqrt(max(weights.shape[-2:])))))


update_weights = batch_update if BATCHED else matrix_update
if COMPILED:
    update_weights = torch.compile(update_weights, fullgraph=True, dynamic=False)


class ExecutionMuon(torch.optim.Muon):
    def __init__(self, params, *, split_qkv=False, **kwargs):
        super().__init__(params, **kwargs)
        self.split_qkv = split_qkv
        self.lr_tensor = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if (
                group["momentum"],
                group["weight_decay"],
                group["ns_steps"],
                group["nesterov"],
                group["adjust_lr_fn"],
            ) != (0.95, 0.1, 5, True, "match_rms_adamw"):
                raise ValueError("W experiments require the fixed L2 Muon recipe")
            buckets = defaultdict(list)
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                if self.lr_tensor is None:
                    self.lr_tensor = torch.zeros(
                        (), device=parameter.device, dtype=torch.float32
                    )
                if self.split_qkv:
                    parts = zip(
                        parameter.chunk(3, 1),
                        parameter.grad.chunk(3, 1),
                        state["momentum_buffer"].chunk(3, 1),
                        strict=True,
                    )
                else:
                    parts = [(parameter, parameter.grad, state["momentum_buffer"])]
                for weight, grad, momentum in parts:
                    buckets[(tuple(weight.shape), weight.dtype, weight.device)].append(
                        (weight, grad, momentum)
                    )
            if not buckets:
                continue
            self.lr_tensor.fill_(group["lr"])
            for entries in buckets.values():
                if BATCHED:
                    weights, grads, momenta = [
                        torch.stack(items) for items in zip(*entries, strict=True)
                    ]
                    update_weights(weights, grads, momenta, self.lr_tensor)
                    for index, (weight, grad, momentum) in enumerate(entries):
                        weight.copy_(weights[index])
                        momentum.copy_(momenta[index])
                else:
                    for weight, grad, momentum in entries:
                        update_weights(weight, grad, momentum, self.lr_tensor)
        return loss


class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, matrices, decay, no_decay, lr, weight_decay, betas, qkv):
        self.muon = ExecutionMuon(
            matrices,
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            adjust_lr_fn="match_rms_adamw",
        )
        self.qkv = ExecutionMuon(
            qkv,
            split_qkv=True,
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            adjust_lr_fn="match_rms_adamw",
        )
        self.adam = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=betas,
            fused=True,
        )
        super().__init__(
            self.muon.param_groups + self.qkv.param_groups + self.adam.param_groups,
            {"lr": lr},
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
        self.param_groups = (
            self.muon.param_groups + self.qkv.param_groups + self.adam.param_groups
        )
