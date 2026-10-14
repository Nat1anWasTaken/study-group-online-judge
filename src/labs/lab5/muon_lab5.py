import torch

POLAR_EXPRESS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@torch.compile(fullgraph=True, dynamic=False)
def orthogonalize(matrix):
    x = matrix.bfloat16()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.mT
    x = x / (x.norm() * 1.02 + 1e-6)
    for a, b, c in POLAR_EXPRESS:
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    return (x.mT if transposed else x).to(matrix.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, momentum):
        super().__init__(params, {"lr": lr, "weight_decay": weight_decay, "momentum": momentum})

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for parameter in group["params"]:
                state = self.state[parameter]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                buffer = state["momentum_buffer"]
                buffer.lerp_(parameter.grad, 1 - group["momentum"])
                update = orthogonalize(parameter.grad.lerp(buffer, group["momentum"]))
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"] * 0.2 * max(parameter.shape) ** 0.5)


class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, model, lr, weight_decay, betas):
        embedding = model.get_input_embeddings().weight
        assert model.get_output_embeddings().weight is embedding
        matrices, no_decay = [], []
        for parameter in model.parameters():
            if parameter is embedding:
                continue
            if parameter.ndim == 2:
                matrices.append(parameter)
            else:
                assert parameter.ndim == 1
                no_decay.append(parameter)
        self.muon = Muon(matrices, lr=lr, weight_decay=2 * weight_decay / 1.5, momentum=0.95)
        self.adam = torch.optim.AdamW(
            [
                {"params": [embedding], "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=betas,
            fused=True,
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
