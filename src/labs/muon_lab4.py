import math

import torch


POLAR_COEFFICIENTS = (
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
)


def polar_express(update):
    transposed = update.shape[0] > update.shape[1]
    x = update.bfloat16()
    if transposed:
        x = x.T
    x = x / (x.norm() * 1.01 + 1e-7)
    for a, b, c in POLAR_COEFFICIENTS:
        a, b, c = a / 1.01, b / 1.01**3, c / 1.01**5
        gram = x @ x.T
        polynomial = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.addmm(x, polynomial, x, beta=a)
    return x.T if transposed else x


class PolarMuon(torch.optim.Muon):
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
                momentum = state['momentum_buffer']
                momentum.lerp_(parameter.grad, 1 - group['momentum'])
                update = (
                    parameter.grad.lerp(momentum, group['momentum'])
                    if group['nesterov'] else momentum
                )
                update = polar_express(update)
                adjusted_lr = group['lr'] * 0.2 * math.sqrt(max(parameter.shape))
                parameter.mul_(1 - group['lr'] * group['weight_decay'])
                parameter.add_(update, alpha=-adjusted_lr)
        return loss


class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, matrices, decay, no_decay, lr, weight_decay, betas):
        self.muon = PolarMuon(
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
