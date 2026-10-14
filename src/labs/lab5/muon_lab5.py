import torch


class SOAP(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas):
        super().__init__(params, {"lr": lr, "weight_decay": weight_decay, "betas": betas})

    @staticmethod
    def rotate(x, bases, back=False):
        left, right = bases
        if left is not None:
            x = (left if back else left.mT) @ x
        if right is not None:
            x = x @ (right.mT if back else right)
        return x

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                grad = parameter.grad.float()
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    state["grams"] = [
                        0.05 * (grad @ grad.mT if side == 0 else grad.mT @ grad) if n <= 2048 else None
                        for side, n in enumerate(grad.shape)
                    ]
                    state["bases"] = [
                        None if gram is None else torch.linalg.eigh(gram)[1].flip(1)
                        for gram in state["grams"]
                    ]
                    continue
                state["step"] += 1
                bases = state["bases"]
                state["exp_avg"].lerp_(grad, 1 - beta1)
                state["exp_avg_sq"].lerp_(self.rotate(grad, bases).square(), 1 - beta2)
                direction = self.rotate(
                    self.rotate(state["exp_avg"], bases) / (state["exp_avg_sq"].sqrt() + 1e-8),
                    bases,
                    back=True,
                )
                correction = (1 - beta2 ** state["step"]) ** 0.5 / (1 - beta1 ** state["step"])
                parameter.add_(direction, alpha=-group["lr"] * correction)
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                for side, gram in enumerate(state["grams"]):
                    if gram is None:
                        continue
                    gram.lerp_(grad @ grad.mT if side == 0 else grad.mT @ grad, 0.05)
                    if state["step"] % 10 == 0:
                        basis = bases[side]
                        product = gram @ basis
                        order = (basis * product).sum(0).argsort(descending=True)
                        state["exp_avg_sq"] = state["exp_avg_sq"].index_select(side, order)
                        bases[side] = torch.linalg.qr(product[:, order]).Q


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
        self.muon = SOAP(matrices, lr=lr, weight_decay=weight_decay, betas=betas)
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
