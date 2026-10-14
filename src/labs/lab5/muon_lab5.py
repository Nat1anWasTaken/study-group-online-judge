import torch


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
        self.muon = torch.optim.Muon(
            matrices,
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            adjust_lr_fn="match_rms_adamw",
        )
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
