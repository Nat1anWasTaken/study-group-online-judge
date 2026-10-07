import math

import torch


def polar_direction(gradient, momentum, momentum_decay, iterations):
    momentum.lerp_(gradient, 1 - momentum_decay)
    direction = gradient.lerp(momentum, momentum_decay).bfloat16()
    transpose = direction.shape[0] > direction.shape[1]
    if transpose:
        direction = direction.T
    direction = direction / (direction.norm() + 1e-7)

    for _ in range(iterations):
        gram = direction @ direction.T
        polynomial = -4.7750 * gram + 2.0315 * (gram @ gram)
        direction = 3.4445 * direction + polynomial @ direction

    if transpose:
        direction = direction.T
    rows, columns = gradient.shape
    return direction * math.sqrt(max(1.0, rows / columns))


def finite_cap_direction(momentum, cap, bisection_steps):
    width = momentum.shape[1]
    magnitudes = momentum.abs()
    normalized = magnitudes / magnitudes.amax(dim=1, keepdim=True).clamp_min(1e-38)
    support_size = normalized.count_nonzero(dim=1)
    active_rows = support_size * cap ** 2 > width
    direction = momentum.sign() * cap
    if not active_rows.any().item():
        return direction

    active_magnitudes = normalized[active_rows]
    lower = torch.zeros_like(active_magnitudes[:, :1])
    upper = math.sqrt(width) / active_magnitudes.square().sum(dim=1, keepdim=True).sqrt()

    for _ in range(128):
        squared_norm = (active_magnitudes * upper).clamp_max(cap).square().sum(dim=1, keepdim=True)
        expand = squared_norm < width
        if not expand.any().item():
            break
        lower = torch.where(expand, upper, lower)
        upper = torch.where(expand, 2 * upper, upper)

    squared_norm = (active_magnitudes * upper).clamp_max(cap).square().sum(dim=1)
    assert torch.isfinite(upper).all().item() and (squared_norm >= width).all().item()

    for _ in range(bisection_steps):
        middle = (lower + upper) / 2
        squared_norm = (active_magnitudes * middle).clamp_max(cap).square().sum(dim=1, keepdim=True)
        below_target = squared_norm < width
        lower = torch.where(below_target, middle, lower)
        upper = torch.where(below_target, upper, middle)

    direction[active_rows] = (
        momentum[active_rows].sign() * (active_magnitudes * lower).clamp_max(cap)
    )
    return direction


class AFMuon(torch.optim.Optimizer):
    def __init__(
        self, model, *, muon_lr, vector_lr, momentum, matrix_weight_decay,
        ns_steps, tied_cap, tied_scale, rho_hidden, rho_output,
        oracle_chunk_rows, oracle_bisection_steps, eps,
    ):
        tied_embedding = model.get_input_embeddings().weight
        assert model.get_output_embeddings().weight is tied_embedding
        matrices, vectors = [], []
        for parameter in model.parameters():
            if parameter is tied_embedding:
                continue
            if parameter.ndim == 2:
                matrices.append(parameter)
            else:
                assert parameter.ndim == 1
                vectors.append(parameter)

        tied_learning_rate = (
            muon_lr * rho_output / rho_hidden * tied_scale / tied_embedding.shape[1]
        )
        super().__init__([
            {"params": matrices, "role": "matrix", "lr": muon_lr},
            {"params": [tied_embedding], "role": "tied", "lr": tied_learning_rate},
            {"params": vectors, "role": "vector", "lr": vector_lr},
        ], {})
        self.momentum = momentum
        self.matrix_weight_decay = matrix_weight_decay
        self.ns_steps = ns_steps
        self.tied_cap = tied_cap
        self.oracle_chunk_rows = oracle_chunk_rows
        self.oracle_bisection_steps = oracle_bisection_steps
        self.eps = eps
        for group in self.param_groups:
            group["peak_lr"] = group["lr"]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                assert parameter.grad is not None
                state = self.state[parameter]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(parameter)
                momentum = state["momentum"]

                if group["role"] == "matrix":
                    direction = polar_direction(
                        parameter.grad, momentum, self.momentum, self.ns_steps
                    )
                    parameter.mul_(1 - group["lr"] * self.matrix_weight_decay)
                    parameter.add_(direction, alpha=-group["lr"])
                    continue

                momentum.mul_(self.momentum).add_(parameter.grad)
                if group["role"] == "vector":
                    rms = momentum.square().mean().sqrt().clamp_min(self.eps)
                    parameter.add_(momentum / rms, alpha=-group["lr"])
                else:
                    for start in range(0, len(parameter), self.oracle_chunk_rows):
                        stop = start + self.oracle_chunk_rows
                        direction = finite_cap_direction(
                            momentum[start:stop], self.tied_cap,
                            self.oracle_bisection_steps,
                        )
                        parameter[start:stop].add_(direction, alpha=-group["lr"])
        return loss
