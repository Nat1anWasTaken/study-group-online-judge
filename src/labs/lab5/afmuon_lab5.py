import math

import torch
import triton
import triton.language as tl


@triton.jit
def _embedding_update_kernel(
    parameter,
    gradient,
    momentum,
    learning_rate,
    momentum_decay,
    WIDTH: tl.constexpr,
    CAP: tl.constexpr,
    BISECTION_STEPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    mask = columns < WIDTH
    offsets = row * WIDTH + columns
    learning_rate = learning_rate.to(tl.float32)
    momentum_decay = momentum_decay.to(tl.float32)
    updated_momentum = tl.load(
        momentum + offsets, mask, other=0
    ) * momentum_decay + tl.load(gradient + offsets, mask, other=0)
    tl.store(momentum + offsets, updated_momentum, mask)
    signs = tl.where(
        updated_momentum > 0, 1.0, tl.where(updated_momentum < 0, -1.0, 0.0)
    )
    magnitudes = tl.abs(updated_momentum)
    normalized = magnitudes / tl.maximum(
        tl.max(magnitudes, 0), tl.full((), 1e-38, tl.float32)
    )
    support = tl.sum((normalized != 0).to(tl.int32), 0)
    direction = signs * CAP
    if support * (CAP * CAP) > WIDTH:
        lower = 0.0
        upper = tl.sqrt(tl.full((), WIDTH, tl.float32)) / tl.sqrt(
            tl.sum(normalized * normalized, 0)
        )
        capped = tl.minimum(normalized * upper, CAP)
        squared_norm = tl.sum(capped * capped, 0)
        expansions = 0
        while (squared_norm < WIDTH) & (expansions < 128):
            lower = upper
            upper = 2.0 * upper
            capped = tl.minimum(normalized * upper, CAP)
            squared_norm = tl.sum(capped * capped, 0)
            expansions += 1
        for _ in range(BISECTION_STEPS):
            middle = (lower + upper) / 2.0
            capped = tl.minimum(normalized * middle, CAP)
            below_target = tl.sum(capped * capped, 0) < WIDTH
            lower = tl.where(below_target, middle, lower)
            upper = tl.where(below_target, upper, middle)
        direction = signs * tl.minimum(normalized * lower, CAP)
    weights = tl.load(parameter + offsets, mask, other=0)
    tl.store(parameter + offsets, weights - learning_rate * direction, mask)


@torch.no_grad()
def embedding_update_(
    parameter, gradient, momentum, learning_rate, momentum_decay, cap, bisection_steps
):
    assert parameter.is_cuda and parameter.dtype == torch.float32
    assert parameter.ndim == 2 and parameter.is_contiguous()
    assert gradient.shape == momentum.shape == parameter.shape
    assert gradient.is_contiguous() and momentum.is_contiguous()
    assert gradient.dtype == momentum.dtype == parameter.dtype
    assert gradient.device == momentum.device == parameter.device
    width = parameter.shape[1]
    with torch.cuda.device(parameter.device):
        _embedding_update_kernel[(parameter.shape[0],)](
            parameter,
            gradient,
            momentum,
            learning_rate,
            momentum_decay,
            WIDTH=width,
            CAP=cap,
            BISECTION_STEPS=bisection_steps,
            BLOCK=triton.next_power_of_2(width),
            num_warps=4,
            enable_fp_fusion=False,
        )


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


compiled_polar_direction = torch.compile(polar_direction, fullgraph=True, dynamic=False)


class AFMuon(torch.optim.Optimizer):
    def __init__(
        self,
        model,
        *,
        muon_lr,
        vector_lr,
        momentum,
        matrix_weight_decay,
        ns_steps,
        tied_cap,
        tied_scale,
        rho_hidden,
        rho_output,
        oracle_bisection_steps,
        eps,
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
        super().__init__(
            [
                {"params": matrices, "role": "matrix", "lr": muon_lr},
                {"params": [tied_embedding], "role": "tied", "lr": tied_learning_rate},
                {"params": vectors, "role": "vector", "lr": vector_lr},
            ],
            {},
        )
        self.momentum = momentum
        self.matrix_weight_decay = matrix_weight_decay
        self.ns_steps = ns_steps
        self.tied_cap = tied_cap
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
                    direction = compiled_polar_direction(
                        parameter.grad, momentum, self.momentum, self.ns_steps
                    )
                    parameter.mul_(1 - group["lr"] * self.matrix_weight_decay)
                    parameter.add_(direction, alpha=-group["lr"])
                    continue

                if group["role"] == "vector":
                    momentum.mul_(self.momentum).add_(parameter.grad)
                    rms = momentum.square().mean().sqrt().clamp_min(self.eps)
                    parameter.add_(momentum / rms, alpha=-group["lr"])
                else:
                    embedding_update_(
                        parameter,
                        parameter.grad,
                        momentum,
                        group["lr"],
                        self.momentum,
                        self.tied_cap,
                        self.oracle_bisection_steps,
                    )
        return loss
