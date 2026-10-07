"""GPU-only correctness and full-size throughput gate; never trains on lab data."""

import json
import math
import os
import socket
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import LlamaConfig, LlamaForCausalLM, set_seed

from afmuon_lab5 import AFMuon, finite_cap_direction, polar_direction, compiled_polar_direction
from train_lab5 import TrainingLoss, evaluation_loss


def main():
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_STEP_ID")
    assert socket.gethostname().split(".")[0] == os.environ["SLURMD_NODENAME"].split(".")[0]
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    dist.init_process_group("nccl", device_id=device)
    print(json.dumps({"host": socket.gethostname(), "torch": torch.__version__,
                      "gpu": torch.cuda.get_device_name(), "allocated_job": os.environ["SLURM_JOB_ID"]}), flush=True)
    set_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True

    for kind in ("dense", "sparse", "zero"):
        momentum = torch.randn(256, 2048, device=device)
        if kind == "sparse":
            momentum[:, 128:] = 0
        if kind == "zero":
            momentum.zero_()
        expected = finite_cap_direction(momentum, 3.0, 32, compile_bisection=False)
        actual = finite_cap_direction(momentum, 3.0, 32)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        assert actual.abs().max().item() <= 3.0
        if kind == "dense":
            torch.testing.assert_close(actual.square().sum(1),
                                       torch.full((256,), 2048.0, device=device), rtol=1e-4, atol=1e-3)
    gradient = torch.randn(512, 2048, device=device)
    expected_momentum = torch.zeros_like(gradient)
    actual_momentum = torch.zeros_like(gradient)
    expected = polar_direction(gradient, expected_momentum, 0.95, 5)
    actual = compiled_polar_direction(gradient, actual_momentum, 0.95, 5)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.08, atol=0.05)
    torch.testing.assert_close(actual_momentum, expected_momentum)
    print("Optimizer oracle and momentum checks passed", flush=True)

    config = LlamaConfig(
        hidden_size=2048, intermediate_size=8192, num_hidden_layers=16,
        num_attention_heads=32, num_key_value_heads=8, vocab_size=128256,
        max_position_embeddings=8192, tie_word_embeddings=True, rope_theta=500000.0,
        rope_scaling={"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0,
                      "high_freq_factor": 4.0, "original_max_position_embeddings": 8192},
        use_cache=False,
    )
    config._attn_implementation = "flash_attention_2"
    model = LlamaForCausalLM(config).to(device=device, dtype=torch.float32)
    optimizer = AFMuon(model, muon_lr=0.02, vector_lr=0.0003, momentum=0.95,
                       matrix_weight_decay=0.1, ns_steps=5, tied_cap=3.0, tied_scale=0.5,
                       rho_hidden=50.0, rho_output=3000.0, oracle_chunk_rows=2048,
                       oracle_bisection_steps=32, eps=1e-8)
    inputs = torch.randint(0, config.vocab_size, (1, 8192), device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected_loss = TrainingLoss(model)(inputs).item()
    compiled = DistributedDataParallel(
        torch.compile(TrainingLoss(model), fullgraph=True, dynamic=False),
        device_ids=[0], broadcast_buffers=False, gradient_as_bucket_view=True,
    )
    for step in range(3):
        torch.cuda.synchronize()
        start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = compiled(input_ids=inputs)
        if step == 0:
            assert math.isclose(loss.item(), expected_loss, rel_tol=1e-3, abs_tol=1e-3), (loss.item(), expected_loss)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(compiled.parameters(), 1.0, error_if_nonfinite=True)
        torch.cuda.synchronize()
        model_seconds = time.monotonic() - start
        optimizer_start = time.monotonic()
        optimizer.step()
        torch.cuda.synchronize()
        optimizer_seconds = time.monotonic() - optimizer_start
        result = {"step": step, "loss": loss.item(), "grad_norm": gradient_norm.item(),
                  "model_seconds": model_seconds, "optimizer_seconds": optimizer_seconds,
                  "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                  "projected_8gpu_accum4_step_seconds": 4 * model_seconds + optimizer_seconds,
                  "note": "Single-GPU synthetic estimate excludes multi-GPU communication and validation"}
        print(json.dumps(result), flush=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.model(input_ids=inputs, use_cache=False).last_hidden_state[0, :-1]
        labels = inputs[0, 1:]
        expected = torch.nn.functional.cross_entropy(model.lm_head(hidden[:256]).float(), labels[:256], reduction="sum")
        actual = evaluation_loss(hidden[:256], model.lm_head.weight, labels[:256])
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-3)
    print("PREFLIGHT_PASSED", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
