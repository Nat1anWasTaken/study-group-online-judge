import json
import math
import os
import random
import shutil
import socket
import subprocess
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from afmuon_lab5 import AFMuon
from datasets import load_from_disk
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM, set_seed
from transformers.integrations.flash_attention import (
    flash_attention_forward as hf_flash_attention_forward,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def flash_attention_with_matching_dtype(
    module, query, key, value, attention_mask, **kwargs
):
    return hf_flash_attention_forward(
        module,
        query.to(value.dtype),
        key.to(value.dtype),
        value,
        attention_mask,
        **kwargs,
    )


ALL_ATTENTION_FUNCTIONS.register(
    "flash_attention_2", flash_attention_with_matching_dtype
)

experiment_name = (
    "afmuon-oracle-rho50-3000-tiedcap3-scale0.5-mlr0.02-vlr0.0003-b262144-s42"
)


class TrainingLoss(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        batch_size, sequence_length = input_ids.shape
        cu_seq_lens = (
            torch.arange(
                batch_size + 1,
                device=input_ids.device,
                dtype=torch.int32,
            )
            * sequence_length
        )
        return self.model(
            input_ids=input_ids,
            labels=input_ids,
            use_cache=False,
            cu_seq_lens_q=cu_seq_lens,
            cu_seq_lens_k=cu_seq_lens,
            max_length_q=sequence_length,
            max_length_k=sequence_length,
        ).loss


@torch.compile(fullgraph=True, dynamic=True)
def evaluation_loss(hidden_states, weight, labels):
    logits = F.linear(hidden_states, weight)
    return F.cross_entropy(logits.float(), labels, reduction="sum")


@torch.inference_mode()
def evaluate(model, documents, device, rank, world_size):
    started = time.monotonic()
    model.eval()
    totals = torch.zeros(2, dtype=torch.float64, device=device)

    for document_index in range(rank, len(documents), world_size):
        input_ids = torch.tensor(
            documents[document_index]["input_ids"], dtype=torch.long, device=device
        ).unsqueeze(0)
        if input_ids.shape[1] < 2:
            continue

        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden_states = model.model(
                input_ids=input_ids, use_cache=False
            ).last_hidden_state[0, :-1]
        labels = input_ids[0, 1:]

        for start in range(0, len(labels), 256):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                negative_log_likelihood = evaluation_loss(
                    hidden_states[start : start + 256],
                    model.lm_head.weight,
                    labels[start : start + 256],
                )
            totals[0] += negative_log_likelihood.double()
        totals[1] += len(labels)

    dist.all_reduce(totals)
    mean_loss = (totals[0] / totals[1]).item()
    model.train()
    return {
        "eval/loss": mean_loss,
        "eval/perplexity": math.exp(mean_loss),
        "eval/seconds": time.monotonic() - started,
    }


def main():
    hostname = socket.gethostname().split(".")[0]
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_STEP_ID")
    assert hostname == os.environ["SLURMD_NODENAME"].split(".")[0]

    data_dir = Path(
        os.environ.get("LAB5_DATA", "/home/nat1andotxyz/lab5/dolma-seed42-8192-v1")
    )
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    tokens_per_step = world_size * 1 * 4 * 8192

    git_commit = (
        os.environ.get("LAB5_GIT_COMMIT")
        or subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"],
            cwd=Path.cwd(),
            text=True,
        ).strip()
    )
    run_name = os.environ.get(
        "LAB5_RUN_NAME",
        f"{os.environ['SLURM_JOB_NAME']}/{experiment_name}-g{git_commit}-j{os.environ['SLURM_JOB_ID']}",
    )
    output_dir = Path(
        os.environ.get(
            "LAB5_OUTPUT",
            f"/home/nat1andotxyz/lab5/runs/{run_name}",
        )
    )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    set_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True

    manifest = json.loads((data_dir / "manifest.json").read_text())
    training_data = load_from_disk(data_dir / "train").with_format("torch")
    evaluation_data = load_from_disk(data_dir / "eval")
    tokenizer = AutoTokenizer.from_pretrained(
        data_dir / "tokenizer", local_files_only=True
    )
    model_config = AutoConfig.from_pretrained(
        data_dir / "model-config", local_files_only=True
    )
    model_config.max_position_embeddings = 8192
    model_config.use_cache = False
    model_config.pad_token_id = tokenizer.pad_token_id
    model_config._attn_implementation = "flash_attention_2"

    raw_model = LlamaForCausalLM(model_config).to(device=device, dtype=torch.float32)
    compiled_loss = torch.compile(
        TrainingLoss(raw_model), fullgraph=True, dynamic=False
    )
    model = DistributedDataParallel(
        compiled_loss,
        device_ids=[local_rank],
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        bucket_cap_mb=128,
    )
    optimizer = AFMuon(
        raw_model,
        muon_lr=0.02,
        vector_lr=0.0003,
        momentum=0.95,
        matrix_weight_decay=0.1,
        ns_steps=5,
        tied_cap=3.0,
        tied_scale=0.5,
        rho_hidden=50.0,
        rho_output=3000.0,
        oracle_bisection_steps=32,
        eps=1e-8,
    )
    sampler = DistributedSampler(
        training_data,
        num_replicas=world_size,
        rank=rank,
        seed=42,
        drop_last=True,
    )
    loader = DataLoader(
        training_data,
        batch_size=1,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    batches = iter(loader)
    max_steps = min(
        6000000000 // tokens_per_step,
        len(loader) // 4,
    )
    if max_steps == 0:
        raise ValueError(
            "The dataset and token budget must allow at least one training step."
        )

    allocation_start = float(os.environ["LAB5_ALLOCATION_START"])
    allocation_seconds = int(os.environ.get("LAB5_ALLOCATION_SECONDS", "7200"))
    training_deadline = allocation_start + allocation_seconds - 600
    eval_steps = min(2000, max(1, max_steps // 10))
    eval_seconds = 600
    profile_run = os.environ.get("LAB5_PROFILE") == "1"
    if profile_run:
        max_steps = min(max_steps, 50)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        training_config = {
            "dataset_id": "allenai/dolma3_mix-150B-1025",
            "dataset_revision": "afa92bfb22366821c5e6cd427cdd036b34b713ef",
            "model_id": "meta-llama/Llama-3.2-1B",
            "model_revision": "4e20de362430cd3b72f300e6b0f18e50e7166e08",
            "seed": 42,
            "holdout_documents": 50000,
            "eval_documents": 1024,
            "sequence_length": 8192,
            "max_train_tokens": 6000000000,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "muon_lr": 0.02,
            "vector_lr": 0.0003,
            "momentum": 0.95,
            "matrix_weight_decay": 0.1,
            "ns_steps": 5,
            "tied_cap": 3.0,
            "tied_scale": 0.5,
            "rho_hidden": 50.0,
            "rho_output": 3000.0,
            "oracle_bisection_steps": 32,
            "eps": 1e-8,
            "max_grad_norm": 1.0,
            "warmup_tokens": 50000000,
            "schedule": "warmup_constant",
            "logging_steps": 10,
            "eval_steps": eval_steps,
            "eval_seconds": eval_seconds,
            "allocation_seconds": allocation_seconds,
            "finalize_reserve_seconds": 600,
            "max_h200_hours": 64,
            "attention_implementation": "flash_attention_2",
            "gradient_checkpointing": False,
            "torch_compile": True,
            "world_size": world_size,
            "ddp_bucket_cap_mb": 128,
            "wandb_project": "lab5-training-llama",
            "optimizer_reference": "https://arxiv.org/abs/2610.01395",
            "optimizer_reference_commit": "2589a530d8a0e99cac2d9f5082a54e31e9146118",
            "effective_batch_tokens": tokens_per_step,
            "model_config": model_config.to_dict(),
            "data_manifest": manifest,
        }
        run = wandb.init(
            project="lab5-training-llama",
            name=run_name,
            config=training_config,
            dir=str(output_dir),
        )
        wandb.define_metric("train/total_tokens_seen")
        wandb.define_metric("train/*", step_metric="train/total_tokens_seen")
        wandb.define_metric("eval/*", step_metric="train/total_tokens_seen")
        (output_dir / "training-config.json").write_text(
            json.dumps(training_config, indent=2)
        )
        print(run.url, flush=True)

    completed_steps = 0
    total_tokens_seen = 0
    interval_steps = 0
    training_seconds = 0.0
    interval_loss = torch.zeros((), dtype=torch.float64, device=device)

    def save_progress():
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device),
        }
        rank_rng_states = [None] * world_size if rank == 0 else None
        dist.gather_object(rng_state, rank_rng_states, dst=0)
        if rank == 0:
            checkpoints = output_dir / "checkpoints"
            checkpoints.mkdir(exist_ok=True)
            latest = checkpoints / "latest"
            previous = latest.resolve() if latest.is_symlink() else None
            staging = Path(tempfile.mkdtemp(prefix=".saving-", dir=checkpoints))
            raw_model.save_pretrained(staging / "model", max_shard_size="2GB")
            tokenizer.save_pretrained(staging / "model")
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "step": completed_steps,
                    "total_tokens_seen": total_tokens_seen,
                    "training_seconds": training_seconds,
                    "world_size": world_size,
                    "sampler_seed": sampler.seed,
                    "sampler_epoch": sampler.epoch,
                    "batches_consumed_per_rank": completed_steps * 4,
                    "rng_states": rank_rng_states,
                },
                staging / "optimizer.pt",
            )
            result = {
                "run_name": run_name,
                "step": completed_steps,
                "total_tokens_seen": total_tokens_seen,
                "wall_hours": (time.time() - allocation_start) / 3600,
                "training_hours": training_seconds / 3600,
                "h200_hours": (time.time() - allocation_start) * world_size / 3600,
                "perplexity": evaluation_metrics["eval/perplexity"],
                "wandb_url": run.url,
            }
            (staging / "result.json").write_text(json.dumps(result, indent=2))
            checkpoint = checkpoints / f"step-{completed_steps:08d}"
            staging.rename(checkpoint)
            for name in ("model", "optimizer.pt", "result.json"):
                exported = output_dir / name
                if not exported.is_symlink():
                    exported.symlink_to(Path("checkpoints/latest") / name)
            pending_link = checkpoints / ".latest-next"
            pending_link.symlink_to(checkpoint.name)
            os.replace(pending_link, latest)
            wandb.run.summary.update(result)
            print(
                json.dumps({"checkpoint": str(checkpoint), "step": completed_steps}),
                flush=True,
            )
            if (
                previous is not None
                and previous != checkpoint.resolve()
                and previous.parent == checkpoints.resolve()
                and previous.name.startswith("step-")
            ):
                shutil.rmtree(previous)
        dist.barrier()

    evaluation_metrics = evaluate(raw_model, evaluation_data, device, rank, world_size)
    evaluation_metrics.update(
        {
            "train/total_tokens_seen": 0,
            "eval/step": 0,
            "eval/h200_hours": (time.time() - allocation_start) * world_size / 3600,
            "eval/wall_hours": (time.time() - allocation_start) / 3600,
            "eval/training_hours": 0.0,
        }
    )
    if rank == 0:
        wandb.log(evaluation_metrics)
        with (output_dir / "eval-curve.jsonl").open("a") as file:
            file.write(json.dumps(evaluation_metrics) + "\n")
        print(json.dumps(evaluation_metrics), flush=True)
    if not profile_run:
        save_progress()
    last_eval_step = 0
    last_eval_time = time.monotonic()

    profiler = None
    if profile_run and rank == 0:

        def write_profile(profile):
            profile.export_chrome_trace(str(output_dir / "profile.json"))
            table = profile.key_averages().table(
                sort_by="self_device_time_total", row_limit=35
            )
            (output_dir / "profile.txt").write_text(table)
            print(table, flush=True)

        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=20, warmup=2, active=3, repeat=1),
            on_trace_ready=write_profile,
        )
        profiler.start()

    for step in range(1, max_steps + 1):
        stop = torch.tensor(int(time.time() >= training_deadline), device=device)
        dist.broadcast(stop, src=0)
        if stop.item():
            break

        torch.cuda.synchronize(device)
        step_start = time.monotonic()
        learning_rate_scale = min(1.0, (total_tokens_seen + tokens_per_step) / 50000000)
        for group in optimizer.param_groups:
            group["lr"] = group["peak_lr"] * learning_rate_scale
        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(4):
            input_ids = next(batches)["input_ids"].to(
                device, dtype=torch.long, non_blocking=True
            )
            sync_gradients = micro_step == 3
            with nullcontext() if sync_gradients else model.no_sync():
                with (
                    torch.profiler.record_function("forward")
                    if profiler
                    else nullcontext()
                ):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = model(input_ids=input_ids)
                with (
                    torch.profiler.record_function("backward")
                    if profiler
                    else nullcontext()
                ):
                    (loss / 4).backward()
            interval_loss += loss.detach().double() / 4

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        with torch.profiler.record_function("optimizer") if profiler else nullcontext():
            optimizer.step()
        torch.cuda.synchronize(device)
        step_seconds = torch.tensor(time.monotonic() - step_start, device=device)
        dist.all_reduce(step_seconds, op=dist.ReduceOp.MAX)
        training_seconds += step_seconds.item()
        completed_steps = step
        total_tokens_seen += tokens_per_step
        interval_steps += 1

        stop = torch.tensor(int(time.time() >= training_deadline), device=device)
        dist.broadcast(stop, src=0)
        if step % 10 == 0 or step == max_steps or stop.item():
            dist.all_reduce(interval_loss)
            if rank == 0:
                training_metrics = {
                    "train/loss": interval_loss.item() / (world_size * interval_steps),
                    "train/grad_norm": gradient_norm.item(),
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/tied_learning_rate": optimizer.param_groups[1]["lr"],
                    "train/vector_learning_rate": optimizer.param_groups[2]["lr"],
                    "train/tokens_per_second": tokens_per_step / step_seconds.item(),
                    "train/total_tokens_seen": total_tokens_seen,
                    "train/step": completed_steps,
                    "train/step_seconds": step_seconds.item(),
                    "train/wall_hours": (time.time() - allocation_start) / 3600,
                    "train/training_hours": training_seconds / 3600,
                    "train/peak_allocated_gb": torch.cuda.max_memory_allocated(device)
                    / 1e9,
                    "train/h200_hours": (time.time() - allocation_start)
                    * world_size
                    / 3600,
                }
                wandb.log(training_metrics)
                with (output_dir / "train-curve.jsonl").open("a") as file:
                    file.write(json.dumps(training_metrics) + "\n")
                print(json.dumps(training_metrics), flush=True)
            interval_loss.zero_()
            interval_steps = 0

        evaluation_due = (
            step - last_eval_step >= eval_steps
            or time.monotonic() - last_eval_time >= eval_seconds
            or step == max_steps
            or stop.item()
        )
        if profile_run:
            evaluation_due = False
        evaluate_now = torch.tensor(int(evaluation_due), device=device)
        dist.broadcast(evaluate_now, src=0)
        if evaluate_now.item():
            evaluation_metrics = evaluate(
                raw_model, evaluation_data, device, rank, world_size
            )
            evaluation_metrics.update(
                {
                    "train/total_tokens_seen": total_tokens_seen,
                    "eval/step": completed_steps,
                    "eval/wall_hours": (time.time() - allocation_start) / 3600,
                    "eval/training_hours": training_seconds / 3600,
                    "eval/h200_hours": (time.time() - allocation_start)
                    * world_size
                    / 3600,
                }
            )
            if rank == 0:
                wandb.log(evaluation_metrics)
                with (output_dir / "eval-curve.jsonl").open("a") as file:
                    file.write(json.dumps(evaluation_metrics) + "\n")
                print(json.dumps(evaluation_metrics), flush=True)
            save_progress()
            last_eval_step = completed_steps
            last_eval_time = time.monotonic()
        if stop.item():
            break
        if profiler:
            profiler.step()

    if profiler:
        profiler.stop()
    if profile_run:
        if rank == 0:
            wandb.finish()
        dist.destroy_process_group()
        return

    if interval_steps:
        dist.all_reduce(interval_loss)
        if rank == 0:
            wandb.log(
                {
                    "train/loss": interval_loss.item() / (world_size * interval_steps),
                    "train/grad_norm": gradient_norm.item(),
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/tokens_per_second": tokens_per_step / step_seconds.item(),
                    "train/total_tokens_seen": total_tokens_seen,
                }
            )

    if last_eval_step != completed_steps:
        evaluation_metrics = evaluate(
            raw_model, evaluation_data, device, rank, world_size
        )
        evaluation_metrics.update(
            {
                "train/total_tokens_seen": total_tokens_seen,
                "eval/step": completed_steps,
                "eval/wall_hours": (time.time() - allocation_start) / 3600,
                "eval/training_hours": training_seconds / 3600,
                "eval/h200_hours": (time.time() - allocation_start) * world_size / 3600,
            }
        )
        if rank == 0:
            wandb.log(evaluation_metrics)
            with (output_dir / "eval-curve.jsonl").open("a") as file:
                file.write(json.dumps(evaluation_metrics) + "\n")
            print(json.dumps(evaluation_metrics), flush=True)

        save_progress()

    if rank == 0:
        wandb.finish()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
