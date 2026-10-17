import json
import math
import os
import socket
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from datasets import load_from_disk
from muon_lab5 import MuonAdamW
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
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

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1 * 1e-3 / LEARNING_RATE / 4
ADAM_BETAS = (0.9, 0.9872585449014338)
ACCUMULATION = 1
WARMUP_STEPS = 800
DECAY_FRACTION = 0.6
SEED = 42
DATA_SEED = 42
EVAL_DOCUMENTS = 1024
FINAL_EVAL_DOCUMENTS = 8192
MAX_DOCUMENTS = 128
TRAIN_STEPS = int(os.environ.get("LAB5_STEPS", "45776"))
SAVE_STEPS = {int(step) for step in os.environ.get("LAB5_SAVE_STEPS", "18310").split(",") if step}
RESUME = os.environ.get("LAB5_RESUME")
DECAY_START = TRAIN_STEPS * (1 - DECAY_FRACTION)
DISTILLATION_ALPHA = 0.25
experiment_name = f"muon-adamw-lr{LEARNING_RATE:g}-wd{WEIGHT_DECAY:g}-muonwd{2 * WEIGHT_DECAY:g}-polarexpress-precond-exactrms0.2-docmask-trunc8k-wsd60-b65536-beta2token-distill8b-a{DISTILLATION_ALPHA:g}-s42"
TAIL_EPSILON = 1e-7
TEACHER_ARRAYS = ("teacher_ids", "teacher_log_probs", "teacher_tail")


def learning_rate_scale(step):
    if step < WARMUP_STEPS:
        return step / WARMUP_STEPS
    if step <= DECAY_START:
        return 1.0
    return (TRAIN_STEPS - step) / (TRAIN_STEPS - DECAY_START)


class TrainingLoss(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, position_ids, cu_seq_lens, teacher_ids, teacher_log_probs, teacher_tail):
        hidden_states = self.model.model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
            cu_seq_lens_q=cu_seq_lens,
            cu_seq_lens_k=cu_seq_lens,
            max_length_q=input_ids.shape[1],
            max_length_k=input_ids.shape[1],
        ).last_hidden_state[0, :-1]
        logits = self.model.lm_head(hidden_states).float()
        log_normalizer = logits.logsumexp(-1, keepdim=True)
        cross_entropy = (log_normalizer - logits.gather(-1, input_ids[0, 1:, None])).mean()
        student_top = logits.gather(-1, teacher_ids[:-1].long()) - log_normalizer
        student_tail = (1 - student_top.exp().sum(-1)).clamp_min(TAIL_EPSILON)
        target_tail = teacher_tail[:-1]
        target_top = teacher_log_probs[:-1].float().exp()
        target_top = target_top * ((1 - target_tail) / target_top.sum(-1))[:, None]
        kl = (
            torch.xlogy(target_top, target_top).sum(-1)
            - (target_top * student_top).sum(-1)
            + torch.xlogy(target_tail, target_tail)
            - target_tail * student_tail.log()
        ).mean()
        loss = DISTILLATION_ALPHA * cross_entropy + (1 - DISTILLATION_ALPHA) * kl
        return loss, cross_entropy.detach(), kl.detach()


class TeacherCacheDataset(Dataset):
    def __init__(self, data, cache_dir, block_ids):
        self.data = data
        self.cache_dir = cache_dir
        self.block_ids = block_ids
        self.arrays = None

    def __len__(self):
        return len(self.block_ids)

    def __getitem__(self, slot):
        if self.arrays is None:
            self.arrays = [np.load(self.cache_dir / f"{name}.npy", mmap_mode="r") for name in TEACHER_ARRAYS]
        return {
            "input_ids": self.data[int(self.block_ids[slot])]["input_ids"],
            **{name: torch.from_numpy(np.array(array[slot])) for name, array in zip(TEACHER_ARRAYS, self.arrays)},
        }


def document_boundaries(input_ids, bos_token_id):
    tokens = input_ids[0]
    is_start = tokens == bos_token_id
    is_start[0] = True
    starts = torch.nonzero_static(is_start, size=MAX_DOCUMENTS, fill_value=len(tokens)).flatten()
    cu_seq_lens = torch.cat([starts, starts.new_tensor([len(tokens)])]).int()
    indices = torch.arange(len(tokens), device=tokens.device)
    return (indices - torch.cummax(indices * is_start, 0).values)[None], cu_seq_lens


@torch.compile(fullgraph=True, dynamic=True)
def evaluation_loss(hidden_states, weight, labels):
    logits = F.linear(hidden_states, weight)
    return F.cross_entropy(logits.float(), labels, reduction="sum")


@torch.inference_mode()
def evaluate(model, documents, limit, device):
    model.eval()
    losses = torch.zeros(limit, dtype=torch.float64, device=device)
    counts = torch.zeros_like(losses)
    for index, input_ids in documents:
        if index >= limit or len(input_ids) < 2:
            continue
        input_ids = torch.tensor(input_ids, device=device)[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden_states = model.model(input_ids=input_ids).last_hidden_state[0, :-1]
            labels = input_ids[0, 1:]
            for start in range(0, len(labels), 256):
                losses[index] += evaluation_loss(
                    hidden_states[start : start + 256],
                    model.lm_head.weight,
                    labels[start : start + 256],
                ).double()
        counts[index] = len(labels)
    totals = torch.stack([losses, counts])
    dist.all_reduce(totals)
    model.train()
    return totals[0], totals[1]


def main():
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_STEP_ID")
    assert socket.gethostname().split(".")[0] == os.environ["SLURMD_NODENAME"].split(".")[0]
    data_dir = Path(os.environ["LAB5_DATA"])
    output_dir = Path(os.environ["LAB5_OUTPUT"])
    cache_dir = Path(os.environ["LAB5_TEACHER_CACHE"])
    cache_metadata = json.loads((cache_dir / "cache.json").read_text())
    git_commit = os.environ["LAB5_GIT_COMMIT"]
    run_name = f"{os.environ['SLURM_JOB_NAME']}/{experiment_name}-g{git_commit}-j{os.environ['SLURM_JOB_ID']}"
    allocation_start = float(os.environ["LAB5_ALLOCATION_START"])
    allocation_seconds = int(os.environ["LAB5_ALLOCATION_SECONDS"])
    training_deadline = allocation_start + allocation_seconds - 300
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    tokens_per_step = world_size * 1 * ACCUMULATION * 8192

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    set_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

    manifest = json.loads((data_dir / "manifest.json").read_text())
    assert cache_metadata["data_manifest"] == manifest
    training_data = load_from_disk(data_dir / "train").with_format("torch")
    tokenizer = AutoTokenizer.from_pretrained(data_dir / "tokenizer", local_files_only=True)
    local_indices = range(rank, FINAL_EVAL_DOCUMENTS, world_size)
    holdout = load_from_disk(data_dir / "judge-holdout").select(local_indices)
    evaluation_documents = list(zip(local_indices, tokenizer(
        list(holdout["text"]), truncation=True, max_length=8192,
        return_attention_mask=False, verbose=False,
    )["input_ids"]))

    if RESUME:
        raw_model = LlamaForCausalLM.from_pretrained(
            Path(RESUME) / "model", dtype=torch.float32,
            attn_implementation="flash_attention_2",
        ).to(device)
    else:
        model_config = AutoConfig.from_pretrained(data_dir / "model-config", local_files_only=True)
        model_config.max_position_embeddings = 8192
        model_config.use_cache = False
        model_config.pad_token_id = tokenizer.pad_token_id
        model_config._attn_implementation = "flash_attention_2"
        raw_model = LlamaForCausalLM(model_config).to(device=device, dtype=torch.float32)
    model = DistributedDataParallel(
        torch.compile(TrainingLoss(raw_model), fullgraph=True, dynamic=False),
        device_ids=[local_rank],
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        bucket_cap_mb=128,
    )
    optimizer = MuonAdamW(raw_model, LEARNING_RATE, WEIGHT_DECAY, ADAM_BETAS)
    start_step = 0
    if RESUME:
        state = torch.load(Path(RESUME) / "optimizer.pt", map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
        assert start_step <= DECAY_START

    block_ids = np.load(cache_dir / "block_ids.npy")
    assert len(block_ids) >= TRAIN_STEPS * ACCUMULATION * world_size
    indices = range(rank, TRAIN_STEPS * ACCUMULATION * world_size, world_size)[start_step * ACCUMULATION :]
    batches = iter(DataLoader(
        TeacherCacheDataset(training_data, cache_dir, block_ids), batch_size=1, sampler=indices, num_workers=2,
        pin_memory=True, persistent_workers=True,
    ))
    max_steps = min(
        TRAIN_STEPS,
        6000000000 // tokens_per_step,
        start_step + len(indices) // ACCUMULATION,
    )
    eval_steps = min(2000, max(1, max_steps // 10))
    eval_seconds = 600

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        training_config = {
            "dataset_id": "allenai/dolma3_mix-150B-1025",
            "dataset_revision": "afa92bfb22366821c5e6cd427cdd036b34b713ef",
            "model_id": "meta-llama/Llama-3.2-1B",
            "model_revision": "4e20de362430cd3b72f300e6b0f18e50e7166e08",
            "seed": SEED,
            "holdout_documents": 50000,
            "eval_documents": EVAL_DOCUMENTS,
            "sequence_length": 8192,
            "max_train_tokens": 6000000000,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": ACCUMULATION,
            "optimizer_recipe": "muon_elementwise_gradient_precond_exact_rms_hidden_adamw_rest",
            "learning_rate": LEARNING_RATE,
            "muon_adjust_lr_fn": "exact_rms_0.2",
            "adamw_betas": ADAM_BETAS,
            "momentum": 0.95,
            "weight_decay": WEIGHT_DECAY,
            "normalization_weight_decay": 0.0,
            "ns_steps": 5,
            "max_grad_norm": 1.0,
            "warmup_ratio": WARMUP_STEPS / TRAIN_STEPS,
            "schedule": f"warmup_{WARMUP_STEPS}_steps_stable_linear_decay_{DECAY_FRACTION:g}",
            "schedule_basis": "optimizer_steps",
            "schedule_steps": TRAIN_STEPS,
            "run_steps": max_steps,
            "tuning_run": False,
            "cosine_min_lr_ratio": 0.0,
            "logging_steps": 10,
            "eval_steps": eval_steps,
            "eval_seconds": eval_seconds,
            "allocation_seconds": allocation_seconds,
            "finalize_reserve_seconds": 300,
            "max_h200_hours": 60,
            "attention_implementation": "flash_attention_2",
            "gradient_checkpointing": False,
            "torch_compile": True,
            "world_size": world_size,
            "ddp_bucket_cap_mb": 128,
            "wandb_project": "lab5-training-llama",
            "optimizer_reference": "src/labs/lab5/muon_lab5.py",
            "effective_batch_tokens": tokens_per_step,
            "model_config": raw_model.config.to_dict(),
            "data_manifest": manifest,
            "distillation": {
                "teacher_id": cache_metadata["teacher_id"],
                "teacher_revision": cache_metadata["teacher_revision"],
                "top_k": cache_metadata["top_k"],
                "alpha": DISTILLATION_ALPHA,
                "tail_epsilon": TAIL_EPSILON,
                "teacher_cache": str(cache_dir),
            },
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
        (output_dir / "training-config.json").write_text(json.dumps(training_config, indent=2))
        print(run.url, flush=True)

    completed_steps = start_step
    training_seconds = 0.0
    interval_steps = 0
    interval_loss = torch.zeros(2, dtype=torch.float64, device=device)

    def record(metrics, curve):
        if rank == 0:
            wandb.log(metrics)
            with (output_dir / curve).open("a") as file:
                file.write(json.dumps(metrics) + "\n")
            print(json.dumps(metrics), flush=True)

    def run_evaluation(limit=EVAL_DOCUMENTS):
        started = time.monotonic()
        losses, counts = evaluate(raw_model, evaluation_documents, limit, device)
        mean_loss = (losses[:EVAL_DOCUMENTS].sum() / counts[:EVAL_DOCUMENTS].sum()).item()
        record({
            "eval/loss": mean_loss,
            "eval/perplexity": math.exp(mean_loss),
            "eval/seconds": time.monotonic() - started,
            "train/total_tokens_seen": completed_steps * tokens_per_step,
            "eval/step": completed_steps,
            "eval/wall_hours": (time.time() - allocation_start) / 3600,
            "eval/training_hours": training_seconds / 3600,
            "eval/h200_hours": (time.time() - allocation_start) * world_size / 3600,
        }, "eval-curve.jsonl")
        return math.exp(mean_loss), losses, counts

    def save_checkpoint():
        if rank == 0:
            checkpoint = output_dir / f"step-{completed_steps:06d}"
            raw_model.save_pretrained(checkpoint / "model", max_shard_size="2GB")
            tokenizer.save_pretrained(checkpoint / "model")
            torch.save({"optimizer": optimizer.state_dict(), "step": completed_steps}, checkpoint / "optimizer.pt")
            print(json.dumps({"checkpoint": str(checkpoint)}), flush=True)
        dist.barrier()

    if start_step == 0:
        run_evaluation()
    last_eval_time = time.monotonic()

    while completed_steps < max_steps:
        stop = torch.tensor(int(time.time() >= training_deadline), device=device)
        dist.broadcast(stop, src=0)
        if stop.item():
            break
        step = completed_steps + 1
        torch.cuda.synchronize(device)
        step_start = time.monotonic()
        for group in optimizer.param_groups:
            group["lr"] = LEARNING_RATE * learning_rate_scale(step)
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(ACCUMULATION):
            batch = next(batches)
            input_ids = batch["input_ids"].to(device, dtype=torch.long, non_blocking=True)
            teacher = [batch[name][0].to(device, non_blocking=True) for name in TEACHER_ARRAYS]
            with nullcontext() if micro_step == ACCUMULATION - 1 else model.no_sync():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, cross_entropy, kl = model(
                        input_ids, *document_boundaries(input_ids, tokenizer.bos_token_id), *teacher
                    )
                (loss / ACCUMULATION).backward()
            interval_loss += torch.stack([cross_entropy, kl]).double() / ACCUMULATION
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize(device)
        step_seconds = torch.tensor(time.monotonic() - step_start, device=device)
        dist.all_reduce(step_seconds, op=dist.ReduceOp.MAX)
        training_seconds += step_seconds.item()
        completed_steps = step
        interval_steps += 1

        if step % 10 == 0 or step == max_steps:
            dist.all_reduce(interval_loss)
            assert interval_loss[1].item() >= -1e-3, "invalid negative KL; check cache"
            if rank == 0:
                record({
                    "train/loss": interval_loss[0].item() / (world_size * interval_steps),
                    "train/distill_loss": interval_loss[1].item() / (world_size * interval_steps),
                    "train/grad_norm": gradient_norm.item(),
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/adamw_learning_rate": optimizer.param_groups[1]["lr"],
                    "train/tokens_per_second": tokens_per_step / step_seconds.item(),
                    "train/total_tokens_seen": completed_steps * tokens_per_step,
                    "train/step": completed_steps,
                    "train/step_seconds": step_seconds.item(),
                    "train/wall_hours": (time.time() - allocation_start) / 3600,
                    "train/training_hours": training_seconds / 3600,
                    "train/peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
                    "train/h200_hours": (time.time() - allocation_start) * world_size / 3600,
                }, "train-curve.jsonl")
            interval_loss.zero_()
            interval_steps = 0

        evaluation_due = torch.tensor(int(
            step % eval_steps == 0
            or time.monotonic() - last_eval_time >= eval_seconds
        ), device=device)
        dist.broadcast(evaluation_due, src=0)
        if evaluation_due.item() and step < max_steps:
            run_evaluation()
            last_eval_time = time.monotonic()
        if step in SAVE_STEPS and step < max_steps:
            save_checkpoint()

    if interval_steps:
        dist.all_reduce(interval_loss)
        if rank == 0:
            wandb.log({
                "train/loss": interval_loss[0].item() / (world_size * interval_steps),
                "train/distill_loss": interval_loss[1].item() / (world_size * interval_steps),
                "train/grad_norm": gradient_norm.item(),
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/tokens_per_second": tokens_per_step / step_seconds.item(),
                "train/total_tokens_seen": completed_steps * tokens_per_step,
            })

    perplexity, losses, counts = run_evaluation(FINAL_EVAL_DOCUMENTS)
    save_checkpoint()
    if rank == 0:
        (output_dir / "model").symlink_to(f"step-{completed_steps:06d}/model")
        torch.save({"losses": losses.cpu(), "counts": counts.cpu()}, output_dir / "eval-documents.pt")
        (output_dir / "eval-8k.json").write_text(json.dumps({
            "step": completed_steps,
            "documents": FINAL_EVAL_DOCUMENTS,
            "perplexity": math.exp((losses.sum() / counts.sum()).item()),
        }, indent=2))
        result = {
            "run_name": run_name,
            "step": completed_steps,
            "total_tokens_seen": completed_steps * tokens_per_step,
            "wall_hours": (time.time() - allocation_start) / 3600,
            "training_hours": training_seconds / 3600,
            "h200_hours": (time.time() - allocation_start) * world_size / 3600,
            "perplexity": perplexity,
            "wandb_url": run.url,
        }
        (output_dir / "result.json").write_text(json.dumps(result, indent=2))
        wandb.run.summary.update(result)
        wandb.finish()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
