import json
import math
import os
import socket
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from datasets import load_from_disk
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM, set_seed

from afmuon_lab5 import AFMuon


@torch.inference_mode()
def evaluate(model, documents, device, rank, world_size):
    model.eval()
    totals = torch.zeros(2, dtype=torch.float64, device=device)

    for document_index in range(rank, len(documents), world_size):
        input_ids = torch.tensor(
            documents[document_index]["input_ids"], dtype=torch.long, device=device
        ).unsqueeze(0)
        if input_ids.shape[1] < 2:
            continue

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=input_ids, use_cache=False).logits[0, :-1]
        labels = input_ids[0, 1:]

        for start in range(0, len(labels), 256):
            negative_log_likelihood = F.cross_entropy(
                logits[start:start + 256].float(),
                labels[start:start + 256],
                reduction="sum",
            )
            totals[0] += negative_log_likelihood.double()
        totals[1] += len(labels)

    dist.all_reduce(totals)
    assert totals[1].item() > 0
    mean_loss = (totals[0] / totals[1]).item()
    assert math.isfinite(mean_loss)
    model.train()
    return {"eval/loss": mean_loss, "eval/perplexity": math.exp(mean_loss)}


def main():
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_STEP_ID")
    hostname = socket.gethostname().split(".")[0]
    assert hostname == os.environ["SLURMD_NODENAME"].split(".")[0]

    config_path = Path(os.environ.get(
        "LAB5_CONFIG", Path(__file__).with_name("lab5_config.json")
    ))
    config = json.loads(config_path.read_text())
    data_dir = Path(os.environ.get(
        "LAB5_DATA", "/home/nat1andotxyz/lab5/dolma-seed42-8192-v1"
    ))
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    accumulation_steps = config["gradient_accumulation_steps"]
    tokens_per_step = (
        world_size * config["micro_batch_size"]
        * accumulation_steps * config["sequence_length"]
    )
    # Run names record matrix/vector LR, batch tokens and seed; choose the HF model name when publishing.
    experiment_name = os.environ.get("LAB5_RUN_NAME", (
        f"afmuon-mlr{config['muon_lr']:g}-vlr{config['vector_lr']:g}"
        f"-b{tokens_per_step}-s{config['seed']}"
    ))
    output_dir = Path(os.environ.get(
        "LAB5_OUTPUT",
        f"/home/nat1andotxyz/lab5/runs/{experiment_name}-j{os.environ['SLURM_JOB_ID']}",
    ))

    assert world_size == config["world_size"] and world_size <= 8
    assert 0 < config["max_train_tokens"] <= 6_000_000_000
    assert world_size * config["allocation_seconds"] / 3600 <= config["max_h200_hours"] <= 64
    assert config["schedule"] == "warmup_constant"
    assert config["warmup_tokens"] > 0
    assert 0 < config["finalize_reserve_seconds"] < config["allocation_seconds"]

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    assert "H200" in torch.cuda.get_device_name(device)
    dist.init_process_group("nccl", device_id=device)
    set_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True

    manifest = json.loads((data_dir / "manifest.json").read_text())
    for key in (
        "dataset_id", "dataset_revision", "model_id", "model_revision",
        "sequence_length", "holdout_documents", "eval_documents",
    ):
        assert manifest[key] == config[key]
    assert manifest["shuffle_seed"] == config["seed"]

    training_data = load_from_disk(data_dir / "train").with_format("torch")
    evaluation_data = load_from_disk(data_dir / "eval")
    assert len(training_data) == manifest["training_blocks"]
    assert len(evaluation_data) == config["eval_documents"]
    tokenizer = AutoTokenizer.from_pretrained(
        data_dir / "tokenizer", local_files_only=True
    )
    model_config = AutoConfig.from_pretrained(
        data_dir / "model-config", local_files_only=True
    )
    assert model_config.tie_word_embeddings
    model_config.max_position_embeddings = config["sequence_length"]
    model_config.use_cache = False
    model_config.pad_token_id = tokenizer.pad_token_id
    model_config._attn_implementation = config["attention_implementation"]

    model = LlamaForCausalLM(model_config).to(device=device, dtype=torch.float32)
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model = DistributedDataParallel(
        model, device_ids=[local_rank], broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )
    optimizer = AFMuon(model.module, config)
    sampler = DistributedSampler(
        training_data, num_replicas=world_size, rank=rank,
        seed=config["seed"], drop_last=True,
    )
    loader = DataLoader(
        training_data, batch_size=config["micro_batch_size"], sampler=sampler,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    batches = iter(loader)
    max_steps = min(
        config["max_train_tokens"] // tokens_per_step,
        len(loader) // accumulation_steps,
    )
    assert max_steps > 0
    assert 0 < config["eval_steps"] <= max(1, max_steps // 10)
    assert config["logging_steps"] > 0 and config["eval_seconds"] > 0

    allocation_start = float(os.environ["LAB5_ALLOCATION_START"])
    training_deadline = (
        allocation_start + config["allocation_seconds"]
        - config["finalize_reserve_seconds"]
    )
    if rank == 0:
        assert not output_dir.exists() or not any(output_dir.iterdir())
        output_dir.mkdir(parents=True, exist_ok=True)
        run = wandb.init(
            project=config["wandb_project"], name=output_dir.name,
            config={
                **config, "effective_batch_tokens": tokens_per_step,
                "model_config": model_config.to_dict(), "data_manifest": manifest,
            },
            dir=str(output_dir),
        )
        wandb.define_metric("train/total_tokens_seen")
        wandb.define_metric("train/*", step_metric="train/total_tokens_seen")
        wandb.define_metric("eval/*", step_metric="train/total_tokens_seen")
        (output_dir / "training-config.json").write_text(json.dumps(config, indent=2))
        print(run.url, flush=True)

    completed_steps = 0
    total_tokens_seen = 0
    interval_steps = 0
    interval_loss = torch.zeros((), dtype=torch.float64, device=device)
    evaluation_metrics = evaluate(model.module, evaluation_data, device, rank, world_size)
    evaluation_metrics.update({
        "train/total_tokens_seen": 0, "eval/step": 0,
        "eval/h200_hours": (time.time() - allocation_start) * world_size / 3600,
    })
    if rank == 0:
        wandb.log(evaluation_metrics)
        with (output_dir / "eval-curve.jsonl").open("a") as file:
            file.write(json.dumps(evaluation_metrics) + "\n")
        print(json.dumps(evaluation_metrics), flush=True)
    last_eval_step = 0
    last_eval_time = time.monotonic()

    for step in range(1, max_steps + 1):
        stop = torch.tensor(int(time.time() >= training_deadline), device=device)
        dist.broadcast(stop, src=0)
        if stop.item():
            break

        torch.cuda.synchronize(device)
        step_start = time.monotonic()
        learning_rate_scale = min(
            1.0, (total_tokens_seen + tokens_per_step) / config["warmup_tokens"]
        )
        for group in optimizer.param_groups:
            group["lr"] = group["peak_lr"] * learning_rate_scale
        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(accumulation_steps):
            input_ids = next(batches)["input_ids"].to(
                device, dtype=torch.long, non_blocking=True
            )
            assert input_ids.shape == (
                config["micro_batch_size"], config["sequence_length"]
            )
            sync_gradients = micro_step == accumulation_steps - 1
            with nullcontext() if sync_gradients else model.no_sync():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(
                        input_ids=input_ids, labels=input_ids, use_cache=False
                    ).loss
                (loss / accumulation_steps).backward()
            interval_loss += loss.detach().double() / accumulation_steps

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config["max_grad_norm"], error_if_nonfinite=True
        )
        optimizer.step()
        torch.cuda.synchronize(device)
        step_seconds = torch.tensor(time.monotonic() - step_start, device=device)
        dist.all_reduce(step_seconds, op=dist.ReduceOp.MAX)
        completed_steps = step
        total_tokens_seen += tokens_per_step
        interval_steps += 1

        stop = torch.tensor(int(time.time() >= training_deadline), device=device)
        dist.broadcast(stop, src=0)
        if step % config["logging_steps"] == 0 or step == max_steps or stop.item():
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
                    "train/h200_hours": (time.time() - allocation_start) * world_size / 3600,
                }
                wandb.log(training_metrics)
                print(json.dumps(training_metrics), flush=True)
            interval_loss.zero_()
            interval_steps = 0

        evaluation_due = (
            step - last_eval_step >= config["eval_steps"]
            or time.monotonic() - last_eval_time >= config["eval_seconds"]
            or step == max_steps or stop.item()
        )
        evaluate_now = torch.tensor(int(evaluation_due), device=device)
        dist.broadcast(evaluate_now, src=0)
        if evaluate_now.item():
            evaluation_metrics = evaluate(
                model.module, evaluation_data, device, rank, world_size
            )
            evaluation_metrics.update({
                "train/total_tokens_seen": total_tokens_seen, "eval/step": completed_steps,
                "eval/h200_hours": (time.time() - allocation_start) * world_size / 3600,
            })
            if rank == 0:
                wandb.log(evaluation_metrics)
                with (output_dir / "eval-curve.jsonl").open("a") as file:
                    file.write(json.dumps(evaluation_metrics) + "\n")
                print(json.dumps(evaluation_metrics), flush=True)
            last_eval_step = completed_steps
            last_eval_time = time.monotonic()
        if stop.item():
            break

    if interval_steps:
        dist.all_reduce(interval_loss)
        if rank == 0:
            wandb.log({
                "train/loss": interval_loss.item() / (world_size * interval_steps),
                "train/grad_norm": gradient_norm.item(),
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/tokens_per_second": tokens_per_step / step_seconds.item(),
                "train/total_tokens_seen": total_tokens_seen,
            })

    if last_eval_step != completed_steps:
        evaluation_metrics = evaluate(
            model.module, evaluation_data, device, rank, world_size
        )
        evaluation_metrics.update({
            "train/total_tokens_seen": total_tokens_seen, "eval/step": completed_steps,
            "eval/h200_hours": (time.time() - allocation_start) * world_size / 3600,
        })
        if rank == 0:
            wandb.log(evaluation_metrics)
            with (output_dir / "eval-curve.jsonl").open("a") as file:
                file.write(json.dumps(evaluation_metrics) + "\n")
            print(json.dumps(evaluation_metrics), flush=True)

    if rank == 0:
        model.module.save_pretrained(output_dir / "model", max_shard_size="2GB")
        tokenizer.save_pretrained(output_dir / "model")
        torch.save({
            "optimizer": optimizer.state_dict(), "step": completed_steps,
            "total_tokens_seen": total_tokens_seen,
        }, output_dir / "optimizer.pt")
        result = {
            "run_name": output_dir.name,
            "step": completed_steps, "total_tokens_seen": total_tokens_seen,
            "h200_hours": (time.time() - allocation_start) * world_size / 3600,
            "perplexity": evaluation_metrics["eval/perplexity"], "wandb_url": run.url,
        }
        (output_dir / "result.json").write_text(json.dumps(result, indent=2))
        wandb.run.summary.update(result)
        wandb.finish()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
