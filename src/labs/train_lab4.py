import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

import wandb
from muon_lab4 import MuonAdamW

TOKENIZER_ID = "openai-community/gpt2"
SEQUENCE_LENGTH = 1024
PER_DEVICE_BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 2
EXPERIMENT = "G"
TRAINING_SECONDS = 25 * 60
LEARNING_RATE = 6e-4
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.1
MAX_GRAD_NORM = 1.0
ADAM_BETAS = (0.9, 0.95)
RESIDUAL_DROPOUT = 0.0
EMBEDDING_DROPOUT = 0.0
ATTENTION_DROPOUT = 0.0
SEED = 42
MAX_STEPS = 100_000
OJ_DOCUMENTS = 100_000
EVAL_DOCUMENTS = 2_048
EVAL_BATCH_SIZE = 8
WORK_DIR = Path(
    os.environ.get("LAB4_WORK_DIR", f"/work/{os.environ.get('USER', 'user')}/lab4")
)
PREPARED_DATA_DIR = WORK_DIR / "c4-packed"
PREPARED_EVAL_DIR = WORK_DIR / "c4-dev"
HF_REPO_ID = os.environ.get("LAB4_HF_REPO_ID", "")


def validation_hash(dataset):
    return hashlib.sha256(json.dumps(list(dataset["input_ids"])).encode()).hexdigest()


def load_validation():
    dataset = load_from_disk(str(PREPARED_EVAL_DIR))
    metadata = json.loads((PREPARED_EVAL_DIR / "validation.json").read_text())
    expected = {
        "seed": SEED,
        "start": OJ_DOCUMENTS,
        "documents": EVAL_DOCUMENTS,
        "max_length": SEQUENCE_LENGTH,
        "tokenizer": TOKENIZER_ID,
        "token_ids_sha256": validation_hash(dataset),
    }
    if len(dataset) != EVAL_DOCUMENTS or any(
        metadata.get(k) != v for k, v in expected.items()
    ):
        raise ValueError("Dev cache changed; run prepare_lab4.py again.")
    return dataset, metadata


def code_revision(directory=None):
    directory = directory or Path(__file__).resolve().parents[2]

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(directory), *args], text=True
        ).strip()

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "training_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }


@torch.inference_mode()
def validation_totals(
    model, dataset, tokenizer, device, batch_size, rank=0, world_size=1
):
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    indices = list(range(rank, len(dataset), world_size))
    was_training = model.training
    forward = getattr(model, "_original_forward", model.forward)
    model.eval()
    try:
        for start in range(0, len(indices), batch_size):
            rows = [
                dataset[i]["input_ids"] for i in indices[start : start + batch_size]
            ]
            valid = [row for row in rows if len(row) > 1]
            totals[3] += len(rows) - len(valid)
            if not valid:
                continue
            inputs = tokenizer.pad(
                {"input_ids": valid}, padding=True, return_tensors="pt"
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            logits = forward(**inputs).logits[:, :-1, :].float().contiguous()
            labels = inputs["input_ids"][:, 1:].contiguous()
            mask = inputs["attention_mask"][:, 1:].bool()
            losses = (
                F.cross_entropy(
                    logits.view(-1, logits.shape[-1]),
                    labels.view(-1),
                    reduction="none",
                )
                .view_as(labels)
                .masked_fill(~mask, 0)
            )
            totals[0] += losses.sum(dim=1).double().sum()
            totals[1] += mask.sum()
            totals[2] += len(valid)
    finally:
        model.train(was_training)
    return totals


def validation_metrics(totals):
    loss, tokens, documents, skipped = totals.tolist()
    if tokens <= 0 or not math.isfinite(loss) or loss / tokens >= 709:
        raise ValueError("Invalid validation loss or token count")
    return {
        "loss": loss / tokens,
        "perplexity": math.exp(loss / tokens),
        "tokens": int(tokens),
        "documents": int(documents),
        "skipped_documents": int(skipped),
    }


class ValidationCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.last_eval_step = None
        self.next_eval = 0.25
        self.progress = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = time.time()
        self.deadline = float(os.environ["LAB4_START_TIME"]) + TRAINING_SECONDS
        self.duration = self.deadline - self.started
        if self.duration <= 0:
            raise RuntimeError("Startup consumed the training budget")

    def update_progress(self, args):
        progress = torch.tensor(
            [(time.time() - self.started) / self.duration],
            dtype=torch.float64,
            device=args.device,
        )
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(progress, src=0)
        self.progress = min(1.0, max(0.0, progress.item()))

    def on_step_begin(self, args, state, control, optimizer=None, **kwargs):
        self.update_progress(args)
        if self.progress < WARMUP_RATIO:
            scale = self.progress / WARMUP_RATIO
        else:
            scale = 0.5 * (1 + math.cos(
                math.pi * (self.progress - WARMUP_RATIO) / (1 - WARMUP_RATIO)
            ))
        self.trainer.lr_scale = scale
        for group in optimizer.param_groups:
            group["lr"] = LEARNING_RATE * scale

    def on_step_end(self, args, state, control, **kwargs):
        self.update_progress(args)
        if self.progress >= 1.0:
            control.should_training_stop = True
        elif self.progress >= self.next_eval and self.next_eval <= 0.75:
            self.evaluate()
            self.next_eval += 0.25

    def evaluate(self):
        trainer = self.trainer
        started = time.monotonic()
        model = trainer.accelerator.unwrap_model(
            trainer.model_wrapped, keep_torch_compile=False
        )
        totals = validation_totals(
            model,
            trainer.eval_dataset,
            trainer.processing_class,
            trainer.args.device,
            trainer.args.per_device_eval_batch_size,
            trainer.accelerator.process_index,
            trainer.accelerator.num_processes,
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(totals)
        metrics = validation_metrics(totals)
        metrics["runtime"] = time.monotonic() - started
        metrics = {f"eval_{key}": value for key, value in metrics.items()}
        self.last_eval_step = trainer.state.global_step
        trainer.log(metrics)
        return metrics


class ValidationWandbCallback(TrainerCallback):
    def __init__(self, provenance, metadata):
        self.provenance = provenance
        self.metadata = metadata
        self.best = math.inf

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero and wandb.run is not None:
            wandb.config.update({**self.provenance, "validation": self.metadata})
            wandb.run.summary.update(self.provenance)
            wandb.run.log_code(
                root=str(Path(__file__).resolve().parent),
                include_fn=lambda path: path.endswith("train_lab4.py"),
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if (
            not state.is_world_process_zero
            or wandb.run is None
            or "eval_perplexity" not in (logs or {})
        ):
            return
        summary = {
            f"last_{key}": value
            for key, value in logs.items()
            if key.startswith("eval_")
        }
        summary["last_eval_step"] = state.global_step
        if logs["eval_perplexity"] < self.best:
            self.best = logs["eval_perplexity"]
            summary.update(
                best_eval_perplexity=self.best, best_eval_step=state.global_step
            )
        wandb.run.summary.update(summary)


def collate(examples):
    batch = default_data_collator(examples)
    batch["labels"] = batch["input_ids"].clone()
    return batch


class TimedTrainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is None:
            decay_names = self.get_decay_parameter_names(self.model)
            matrices, decay, no_decay = [], [], []
            for name, parameter in self.model.named_parameters():
                if name.startswith("transformer.h.") and parameter.ndim == 2:
                    matrices.append(parameter)
                elif name in decay_names:
                    decay.append(parameter)
                else:
                    no_decay.append(parameter)
            self.optimizer = MuonAdamW(matrices, decay, no_decay,
                                       LEARNING_RATE, WEIGHT_DECAY, ADAM_BETAS)
        return self.optimizer

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.lr_scheduler is None:
            self.lr_scale = 0.0
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer or self.optimizer, lambda step: self.lr_scale
            )
        return self.lr_scheduler


def train():
    if not HF_REPO_ID:
        raise ValueError("Set LAB4_HF_REPO_ID to Cerulean's actual org/model-name.")

    os.environ["WANDB_PROJECT"] = "gpt2-training"
    set_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dataset = load_from_disk(str(PREPARED_DATA_DIR)).with_format("torch")
    print(f"Training on {len(dataset):,} shared blocks.", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(PREPARED_DATA_DIR / "tokenizer"))
    eval_dataset, eval_metadata = load_validation()
    provenance = code_revision()
    run_name = f"lab4-{EXPERIMENT}-{provenance['git_commit'][:8]}-{os.environ.get('SLURM_JOB_ID', 'local')}"
    output_dir = WORK_DIR / "runs" / run_name
    hf_repo_id = f"{HF_REPO_ID}-{EXPERIMENT.lower()}-{provenance['git_commit'][:8]}"
    provenance.update(
        experiment=EXPERIMENT,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        hf_model_id=hf_repo_id,
        training_budget_seconds=TRAINING_SECONDS,
        training_dataset_fingerprint=dataset._fingerprint,
        training_blocks=len(dataset),
        schedule="cosine",
        optimizer_recipe="muon_hidden_adamw_rest",
        muon_momentum=0.95,
        muon_ns_steps=5,
        muon_adjust_lr_fn="match_rms_adamw",
    )
    config = GPT2Config(
        vocab_size=50304,
        n_positions=SEQUENCE_LENGTH,
        n_ctx=SEQUENCE_LENGTH,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=3072,
        resid_pdrop=RESIDUAL_DROPOUT,
        embd_pdrop=EMBEDDING_DROPOUT,
        attn_pdrop=ATTENTION_DROPOUT,
        initializer_range=0.02,
        bos_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False,
    )

    model = GPT2LMHeadModel(config)
    trainer = TimedTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            max_steps=MAX_STEPS,
            learning_rate=LEARNING_RATE,
            warmup_ratio=WARMUP_RATIO,
            lr_scheduler_type="cosine",
            optim="adamw_torch_fused",
            adam_beta1=ADAM_BETAS[0],
            adam_beta2=ADAM_BETAS[1],
            weight_decay=WEIGHT_DECAY,
            max_grad_norm=MAX_GRAD_NORM,
            bf16=True,
            tf32=True,
            seed=SEED,
            data_seed=SEED,
            dataloader_num_workers=4,
            dataloader_drop_last=True,
            ddp_find_unused_parameters=False,
            logging_steps=10,
            report_to="wandb",
            run_name=run_name,
            include_num_input_tokens_seen="all",
            save_strategy="no",
            eval_strategy="no",
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            torch_compile=True,
        ),
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=collate,
        processing_class=tokenizer,
        callbacks=[
            ValidationWandbCallback(provenance, eval_metadata),
        ],
    )
    validation = ValidationCallback(trainer)
    trainer.add_callback(validation)
    allocation_started = float(os.environ["LAB4_START_TIME"])
    result = trainer.train()
    if validation.last_eval_step != trainer.state.global_step:
        validation.evaluate()

    result.metrics["train_tokens_per_second"] = (
        trainer.state.num_input_tokens_seen / result.metrics["train_runtime"]
    )

    trainer.log_metrics("train", result.metrics)
    trainer.log(result.metrics)
    model.config.use_cache = True
    trainer.save_model()

    if trainer.is_world_process_zero():
        model.push_to_hub(hf_repo_id)
        tokenizer.push_to_hub(hf_repo_id)
        wandb.run.summary.update(
            final_eval_perplexity=wandb.run.summary.get("last_eval_perplexity"),
            final_eval_loss=wandb.run.summary.get("last_eval_loss"),
            allocation_elapsed_seconds=time.time() - allocation_started,
            hf_model_id=hf_repo_id,
        )
        print(f"Final model repository: {hf_repo_id}", flush=True)
        wandb.finish()
    trainer.accelerator.wait_for_everyone()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    train()
