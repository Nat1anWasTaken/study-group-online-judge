import hashlib
import math
import os
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed
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

SEQUENCE_LENGTH = 1024
PER_DEVICE_BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 1
EXPERIMENT = "Q4"
JOB_SECONDS = 30 * 60
FINALIZE_RESERVE_SECONDS = 3 * 60
LEARNING_RATE = 4e-3
COOLDOWN_SHAPE = "linear"
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.1
MAX_GRAD_NORM = 1.0
ADAM_BETAS = (0.9, 0.95)
RESIDUAL_DROPOUT = 0.0
EMBEDDING_DROPOUT = 0.0
ATTENTION_DROPOUT = 0.0
SEED = 42
MAX_STEPS = 100_000
WORK_DIR = Path(
    os.environ.get("LAB4_WORK_DIR", f"/work/{os.environ.get('USER', 'user')}/lab4")
)
PREPARED_DATA_DIR = WORK_DIR / "c4-packed"
HF_REPO_ID = os.environ.get("LAB4_HF_REPO_ID", "")


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


class TimeCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.progress = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = time.time()
        self.deadline = (
            float(os.environ["LAB4_START_TIME"])
            + JOB_SECONDS
            - FINALIZE_RESERVE_SECONDS
        )
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
        elif self.progress < 0.8:
            scale = 1.0
        else:
            cooldown_progress = (self.progress - 0.8) / 0.2
            scale = (
                1.0 - math.sqrt(cooldown_progress)
                if COOLDOWN_SHAPE == "sqrt"
                else 1.0 - cooldown_progress
            )
        self.trainer.lr_scale = scale
        for group in optimizer.param_groups:
            group["lr"] = LEARNING_RATE * scale

    def on_step_end(self, args, state, control, **kwargs):
        self.update_progress(args)
        if self.progress >= 1.0:
            control.should_training_stop = True


class ProvenanceCallback(TrainerCallback):
    def __init__(self, provenance):
        self.provenance = provenance

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero and wandb.run is not None:
            wandb.config.update(self.provenance)
            wandb.run.summary.update(self.provenance)
            wandb.run.log_code(
                root=str(Path(__file__).resolve().parent),
                include_fn=lambda path: path.endswith(".py"),
            )


def collate(examples):
    batch = default_data_collator(examples)
    batch["labels"] = batch["input_ids"].clone()
    return batch


class TimedTrainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is None:
            decay_names = self.get_decay_parameter_names(self.model)
            matrices, qkv, decay, no_decay = [], [], [], []
            for name, parameter in self.model.named_parameters():
                if name.endswith("attn.c_attn.weight"):
                    if tuple(parameter.shape) != (768, 2304):
                        raise ValueError(f"Unexpected QKV layout: {name} {parameter.shape}")
                    qkv.append(parameter)
                elif name.startswith("transformer.h.") and parameter.ndim == 2:
                    matrices.append(parameter)
                elif name in decay_names:
                    decay.append(parameter)
                else:
                    no_decay.append(parameter)
            if len(qkv) != 12:
                raise ValueError(f"Expected 12 fused QKV weights, found {len(qkv)}")
            self.optimizer = MuonAdamW(
                matrices, decay, no_decay, LEARNING_RATE, WEIGHT_DECAY, ADAM_BETAS, qkv
            )
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
    provenance = code_revision()
    run_name = f"lab4-{EXPERIMENT}-{provenance['git_commit'][:8]}-{os.environ.get('SLURM_JOB_ID', 'local')}"
    output_dir = WORK_DIR / "runs" / run_name
    hf_repo_id = f"{HF_REPO_ID}-{EXPERIMENT.lower()}-{provenance['git_commit'][:8]}"
    provenance.update(
        experiment=EXPERIMENT,
        baseline_commit="3f4c5e1",
        baseline_experiment="O-final",
        optimizer_source_sha256=hashlib.sha256(
            Path(__file__).with_name("muon_lab4.py").read_bytes()
        ).hexdigest(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        hf_model_id=hf_repo_id,
        job_budget_seconds=JOB_SECONDS,
        training_budget_seconds=JOB_SECONDS - FINALIZE_RESERVE_SECONDS,
        finalize_reserve_seconds=FINALIZE_RESERVE_SECONDS,
        training_dataset_fingerprint=dataset._fingerprint,
        training_blocks=len(dataset),
        schedule=f"wsd_5_75_20_{COOLDOWN_SHAPE}",
        cooldown_shape=COOLDOWN_SHAPE,
        muon_learning_rate=LEARNING_RATE,
        adamw_learning_rate=LEARNING_RATE,
        training_seed=SEED,
        data_seed=SEED,
        optimizer_recipe="split_qkv_muon_hidden_adamw_rest",
        qkv_split_axis=1,
        qkv_parts=3,
        qkv_submatrix_shape=[768, 768],
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
            torch_compile=True,
        ),
        train_dataset=dataset,
        data_collator=collate,
        processing_class=tokenizer,
        callbacks=[ProvenanceCallback(provenance)],
    )
    timing = TimeCallback(trainer)
    trainer.add_callback(timing)
    allocation_started = float(os.environ["LAB4_START_TIME"])
    result = trainer.train()
    training_finished = time.time()

    result.metrics["train_steps_per_second"] = (
        trainer.state.global_step / result.metrics["train_runtime"]
    )
    result.metrics["train_samples_per_second"] = (
        trainer.state.global_step
        * PER_DEVICE_BATCH_SIZE
        * GRADIENT_ACCUMULATION_STEPS
        * trainer.accelerator.num_processes
        / result.metrics["train_runtime"]
    )
    result.metrics["train_tokens_per_second"] = (
        trainer.state.num_input_tokens_seen / result.metrics["train_runtime"]
    )

    model.config.use_cache = True
    save_started = time.monotonic()
    trainer.save_model()
    trainer.accelerator.wait_for_everyone()
    save_seconds = time.monotonic() - save_started

    trainer.log_metrics("train", result.metrics)
    trainer.log(result.metrics)
    if trainer.is_world_process_zero():
        wandb.run.summary.update(dict(
            startup_seconds=timing.started - allocation_started,
            train_loop_seconds=training_finished - timing.started,
            save_seconds=save_seconds,
            allocation_elapsed_seconds=time.time() - allocation_started,
            final_train_step=trainer.state.global_step,
            training_completed=True,
            evaluation_completed=False,
            hf_upload_completed=False,
            hf_model_id=hf_repo_id,
        ))
        print(
            f"Saved final model at {output_dir}; evaluate and publish separately to {hf_repo_id}",
            flush=True,
        )
        wandb.finish()
    trainer.accelerator.wait_for_everyone()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    train()
