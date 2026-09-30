import argparse
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
from datasets import Dataset, Features, Sequence, Value, load_dataset, load_from_disk
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

TOKENIZER_ID = "openai-community/gpt2"
SEQUENCE_LENGTH = 1024
PER_DEVICE_BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 2
TRAINING_SECONDS = 26 * 60
LEARNING_RATE = 6e-4
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.1
MAX_GRAD_NORM = 1.0
ADAM_BETAS = (0.9, 0.95)
RESIDUAL_DROPOUT = 0.0
EMBEDDING_DROPOUT = 0.0
ATTENTION_DROPOUT = 0.0
SEED = 42
MAX_STEPS = 4_800
OJ_DOCUMENTS = 100_000
EVAL_DOCUMENTS = 2_048
EVAL_STEPS = 1_200
EVAL_BATCH_SIZE = 8
# Enough distinct blocks for one pass at the configured batch size on two GPUs.
PREPARED_BLOCKS = MAX_STEPS * PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * 2
WORK_DIR = Path(
    os.environ.get("LAB4_WORK_DIR", f"/work/{os.environ.get('USER', 'user')}/lab4")
)
PREPARED_DATA_DIR = WORK_DIR / "c4-packed"
PREPARED_EVAL_DIR = WORK_DIR / "c4-dev"
OUTPUT_DIR = WORK_DIR / "gpt2-small"
HF_REPO_ID = os.environ.get("LAB4_HF_REPO_ID", "")


def pack_documents(documents, tokenizer, block_count):
    """Pack one partition, retaining partial blocks across document batches."""
    pending = []
    count = 0
    for batch in documents.iter(batch_size=256):
        tokenized = tokenizer(
            batch["text"],
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            verbose=False,
        )
        for tokens in tokenized["input_ids"]:
            pending.extend(tokens)
            pending.append(tokenizer.eos_token_id)
            end = len(pending) // SEQUENCE_LENGTH * SEQUENCE_LENGTH
            for start in range(0, end, SEQUENCE_LENGTH):
                yield {"input_ids": pending[start : start + SEQUENCE_LENGTH]}
                count += 1
                if count == block_count:
                    return
            pending = pending[end:]
    raise ValueError(
        f"Partition exhausted: expected {block_count:,} blocks, got {count:,}."
    )


def packed_examples(worker_ids, documents, num_workers, total_blocks, tokenizer):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(1)
    base, extra = divmod(total_blocks, num_workers)

    for worker_id in worker_ids:
        partition = documents.shard(num_shards=num_workers, index=worker_id)
        partition = partition.shuffle(seed=SEED, buffer_size=10_000)
        yield from pack_documents(partition, tokenizer, base + (worker_id < extra))


def prepare():
    started = time.monotonic()
    available_cpus = os.process_cpu_count() or 1
    allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", available_cpus))
    num_workers = min(available_cpus, allocated_cpus)
    if num_workers < 1:
        raise ValueError("prepare workers must be at least 1")

    documents = load_dataset("allenai/c4", "en", split="train", streaming=True)
    num_workers = min(num_workers, documents.num_shards, PREPARED_BLOCKS)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(
        f"Preparing {PREPARED_BLOCKS:,} blocks with {num_workers} workers "
        f"from {documents.num_shards} source shards.",
        flush=True,
    )

    dataset = Dataset.from_generator(
        packed_examples,
        gen_kwargs={
            "worker_ids": list(range(num_workers)),
            "documents": documents,
            "num_workers": num_workers,
            "total_blocks": PREPARED_BLOCKS,
            "tokenizer": tokenizer,
        },
        num_proc=num_workers,
        features=Features(
            {"input_ids": Sequence(Value("int32"), length=SEQUENCE_LENGTH)}
        ),
        cache_dir=str(WORK_DIR / "prepare-cache-parallel"),
    )

    if len(dataset) != PREPARED_BLOCKS:
        raise ValueError(f"Expected {PREPARED_BLOCKS:,} blocks, got {len(dataset):,}.")

    dataset.save_to_disk(str(PREPARED_DATA_DIR))
    tokenizer.save_pretrained(str(PREPARED_DATA_DIR / "tokenizer"))
    prepare_validation()

    elapsed = time.monotonic() - started
    print(
        f"Prepared {len(dataset):,} blocks at {PREPARED_DATA_DIR} in {elapsed:.1f}s "
        f"({len(dataset) * SEQUENCE_LENGTH / elapsed:,.0f} tokens/s).",
        flush=True,
    )


def prepare_validation():
    dataset = (
        load_dataset(
            "allenai/c4",
            "en",
            data_files={"validation": "en/c4-validation.*.json.gz"},
            split="validation",
            verification_mode="no_checks",
        )
        .shuffle(seed=SEED)
        .select(range(OJ_DOCUMENTS, OJ_DOCUMENTS + EVAL_DOCUMENTS))
    )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    dataset = dataset.map(
        lambda batch: tokenizer(
            batch["text"],
            truncation=True,
            max_length=SEQUENCE_LENGTH,
            return_attention_mask=False,
        ),
        batched=True,
        remove_columns=dataset.column_names,
    )
    dataset.save_to_disk(str(PREPARED_EVAL_DIR))
    metadata = {
        "dataset": "allenai/c4",
        "subset": "en",
        "split": "validation",
        "seed": SEED,
        "start": OJ_DOCUMENTS,
        "documents": EVAL_DOCUMENTS,
        "max_length": SEQUENCE_LENGTH,
        "tokenizer": TOKENIZER_ID,
        "token_ids_sha256": validation_hash(dataset),
    }
    (PREPARED_EVAL_DIR / "validation.json").write_text(json.dumps(metadata, indent=2))
    print(f"Prepared {len(dataset)} dev documents at {PREPARED_EVAL_DIR}", flush=True)


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
        raise ValueError("Dev cache changed; run prepare again.")
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

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % EVAL_STEPS == 0:
            self.evaluate()

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


class Deadline(TrainerCallback):
    def __init__(self, seconds):
        self.deadline = time.monotonic() + seconds

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50:
            return
        flag = torch.tensor(
            [time.monotonic() > self.deadline], device=args.device, dtype=torch.int32
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
        if flag.item():
            control.should_training_stop = True


def train():
    if not HF_REPO_ID:
        raise ValueError("Set LAB4_HF_REPO_ID to Cerulean's actual org/model-name.")

    os.environ["WANDB_PROJECT"] = "gpt2-training"
    set_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dataset = load_from_disk(str(PREPARED_DATA_DIR)).with_format("torch")
    if len(dataset) != PREPARED_BLOCKS:
        raise ValueError(f"Expected {PREPARED_BLOCKS:,} blocks. Run prepare again.")
    print(f"Training on {len(dataset):,} blocks (one pass on 2 GPUs).", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(PREPARED_DATA_DIR / "tokenizer"))
    eval_dataset, eval_metadata = load_validation()
    provenance = code_revision()
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
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(OUTPUT_DIR),
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
            run_name=f"lab4-{provenance['git_commit'][:8]}"
            + ("-dirty" if provenance["git_dirty"] else ""),
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
            Deadline(seconds=TRAINING_SECONDS - 60),
            ValidationWandbCallback(provenance, eval_metadata),
        ],
    )
    validation = ValidationCallback(trainer)
    trainer.add_callback(validation)
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
        model.push_to_hub(HF_REPO_ID)
        tokenizer.push_to_hub(HF_REPO_ID)
        print(f"Final model repository: {HF_REPO_ID}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lab 4: prepare C4 on CPU first, then train GPT-2 Small on two H200s."
    )
    parser.add_argument("mode", choices=["prepare", "train"])
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    else:
        train()
