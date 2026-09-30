import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed
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

TOKENIZER_ID = "openai-community/gpt2"
SEQUENCE_LENGTH = 1024
PER_DEVICE_BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 2
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
MAX_STEPS = 4_400
PREPARED_BLOCKS = 1_126_400
WORK_DIR = Path(
    os.environ.get("LAB4_WORK_DIR", f"/work/{os.environ.get('USER', 'user')}/lab4")
)
PREPARED_DATA_DIR = WORK_DIR / "c4-packed"
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

    elapsed = time.monotonic() - started
    print(
        f"Prepared {len(dataset):,} blocks at {PREPARED_DATA_DIR} in {elapsed:.1f}s "
        f"({len(dataset) * SEQUENCE_LENGTH / elapsed:,.0f} tokens/s).",
        flush=True,
    )


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
            run_name="lab4-gpt2-small",
            include_num_input_tokens_seen="all",
            save_strategy="no",
            eval_strategy="no",
            torch_compile=True,
        ),
        train_dataset=dataset,
        data_collator=collate,
        processing_class=tokenizer,
        callbacks=[Deadline(seconds=TRAINING_SECONDS)],
    )
    result = trainer.train()

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
