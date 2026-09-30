import hashlib
import json
import os
import time
from pathlib import Path

import torch
from datasets import Dataset, Features, Sequence, Value, load_dataset
from transformers import AutoTokenizer

TOKENIZER_ID = "openai-community/gpt2"
SEQUENCE_LENGTH = 1024
PREPARED_BLOCKS = 4_800 * 64 * 2 * 2
SEED = 42
OJ_DOCUMENTS = 100_000
EVAL_DOCUMENTS = 2_048
WORK_DIR = Path(
    os.environ.get("LAB4_WORK_DIR", f"/work/{os.environ.get('USER', 'user')}/lab4")
)
PREPARED_DATA_DIR = WORK_DIR / "c4-packed"
PREPARED_EVAL_DIR = WORK_DIR / "c4-dev"


def pack_documents(documents, tokenizer, block_count):
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


if __name__ == "__main__":
    prepare()
