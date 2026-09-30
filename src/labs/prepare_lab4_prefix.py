import json
import os
from pathlib import Path

import torch
from datasets import Dataset, Features, Sequence, Value, load_dataset
from transformers import AutoTokenizer

WORK_DIR = Path("/work/nat1andotxyz/lab4")
TOTAL_TOKENS = 256_000_000
SEED = 42


def examples(worker_ids, documents, workers, tokenizer):
    torch.set_num_threads(1)
    for worker in worker_ids:
        partition = documents.shard(num_shards=workers, index=worker)
        partition = partition.shuffle(seed=SEED, buffer_size=10_000)
        count = 0
        for batch in partition.iter(batch_size=256):
            tokenized = tokenizer(batch["text"], truncation=True, max_length=1024,
                                  add_special_tokens=False, return_attention_mask=False)
            for tokens in tokenized["input_ids"]:
                if len(tokens) < 2:
                    continue
                bucket = next(size for size in (128, 256, 512, 1024) if len(tokens) <= size)
                yield {"input_ids": tokens, "bucket": bucket, "targets": len(tokens) - 1}
                count += len(tokens) - 1
                if count >= (TOTAL_TOKENS + workers - 1) // workers:
                    break
            else:
                continue
            break
        if count < (TOTAL_TOKENS + workers - 1) // workers:
            raise RuntimeError("Prefix partition exhausted")


def main():
    workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    documents = load_dataset("allenai/c4", "en", split="train", streaming=True)
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    dataset = Dataset.from_generator(
        examples,
        gen_kwargs={"worker_ids": list(range(workers)), "documents": documents,
                    "workers": workers, "tokenizer": tokenizer},
        num_proc=workers,
        features=Features({"input_ids": Sequence(Value("int32")),
                           "bucket": Value("int32"), "targets": Value("int32")}),
        cache_dir=str(WORK_DIR / "prefix-prepare-cache"),
    )
    output = WORK_DIR / "c4-prefix"
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for size in (128, 256, 512, 1024):
        bucket = dataset.filter(lambda row: row["bucket"] == size, num_proc=workers)
        counts[str(size)] = len(bucket)
        bucket.remove_columns(["bucket", "targets"]).save_to_disk(str(output / str(size)))
    metadata = {"split": "train", "seed": SEED, "counts": counts,
                "target_tokens": sum(dataset["targets"]), "fingerprint": dataset._fingerprint}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(metadata, flush=True)


if __name__ == "__main__":
    main()
