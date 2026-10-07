import json
import os
import socket
from pathlib import Path

from datasets import Dataset, Features, Sequence, Value, load_dataset
from transformers import AutoConfig, AutoTokenizer


def pack_documents(worker_ids, documents, tokenizer_path, workers, blocks, length):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    for worker in worker_ids:
        target_blocks = blocks // workers + int(worker < blocks % workers)
        partition = documents.shard(workers, worker, contiguous=True)
        pending_tokens = []
        emitted_blocks = 0

        for batch in partition.iter(batch_size=128):
            tokenized_documents = tokenizer(
                batch["text"], add_special_tokens=True, truncation=False,
                return_attention_mask=False, verbose=False,
            )["input_ids"]

            for document_tokens in tokenized_documents:
                pending_tokens.extend(document_tokens)
                pending_tokens.append(tokenizer.eos_token_id)
                complete_length = len(pending_tokens) // length * length

                for start in range(0, complete_length, length):
                    yield {"input_ids": pending_tokens[start:start + length]}
                    emitted_blocks += 1
                    if emitted_blocks == target_blocks:
                        break

                pending_tokens = pending_tokens[complete_length:]
                if emitted_blocks == target_blocks:
                    break
            if emitted_blocks == target_blocks:
                break

        assert emitted_blocks == target_blocks, (worker, emitted_blocks, target_blocks)


def main():
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_STEP_ID")
    hostname = socket.gethostname().split(".")[0]
    assert hostname == os.environ["SLURMD_NODENAME"].split(".")[0]

    config_path = Path(os.environ.get(
        "LAB5_CONFIG", Path(__file__).with_name("lab5_config.json")
    ))
    config = json.loads(config_path.read_text())
    output_dir = Path(os.environ.get(
        "LAB5_DATA", "/home/nat1andotxyz/lab5/dolma-seed42-8192-v1"
    ))
    cache_dir = Path(os.environ["LAB5_CACHE"])
    assert not output_dir.exists() or not any(output_dir.iterdir())
    assert 0 < config["prepare_tokens"] <= 6_000_000_000
    assert 0 < config["eval_documents"] <= config["holdout_documents"] // 5
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["model_revision"]
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(output_dir / "tokenizer")
    model_config = AutoConfig.from_pretrained(
        config["model_id"], revision=config["model_revision"]
    )
    model_config.save_pretrained(output_dir / "model-config")

    workers = min(24, int(os.environ["SLURM_CPUS_PER_TASK"]))
    documents = load_dataset(
        config["dataset_id"], revision=config["dataset_revision"], split="train",
        cache_dir=str(cache_dir / "datasets"), num_proc=workers,
    )
    documents = documents.shuffle(seed=config["seed"])
    training_end = len(documents) - config["holdout_documents"]
    assert training_end > 0
    training_documents = documents.select(range(training_end)).select_columns(["text"])
    heldout_documents = documents.select(range(training_end, len(documents)))
    evaluation_documents = heldout_documents.select(range(config["eval_documents"]))
    heldout_rows = documents._indices.column(0).slice(training_end).to_pylist()
    (output_dir / "holdout-source-rows.json").write_text(json.dumps(heldout_rows))

    evaluation_data = evaluation_documents.map(
        lambda batch: tokenizer(
            batch["text"], add_special_tokens=True, truncation=True,
            max_length=config["sequence_length"], return_attention_mask=False,
        ),
        batched=True, remove_columns=evaluation_documents.column_names,
    )
    evaluation_data.save_to_disk(output_dir / "eval")

    total_blocks = config["prepare_tokens"] // config["sequence_length"]
    assert total_blocks >= workers
    training_data = Dataset.from_generator(
        pack_documents,
        gen_kwargs={
            "worker_ids": list(range(workers)), "documents": training_documents,
            "tokenizer_path": str(output_dir / "tokenizer"), "workers": workers,
            "blocks": total_blocks, "length": config["sequence_length"],
        },
        num_proc=workers, cache_dir=str(cache_dir / "packed"),
        features=Features({
            "input_ids": Sequence(Value("int32"), length=config["sequence_length"])
        }),
    )
    assert len(training_data) == total_blocks
    training_data.save_to_disk(output_dir / "train", max_shard_size="1GB")

    manifest = {
        "dataset_id": config["dataset_id"], "dataset_revision": config["dataset_revision"],
        "model_id": config["model_id"], "model_revision": config["model_revision"],
        "shuffle_seed": config["seed"], "raw_documents": len(documents),
        "holdout_documents": config["holdout_documents"],
        "eval_documents": len(evaluation_data), "sequence_length": config["sequence_length"],
        "training_blocks": len(training_data),
        "prepared_tokens": len(training_data) * config["sequence_length"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
