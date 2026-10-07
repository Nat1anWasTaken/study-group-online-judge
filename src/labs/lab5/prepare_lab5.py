import hashlib
import io
import json
import os
import socket
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import pyarrow as pa
import pyarrow.json as paj

from datasets import (
    Dataset, Features, Sequence, Value, concatenate_datasets, load_dataset_builder,
)
from datasets.arrow_writer import ArrowWriter
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer


DATASET_ID = "allenai/dolma3_mix-150B-1025"
DATASET_REVISION = "afa92bfb22366821c5e6cd427cdd036b34b713ef"


def convert_text_shard(task):
    source, destination = map(Path, task)
    if destination.exists():
        return str(destination), len(Dataset.from_file(str(destination)))

    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    schema = Features({"text": Value("string")}).arrow_schema
    temporary = destination.with_suffix(".incomplete")
    rows = 0
    with pa.input_stream(str(source), compression="zstd") as compressed:
        with io.BufferedReader(compressed) as stream:
            with pa.OSFile(str(temporary), "wb") as sink:
                with pa.ipc.new_stream(sink, schema) as writer:
                    while True:
                        chunk = stream.read(10 << 20)
                        if not chunk:
                            break
                        chunk += stream.readline()
                        table = paj.read_json(
                            io.BytesIO(chunk),
                            read_options=paj.ReadOptions(
                                use_threads=False, block_size=len(chunk),
                            ),
                            parse_options=paj.ParseOptions(
                                explicit_schema=schema,
                                unexpected_field_behavior="ignore",
                            ),
                        )
                        if table.column("text").null_count:
                            raise ValueError(f"Missing text in {source}")
                        writer.write_table(table.cast(schema))
                        rows += table.num_rows
    temporary.replace(destination)
    return str(destination), rows


def load_text_documents(cache_dir, workers):
    builder = load_dataset_builder(
        DATASET_ID, revision=DATASET_REVISION,
        cache_dir=str(cache_dir / "datasets"),
    )
    source_files = builder.config.data_files["train"]
    snapshot = Path(snapshot_download(
        DATASET_ID, repo_type="dataset", revision=DATASET_REVISION,
        allow_patterns=["README.md", "data/**/*.jsonl.zst"],
        max_workers=workers,
    ))
    prefix = f"hf://datasets/{DATASET_ID}@{DATASET_REVISION}/"
    converted_dir = cache_dir / "text-v1" / DATASET_REVISION
    converted_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for source in source_files:
        if not source.startswith(prefix):
            raise ValueError(f"Unexpected dataset source: {source}")
        relative = source[len(prefix):]
        destination = converted_dir / (
            hashlib.sha256(relative.encode()).hexdigest() + ".arrow"
        )
        tasks.append((str(snapshot / relative), str(destination)))

    print(json.dumps({"stage": "convert_text", "workers": workers,
                      "source_shards": len(tasks)}), flush=True)
    ordered_paths = [None] * len(tasks)
    converted_rows = 0
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn"),
    ) as pool:
        schedule = sorted(range(len(tasks)),
                          key=lambda index: Path(tasks[index][0]).stat().st_size,
                          reverse=True)
        pending = {pool.submit(convert_text_shard, tasks[index]): index
                   for index in schedule}
        for completed, future in enumerate(as_completed(pending), 1):
            path, rows = future.result()
            ordered_paths[pending[future]] = path
            converted_rows += rows
            if completed % 100 == 0 or completed == len(tasks):
                print(json.dumps({"stage": "convert_text",
                                  "completed_shards": completed,
                                  "total_shards": len(tasks),
                                  "converted_documents": converted_rows}), flush=True)
    return concatenate_datasets([Dataset.from_file(path) for path in ordered_paths])


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


def init_packing(documents, tokenizer_path, workers, blocks, length, directory):
    global packing_args
    packing_args = documents, tokenizer_path, workers, blocks, length, directory
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


def pack_partition(worker):
    documents, tokenizer_path, workers, blocks, length, directory = packing_args
    path = Path(directory) / f"worker-{worker:02d}.arrow"
    temporary = path.with_suffix(".incomplete")
    features = Features({"input_ids": Sequence(Value("int32"), length=length)})
    with ArrowWriter(path=str(temporary), features=features, writer_batch_size=256) as writer:
        for row in pack_documents([worker], documents, tokenizer_path, workers, blocks, length):
            writer.write(row)
        rows, _ = writer.finalize()
    temporary.replace(path)
    print(f"Packed worker {worker}: {rows} blocks", flush=True)
    return str(path)


def main():
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_STEP_ID")
    hostname = socket.gethostname().split(".")[0]
    assert hostname == os.environ["SLURMD_NODENAME"].split(".")[0]

    output_dir = Path(os.environ.get(
        "LAB5_DATA", "/home/nat1andotxyz/lab5/dolma-seed42-8192-v1"
    ))
    cache_dir = Path(os.environ["LAB5_CACHE"])
    assert not (output_dir / "manifest.json").exists(), "Data is already prepared"
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B", revision="4e20de362430cd3b72f300e6b0f18e50e7166e08"
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(output_dir / "tokenizer")
    model_config = AutoConfig.from_pretrained(
        "meta-llama/Llama-3.2-1B", revision="4e20de362430cd3b72f300e6b0f18e50e7166e08"
    )
    model_config.save_pretrained(output_dir / "model-config")

    workers = min(24, int(os.environ["SLURM_CPUS_PER_TASK"]))
    documents = load_text_documents(cache_dir, workers)
    print(json.dumps({"stage": "shuffle", "documents": len(documents)}), flush=True)
    documents = documents.shuffle(seed=42)
    training_end = len(documents) - 50000
    assert training_end > 0
    training_documents = documents.select(range(training_end)).select_columns(["text"])
    heldout_documents = documents.select(range(training_end, len(documents)))
    evaluation_documents = Dataset.from_dict(heldout_documents[:1024])
    heldout_rows = documents._indices.column(0).slice(training_end).to_pylist()
    (output_dir / "holdout-source-rows.json").write_text(json.dumps(heldout_rows))

    print("Tokenizing 1024 validation documents", flush=True)
    evaluation_data = evaluation_documents.map(
        lambda batch: tokenizer(
            batch["text"], add_special_tokens=True, truncation=True,
            max_length=8192, return_attention_mask=False,
        ),
        batched=True, num_proc=workers,
        remove_columns=evaluation_documents.column_names,
    )
    evaluation_data.save_to_disk(output_dir / "eval")

    total_blocks = 6000000000 // 8192
    assert total_blocks >= workers
    print(json.dumps({"stage": "pack", "workers": workers,
                      "training_blocks": total_blocks}), flush=True)
    packed_dir = cache_dir / "packed"
    packed_dir.mkdir(exist_ok=True)
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("fork"),
        initializer=init_packing,
        initargs=(training_documents, str(output_dir / "tokenizer"),
                  workers, total_blocks, 8192, str(packed_dir)),
    ) as pool:
        paths = list(pool.map(pack_partition, range(workers)))
    training_data = concatenate_datasets([Dataset.from_file(path) for path in paths])
    assert len(training_data) == total_blocks
    training_data.save_to_disk(output_dir / "train", max_shard_size="1GB")

    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "preparation_format": "text-v1",
        "preparation_workers": workers,
        "model_id": "meta-llama/Llama-3.2-1B",
        "model_revision": "4e20de362430cd3b72f300e6b0f18e50e7166e08",
        "shuffle_seed": 42, "raw_documents": len(documents),
        "holdout_documents": 50000,
        "eval_documents": len(evaluation_data), "sequence_length": 8192,
        "training_blocks": len(training_data),
        "prepared_tokens": len(training_data) * 8192,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
