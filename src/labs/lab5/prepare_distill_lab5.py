import hashlib
import json
import os
import socket
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from datasets import load_from_disk
from huggingface_hub import HfApi, snapshot_download
from torch.utils.data import DistributedSampler
from train_lab5 import ACCUMULATION, DATA_SEED, document_boundaries
from transformers import AutoModelForCausalLM, AutoTokenizer

TEACHER_ID = "meta-llama/Llama-3.1-8B"
TEMPERATURE = 1.0
TOP_K = 64
VOCAB_SIZE = 128256
SEQUENCE_LENGTH = 8192
STUDENT_STEPS = 4000
STUDENT_WORLD_SIZE = 2
LOGIT_CHUNK = 256
BYTES_PER_POSITION = TOP_K * 4 + TOP_K * 2 + 4
CACHE_ARRAYS = {
    "teacher_ids": (np.int32, 3),
    "teacher_log_probs": (np.float16, 3),
    "teacher_tail": (np.float32, 2),
}
ENCODING_EXAMPLES = [
    "Hello, world!",
    " The quick brown fox jumps over the lazy dog.\n\n",
    "def square(x):\n    return x ** 2\n",
    "Unicode: naïve café, 日本語, Ελληνικά, emoji 🙂",
    "<|begin_of_text|>special<|end_of_text|>",
    "   leading spaces and\ttabs 12345 67.89",
]


def tokenizer_fingerprint(tokenizer):
    return hashlib.sha256(json.dumps(
        [sorted(tokenizer.get_vocab().items()), tokenizer.bos_token_id, tokenizer.eos_token_id]
    ).encode()).hexdigest()


@torch.inference_mode()
def teacher_targets(teacher, input_ids, bos_token_id):
    position_ids, cu_seq_lens = document_boundaries(input_ids, bos_token_id)
    hidden_states = teacher.model(
        input_ids=input_ids,
        position_ids=position_ids,
        use_cache=False,
        cu_seq_lens_q=cu_seq_lens,
        cu_seq_lens_k=cu_seq_lens,
        max_length_q=input_ids.shape[1],
        max_length_k=input_ids.shape[1],
    ).last_hidden_state[0]
    ids = torch.empty(len(hidden_states), TOP_K, dtype=torch.int32, device=input_ids.device)
    log_probs = torch.empty(len(hidden_states), TOP_K, dtype=torch.float16, device=input_ids.device)
    tail = torch.empty(len(hidden_states), dtype=torch.float32, device=input_ids.device)
    for start in range(0, len(hidden_states), LOGIT_CHUNK):
        logits = teacher.lm_head(hidden_states[start : start + LOGIT_CHUNK]).float() / TEMPERATURE
        values, indices = logits.topk(TOP_K, dim=-1)
        top = values - logits.logsumexp(-1, keepdim=True)
        ids[start : start + LOGIT_CHUNK] = indices
        log_probs[start : start + LOGIT_CHUNK] = top
        tail[start : start + LOGIT_CHUNK] = (1 - top.exp().sum(-1)).clamp_min(0)
    return ids, log_probs, tail


def main():
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_STEP_ID")
    assert socket.gethostname().split(".")[0] == os.environ["SLURMD_NODENAME"].split(".")[0]
    started = time.time()
    data_dir = Path(os.environ["LAB5_DATA"])
    cache_dir = Path(os.environ["LAB5_TEACHER_CACHE"])
    blocks = int(os.environ.get("LAB5_CACHE_BLOCKS", "128"))
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    student_blocks = STUDENT_STEPS * ACCUMULATION * STUDENT_WORLD_SIZE
    assert 0 < blocks <= student_blocks

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)

    manifest = json.loads((data_dir / "manifest.json").read_text())
    training_data = load_from_disk(data_dir / "train").with_format("torch")
    tokenizer = AutoTokenizer.from_pretrained(data_dir / "tokenizer", local_files_only=True)
    permutation = torch.randperm(len(training_data), generator=torch.Generator().manual_seed(DATA_SEED))
    for student_rank in range(STUDENT_WORLD_SIZE):
        sampler = DistributedSampler(
            training_data, num_replicas=STUDENT_WORLD_SIZE, rank=student_rank, seed=DATA_SEED, drop_last=True
        )
        assert list(sampler)[: STUDENT_STEPS * ACCUMULATION] == permutation[student_rank:student_blocks:STUDENT_WORLD_SIZE].tolist()
    block_ids = permutation[:blocks].numpy()

    revision = None
    if rank == 0:
        cache_dir.mkdir(parents=True, exist_ok=False)
        revision = HfApi().model_info(TEACHER_ID).sha
        snapshot_download(TEACHER_ID, revision=revision, allow_patterns=["*.json", "*.safetensors"])
    shared = [revision]
    dist.broadcast_object_list(shared, src=0)
    revision = shared[0]

    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID, revision=revision)
    assert len(tokenizer.get_vocab()) == VOCAB_SIZE
    assert teacher_tokenizer.get_vocab() == tokenizer.get_vocab()
    assert teacher_tokenizer.bos_token_id == tokenizer.bos_token_id
    assert teacher_tokenizer.eos_token_id == tokenizer.eos_token_id
    examples = ENCODING_EXAMPLES + [tokenizer.decode(training_data[int(block_ids[0])]["input_ids"])]
    for add_special_tokens in (True, False):
        assert teacher_tokenizer(examples, add_special_tokens=add_special_tokens, verbose=False)["input_ids"] == tokenizer(
            examples, add_special_tokens=add_special_tokens, verbose=False
        )["input_ids"]
    fingerprint = tokenizer_fingerprint(tokenizer)
    assert tokenizer_fingerprint(teacher_tokenizer) == fingerprint

    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_ID, revision=revision, dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map={"": device},
    ).eval()
    assert teacher.config.vocab_size == VOCAB_SIZE and teacher.lm_head.weight.dtype == torch.bfloat16

    shapes = {name: ([blocks, SEQUENCE_LENGTH, TOP_K][:ndim], dtype) for name, (dtype, ndim) in CACHE_ARRAYS.items()}
    if rank == 0:
        np.save(cache_dir / "block_ids.npy", block_ids)
        for name, (shape, dtype) in shapes.items():
            np.lib.format.open_memmap(cache_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=tuple(shape)).flush()
    dist.barrier()
    arrays = [np.load(cache_dir / f"{name}.npy", mmap_mode="r+") for name in shapes]
    load_seconds = time.time() - started

    slots = range(rank, blocks, world_size)
    block_seconds = []
    retained = torch.zeros((), dtype=torch.float64, device=device)
    torch.cuda.synchronize(device)
    processing_start = time.monotonic()
    for slot in slots:
        block_start = time.monotonic()
        input_ids = training_data[int(block_ids[slot])]["input_ids"].to(device, dtype=torch.long)[None]
        targets = teacher_targets(teacher, input_ids, tokenizer.bos_token_id)
        retained += (1 - targets[2]).sum(dtype=torch.float64)
        for array, values in zip(arrays, targets):
            array[slot] = values.cpu().numpy()
        torch.cuda.synchronize(device)
        block_seconds.append(time.monotonic() - block_start)
    flush_start = time.monotonic()
    for array in arrays:
        array.flush()
    flush_seconds = time.monotonic() - flush_start
    torch.cuda.synchronize(device)
    processing_seconds = time.monotonic() - processing_start
    del arrays

    tokens = len(slots) * SEQUENCE_LENGTH
    steady_seconds = sum(block_seconds[1:])
    local = {
        "rank": rank,
        "node": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(device),
        "blocks": len(slots),
        "tokens": tokens,
        "bytes": tokens * BYTES_PER_POSITION,
        "load_seconds": load_seconds,
        "processing_seconds": processing_seconds,
        "flush_seconds": flush_seconds,
        "first_block_seconds": block_seconds[0] if block_seconds else None,
        "processing_tokens_per_second": tokens / processing_seconds,
        "steady_tokens_per_second": (tokens - SEQUENCE_LENGTH) / steady_seconds if steady_seconds else None,
        "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
    }
    print(json.dumps(local), flush=True)
    dist.barrier()
    ranks = [None] * world_size
    dist.all_gather_object(ranks, local)
    dist.all_reduce(retained)

    if rank == 0:
        wall_seconds = time.time() - started
        total_tokens = blocks * SEQUENCE_LENGTH
        processing = max(row["processing_seconds"] for row in ranks)
        steady_rate = sum(row["steady_tokens_per_second"] for row in ranks) if all(row["steady_tokens_per_second"] for row in ranks) else None

        def projection(projected_tokens):
            return {
                "tokens": projected_tokens,
                "processing_wall_hours": projected_tokens / steady_rate / 3600 if steady_rate else None,
                "processing_h200_hours": projected_tokens / steady_rate / 3600 * world_size if steady_rate else None,
                "cache_bytes": projected_tokens * BYTES_PER_POSITION,
            }

        benchmark = {
            "world_size": world_size,
            "blocks": blocks,
            "tokens": total_tokens,
            "cache_bytes": sum((cache_dir / f"{name}.npy").stat().st_size for name in shapes),
            "cache_bytes_per_position": BYTES_PER_POSITION,
            "wall_seconds": wall_seconds,
            "wall_h200_hours": wall_seconds * world_size / 3600,
            "max_load_seconds": max(row["load_seconds"] for row in ranks),
            "max_processing_seconds": processing,
            "processing_h200_hours": processing * world_size / 3600,
            "aggregate_tokens_per_second": total_tokens / processing,
            "steady_aggregate_tokens_per_second": steady_rate,
            "retained_top_k_mass_mean": retained.item() / total_tokens,
            "projections": {"262M": projection(262144000), "6B": projection(6000000000)},
            "ranks": ranks,
        }
        metadata = {
            "complete": True,
            "teacher_id": TEACHER_ID,
            "teacher_revision": revision,
            "teacher_dtype": "bfloat16",
            "teacher_attention": "flash_attention_2",
            "teacher_config": teacher.config.to_dict(),
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "vocab_size": VOCAB_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "position_stride": 1,
            "positions": "all input positions; student pairs positions [:-1] with labels [1:]",
            "log_probs": "full-vocabulary-normalized top-k log-probabilities at temperature 1, not renormalized over top-k",
            "tail": "1 - sum(exp(fp32 top-k log-probabilities)), clamped at 0",
            "attention": "document_boundaries from train_lab5: BOS-delimited varlen attention and reset position ids",
            "blocks": blocks,
            "tokens": total_tokens,
            "student_steps": STUDENT_STEPS,
            "student_world_size": STUDENT_WORLD_SIZE,
            "student_accumulation": ACCUMULATION,
            "student_blocks": student_blocks,
            "full_student_coverage": blocks == student_blocks,
            "sampler": {
                "type": "DistributedSampler",
                "shuffle": True,
                "drop_last": True,
                "seed": DATA_SEED,
                "num_replicas": STUDENT_WORLD_SIZE,
                "slot_order": "slot i is position i of torch.randperm(len(train), seed 42); student rank r micro-batch j reads slot 2j + r",
            },
            "data_seed": DATA_SEED,
            "dataset_length": len(training_data),
            "dataset_revision": manifest["dataset_revision"],
            "data_manifest": manifest,
            "data_dir": str(data_dir),
            "block_ids_sha256": hashlib.sha256(block_ids.tobytes()).hexdigest(),
            "tokenizer_fingerprint": fingerprint,
            "teacher_tokenizer_fingerprint": tokenizer_fingerprint(teacher_tokenizer),
            "tokenizer_encoding_examples": len(examples),
            "model_config_sha256": hashlib.sha256((data_dir / "model-config" / "config.json").read_bytes()).hexdigest(),
            "arrays": {name: {"shape": shape, "dtype": np.dtype(dtype).name} for name, (shape, dtype) in shapes.items()},
            "versions": {name: version(name) for name in ("torch", "transformers", "huggingface_hub", "datasets", "numpy")},
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "git_commit": os.environ["LAB5_GIT_COMMIT"],
            "source": str(Path(__file__).resolve().parent),
            "benchmark": benchmark,
        }
        (cache_dir / "cache.json.incomplete").write_text(json.dumps(metadata, indent=2))
        (cache_dir / "cache.json.incomplete").replace(cache_dir / "cache.json")
        print(json.dumps(benchmark, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
