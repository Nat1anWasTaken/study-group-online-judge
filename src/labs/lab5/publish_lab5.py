import argparse
import json
import math
import os
import socket
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


def main():
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_STEP_ID")
    hostname = socket.gethostname().split(".")[0]
    assert hostname == os.environ["SLURMD_NODENAME"].split(".")[0]

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs=2, action="append", required=True,
                        metavar=("RUN_DIRECTORY", "HF_REPOSITORY"))
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    api = HfApi()

    for run_directory, repository in args.model:
        run_dir = Path(run_directory).resolve()
        model_dir = run_dir / "model"
        config = json.loads((run_dir / "training-config.json").read_text())
        result = json.loads((run_dir / "result.json").read_text())
        assert result["step"] > 0
        assert 0 < result["total_tokens_seen"] <= 6_000_000_000
        assert result["h200_hours"] <= 64
        assert math.isfinite(result["perplexity"])
        assert (model_dir / "config.json").is_file()
        assert (model_dir / "tokenizer.json").is_file()
        assert (model_dir / "tokenizer_config.json").is_file()
        assert any(model_dir.glob("*.safetensors"))
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.exists():
            weight_files = json.loads(index_path.read_text())["weight_map"].values()
            assert all((model_dir / filename).is_file() for filename in weight_files)

        batch_tokens = (
            config["world_size"] * config["micro_batch_size"]
            * config["gradient_accumulation_steps"] * config["sequence_length"]
        )
        model_card = f"""---
license: llama3.2
library_name: transformers
pipeline_tag: text-generation
datasets:
- {config['dataset_id']}
tags:
- lab5
- muon
- adamw
- from-scratch
---

# {repository.split('/')[-1]}

Llama-3.2-1B architecture trained from random initialization with Muon for hidden
matrices and AdamW for the tied embedding/output table and normalization parameters.

| Setting | Value |
| --- | --- |
| Source run | {run_dir.name} |
| Context length | {config['sequence_length']} |
| Tokens per update | {batch_tokens} |
| Peak learning rate (Muon and AdamW) | {config['learning_rate']} |
| Muon learning-rate adjustment | {config['muon_adjust_lr_fn']} |
| AdamW betas | {config['adamw_betas']} |
| Momentum | {config['momentum']} |
| Matrix and embedding weight decay | {config['weight_decay']} |
| Normalization weight decay | {config['normalization_weight_decay']} |
| Schedule | {config['schedule']} |
| Warmup fraction | {config['warmup_ratio']} |
| Schedule basis | {config['schedule_basis']} |
| Training tokens processed | {result['total_tokens_seen']} |
| H200-hours | {result['h200_hours']:.4f} |
| Heldout perplexity | {result['perplexity']:.4f} |
| Evaluated documents | {config['eval_documents']} |

[W&B run]({result['wandb_url']})

Documents were shuffled with seed {config['seed']} before tokenization, and the
last {config['holdout_documents']} documents were excluded from training.
Perplexity is token-weighted corpus perplexity on a fixed subset of the heldout
set, with individual documents truncated to {config['sequence_length']} tokens.
Evaluation ran inside training. This is not an OJ score on all heldout documents.

The full recipe is in `training-config.json`, the final result is in `result.json`,
and the in-training evaluation curve is in `eval-curve.jsonl`.
"""
        operations = [
            CommitOperationAdd(path_in_repo=file.name, path_or_fileobj=file)
            for file in sorted(model_dir.iterdir()) if file.is_file()
        ]
        for filename in ("training-config.json", "result.json", "eval-curve.jsonl"):
            operations.append(CommitOperationAdd(
                path_in_repo=filename, path_or_fileobj=run_dir / filename,
            ))
        operations.append(CommitOperationAdd(
            path_in_repo="README.md", path_or_fileobj=model_card.encode(),
        ))

        api.create_repo(repository, repo_type="model", private=args.private, exist_ok=True)
        commit = api.create_commit(
            repo_id=repository, repo_type="model", operations=operations,
            commit_message=f"Publish {run_dir.name}", num_threads=2,
        )
        publication = {
            "repo_id": repository, "revision": commit.oid,
            "commit_url": commit.commit_url, "wandb_url": result["wandb_url"],
        }
        (run_dir / "publication.json").write_text(json.dumps(publication, indent=2))
        print(json.dumps(publication, indent=2), flush=True)
        print(f'eval_model_id = "{repository}"', flush=True)


if __name__ == "__main__":
    main()
