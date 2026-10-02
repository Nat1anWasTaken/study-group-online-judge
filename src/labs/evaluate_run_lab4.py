import os
import sys
import time
from pathlib import Path

import torch
from eval_lab4 import (
    EVAL_BATCH_SIZE,
    load_validation,
    validation_metrics,
    validation_totals,
)
from transformers import AutoTokenizer, GPT2LMHeadModel

import wandb

training_job = sys.argv[1]
runs = list(
    wandb.Api().runs(
        "cerulean-labs/gpt2-training", filters={"config.slurm_job_id": training_job}
    )
)
if len(runs) != 1:
    raise RuntimeError(
        f"Expected one W&B run for training job {training_job}, found {len(runs)}"
    )
run = runs[0]
if not run.summary.get("training_completed"):
    raise RuntimeError(f"Training job {training_job} has not completed")

work_dir = Path(os.environ.get("LAB4_WORK_DIR", f"/work/{os.environ['USER']}/lab4"))
output_dir = work_dir / "runs" / run.name
model_id = run.config["hf_model_id"]
final_step = run.summary["final_train_step"]

started = time.monotonic()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
model = GPT2LMHeadModel.from_pretrained(
    output_dir, dtype=torch.float32, attn_implementation="sdpa"
).to("cuda")
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(output_dir)
dataset, metadata = load_validation(work_dir / "c4-dev")
wandb.init(entity="cerulean-labs", project="gpt2-training", id=run.id, resume="must")
wandb.config.update({"validation": metadata})
wandb.run.summary.update(
    {
        "evaluation_completed": False,
        "hf_upload_completed": False,
        "evaluation_job_id": os.environ["SLURM_JOB_ID"],
    }
)
try:
    eval_started = time.monotonic()
    totals = validation_totals(model, dataset, tokenizer, "cuda", EVAL_BATCH_SIZE)
    metrics = validation_metrics(totals)
    metrics["runtime"] = time.monotonic() - eval_started
    metrics = {f"eval_{key}": value for key, value in metrics.items()}
    wandb.log({**metrics, "evaluated_train_step": final_step})
    wandb.run.summary.update(
        {
            **{f"last_{key}": value for key, value in metrics.items()},
            "last_eval_step": final_step,
            "final_eval_perplexity": metrics["eval_perplexity"],
            "final_eval_loss": metrics["eval_loss"],
            "evaluation_completed": True,
        }
    )
    print(f"Final dev perplexity: {metrics['eval_perplexity']}", flush=True)

    wandb.run.summary.update({"evaluation_runtime": time.monotonic() - started})
    print(
        f"Evaluated {output_dir}; final dev PPL={metrics['eval_perplexity']}",
        flush=True,
    )
finally:
    wandb.finish()
