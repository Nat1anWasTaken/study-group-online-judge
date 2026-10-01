import os
import sys
import time
from pathlib import Path

import torch
import wandb
from transformers import AutoTokenizer, GPT2LMHeadModel

from eval_lab4 import (
    EVAL_BATCH_SIZE,
    load_validation,
    validation_metrics,
    validation_totals,
)

training_job = sys.argv[1]
runs = list(wandb.Api().runs(
    "cerulean-labs/gpt2-training", filters={"config.slurm_job_id": training_job}
))
if len(runs) != 1:
    raise RuntimeError(f"Expected one W&B run for training job {training_job}, found {len(runs)}")
run = runs[0]
if not run.summary.get("training_completed") or not run.summary.get("averaging_completed"):
    raise RuntimeError(f"Training job {training_job} has not saved both checkpoints")

work_dir = Path(os.environ.get("LAB4_WORK_DIR", f"/work/{os.environ['USER']}/lab4"))
output_dir = work_dir / "runs" / run.name
final_step = run.summary["final_train_step"]
checkpoints = [
    ("final", output_dir, run.config["hf_model_id"]),
    ("avg", output_dir / "avg", run.config["hf_avg_model_id"]),
]
for variant, directory, model_id in checkpoints:
    if not (directory / "model.safetensors").is_file():
        raise RuntimeError(f"Missing {variant} checkpoint: {directory}")

started = time.monotonic()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
dataset, metadata = load_validation(work_dir / "c4-dev")
wandb.init(entity="cerulean-labs", project="gpt2-training", id=run.id, resume="must")
wandb.config.update({"validation": metadata})
wandb.run.summary.update(dict(
    evaluation_completed=False,
    hf_upload_completed=False,
    publish_job_id=os.environ["SLURM_JOB_ID"],
))
try:
    for variant, directory, model_id in checkpoints:
        model = GPT2LMHeadModel.from_pretrained(directory, dtype=torch.float32).to("cuda")
        model.config.use_cache = False
        tokenizer = AutoTokenizer.from_pretrained(directory)
        eval_started = time.monotonic()
        totals = validation_totals(model, dataset, tokenizer, "cuda", EVAL_BATCH_SIZE)
        metrics = validation_metrics(totals)
        metrics["runtime"] = time.monotonic() - eval_started
        metrics = {f"{variant}_eval_{key}": value for key, value in metrics.items()}
        wandb.log({**metrics, "evaluated_train_step": final_step})
        wandb.run.summary.update({**metrics, f"{variant}_evaluation_completed": True})
        print(f"{variant} dev perplexity: {metrics[f'{variant}_eval_perplexity']}", flush=True)
        model.config.use_cache = True
        model.cpu()
        model.push_to_hub(model_id)
        tokenizer.push_to_hub(model_id)
        wandb.run.summary.update({f"{variant}_hf_upload_completed": True})
        print(f"Published {model_id}", flush=True)
        del model
        torch.cuda.empty_cache()
    wandb.run.summary.update(dict(
        evaluation_completed=True,
        hf_upload_completed=True,
        publish_runtime=time.monotonic() - started,
    ))
    wandb.log({"publication_completed": 1})
finally:
    wandb.finish()
