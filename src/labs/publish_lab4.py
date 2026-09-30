import os
import sys
import time
from pathlib import Path

import wandb
from transformers import AutoTokenizer, GPT2LMHeadModel

training_job = sys.argv[1]
run = list(wandb.Api().runs(
    "cerulean-labs/gpt2-training", filters={"config.slurm_job_id": training_job}
))[0]
output_dir = Path("/work/nat1andotxyz/lab4/runs") / run.name
model_id = run.config["hf_model_id"]
started = time.monotonic()
model = GPT2LMHeadModel.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(output_dir)
model.push_to_hub(model_id)
tokenizer.push_to_hub(model_id)
summary = dict(run.summary)
wandb.init(entity="cerulean-labs", project="gpt2-training", id=run.id, resume="must")
metrics = {
    "final_eval_perplexity": summary["last_eval_perplexity"],
    "final_eval_loss": summary["last_eval_loss"],
    "hf_upload_completed": True,
    "training_completed": True,
    "publish_job_id": os.environ["SLURM_JOB_ID"],
    "publish_runtime": time.monotonic() - started,
    "hf_model_id": model_id,
    "train_steps_per_second": summary["train/global_step"] / summary["train_runtime"],
    "train_samples_per_second": (
        summary["train/global_step"] * run.config["per_device_train_batch_size"]
        * run.config["gradient_accumulation_steps"] * 2 / summary["train_runtime"]
    ),
}
if training_job in ("467196", "467197", "467198", "467199", "467200"):
    metrics["recovered_after_training_job_failure"] = True
    metrics["training_job_failure_reason"] = (
        "final summary update" if training_job in ("467196", "467197") else "upload timeout"
    )
wandb.run.summary.update(metrics)
wandb.log({"publication_completed": 1})
print(f"Published {model_id}; final dev PPL={metrics['final_eval_perplexity']}", flush=True)
wandb.finish()
