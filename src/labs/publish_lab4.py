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
work_dir = Path(os.environ.get("LAB4_WORK_DIR", f"/work/{os.environ['USER']}/lab4"))
output_dir = work_dir / "runs" / run.name
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
}
wandb.run.summary.update(metrics)
wandb.log({"publication_completed": 1})
print(f"Published {model_id}; final dev PPL={metrics['final_eval_perplexity']}", flush=True)
wandb.finish()
