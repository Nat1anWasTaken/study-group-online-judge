import json
import os
import sys
import tempfile
from pathlib import Path

import wandb
from judge.tasks.lab4 import Lab4

model_id = sys.argv[1]
training_commit = sys.argv[2]
wandb.init(
    entity="cerulean-labs", project="gpt2-training", job_type="oj-evaluation",
    name=f"oj-{model_id.split('/')[-1]}-{os.environ['SLURM_JOB_ID']}",
    config={"hf_model_id": model_id, "training_commit": training_commit,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "documents": 100_000, "shuffle_seed": 42},
)
with tempfile.TemporaryDirectory(prefix="lab4-oj-") as directory:
    submission = Path(directory)
    (submission / "src/labs").mkdir(parents=True)
    (submission / "src/labs/lab4.py").write_text(f"eval_model_id = {model_id!r}\n")
    result = Lab4().evaluate(submission)
    print(json.dumps(result.model_dump(), indent=2), flush=True)
    wandb.log({**result.metrics, "score": result.score, "passed": result.passed})
    wandb.run.summary.update(result.metrics)
    wandb.run.summary.update(score=result.score, passed=result.passed)
wandb.finish()
