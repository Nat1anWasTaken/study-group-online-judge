"""Submit a participant's sbatch file to Nano4 without Docker."""

import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from judge.models import Resources
from judge.repository import SECRET_ENVIRONMENT_VARIABLES

SLURM_JOB_ID = re.compile(r"[0-9]+")
TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
}


@dataclass(frozen=True)
class SlurmState:
    state: str
    exit_code: str | None = None

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def succeeded(self) -> bool:
        return self.state == "COMPLETED" and self.exit_code == "0:0"


class SlurmExecutor:
    def __init__(
        self,
        *,
        account: str = "ACD115198",
        gpu_resource: str = "gpu",
        trusted_root: Path,
        sbatch_binary: str = "sbatch",
        sacct_binary: str = "sacct",
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", account):
            raise ValueError("Invalid Slurm account")
        if not re.fullmatch(r"[A-Za-z0-9_:-]+", gpu_resource):
            raise ValueError("Invalid Slurm GPU resource")
        self.account = account
        self.gpu_resource = gpu_resource
        self.trusted_root = trusted_root.resolve()
        self.sbatch_binary = sbatch_binary
        self.sacct_binary = sacct_binary

    def build_submit_command(
        self,
        *,
        task_id: str,
        resources: Resources,
        submission: Path,
        output_directory: Path,
    ) -> list[str]:
        if re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]*", task_id) is None:
            raise ValueError("Invalid task ID for Slurm")
        if resources.gpus < 1 or resources.gpus > 8:
            raise ValueError("Nano4 H200 jobs require 1 to 8 GPUs")
        if resources.cpus > 12 * resources.gpus:
            raise ValueError("Nano4 H200 allows at most 12 CPUs per GPU")
        if resources.memory_gb > 200 * resources.gpus:
            raise ValueError("Nano4 H200 allows at most 200 GiB per GPU")
        if resources.timeout_seconds > 48 * 3600:
            raise ValueError("Nano4 H200 jobs cannot exceed 48 hours")

        script = submission / "src" / "labs" / f"{task_id}.sbatch"
        if not script.is_file():
            raise FileNotFoundError(f"Expected src/labs/{task_id}.sbatch")
        partition = "dev" if resources.timeout_seconds <= 4 * 3600 else "8gpus"
        return [
            self.sbatch_binary,
            "--parsable",
            f"--account={self.account}",
            f"--partition={partition}",
            f"--gres={self.gpu_resource}:{resources.gpus}",
            f"--cpus-per-task={resources.cpus}",
            f"--mem={resources.memory_gb}G",
            f"--time={math.ceil(resources.timeout_seconds / 60)}",
            f"--job-name=judge-{task_id}",
            f"--chdir={submission.resolve()}",
            f"--output={(output_directory / 'slurm.log').resolve()}",
            f"--error={(output_directory / 'slurm.log').resolve()}",
            str(script.resolve()),
        ]

    def submit(
        self,
        *,
        task_id: str,
        resources: Resources,
        submission: Path,
        output_directory: Path,
    ) -> str:
        output_directory.mkdir(parents=True, exist_ok=True)
        command = self.build_submit_command(
            task_id=task_id,
            resources=resources,
            submission=submission,
            output_directory=output_directory,
        )
        environment = os.environ.copy()
        for variable in SECRET_ENVIRONMENT_VARIABLES:
            environment.pop(variable, None)
        environment.update(
            {
                "JUDGE_TASK_ID": task_id,
                "JUDGE_SUBMISSION_DIR": str(submission.resolve()),
                "JUDGE_OUTPUT_DIR": str(output_directory.resolve()),
                "JUDGE_TRUSTED_ROOT": str(self.trusted_root),
                "HF_HOME": str(Path(os.environ.get("JUDGE_HF_HOME", "/work/hf-cache"))),
                "UV_CACHE_DIR": str(
                    Path(os.environ.get("JUDGE_UV_CACHE_DIR", "/work/uv-cache"))
                ),
            }
        )
        process = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=environment,
            timeout=30,
        )
        job_id = process.stdout.strip().split(";", maxsplit=1)[0]
        if SLURM_JOB_ID.fullmatch(job_id) is None:
            raise RuntimeError(
                f"Unrecognized sbatch response: {process.stdout.strip()}"
            )
        return job_id

    def status(self, slurm_job_id: str) -> SlurmState | None:
        if SLURM_JOB_ID.fullmatch(slurm_job_id) is None:
            raise ValueError("Invalid Slurm job ID")
        process = subprocess.run(
            [
                self.sacct_binary,
                "--jobs",
                slurm_job_id,
                "--format=JobIDRaw,State,ExitCode",
                "--parsable2",
                "--noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        for line in process.stdout.splitlines():
            fields = line.split("|")
            if len(fields) >= 3 and fields[0] == slurm_job_id:
                return SlurmState(fields[1].split(" ", maxsplit=1)[0], fields[2])
        return None
