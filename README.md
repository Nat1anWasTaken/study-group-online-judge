# Study Group Online Judge

A small online judge for machine-learning study groups. Participants submit a
commit from their fork, and the judge evaluates it with its own task tests.

## Participants

### Submission flow

1. Fork `cerulean-works/study-group-online-judge` into your GitHub account.
2. Implement the task in your fork and push your changes to `main`.
3. In your fork, open **Actions** → **Submit to study group judge** → **Run
   workflow**, select the task, and run it on `main`.

Your fork needs an Actions secret named `JUDGE_API_TOKEN` and a repository
variable named `WANDB_PROJECT_URL`. Ask the study-group organizer for the token
and project URL if they are not already configured. The workflow submits the
exact commit you selected and succeeds once the judge queues it; this does not
mean the submission passed.

### Where to implement

Put each implementation in the `src/labs/` file specified by its task. The
judge checks out the submitted commit, then its task loads that file directly
as a Python module and calls the required function. For example, `lab1` loads
`src/labs/lab1.py` and calls `gpt2_complete`; it checks the completions and
logits against GPT-2 Small using 20 Tiny Shakespeare prompts.

Keep your implementation in your fork. Changing the judge's task code in your
fork does not change the evaluation used by the deployed judge.

### View results

The submission workflow prints the queued job ID and a link to its W&B run in
the run summary. The link becomes active when the worker starts the job. Open
it for progress, logs, metrics, and the final pass/fail result.

## Contributors

### Defining tasks

Each task is a `Task` subclass in `src/judge/tasks/`. It declares resource
limits and evaluates the checked-out participant submission:

```python
from pathlib import Path

from judge.models import JudgeResult, Resources
from judge.tasks.base import Task


class Assignment01(Task):
    id = "assignment-01"
    resources = Resources(cpus=2, memory_gb=4, timeout_seconds=60)

    def evaluate(self, submission: Path) -> JudgeResult:
        # Import the participant implementation and evaluate it here.
        ...
```

Register the task instance in the `TASKS` dictionary in
`src/judge/tasks/__init__.py`. Add its ID to the `task_id` choices in
`.github/workflows/submit.yaml` so participants can select it. Return
`JudgeResult(passed=...)` for correctness tasks, or include `score` and
`metrics` for benchmarks. Set `gpus` in `Resources` when a task needs a GPU.

### Development

```console
uv sync --frozen
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run python -W error::ResourceWarning -m unittest discover -s tests -v
```

The API exposes `GET /healthz`, `POST /submissions`, and `GET /jobs/{job_id}`.
Submission and job routes require `Authorization: Bearer <JUDGE_API_TOKEN>`.

### Deployment

Requirements: Docker with Compose, access to the GHCR judge image, a W&B
project and API key, and a host work directory writable by the judge container's
UID/GID `10001`. GPU tasks also require the NVIDIA Container Toolkit.

Copy `.env.example` to `.env` and set its required values. `JUDGE_WORK_ROOT`
must be an absolute host path, mounted at the same path in the worker and its
evaluator containers. For example:

```console
sudo mkdir -p /var/lib/study-group-online-judge/work
sudo chown 10001:10001 /var/lib/study-group-online-judge/work
stat -c '%g' /var/run/docker.sock
```

Set `DOCKER_GID` to the reported Docker socket group. The `999` in
`.env.example` is an example socket group, not the judge user's UID or primary
GID. Docker Desktop and OrbStack commonly report group `0`. Generate a
submission token, then start the services:

```console
openssl rand -hex 32
docker compose up -d
docker compose ps
```

Compose does not publish an API port on the host. The `api` container listens
on port `8000` internally; configure Dokploy to route to that service and
port. SQLite and local W&B files live in `judge-data`. Checked-out repositories
and result files live under `JUDGE_WORK_ROOT`. The HF and uv cache volumes
persist downloads across submissions and container restarts.

```console
docker compose logs --follow api worker
docker compose restart worker
docker compose down
```
