# Study Group Online Judge

A small online judge for machine-learning study groups. Students fork the starter
repository, implement an assignment, and submit an exact commit from GitHub Actions.
The judge checks out that commit and runs its own trusted evaluator in an isolated
Docker container.

## How it works

```text
GitHub Actions
      |
      | POST repository + commit SHA + task ID
      v
FastAPI -> SQLite queue -> trusted worker -> isolated evaluator container
                                              |
                                              +-> result.json
                                              +-> W&B run
```

The evaluator comes from the deployed judge image, not the student's fork. A student
can modify their copy of the evaluator, but the server never executes that copy.

The API exposes:

- `GET /healthz`
- `POST /submissions`
- `GET /jobs/{job_id}`

Submission and job routes require `Authorization: Bearer <JUDGE_API_TOKEN>`.

## Define a task

Each assignment is an ordinary `Task` subclass owned by the judge. It declares its
resource limits and evaluates the checked-out submission:

```python
from pathlib import Path

from judge.models import JudgeResult, Resources
from judge.tasks.base import Task


class Assignment01(Task):
    id = "assignment-01"
    resources = Resources(cpus=2, memory_gb=4, timeout_seconds=60)

    def evaluate(self, submission: Path) -> JudgeResult:
        # Load only the student implementation you intend to evaluate.
        ...
```

Register the trusted instance in `src/judge/tasks/__init__.py`:

```python
from judge.tasks.assignment_01 import Assignment01


TASKS = {
    Assignment01.id: Assignment01(),
}
```

Return `JudgeResult(passed=...)` for correctness tasks or include `score` and
`metrics` for benchmarks. GPU tasks set `gpus` in `Resources`; the executor then
adds Docker's `--gpus` option.

## Deploy

Requirements:

- Docker with Compose
- A host directory writable by UID/GID `10001` for temporary checkouts
- A W&B project and API key
- NVIDIA Container Toolkit when any task requests a GPU

Copy `.env.example` to `.env` and set every required value. `JUDGE_WORK_ROOT` must
be an absolute host path. The worker mounts it at the same path because evaluator
containers are siblings created by the host Docker daemon.

Create the work directory and make it available to the judge user:

```console
sudo mkdir -p /var/lib/study-group-online-judge/work
sudo chown 10001:10001 /var/lib/study-group-online-judge/work
```

`DOCKER_GID` must match the group of the Docker socket as it appears inside a
container. On a typical Linux Docker host, this returns the value:

```console
stat -c '%g' /var/run/docker.sock
```

Docker Desktop and OrbStack commonly expose the mounted socket with group `0`.
The worker remains UID `10001`; the configured group is supplemental.

Generate a submission token and start both services:

```console
openssl rand -hex 32
docker compose up --build --detach
docker compose ps
```

The API listens on `127.0.0.1:8000` by default. Put TLS and authentication-aware
rate limiting in a reverse proxy before exposing it publicly. Change
`JUDGE_BIND_ADDRESS` only when the host firewall and proxy arrangement require it.

Useful operator commands:

```console
docker compose logs --follow api worker
docker compose restart worker
docker compose down
```

SQLite and local W&B files live in the `judge-data` volume. Checked-out repositories
and result files live under `JUDGE_WORK_ROOT`.

## Configure a student fork

In the fork's GitHub settings, add:

- Repository variable `JUDGE_URL`, such as `https://judge.example.org`
- Repository variable `WANDB_PROJECT_URL`, such as
  `https://wandb.ai/cerulean/study-group-online-judge`
- Actions secret `JUDGE_API_TOKEN`, matching the deployed judge

Then open **Actions**, choose **Submit to study group judge**, select **Run
workflow** on `main`, and enter the task ID. The workflow submits the current commit,
prints the queued job, provides its W&B run link, and exits as soon as the judge
accepts it. The link becomes active when the worker claims the job. An Actions success
therefore means "queued successfully," not "passed judging." Progress, logs, metrics,
and the final result are available in W&B.

The equivalent request is:

```console
curl --fail-with-body \
  --request POST \
  --header "Authorization: Bearer $JUDGE_API_TOKEN" \
  --header "Content-Type: application/json" \
  --data '{
    "repo_url": "https://github.com/student/study-group.git",
    "commit_sha": "0123456789abcdef0123456789abcdef01234567",
    "task_id": "assignment-01",
    "github_actor": "student"
  }' \
  "$JUDGE_URL/submissions"
```

## Security boundary

Student code runs as a non-root user with no network, a read-only root filesystem,
a read-only submission mount, dropped Linux capabilities, `no-new-privileges`, and
CPU, memory, process, and time limits. The container receives neither the judge token
nor W&B credentials.

The trusted worker is different: access to the Docker socket is effectively
host-level control. Do not run student-controlled worker code or expose that socket
to evaluator containers.

This MVP intentionally has no automatic recovery for a worker that dies after
claiming a job. Such a job remains `running` and requires operator intervention.

## Development

```console
uv sync --frozen
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run python -W error::ResourceWarning -m unittest discover -s tests -v
```
