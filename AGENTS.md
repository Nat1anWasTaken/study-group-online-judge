# Agent guide

## Compute

- Never run computation on the login node: training, eval, data processing, analysis, builds, benchmarks, and small trial runs all go through Slurm (`sbatch`, `salloc` + `srun`).
- The login node is for file inspection and editing, git, job submission and monitoring, and Flock API reads.
- Training env: `work/lab4-fa2-env` (CUDA torch 2.11, Python 3.14) via `UV_PROJECT_ENVIRONMENT`, launched with `uv run --project "$REPO" --no-sync`. Never `uv sync` to change training packages. The repo `.venv` is the judge env with CPU torch.
- Each sbatch asserts it runs from its own worktree and on 8 GPUs, exports `*_GIT_COMMIT="$(git rev-parse --short=8 HEAD)"`, copies the training sources and `git diff HEAD` into `$OUTPUT/source/`, and runs the copied script.

## Research-style scripts

- One flat script per task under `src/labs/<lab>/`, plus at most one sibling module such as `muon_lab5.py`. No shared harness, no framework, no config classes.
- No comments, no docstrings, no tests.
- Hyperparameters are module-level constants. Only values that vary between runs of one recipe (seed, steps, data/output paths, resume) come from environment variables with the lab prefix, such as `LAB5_STEPS` or `MIDTERM_SEED`.
- An experiment changes the script directly. Do not add flags or switches to keep the old path alive; the branch is the switch.
- Fail loudly with `assert` instead of fallbacks.
- Keep metric keys, config keys and run-name format identical across runs of a lab. Extra measurements go to local files in the run directory.
- When porting external code, keep numerics byte-identical: same math and floating-point operation order.
- After each substantive code change, run `/simplify` and apply what preserves behavior.

## Experiment workflow

- One branch per experiment, cut from the current `main`: `<lab>/<wave>-<name>`, for example `lab5/w15-teacher3b-3b` or `lab4/u12`.
- One worktree per branch under `/work/u9216411/acd115198/<lab>-worktrees/<name>`:

  ```bash
  git worktree add -b lab5/w16-name /work/u9216411/acd115198/lab5-worktrees/w16-name main
  ```

- The sbatch job name matches the experiment, `<lab>-<wave>-<name>`, and is submitted from that worktree.
- Plan by dependency order, not by calendar day. Launch each stage when its dependencies finish and parallelize. Do not gate experiments on wall-clock speed; compare at matched tokens or steps.
- Report each wave's results to the user before starting the next wave, combining recipes, or launching a final run.

## Logging results

- Runs log through the wandb SDK to Flock (`WANDB_BASE_URL=https://flock-api.cerulean.works`, entity `cerulean-labs`, credentials from `~/.config/wandb/env.sh`). Flock has no artifacts, sweeps, reports or `wandb.Api()`; read runs through its REST API at `/api/v1/projects/<entity>/<project>/runs`.
- Projects: `lab5-training-llama` for Lab 5, `midterm-nanogpt` for the midterm. The midterm leaderboard reads `midterm-nanogpt` directly: a run counts when it is `finished`, its summary has an integer `total_train_steps` and `val_loss < 3.28`, and no `system.gpu.N.memoryAllocatedBytes` sample exceeds 80 GiB. Keep wandb system stats on.
- Run name: `<slurm job name>/<experiment_name>-g<commit8>-j<jobid>`, where `experiment_name` encodes the recipe.
- Run directory: `/work/u9216411/acd115198/<lab>-runs/<job name>-j<jobid>[-s<seed>]/`, holding `source/`, curve files (`train-curve.jsonl`, `eval-curve.jsonl`, or `log.txt` and `results.json`), and `result.json` with the run URL.
- Cross-run analysis scripts and their outputs live in `/work/u9216411/acd115198/<lab>-runs/tools/` and also run through Slurm.
- Judge experiments by full curves, not by the final number: report the early gap, the stable-phase gap and its trend, and the decay gain against the reference, and give a mechanism hypothesis for winners and losers. One untuned run of a bold idea is inconclusive.

## Commits

- Commit only when asked. Every commit message ends with the attribution lines supplied for the session.
- Setup commit on an experiment branch: `<Lab> <wave>: <what changes>`, for example `Lab5 W15: Llama-3.2-3B teacher at 3B`, with a body stating the change and the reason.
- After the run, add a `Result:` commit on the branch. Subject: `Result: <wave> <arm> <outcome with the headline number>`. Body, in order:
  - job id, steps, tokens, compute (H200-hours, wall hours, GPUs), and the recipe delta;
  - the metric against the reference with its delta and confidence interval, stating seeds and what the interval covers;
  - curve findings: early, stable-phase and post-decay gaps;
  - `Verdict:` adopted or not, and `Mechanism:` the hypothesis;
  - run URL, run directory, and comparison file path.
- Adopted branches merge into `main` with `git merge --no-ff`, subject `Merge <branch>: <one-line reason>`, body with the headline number. Push to `origin` only when the user asks.
- Infrastructure changes on `main` use a plain imperative subject, such as `Upgrade wandb to 0.31.0 for Flock compatibility`.
