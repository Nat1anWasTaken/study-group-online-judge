import argparse
from pathlib import Path

from judge.models import JudgeResult
from judge.tasks import TASKS


def run_task(task_id: str, submission: Path, output: Path) -> JudgeResult:
    """Run one trusted evaluator and atomically write its result."""

    if not submission.is_dir():
        raise FileNotFoundError(f"Submission directory does not exist: {submission}")

    try:
        task = TASKS[task_id]
    except KeyError:
        raise ValueError(f"Unknown task: {task_id}") from None

    result = task.evaluate(submission)
    temporary_output = output.with_suffix(f"{output.suffix}.tmp")
    temporary_output.write_text(result.model_dump_json(indent=2))
    temporary_output.replace(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    run_task(arguments.task, arguments.submission, arguments.output)


if __name__ == "__main__":
    main()
