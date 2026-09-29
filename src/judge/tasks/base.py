import importlib.util
from abc import ABC, abstractmethod
from pathlib import Path
from types import ModuleType
from typing import ClassVar

from judge.models import JudgeResult, MetricDirection, Resources


def load_student_function(submission: Path, lab_id: str) -> ModuleType:
    source = submission / "src" / "labs" / f"{lab_id}.py"
    if not source.is_file():
        raise FileNotFoundError(f"Expected src/labs/{lab_id}.py in the submission")

    spec = importlib.util.spec_from_file_location("student_lab", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import src/labs/{lab_id}.py")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Task(ABC):
    """Describe how to evaluate one task and the resources it requires."""

    id: ClassVar[str]
    resources: ClassVar[Resources] = Resources()
    primary_metric: ClassVar[str | None] = None
    metric_direction: ClassVar[MetricDirection | None] = None

    @abstractmethod
    def evaluate(self, submission: Path) -> JudgeResult:
        """Evaluate a checked-out student submission."""
