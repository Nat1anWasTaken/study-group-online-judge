from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from judge.models import JudgeResult, MetricDirection, Resources


class Task(ABC):
    """Describe how to evaluate one task and the resources it requires."""

    id: ClassVar[str]
    resources: ClassVar[Resources] = Resources()
    primary_metric: ClassVar[str | None] = None
    metric_direction: ClassVar[MetricDirection | None] = None

    @abstractmethod
    def evaluate(self, submission: Path) -> JudgeResult:
        """Evaluate a checked-out student submission."""
