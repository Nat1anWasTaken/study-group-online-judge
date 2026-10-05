import json
from types import ModuleType

from datasets import Dataset, load_dataset
from transformers import AutoConfig, AutoTokenizer

from judge.evaluators import PerplexityEvaluator
from judge.models import GradingType, MetricDirection, Resources, TestResult
from judge.tasks.lab5_requirements import (
    CONTEXT_LENGTH,
    DATASET,
    HOLDOUT_DOCUMENTS,
    REFERENCE_MODEL,
    TrainingEvidence,
    validate_model_config,
    validate_run_url,
)
from judge.tasks.model import ModelEvaluationTask

SPLIT = "train"
VALIDATION_SAMPLES = HOLDOUT_DOCUMENTS
SHUFFLE_SEED = 42


class Lab5(ModelEvaluationTask):
    grading_type = GradingType.SCORE
    id = "lab5"
    resources = Resources(cpus=8, memory_gb=32, gpus=1, timeout_seconds=4 * 3600)
    primary_metric = "score"
    metric_direction = MetricDirection.MINIMIZE
    evaluator = PerplexityEvaluator(
        tokenizer_id=REFERENCE_MODEL, batch_size=1, max_length=CONTEXT_LENGTH
    )

    def validate_submission(
        self, module: ModuleType, model_id: str
    ) -> list[TestResult]:
        evidence = TrainingEvidence.model_validate(
            getattr(module, "training_config", None)
        )
        evidence.validate_lab()
        run_url = validate_run_url(getattr(module, "training_run_url", None))
        config = AutoConfig.from_pretrained(model_id)
        reference = AutoConfig.from_pretrained(REFERENCE_MODEL)
        validate_model_config(config, reference)
        # Require the uploaded tokenizer and check its vocabulary/special IDs
        # against the fixed tokenizer used for comparable leaderboard scores.
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        reference_tokenizer = AutoTokenizer.from_pretrained(REFERENCE_MODEL)
        if (
            tokenizer.get_vocab() != reference_tokenizer.get_vocab()
            or tokenizer.bos_token_id != reference_tokenizer.bos_token_id
            or tokenizer.eos_token_id != reference_tokenizer.eos_token_id
        ):
            raise ValueError(
                "Uploaded tokenizer must match the Llama 3.2 vocabulary and BOS/EOS IDs"
            )
        return [
            TestResult(name="model_architecture", passed=True, message=REFERENCE_MODEL),
            TestResult(
                name="training_evidence",
                passed=True,
                message=json.dumps(
                    {
                        "training_run_url": run_url,
                        "training_config": evidence.model_dump(mode="json"),
                        "verification": "Submitted evidence requires organizer review; training provenance is not verified by checkpoint scoring.",
                    },
                    allow_nan=False,
                ),
            ),
        ]

    def load_dataset(self) -> Dataset:
        print("[lab5] loading Dolma 3 mix", flush=True)
        dataset = load_dataset(DATASET, split=SPLIT)
        if len(dataset) < VALIDATION_SAMPLES:
            raise ValueError(
                f"Lab 5 requires at least {VALIDATION_SAMPLES} documents; "
                f"got {len(dataset)}"
            )
        # The held-out tail is defined after a full deterministic shuffle, not
        # a streaming buffer shuffle or a shuffle of the original tail alone.
        dataset = dataset.shuffle(seed=SHUFFLE_SEED)
        return dataset.select(range(len(dataset) - VALIDATION_SAMPLES, len(dataset)))
