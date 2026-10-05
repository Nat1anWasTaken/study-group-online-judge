import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
from datasets import Dataset
from transformers import LlamaConfig

from judge.evaluators import PerplexityEvaluator
from judge.models import JudgeResult
from judge.tasks import TASKS
from judge.tasks.lab4 import Lab4
from judge.tasks.lab5 import Lab5
from judge.tasks.lab5_requirements import (
    TrainingEvidence,
    validate_model_config,
    validate_run_url,
)


def training_config():
    return {
        "optimizer": {"name": "AdamW", "learning_rate": 0.001},
        "context_length": 8192,
        "initialization": "random",
        "dataset": "allenai/dolma3_mix-150B-1025",
        "shuffle_seed": 42,
        "holdout_documents": 50_000,
        "holdout_excluded_from_training": True,
        "non_padding_tokens_seen": 1_000_000,
        "repeated_tokens_counted": True,
        "duration_seconds": 600.0,
        "gpu_type": "H200",
        "gpu_count": 1,
        "validation_documents": 10_000,
    }


def llama_config():
    return LlamaConfig(
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        vocab_size=128256,
        max_position_embeddings=131072,
        tie_word_embeddings=True,
        rope_theta=500000.0,
        rope_scaling={
            "rope_type": "llama3",
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
    )


def tokenizer_stub():
    return SimpleNamespace(
        get_vocab=lambda: {"token": 0}, bos_token_id=1, eos_token_id=2
    )


def test_lab5_selects_tail_after_full_shuffle():
    dataset = Dataset.from_dict({"text": [str(i) for i in range(60_000)]})
    with patch("judge.tasks.lab5.load_dataset", return_value=dataset) as load:
        selected = Lab5().load_dataset()
    load.assert_called_once_with("allenai/dolma3_mix-150B-1025", split="train")
    assert len(selected) == 50_000
    expected = dataset.shuffle(seed=42).select(range(10_000, 60_000))
    assert selected["text"] == expected["text"]
    assert (
        selected["text"]
        != dataset.select(range(10_000, 60_000)).shuffle(seed=42)["text"]
    )
    assert not set(selected["text"]) & set(
        dataset.shuffle(seed=42).select(range(10_000))["text"]
    )


def test_lab5_rejects_insufficient_documents():
    with (
        patch(
            "judge.tasks.lab5.load_dataset",
            return_value=Dataset.from_dict({"text": ["short"]}),
        ),
        pytest.raises(ValueError, match="requires at least 50000 documents"),
    ):
        Lab5().load_dataset()


def test_lab5_is_registered_and_requires_gpu():
    task = TASKS["lab5"]
    assert isinstance(task, Lab5)
    assert task.resources.gpus == 1
    assert task.primary_metric == "score"
    assert task.metric_direction.value == "minimize"
    assert isinstance(task.evaluator, PerplexityEvaluator)
    assert task.evaluator.tokenizer_id == "meta-llama/Llama-3.2-1B"
    assert task.evaluator.batch_size == 1
    assert task.evaluator.max_length == 8192


def test_lab5_loads_its_own_submission_and_uses_shared_evaluator(tmp_path):
    source = tmp_path / "src" / "labs" / "lab5.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        'eval_model_id = "student/lab5-model"\n'
        f"training_config = {training_config()!r}\n"
        'training_run_url = "https://wandb.ai/cerulean/lab5-training-llama/runs/training"\n'
    )
    dataset = Dataset.from_dict({"text": ["document"]})
    expected = JudgeResult(passed=True, score=2.0)
    with (
        patch("judge.tasks.model.torch.set_num_threads"),
        patch("judge.tasks.model.torch.cuda.is_available", return_value=True),
        patch.object(Lab5, "load_dataset", return_value=dataset),
        patch(
            "judge.tasks.lab5.AutoConfig.from_pretrained", return_value=llama_config()
        ),
        patch(
            "judge.tasks.lab5.AutoTokenizer.from_pretrained",
            return_value=tokenizer_stub(),
        ),
        patch(
            "judge.evaluators.PerplexityEvaluator.evaluate", return_value=expected
        ) as evaluate,
    ):
        result = Lab5().evaluate(tmp_path)
    assert result is expected
    assert any(check.name == "training_evidence" for check in result.tests)
    message = result.tests[-1].message
    assert message is not None and "organizer review" in message
    evaluate.assert_called_once_with("student/lab5-model", dataset)


@pytest.mark.parametrize("task_type", [Lab4, Lab5])
def test_model_tasks_fail_without_cuda_before_dataset_or_participant_load(task_type):
    with (
        patch("judge.tasks.model.torch.cuda.is_available", return_value=False),
        patch.object(task_type, "load_dataset") as load_data,
        patch("judge.tasks.model.load_student_function") as load_participant,
        pytest.raises(RuntimeError, match="requires a CUDA GPU"),
    ):
        task_type().evaluate(Path("submission"))
    load_data.assert_not_called()
    load_participant.assert_not_called()


@pytest.mark.parametrize("task_type", [Lab4, Lab5])
@pytest.mark.parametrize("model_id", ["", "  ", None, 42])
def test_model_tasks_reject_invalid_model_id_before_dataset_load(task_type, model_id):
    with (
        patch("judge.tasks.model.torch.cuda.is_available", return_value=True),
        patch("judge.tasks.model.torch.set_num_threads"),
        patch(
            "judge.tasks.model.load_student_function",
            return_value=SimpleNamespace(eval_model_id=model_id),
        ),
        patch.object(task_type, "load_dataset") as load,
    ):
        result = task_type().evaluate(Path("submission"))
    assert not result.passed
    assert result.tests[0].name == "model_id"
    load.assert_not_called()


@pytest.mark.parametrize("task_type", [Lab4, Lab5])
def test_model_tasks_report_participant_failure_and_restore_import_path(task_type):
    original_path = sys.path.copy()
    with (
        patch("judge.tasks.model.torch.cuda.is_available", return_value=True),
        patch("judge.tasks.model.torch.set_num_threads"),
        patch(
            "judge.tasks.model.load_student_function",
            side_effect=ImportError("bad module"),
        ),
        patch.object(task_type, "load_dataset") as load,
    ):
        result = task_type().evaluate(Path("submission"))
    assert not result.passed
    assert result.tests[0].name == "participant_module"
    assert "bad module" in result.tests[0].message
    assert sys.path == original_path
    load.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("hidden_size", 3072),
        ("num_hidden_layers", 28),
        ("vocab_size", 50257),
        ("tie_word_embeddings", False),
        ("rope_theta", 10000.0),
        ("model_type", "gpt2"),
        ("max_position_embeddings", 4096),
    ],
)
def test_lab5_rejects_wrong_architecture_or_short_context(field, value):
    config = llama_config()
    setattr(config, field, value)
    with pytest.raises(ValueError, match="Llama 3.2 1B|8192 context"):
        validate_model_config(config, llama_config())


def test_lab5_accepts_8192_context_reference_architecture():
    config = llama_config()
    config.max_position_embeddings = 8192
    validate_model_config(config, llama_config())


@pytest.mark.parametrize(
    "field,value",
    [
        ("context_length", 1024),
        ("initialization", "pretrained"),
        ("shuffle_seed", 123),
        ("holdout_documents", 20_000),
        ("holdout_excluded_from_training", False),
        ("repeated_tokens_counted", False),
        ("gpu_type", "A100"),
        ("validation_documents", 10_001),
        ("non_padding_tokens_seen", 0),
        ("duration_seconds", float("inf")),
        ("optimizer", {}),
        ("gpu_count", 0),
    ],
)
def test_lab5_rejects_invalid_training_evidence(field, value):
    config = training_config()
    config[field] = value
    with pytest.raises(ValueError):
        evidence = TrainingEvidence.model_validate(config)
        evidence.validate_lab()


@pytest.mark.parametrize(
    "url",
    [
        None,
        "",
        "https://wandb.ai/team/other-project/runs/run",
        "https://wandb.ai/team/lab5-training-llama",
        "https://example.com/team/lab5-training-llama/runs/run",
    ],
)
def test_lab5_rejects_missing_or_wrong_project_run_links(url):
    with pytest.raises(ValueError, match="training_run_url"):
        validate_run_url(url)


def test_lab5_missing_training_evidence_fails_before_dataset_download():
    with (
        patch("judge.tasks.model.torch.cuda.is_available", return_value=True),
        patch("judge.tasks.model.torch.set_num_threads"),
        patch(
            "judge.tasks.model.load_student_function",
            return_value=SimpleNamespace(eval_model_id="model"),
        ),
        patch.object(Lab5, "load_dataset") as load,
        patch("judge.tasks.lab5.AutoConfig.from_pretrained") as config,
    ):
        result = Lab5().evaluate(Path("submission"))
    assert not result.passed
    assert result.tests[0].name == "submission_requirements"
    load.assert_not_called()
    config.assert_not_called()


def test_lab5_rejects_incompatible_uploaded_tokenizer():
    module = ModuleType("student_lab5")
    module.__dict__.update(
        training_config=training_config(),
        training_run_url="https://wandb.ai/team/lab5-training-llama/runs/run",
    )
    incorrect = tokenizer_stub()
    incorrect.get_vocab = lambda: {"different": 0}
    with (
        patch(
            "judge.tasks.lab5.AutoConfig.from_pretrained", return_value=llama_config()
        ),
        patch(
            "judge.tasks.lab5.AutoTokenizer.from_pretrained",
            side_effect=[incorrect, tokenizer_stub()],
        ),
        pytest.raises(ValueError, match="Uploaded tokenizer"),
    ):
        Lab5().validate_submission(module, "student/model")


def test_lab5_wrong_checkpoint_fails_before_dataset_download():
    module = ModuleType("student_lab5")
    module.__dict__.update(
        eval_model_id="student/wrong-model",
        training_config=training_config(),
        training_run_url="https://wandb.ai/team/lab5-training-llama/runs/run",
    )
    incorrect = llama_config()
    incorrect.num_hidden_layers = 28
    with (
        patch("judge.tasks.model.torch.cuda.is_available", return_value=True),
        patch("judge.tasks.model.torch.set_num_threads"),
        patch("judge.tasks.model.load_student_function", return_value=module),
        patch(
            "judge.tasks.lab5.AutoConfig.from_pretrained",
            side_effect=[incorrect, llama_config()],
        ),
        patch("judge.tasks.lab5.AutoTokenizer.from_pretrained") as tokenizer,
        patch.object(Lab5, "load_dataset") as load,
    ):
        result = Lab5().evaluate(Path("submission"))
    assert not result.passed
    assert result.tests[0].name == "submission_requirements"
    load.assert_not_called()
    tokenizer.assert_not_called()
