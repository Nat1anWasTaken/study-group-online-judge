"""Lab 5 checkpoint checks and training evidence for organizer review."""

import re

from pydantic import BaseModel, ConfigDict, Field, JsonValue
from transformers import PretrainedConfig

REFERENCE_MODEL = "meta-llama/Llama-3.2-1B"
DATASET = "allenai/dolma3_mix-150B-1025"
HOLDOUT_DOCUMENTS = 50_000
CONTEXT_LENGTH = 8192


class TrainingEvidence(BaseModel):
    """Reported training facts; checkpoint scoring cannot prove their provenance."""

    model_config = ConfigDict(extra="allow", strict=True)

    optimizer: dict[str, JsonValue] = Field(min_length=1)
    context_length: int
    initialization: str
    dataset: str
    shuffle_seed: int
    holdout_documents: int
    holdout_excluded_from_training: bool
    non_padding_tokens_seen: int = Field(gt=0)
    repeated_tokens_counted: bool
    duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    gpu_type: str
    gpu_count: int = Field(gt=0)
    validation_documents: int = Field(ge=0, le=HOLDOUT_DOCUMENTS // 5)

    def validate_lab(self) -> None:
        required = {
            "context_length": CONTEXT_LENGTH,
            "initialization": "random",
            "dataset": DATASET,
            "shuffle_seed": 42,
            "holdout_documents": HOLDOUT_DOCUMENTS,
            "holdout_excluded_from_training": True,
            "repeated_tokens_counted": True,
            "gpu_type": "H200",
        }
        for field, expected in required.items():
            if getattr(self, field) != expected:
                raise ValueError(f"training_config.{field} must be {expected!r}")


def validate_run_url(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"https://wandb\.ai/[A-Za-z0-9_-]+/lab5-training-llama/runs/[A-Za-z0-9_-]+/?",
        value,
    ):
        raise ValueError(
            "training_run_url must link to a W&B run in the lab5-training-llama project"
        )
    return value


def validate_model_config(
    config: PretrainedConfig, reference: PretrainedConfig
) -> None:
    # Compare architecture and RoPE settings to the published reference rather
    # than accepting any model that happens to use the Llama tokenizer.
    fields = (
        "model_type",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "hidden_act",
        "rms_norm_eps",
        "rope_theta",
        "rope_scaling",
        "tie_word_embeddings",
        "attention_bias",
        "mlp_bias",
        "attention_dropout",
    )
    mismatches = [
        field
        for field in fields
        if getattr(config, field, None) != getattr(reference, field, None)
    ]
    if mismatches:
        raise ValueError(
            "Lab 5 requires the Llama 3.2 1B architecture; differing fields: "
            + ", ".join(mismatches)
        )
    context_length = getattr(config, "max_position_embeddings", 0)
    if not isinstance(context_length, int) or context_length < CONTEXT_LENGTH:
        raise ValueError("Lab 5 model must support at least 8192 context tokens")
