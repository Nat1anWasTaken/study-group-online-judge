import math
import sys
from dataclasses import dataclass
from itertools import chain

import numpy as np
import torch
import torch._dynamo.config
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from judge.evaluators.base import Evaluator
from judge.models import JudgeResult, TestResult

CUDA_DEVICE = torch.device("cuda:0")


class CompilationRequiredModel(torch.nn.Module):
    """Reject an eager forward even if compilation was disabled externally."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, **inputs):
        if not torch.compiler.is_compiling():
            raise RuntimeError("Perplexity model forward requires torch.compile")
        return self.model(**inputs)


@torch.inference_mode()
def evaluate_perplexity(
    batch: dict[str, list], model, tokenizer, *, max_length: int = 1024
) -> dict[str, list]:
    inputs = tokenizer(
        batch["text"],
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(CUDA_DEVICE) for k, v in inputs.items()}
    output = model(**inputs, use_cache=False)
    if output.logits.device != CUDA_DEVICE:
        raise RuntimeError("Perplexity model returned logits outside cuda:0")
    return score_logits(output.logits, inputs["input_ids"], inputs["attention_mask"])


def score_logits(
    logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> dict[str, list]:
    """Compute padding-aware document losses from next-token logits."""
    logits = logits[:, :-1, :].float().contiguous()
    labels = input_ids[:, 1:].contiguous()
    mask = attention_mask[:, 1:].bool()

    losses = F.cross_entropy(
        logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="none"
    )

    losses = losses.view_as(labels).masked_fill(~mask, 0)
    loss_sums = losses.sum(dim=1).tolist()
    token_counts = mask.sum(dim=1).tolist()
    document_perplexities = [
        (
            math.exp(loss_sum / token_count)
            if loss_sum / token_count < math.log(sys.float_info.max)
            else math.inf
        )
        if token_count
        else None
        for loss_sum, token_count in zip(loss_sums, token_counts, strict=True)
    ]
    return {
        "loss_sum": loss_sums,
        "token_count": token_counts,
        "document_perplexity": document_perplexities,
    }


@dataclass(frozen=True)
class PerplexityEvaluator(Evaluator):
    """Token-weighted causal-LM perplexity with mandatory CUDA and Inductor."""

    batch_size: int = 32
    tokenizer_id: str | None = None
    max_length: int = 1024

    def validate_runtime(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Perplexity evaluation requires a CUDA GPU; CPU/MPS fallback is disabled"
            )
        if self.batch_size < 1:
            raise ValueError("Perplexity batch_size must be positive")
        if self.max_length < 2:
            raise ValueError("Perplexity max_length must be at least two tokens")

    @torch.inference_mode()
    def evaluate(self, model_id: str, dataset: Dataset) -> JudgeResult:
        self.validate_runtime()
        print(f"[perplexity] loading model {model_id} on {CUDA_DEVICE}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        max_length = self.max_length
        context_length = getattr(model.config, "max_position_embeddings", None)
        if isinstance(context_length, int) and context_length > 0:
            max_length = min(max_length, context_length)
        torch.nn.Module.to(model, device=CUDA_DEVICE)
        model.eval()
        if any(
            tensor.device != CUDA_DEVICE
            for tensor in chain(model.parameters(), model.buffers())
        ):
            raise RuntimeError(
                "All perplexity model parameters and buffers must be on cuda:0"
            )
        print(
            "[perplexity] torch.compile enabled (inductor, fullgraph, dynamic)",
            flush=True,
        )
        model = torch.compile(
            CompilationRequiredModel(model),
            backend="inductor",
            fullgraph=True,
            dynamic=True,
        )
        tokenizer_id = self.tokenizer_id or model_id
        print(f"[perplexity] loading tokenizer {tokenizer_id}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                raise ValueError(
                    "Perplexity tokenizer requires a pad token or EOS token"
                )
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        total_loss = 0.0
        total_tokens = 0
        document_perplexities = []
        with torch._dynamo.config.patch(
            suppress_errors=False, fail_on_recompile_limit_hit=True
        ):
            for start in tqdm(
                range(0, len(dataset), self.batch_size),
                desc="[perplexity] evaluating documents",
                file=sys.stdout,
                mininterval=5,
            ):
                batch = dataset[start : start + self.batch_size]
                results = evaluate_perplexity(
                    batch, model, tokenizer, max_length=max_length
                )
                total_loss += sum(results["loss_sum"])
                total_tokens += sum(results["token_count"])
                document_perplexities.extend(
                    value
                    for value in results["document_perplexity"]
                    if value is not None
                )
        if total_tokens == 0:
            return JudgeResult(
                passed=False,
                tests=[
                    TestResult(
                        name="validation_data",
                        passed=False,
                        message="No validation tokens remained after tokenization",
                    )
                ],
            )
        mean_loss = total_loss / total_tokens
        corpus_perplexity = (
            math.exp(mean_loss)
            if mean_loss < math.log(sys.float_info.max)
            else math.inf
        )
        if not math.isfinite(corpus_perplexity) or not all(
            math.isfinite(value) for value in document_perplexities
        ):
            return JudgeResult(
                passed=False,
                tests=[
                    TestResult(
                        name="perplexity",
                        passed=False,
                        message="Model produced non-finite perplexity",
                    )
                ],
            )
        print(
            f"[perplexity] evaluated {len(document_perplexities)}/{len(dataset)} documents "
            f"and {total_tokens} tokens; corpus perplexity={corpus_perplexity:.4f}",
            flush=True,
        )

        return JudgeResult(
            passed=True,
            score=corpus_perplexity,
            metrics={
                "corpus_perplexity": corpus_perplexity,
                "p90_document_perplexity": float(
                    np.percentile(document_perplexities, 90)
                ),
                "p99_document_perplexity": float(
                    np.percentile(document_perplexities, 99)
                ),
                "evaluated_documents": float(len(document_perplexities)),
                "evaluated_tokens": float(total_tokens),
                "skipped_documents": float(len(dataset) - len(document_perplexities)),
            },
        )
