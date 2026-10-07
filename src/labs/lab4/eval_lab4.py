import hashlib
import json
import math

import torch
import torch.nn.functional as F
from datasets import load_from_disk

TOKENIZER_ID = "openai-community/gpt2"
SEQUENCE_LENGTH = 1024
SEED = 42
OJ_DOCUMENTS = 100_000
EVAL_DOCUMENTS = 2_048
EVAL_BATCH_SIZE = 8


def validation_hash(dataset):
    return hashlib.sha256(json.dumps(list(dataset["input_ids"])).encode()).hexdigest()


def load_validation(directory):
    dataset = load_from_disk(str(directory))
    metadata = json.loads((directory / "validation.json").read_text())
    expected = {
        "seed": SEED,
        "start": OJ_DOCUMENTS,
        "documents": EVAL_DOCUMENTS,
        "max_length": SEQUENCE_LENGTH,
        "tokenizer": TOKENIZER_ID,
        "token_ids_sha256": validation_hash(dataset),
    }
    if len(dataset) != EVAL_DOCUMENTS or any(
        metadata.get(k) != v for k, v in expected.items()
    ):
        raise ValueError("Dev cache changed; run prepare_lab4.py again.")
    return dataset, metadata


@torch.inference_mode()
def validation_totals(model, dataset, tokenizer, device, batch_size):
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    model.eval()
    for start in range(0, len(dataset), batch_size):
        rows = dataset[start : start + batch_size]["input_ids"]
        valid = [row for row in rows if len(row) > 1]
        totals[3] += len(rows) - len(valid)
        if not valid:
            continue
        inputs = tokenizer.pad(
            {"input_ids": valid}, padding=True, return_tensors="pt"
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        logits = model(**inputs).logits[:, :-1, :].float().contiguous()
        labels = inputs["input_ids"][:, 1:].contiguous()
        mask = inputs["attention_mask"][:, 1:].bool()
        losses = (
            F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1),
                reduction="none",
            )
            .view_as(labels)
            .masked_fill(~mask, 0)
        )
        totals[0] += losses.sum(dim=1).double().sum()
        totals[1] += mask.sum()
        totals[2] += len(valid)
    return totals


def validation_metrics(totals):
    loss, tokens, documents, skipped = totals.tolist()
    if tokens <= 0 or not math.isfinite(loss) or loss / tokens >= 709:
        raise ValueError("Invalid validation loss or token count")
    return {
        "loss": loss / tokens,
        "perplexity": math.exp(loss / tokens),
        "tokens": int(tokens),
        "documents": int(documents),
        "skipped_documents": int(skipped),
    }
