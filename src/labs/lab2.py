import json
from hashlib import sha256

import torch
from datasets import load_dataset
from tqdm import tqdm

from labs.lab1 import (
    gpt2_logits,
    load_embedding_layers,
    load_final_layer_norm,
    load_tokenizer,
    load_weights,
)


def get_prompt(subject, e1, e2, e3, e4, row):
    letters = "ABCD"
    return f"""The following are multiple choice questions about {subject}.

{e1["question"]}
(A) {e1["choices"][0]} (B) {e1["choices"][1]} (C) {e1["choices"][2]} (D) {e1["choices"][3]}
Answer: {letters[e1["answer"]]}

{e2["question"]}
(A) {e2["choices"][0]} (B) {e2["choices"][1]} (C) {e2["choices"][2]} (D) {e2["choices"][3]}
Answer: {letters[e2["answer"]]}

{e3["question"]}
(A) {e3["choices"][0]} (B) {e3["choices"][1]} (C) {e3["choices"][2]} (D) {e3["choices"][3]}
Answer: {letters[e3["answer"]]}

{e4["question"]}
(A) {e4["choices"][0]} (B) {e4["choices"][1]} (C) {e4["choices"][2]} (D) {e4["choices"][3]}
Answer: {letters[e4["answer"]]}

{row["question"]}
(A) {row["choices"][0]} (B) {row["choices"][1]} (C) {row["choices"][2]} (D) {row["choices"][3]}
Answer: """


def prepare_gpt2():
    tokenizer = load_tokenizer()
    weights = load_weights()
    wte, wpe = load_embedding_layers(weights)
    final_ln = load_final_layer_norm(weights)

    return tokenizer, weights, wte, wpe, final_ln


def mmlu_eval() -> dict[str, str]:
    """Return GPT-2's A/B/C/D prediction for every MMLU test question.

    Load ``cais/mmlu`` at revision
    ``c30699e8356da336a370243923dbaf21066bb9fe``. For each subject, use
    its first four ``dev`` questions as exemplars and evaluate its ``test``
    questions. Format the prompt as specified in the Lab 2 assignment. If a
    prompt exceeds GPT-2's context window, retain its final 1024 tokens.

    Each key is the SHA-256 of a compact UTF-8 JSON object with keys
    ``index``, ``subject``, ``question``, and ``choices`` (sorted keys,
    ``ensure_ascii=False``, compact separators). ``index`` is the zero-based
    row number of the pinned ``all`` test split. This disambiguates repeated
    questions, including 27 identical subject/question/choice rows. Each
    value is one of A/B/C/D, selected from the corresponding next-token
    logits. No question labels should be used to choose a prediction.
    """
    revision = "c30699e8356da336a370243923dbaf21066bb9fe"
    dataset = load_dataset("cais/mmlu", "all", split="test", revision=revision)
    dev = load_dataset("cais/mmlu", "all", split="dev", revision=revision)
    tokenizer, weights, wte, wpe, final_ln = prepare_gpt2()
    predictions = {}
    exemplars = {}

    for example in dev:
        examples = exemplars.setdefault(example["subject"], [])
        if len(examples) < 4:
            examples.append(example)

    answer_token_ids = [
        tokenizer.encode(letter, add_special_tokens=False).ids[0] for letter in "ABCD"
    ]

    for index, row in enumerate(tqdm(dataset, desc="MMLU")):
        e1, e2, e3, e4 = exemplars[row["subject"]]
        subject = row["subject"].replace("_", " ")
        prompt = get_prompt(subject, e1, e2, e3, e4, row)
        ids = tokenizer.encode(prompt, add_special_tokens=False).ids[-1024:]
        token_ids = torch.tensor([ids], dtype=torch.long, device=wte.weight.device)
        attention_mask = torch.ones_like(token_ids, dtype=torch.bool)

        with torch.inference_mode():
            logits = gpt2_logits(token_ids, attention_mask, weights, wte, wpe, final_ln)

        choice_logits = logits[0, answer_token_ids]
        payload = {
            "index": index,
            "subject": row["subject"],
            "question": row["question"],
            "choices": row["choices"],
        }
        question_hash = sha256(
            json.dumps(
                payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        predictions[question_hash] = "ABCD"[int(choice_logits.argmax())]

    return predictions
