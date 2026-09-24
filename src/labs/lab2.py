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
    raise NotImplementedError("Implement MMLU evaluation here")
