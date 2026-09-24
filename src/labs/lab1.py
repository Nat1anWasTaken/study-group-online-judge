import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tokenizers import Tokenizer
from torch import nn

MODEL_ID = "openai-community/gpt2"


def load_weights() -> dict[str, torch.Tensor]:
    weights = load_file(hf_hub_download(MODEL_ID, "model.safetensors"))

    return weights


def load_tokenizer() -> Tokenizer:
    return Tokenizer.from_file(hf_hub_download(MODEL_ID, "tokenizer.json"))


def load_first_layer_norm(
    weights: dict[str, torch.Tensor], layer_index: int
) -> nn.LayerNorm:
    layer = nn.LayerNorm(768)

    with torch.no_grad():
        layer.weight.copy_(weights[f"h.{layer_index}.ln_1.weight"])
        layer.bias.copy_(weights[f"h.{layer_index}.ln_1.bias"])

    return layer


def load_second_layer_norm(
    weights: dict[str, torch.Tensor], layer_index: int
) -> nn.LayerNorm:
    layer = nn.LayerNorm(768)

    with torch.no_grad():
        layer.weight.copy_(weights[f"h.{layer_index}.ln_2.weight"])
        layer.bias.copy_(weights[f"h.{layer_index}.ln_2.bias"])

    return layer


def load_final_layer_norm(weights: dict[str, torch.Tensor]) -> nn.LayerNorm:
    layer = nn.LayerNorm(768)

    with torch.no_grad():
        layer.weight.copy_(weights["ln_f.weight"])
        layer.bias.copy_(weights["ln_f.bias"])

    return layer


def load_embedding_layers(
    weights: dict[str, torch.Tensor],
) -> tuple[nn.Embedding, nn.Embedding]:
    word_token_embedding = nn.Embedding(50257, 768)
    word_position_embedding = nn.Embedding(1024, 768)

    with torch.no_grad():
        word_token_embedding.weight.copy_(weights["wte.weight"])
        word_position_embedding.weight.copy_(weights["wpe.weight"])

    return word_token_embedding, word_position_embedding


def embed_token_ids(
    ids: list[int], wte: nn.Embedding, wpe: nn.Embedding
) -> torch.Tensor:
    token_ids = torch.tensor(ids, dtype=torch.long, device=wte.weight.device)

    if token_ids.numel() > 1024:
        raise ValueError("Token sequence exceeds GPT-2's 1024-token context length")

    positions = torch.arange(token_ids.numel(), device=token_ids.device)
    return wte(token_ids) + wpe(positions)


def mlp_first_projection(
    normalized: torch.Tensor, weights: dict[str, torch.Tensor], layer_index: int
) -> torch.Tensor:
    return (
        torch.matmul(normalized, weights[f"h.{layer_index}.mlp.c_fc.weight"])
        + weights[f"h.{layer_index}.mlp.c_fc.bias"]
    )


def mlp_second_projection(
    activated: torch.Tensor, weights: dict[str, torch.Tensor], layer_index: int
) -> torch.Tensor:
    return (
        torch.matmul(activated, weights[f"h.{layer_index}.mlp.c_proj.weight"])
        + weights[f"h.{layer_index}.mlp.c_proj.bias"]
    )


def transformer_block(
    hidden: torch.Tensor, weights: dict[str, torch.Tensor], layer_index: int
) -> torch.Tensor:
    prefix = f"h.{layer_index}"
    normalized = load_first_layer_norm(weights, layer_index)(hidden)

    qkv = (
        torch.matmul(normalized, weights[f"{prefix}.attn.c_attn.weight"])
        + weights[f"{prefix}.attn.c_attn.bias"]
    )
    q, k, v = qkv.chunk(3, dim=-1)

    query = q.reshape(q.shape[0], 12, 64).transpose(0, 1)
    key = k.reshape(k.shape[0], 12, 64).transpose(0, 1)
    value = v.reshape(v.shape[0], 12, 64).transpose(0, 1)

    attention_scores = torch.matmul(query, key.transpose(-2, -1)) / 8
    token_count = attention_scores.shape[-1]
    causal_mask = torch.ones(
        token_count, token_count, dtype=torch.bool, device=attention_scores.device
    ).tril()
    attention_weights = torch.softmax(
        attention_scores.masked_fill(~causal_mask, float("-inf")), dim=-1
    )
    context = torch.matmul(attention_weights, value)
    combined_context = context.transpose(0, 1).reshape(token_count, 768)

    projected_attention = (
        torch.matmul(combined_context, weights[f"{prefix}.attn.c_proj.weight"])
        + weights[f"{prefix}.attn.c_proj.bias"]
    )
    after_attention = hidden + projected_attention

    normalized_for_mlp = load_second_layer_norm(weights, layer_index)(after_attention)
    mlp_expanded = mlp_first_projection(normalized_for_mlp, weights, layer_index)
    mlp_activated = torch.nn.functional.gelu(mlp_expanded, approximate="tanh")
    mlp_projected = mlp_second_projection(mlp_activated, weights, layer_index)
    return after_attention + mlp_projected


def project_to_vocabulary(hidden: torch.Tensor, wte: nn.Embedding) -> torch.Tensor:
    return torch.matmul(hidden, wte.weight.T)


def gpt2_logits(
    token_ids: list[int],
    weights: dict[str, torch.Tensor],
    wte: nn.Embedding,
    wpe: nn.Embedding,
    final_ln: nn.LayerNorm,
) -> torch.Tensor:
    hidden = embed_token_ids(token_ids, wte, wpe)

    for layer_index in range(12):
        hidden = transformer_block(hidden, weights, layer_index)

    return project_to_vocabulary(final_ln(hidden[-1]), wte)


def gpt2_complete(
    input: list[str],
    max_seq_length: int = 1024,
) -> tuple[list[str], torch.Tensor]:
    """Generate greedy completions with a from-scratch GPT-2 Small implementation.

    Loads the pretrained GPT-2 Small weights into a manually implemented
    transformer consisting of token and positional embeddings, causal
    multi-head self-attention, feed-forward layers, residual connections,
    layer normalization, and a language-modeling output head.

    Generation is performed for the entire batch simultaneously. At each
    decoding step, the highest-logit token is selected for every unfinished
    sequence. A sequence stops generating after producing the EOS token, and
    generation terminates once every sequence has either produced EOS or
    reached ``max_seq_length``.

    Args:
        input: Batch of input strings to complete.
        max_seq_length: Maximum total tokenized sequence length, including
            both prompt and generated tokens.

    Returns:
        A tuple containing:
            - The decoded completion for each input string.
            - The model logits used during greedy generation.
    """
    if not 1 <= max_seq_length <= 1024:
        raise ValueError("max_seq_length must be between 1 and 1024")

    tokenizer = load_tokenizer()
    eos_id = tokenizer.token_to_id("<|endoftext|>")

    if eos_id is None:
        raise ValueError("Tokenizer has no <|endoftext|> token")

    weights = load_weights()
    wte, wpe = load_embedding_layers(weights)
    final_ln = load_final_layer_norm(weights)

    sequences = [
        tokenizer.encode(prompt, add_special_tokens=False).ids for prompt in input
    ]

    for token_ids in sequences:
        if not token_ids:
            token_ids.append(eos_id)
        if len(token_ids) > max_seq_length:
            raise ValueError("Prompt exceeds max_seq_length")

    generated_ids: list[list[int]] = [[] for _ in sequences]
    step_logits: list[list[torch.Tensor]] = [[] for _ in sequences]
    finished = [len(token_ids) == max_seq_length for token_ids in sequences]

    with torch.inference_mode():
        while not all(finished):
            for index, token_ids in enumerate(sequences):
                if finished[index]:
                    continue

                next_logits = gpt2_logits(token_ids, weights, wte, wpe, final_ln)
                step_logits[index].append(next_logits)
                next_id = int(next_logits.argmax())

                if next_id == eos_id:
                    finished[index] = True
                    continue

                token_ids.append(next_id)
                generated_ids[index].append(next_id)
                finished[index] = len(token_ids) == max_seq_length

    completions = [
        tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids
    ]

    all_logits = [
        torch.stack(logits) if logits else wte.weight.new_empty((0, wte.num_embeddings))
        for logits in step_logits
    ]

    if not all_logits:
        return completions, wte.weight.new_empty((0, 0, wte.num_embeddings))

    return completions, nn.utils.rnn.pad_sequence(all_logits, batch_first=True)


def main():
    prompt = "How can"
    completions, logits = gpt2_complete([prompt], max_seq_length=16)
    print(f"Prompt: {prompt}")
    print(f"Completion: {completions[0]}")
    print(f"Logits shape: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
