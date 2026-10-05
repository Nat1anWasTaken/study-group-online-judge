"""Submit your Llama 3.2 model for perplexity evaluation on the Dolma 3 holdout.

Upload your weights and tokenizer, then fill all fields below. Report actual
training totals; the organizer reviews the configuration and linked W&B run.
"""

eval_model_id = ""
training_run_url = ""  # https://wandb.ai/<entity>/lab5-training-llama/runs/<run-id>
training_config = {
    "optimizer": {},  # Include all parameter groups, hyperparameters, and schedules.
    "context_length": 8192,
    "initialization": "random",
    "dataset": "allenai/dolma3_mix-150B-1025",
    "shuffle_seed": 42,
    "holdout_documents": 50_000,
    "holdout_excluded_from_training": True,
    "non_padding_tokens_seen": None,  # Count repeated tokens on every pass.
    "repeated_tokens_counted": True,
    "duration_seconds": None,
    "gpu_type": "H200",
    "gpu_count": None,
    "validation_documents": None,  # At most 10,000 of the 50,000 held-out documents.
}
