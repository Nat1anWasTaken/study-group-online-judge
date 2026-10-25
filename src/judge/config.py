import os


def allow_code_submissions() -> bool:
    """Code execution is opt-in; unset or unrecognized values disable it."""
    return os.getenv("JUDGE_ALLOW_CODE_SUBMISSIONS", "false").strip().lower() == "true"
