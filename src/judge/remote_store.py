"""Durable remote records and locks that survive workspace retention."""

import json
import os
from contextlib import contextmanager
from fcntl import LOCK_EX, flock
from pathlib import Path


def save_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(record, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def job_lock(workspace: Path):
    locks = workspace.parent.parent / "job-locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / f"{workspace.name}.lock").open("a") as lock:
        flock(lock, LOCK_EX)
        yield
