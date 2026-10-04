"""Recover, prepare, and monitor GPU jobs through Tailscale SSH."""

import argparse
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from fcntl import LOCK_EX, LOCK_NB, flock
from pathlib import Path

from judge.database import (
    begin_remote_job,
    get_job,
    migrate_database,
    pending_ssh_jobs,
    record_ssh_snapshot,
)
from judge.ssh import RemoteRequest, RemoteSetupError, RemoteSnapshot, TailscaleSSH


def poll_job(
    database_path: Path, transport: TailscaleSSH, request: RemoteRequest
) -> None:
    begin_remote_job(database_path, request.job_id)
    try:
        snapshot = transport.poll(request)
    except RemoteSetupError as error:
        snapshot = RemoteSnapshot(
            setup_offset=request.setup_offset,
            log_offset=request.log_offset,
            error=f"Trusted repository setup failed: {error}"[-8192:],
        )
    except Exception as error:  # noqa: BLE001 - retry transport errors safely
        print(
            f"[judge] SSH retry for {request.job_id}: {type(error).__name__}: {error}",
            flush=True,
        )
        return
    record_ssh_snapshot(database_path, request, snapshot)


def poll_jobs(database_path: Path, transport: TailscaleSSH) -> None:
    """Process one cycle synchronously, for development and tests."""
    for request in pending_ssh_jobs(database_path):
        poll_job(database_path, transport, request)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    arguments = parser.parse_args()
    database_path = Path(os.environ.get("JUDGE_DATABASE_PATH", "data/judge.db"))
    migrate_database(database_path)
    transport = TailscaleSSH(
        os.environ.get("TS_SOCKET", "/var/run/tailscale/tailscaled.sock")
    )
    # One worker owns this DB. A remote per-job lock provides a second guard.
    with database_path.with_suffix(".ssh-worker.lock").open("a") as lock:
        flock(lock, LOCK_EX | LOCK_NB)
        if arguments.once:
            poll_jobs(database_path, transport)
            return
        # Slow dependency setup cannot hold up monitoring of accepted Slurm jobs.
        active: dict[str, Future] = {}
        with (
            ThreadPoolExecutor(max_workers=2) as preparation,
            ThreadPoolExecutor(max_workers=4) as monitoring,
        ):
            while True:
                for job_id, future in list(active.items()):
                    if future.done():
                        try:
                            future.result()
                        except Exception as error:  # noqa: BLE001 - recover after a failed poll
                            print(
                                f"[judge] worker retry for {job_id}: {error}",
                                flush=True,
                            )
                        del active[job_id]
                for request in pending_ssh_jobs(database_path):
                    if request.job_id in active:
                        continue
                    job = get_job(database_path, request.job_id)
                    pool = monitoring if job and job.slurm_job_id else preparation
                    active[request.job_id] = pool.submit(
                        poll_job, database_path, transport, request
                    )
                time.sleep(30)


if __name__ == "__main__":
    main()
