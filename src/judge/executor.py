import os
import re
import subprocess
import threading
from collections.abc import Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import IO
from uuid import uuid4

from judge.models import Resources


@dataclass(frozen=True)
class ExecutionResult:
    returncode: int


class ExecutionTimeout(RuntimeError):
    pass


def _output_lines(stream: IO[str]) -> Iterator[str]:
    buffer: list[str] = []
    while character := stream.read(1):
        if character in "\r\n":
            if buffer:
                yield "".join(buffer) + "\n"
                buffer.clear()
        else:
            buffer.append(character)
    if buffer:
        yield "".join(buffer) + "\n"


class DockerExecutor:
    def __init__(
        self,
        image: str,
        *,
        user_id: int | None = None,
        group_id: int | None = None,
        docker_binary: str = "docker",
        hf_cache_volume: str | None = None,
        uv_cache_volume: str | None = None,
    ) -> None:
        self.image = image
        self.user_id = os.getuid() if user_id is None else user_id
        self.group_id = os.getgid() if group_id is None else group_id
        self.docker_binary = docker_binary
        self.hf_cache_volume = hf_cache_volume
        self.uv_cache_volume = uv_cache_volume

        if not image:
            raise ValueError("image must not be empty")
        if self.user_id == 0 or self.group_id == 0:
            raise ValueError(
                "the Docker executor must run submissions as a non-root user"
            )
        for volume in (hf_cache_volume, uv_cache_volume):
            if (
                volume is not None
                and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", volume) is None
            ):
                raise ValueError("cache volume names must be valid Docker volume names")

    def run(
        self,
        *,
        task_id: str,
        resources: Resources,
        submission: Path,
        output_directory: Path,
        on_output: Callable[[str], None],
    ) -> ExecutionResult:
        """Run one trusted task runner in an isolated Docker container."""

        output_directory.mkdir(parents=True)
        container_name = f"judge-{uuid4().hex}"
        command = self.build_command(
            task_id=task_id,
            resources=resources,
            submission=submission,
            output_directory=output_directory,
            container_name=container_name,
        )
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        timed_out = threading.Event()

        def stop_after_timeout() -> None:
            if process.poll() is not None:
                return
            timed_out.set()
            try:
                self._remove_container(container_name)
            finally:
                with suppress(ProcessLookupError):
                    process.kill()

        timer = threading.Timer(resources.timeout_seconds, stop_after_timeout)
        timer.daemon = True
        timer.start()

        try:
            assert process.stdout is not None
            for line in _output_lines(process.stdout):
                on_output(line)
            returncode = process.wait()
        except BaseException:
            try:
                self._remove_container(container_name)
            finally:
                with suppress(ProcessLookupError):
                    process.kill()
            raise
        finally:
            timer.cancel()
            timer.join()

        if timed_out.is_set():
            raise ExecutionTimeout(
                f"Task {task_id!r} exceeded {resources.timeout_seconds} seconds"
            )
        return ExecutionResult(returncode=returncode)

    def build_command(
        self,
        *,
        task_id: str,
        resources: Resources,
        submission: Path,
        output_directory: Path,
        container_name: str,
    ) -> list[str]:
        """Build the complete Docker command for one task execution."""

        submission = submission.resolve()
        output_directory = output_directory.resolve()
        for path in (submission, output_directory):
            if "," in str(path):
                raise ValueError("Docker mount paths must not contain commas")

        command = [
            self.docker_binary,
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "bridge",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "256",
            "--cpus",
            str(resources.cpus),
            "--memory",
            f"{resources.memory_gb}g",
            "--user",
            f"{self.user_id}:{self.group_id}",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=256m",
            "--mount",
            f"type=bind,source={submission},target=/submission,readonly",
            "--mount",
            f"type=bind,source={output_directory},target=/output",
        ]
        for volume, target in (
            (self.hf_cache_volume, "/home/judge/.cache/huggingface"),
            (self.uv_cache_volume, "/home/judge/.cache/uv"),
        ):
            if volume:
                command.extend(
                    ["--mount", f"type=volume,source={volume},target={target}"]
                )
        if resources.gpus:
            command.extend(["--gpus", str(resources.gpus)])
        command.extend(
            [
                self.image,
                "python",
                "-u",
                "-m",
                "judge.run_task",
                "--task",
                task_id,
                "--submission",
                "/submission",
                "--output",
                "/output/result.json",
            ]
        )
        return command

    def _remove_container(self, container_name: str) -> None:
        try:
            subprocess.run(
                [self.docker_binary, "rm", "--force", container_name],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except OSError, subprocess.TimeoutExpired:
            pass
