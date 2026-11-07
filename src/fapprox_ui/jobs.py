"""Experiment runner: one experiment script at a time in a child process, with its log."""
from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

MAX_LOG_LINES = 5000
Command = Callable[[str, str, int], list[str]]


@dataclass
class Job:
    experiment: str
    profile: str
    workers: int
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    status: str = "queued"       # queued | running | done | failed | cancelled
    log: list[str] = field(default_factory=list)
    returncode: int | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    cancel_requested: bool = False
    process: subprocess.Popen | None = None

    @property
    def over(self) -> bool:
        return self.status in ("done", "failed", "cancelled")

    def summary(self) -> dict:
        return dict(id=self.id, experiment=self.experiment, profile=self.profile, workers=self.workers, status=self.status,
                    returncode=self.returncode, created=self.created, started=self.started, finished=self.finished,
                    last_line=self.log[-1] if self.log else None)

    def full(self) -> dict:
        return dict(self.summary(), log=list(self.log))


def _kill_tree(process: subprocess.Popen) -> None:
    # experiments start worker processes of their own; they must go too
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
    else:
        os.killpg(process.pid, 15)


class JobQueue:
    def __init__(self, command: Command, after: list[str] | None = None, cwd: str | None = None):
        self.command, self.after, self.cwd = command, after, cwd
        self.jobs: dict[str, Job] = {}
        self._queue: queue.Queue[Job] = queue.Queue()
        self._worker: threading.Thread | None = None

    def submit(self, experiment: str, profile: str, workers: int) -> Job:
        job = Job(experiment, profile, workers)
        self.jobs[job.id] = job
        self._queue.put(job)
        if self._worker is None:
            self._worker = threading.Thread(target=self._loop, name="jobs", daemon=True)
            self._worker.start()
        return job

    def cancel(self, job: Job) -> None:
        job.cancel_requested = True
        if job.process is not None and job.process.poll() is None:
            _kill_tree(job.process)

    def shutdown(self) -> None:
        for job in self.jobs.values():
            if not job.over:
                self.cancel(job)

    def _loop(self) -> None:
        while True:
            job = self._queue.get()
            if job.cancel_requested:
                job.status, job.finished = "cancelled", time.time()
                continue
            self._run(job)

    def _spawn(self, cmd: list[str]) -> subprocess.Popen:
        env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
        return subprocess.Popen(cmd, cwd=self.cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", start_new_session=os.name != "nt")

    def _run(self, job: Job) -> None:
        job.status, job.started = "running", time.time()
        try:
            job.process = self._spawn(self.command(job.experiment, job.profile, job.workers))
            for line in job.process.stdout:
                job.log.append(line.rstrip("\r\n"))
                del job.log[:-MAX_LOG_LINES]
            job.returncode = job.process.wait()
        except OSError as exc:
            job.log.append(f"could not start: {exc}")
            job.returncode = -1
        if job.cancel_requested:
            job.status = "cancelled"
        elif job.returncode == 0:
            self._collect(job)
            job.status = "done"
        else:
            job.status = "failed"
        job.finished = time.time()

    def _collect(self, job: Job) -> None:
        """Refresh docs/RESULTS.md, as run_all.py does after a batch."""
        if self.after:
            done = subprocess.run(self.after, cwd=self.cwd, capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", env=dict(os.environ, PYTHONUTF8="1"), check=False)
            job.log.append(done.stdout.strip() or done.stderr.strip())
