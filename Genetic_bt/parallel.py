"""Tiny helper: run jobs inline (workers<=1) or in a process pool."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Iterable


def run_jobs(func: Callable[[Any], Any], jobs: Iterable[Any], *, workers: int,
             initializer: Callable[..., None], initargs: tuple) -> list[Any]:
    jobs = list(jobs)
    if workers <= 1 or len(jobs) <= 1:
        initializer(*initargs)
        return [func(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers, initializer=initializer, initargs=initargs) as pool:
        return list(pool.map(func, jobs, chunksize=1))
