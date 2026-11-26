"""Run jobs inline (workers<=1) or in a process pool; results stream as they finish."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Iterable, Iterator


def iter_jobs(func: Callable[[Any], Any], jobs: Iterable[Any], *, workers: int,
              initializer: Callable[..., None], initargs: tuple) -> Iterator[Any]:
    """Yield func(job) results as soon as each finishes (order not guaranteed)."""
    jobs = list(jobs)
    if workers <= 1 or len(jobs) <= 1:
        initializer(*initargs)
        for job in jobs:
            yield func(job)
        return
    pool = ProcessPoolExecutor(max_workers=workers, initializer=initializer, initargs=initargs)
    try:
        futures = [pool.submit(func, job) for job in jobs]
        for future in as_completed(futures):
            yield future.result()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def run_jobs(func: Callable[[Any], Any], jobs: Iterable[Any], *, workers: int,
             initializer: Callable[..., None], initargs: tuple) -> list[Any]:
    """Like iter_jobs but returns results in the order of ``jobs``."""
    jobs = list(jobs)
    if workers <= 1 or len(jobs) <= 1:
        initializer(*initargs)
        return [func(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers, initializer=initializer, initargs=initargs) as pool:
        return list(pool.map(func, jobs, chunksize=1))
