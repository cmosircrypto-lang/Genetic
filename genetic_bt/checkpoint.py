"""Resumable work-unit store + progress reporting for long backtests."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable


class CheckpointMismatch(RuntimeError):
    """The cache on disk was produced by different inputs / code."""


def code_hash(paths: Iterable[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(paths):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _fmt(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


class UnitStore:
    """One JSON file per finished work unit, written atomically (tmp + replace).

    ``directory=None`` keeps everything in memory (used by tests). A cache is only
    reused when its ``meta.json`` fingerprint equals the current one exactly.
    """

    def __init__(self, directory: str | Path | None, fingerprint: dict[str, Any], *, fresh: bool = False):
        self.dir = Path(directory) if directory else None
        self.fingerprint = json.loads(json.dumps(fingerprint, sort_keys=True))
        self._memory: dict[str, Any] = {}
        if self.dir is None:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        meta = self.dir / "meta.json"
        if fresh:
            for f in self.dir.glob("unit_*.json"):
                f.unlink()
            if meta.exists():
                meta.unlink()
        if meta.exists():
            old = json.loads(meta.read_text(encoding="utf-8"))
            if old != self.fingerprint:
                diff = sorted(k for k in set(old) | set(self.fingerprint) if old.get(k) != self.fingerprint.get(k))
                raise CheckpointMismatch(
                    f"Existing cache in '{self.dir}' was made with different settings/data/code "
                    f"(differs in: {diff}). Re-run with --fresh to discard it, or use another --output-dir.")
        else:
            self._atomic_write(meta, self.fingerprint)

    @staticmethod
    def _atomic_write(path: Path, payload: Any) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)

    def _path(self, uid: str) -> Path:
        return self.dir / f"unit_{uid}.json"

    def done_ids(self) -> set[str]:
        if self.dir is None:
            return set(self._memory)
        ok = set()
        for f in self.dir.glob("unit_*.json"):
            try:
                json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):      # half-written / corrupt file => redo this unit
                f.unlink(missing_ok=True)
                continue
            ok.add(f.stem[len("unit_"):])
        return ok

    def save(self, uid: str, payload: Any) -> None:
        if self.dir is None:
            self._memory[uid] = payload
        else:
            self._atomic_write(self._path(uid), payload)

    def load(self, uid: str) -> Any:
        if self.dir is None:
            return self._memory[uid]
        return json.loads(self._path(uid).read_text(encoding="utf-8"))


class Progress:
    """Prints done/remaining percent and an ETA; also mirrors it to progress.json."""

    def __init__(self, label: str, total: int, already_done: int = 0, mirror: Path | None = None):
        self.label, self.total, self.done = label, max(1, total), already_done
        self.already, self.t0, self.mirror = already_done, time.time(), mirror
        self._last_print = 0.0
        if already_done:
            print(f"[RESUME] {label}: {already_done}/{self.total} units already finished", flush=True)
        self.show(force=True)

    def tick(self, n: int = 1) -> None:
        self.done += n
        self.show(force=self.done >= self.total)

    def show(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_print < 1.0:
            return
        self._last_print = now
        pct = self.done / self.total * 100
        new = self.done - self.already
        elapsed = now - self.t0
        eta = elapsed / new * (self.total - self.done) if new > 0 else None
        print(f"[PROGRESS] {self.label}: done {pct:5.1f}% | remaining {100 - pct:5.1f}% | "
              f"{self.done}/{self.total} units | elapsed {_fmt(elapsed)} | "
              f"ETA {_fmt(eta) if eta is not None else '--:--:--'}", flush=True)
        if self.mirror is not None:
            try:
                self.mirror.write_text(json.dumps({
                    "label": self.label, "done_pct": round(pct, 2), "remaining_pct": round(100 - pct, 2),
                    "units_done": self.done, "units_total": self.total,
                    "eta_seconds": None if eta is None else int(eta), "updated": time.time()}), encoding="utf-8")
            except OSError:
                pass
