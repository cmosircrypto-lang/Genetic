"""Common measured window shared by every candidate/symbol (no per-symbol drift)."""
from __future__ import annotations

from typing import Any, Iterable, Mapping

import pandas as pd

from .historical_evaluator import DAY_MS, TF_MS, _frames_of, _normalize

WARMUP_CANDLES_15M = 500


def prepare_window(
    data: Mapping[str, Any], symbols: Iterable[str], *, min_history_days: float,
    smoke_days: float | None = None, dev_fraction: float = 0.85,
) -> tuple[dict[str, dict[str, pd.DataFrame]], int, int, int, float]:
    """Return (normalized frames, measured_start, split, measured_end, span_days).

    All timestamps are candle OPEN times in ms. ``smoke_days`` keeps only the last N
    days (quick test) and disables the minimum-history check.
    """
    starts: list[int] = []
    ends: list[int] = []
    normalized: dict[str, dict[str, pd.DataFrame]] = {}
    for symbol in sorted(set(symbols)):
        if symbol not in data:
            raise ValueError(f"Missing required symbol data: {symbol}")
        frames = _frames_of(data[symbol])
        current = {tf: _normalize(frames[tf]) for tf in ("15m", "1h", "4h")}
        normalized[symbol] = current
        for tf in ("15m", "1h", "4h"):
            starts.append(int(current[tf]["timestamp"].iloc[0]))
            ends.append(int(current[tf]["timestamp"].iloc[-1]) + TF_MS[tf])
    step = TF_MS["15m"]
    measured_start = max(starts) + WARMUP_CANDLES_15M * step
    measured_start = ((measured_start + step - 1) // step) * step
    measured_end = (min(ends) // step) * step
    if smoke_days:
        measured_start = max(measured_start, measured_end - int(smoke_days * DAY_MS))
        measured_start = ((measured_start + step - 1) // step) * step
    span_days = (measured_end - measured_start) / DAY_MS
    if not smoke_days and span_days < min_history_days:
        raise ValueError(
            f"Data covers only {span_days:.1f} common days; at least {min_history_days:g} are "
            "required. Lower --min-history-days only if you accept a shorter test.")
    if span_days <= 0:
        raise ValueError("No common window across the required symbols")
    split = measured_start + int((measured_end - measured_start) * dev_fraction)
    split = (split // step) * step
    if split <= measured_start or split >= measured_end:
        raise ValueError("Invalid chronological split")
    return normalized, measured_start, split, measured_end, span_days
