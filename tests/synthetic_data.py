"""Synthetic OKX-format dataset + manifest for offline smoke tests (NOT real market data)."""
from __future__ import annotations

import gzip
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from genetic_bt.genome import StrategyGenome

START_MS = 1_735_689_600_000  # 2025-01-01T00:00:00Z (aligned to 15m/1h/4h/1d)
SYMBOLS = ("BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP")


def _walk(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    regime = np.repeat(rng.choice([-1.0, 0.0, 1.0], size=n // 400 + 1), 400)[:n]
    vol = 0.0015 * (1 + 0.8 * np.abs(np.sin(np.arange(n) / 900.0)))
    ret = regime * 0.0002 + rng.normal(0, 1, n) * vol
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[100.0], close[:-1]])
    spread = np.abs(rng.normal(0, 1, n)) * vol * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.lognormal(3, 0.6, n) * (1 + 2 * (rng.random(n) > 0.95))
    ts = START_MS + np.arange(n, dtype=np.int64) * 900_000
    return pd.DataFrame({"ts": ts, "open": open_, "high": high, "low": low, "close": close, "volume": volume})


def _resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    step = minutes * 60_000
    g = df.groupby((df["ts"] // step) * step)
    out = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                close=("close", "last"), volume=("volume", "sum"), n=("ts", "size"))
    out = out[out["n"] == minutes // 15].drop(columns="n")
    out.index.name = "ts"
    return out.reset_index()


def make_dataset(root: Path, days: int = 150, symbols=SYMBOLS) -> None:
    n = days * 96
    for k, symbol in enumerate(symbols):
        base = _walk(n, seed=100 + k)
        for tf, minutes in (("15m", 15), ("1h", 60), ("4h", 240)):
            frame = base if minutes == 15 else _resample(base, minutes)
            folder = root / tf
            folder.mkdir(parents=True, exist_ok=True)
            with gzip.open(folder / f"{symbol}.csv.gz", "wt") as handle:
                frame.to_csv(handle, index=False)


def make_manifest(path: Path, symbols=SYMBOLS) -> None:
    base = StrategyGenome.baseline()
    variants = [
        ("MOMENTUM_BREAKOUT", dict(momentum_lookback=10, momentum_breakout_buffer_atr=0.0,
                                   momentum_min_adx=14.0, momentum_volume_mult=0.8)),
        ("MEAN_REVERSION", dict(mean_reversion_zscore=1.5, mean_reversion_rsi_low=35.0,
                                mean_reversion_rsi_high=65.0)),
        ("VOLATILITY_SQUEEZE", dict(squeeze_width_percentile=30.0, squeeze_expansion_atr=0.8)),
        ("EMA_CROSS_PRO", dict(ema_min_adx_4h=12.0, ema_min_adx_1h=12.0, ema_volume_mult=1.0,
                               ema_slope_atr=0.04, ema_cross_lookback=12)),
    ]
    candidates = []
    for i, symbol in enumerate(symbols):
        family, params = variants[i % len(variants)]
        genome = replace(base, signal_families=(family,), family_combination="SINGLE",
                         symbols=(symbol,), **params)
        for direction in ("LONG", "SHORT"):
            candidates.append({
                "candidate_id": f"{symbol}__{direction}", "symbol": symbol, "direction": direction,
                "genome": genome.to_dict(), "historical_reference_metrics": {}, "source_summary": "synthetic"})
    manifest = {"schema_version": 1, "frozen": True, "selection_rule": "synthetic smoke test",
                "source": {}, "candidates": candidates, "excluded": []}
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
