"""Causal historical evaluator for strategy-family genomes (backtest only).

Only 15m/1h/4h closed candles are used. A signal is decided at the CLOSE of the
candidate's event timeframe and traded at the NEXT 15m open. Same-candle SL/TP
collisions resolve SL-first. All fees, slippage and funding are ZERO by design.

Refactor note (behaviour-preserving): the original ``_simulate_fold`` is split into
  1. ``collect_raw_signals``  - evaluate the genome at every decision time,
  2. ``dedupe_signals``       - drop repeated signal signatures (chronological),
  3. ``run_trades``           - closed-candle trade simulation.
``_simulate_fold`` is still available and is the composition of the three.
``simulate_segments`` evaluates signals ONCE for several time segments, which is
mathematically identical to separate runs (dedup is applied per segment) but ~2x
faster for the dev / holdout / combined replay.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import math

import pandas as pd

from .family_signals import evaluate_genome_signals
from .genome import StrategyGenome

DAY_MS = 86_400_000
TF_MS = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": DAY_MS}
SYMBOL_TIMEFRAMES = ("15m", "1h", "4h")


@dataclass(frozen=True)
class FoldResult:
    start_ms: int
    end_ms: int
    trade_count: int
    net_pnl: float
    average_r: float
    profit_factor: float
    max_drawdown_pct: float
    gross_profit: float
    gross_loss: float


# --------------------------------------------------------------------------- data
def _frames_of(value: Any) -> Mapping[str, pd.DataFrame]:
    frames = value.frames if hasattr(value, "frames") else value
    if not isinstance(frames, Mapping):
        raise TypeError("symbol data must expose a timeframe-to-frame mapping")
    return frames


def _normalize(frame: Any) -> pd.DataFrame:
    df = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
    if "timestamp" not in df.columns and "ts" in df.columns:
        df = df.rename(columns={"ts": "timestamp"})
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"OHLCV frame missing columns: {sorted(required-set(df.columns))}")
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="raise")
    df = df.sort_values("timestamp", kind="stable").drop_duplicates("timestamp", keep="last")
    return df.reset_index(drop=True)


def build_complete_btc_daily(hourly: Any) -> pd.DataFrame:
    """Aggregate only complete, contiguous UTC days from closed hourly candles."""
    frame = _normalize(hourly)
    rows = []
    for day, group in frame.groupby((frame["timestamp"].astype("int64") // DAY_MS) * DAY_MS, sort=True):
        group = group.sort_values("timestamp")
        stamps = group["timestamp"].to_numpy(dtype="int64")
        expected = int(day) + pd.RangeIndex(24).to_numpy(dtype="int64") * TF_MS["1h"]
        if len(stamps) != 24 or not (stamps == expected).all():
            continue
        rows.append({
            "timestamp": int(day), "open": float(group.iloc[0]["open"]),
            "high": float(group["high"].max()), "low": float(group["low"].min()),
            "close": float(group.iloc[-1]["close"]), "volume": float(group["volume"].sum()),
        })
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _closed_slice(frame: pd.DataFrame, timeframe: str, decision_ms: int, limit: int = 500) -> pd.DataFrame:
    """Binary-search the last fully closed bar; avoid rescanning the whole history."""
    duration = TF_MS[timeframe]
    close_times = frame["timestamp"].to_numpy(dtype="int64") + duration
    end = int(close_times.searchsorted(int(decision_ms), side="right"))
    start = max(0, end-limit)
    return frame.iloc[start:end].reset_index(drop=True)


def _find_btc_frames(all_data: Mapping[str, Any] | None) -> dict[str, pd.DataFrame] | None:
    if not all_data:
        return None
    names = [name for name in all_data if "BTC" in name.upper() and "USDT" in name.upper() and "SWAP" in name.upper()]
    if len(names) != 1:
        return None
    frames = _frames_of(all_data[names[0]])
    if not all(tf in frames for tf in ("1h", "4h")):
        return None
    hourly, four_hour = _normalize(frames["1h"]), _normalize(frames["4h"])
    daily = build_complete_btc_daily(hourly)
    if daily.empty:
        return None
    return {"1h": hourly, "4h": four_hour, "1d": daily}


def _fold_ranges(length: int, *, oos_fraction: float, warmup_fraction: float, fold_count: int):
    if not 0.0 < oos_fraction < 0.5 or not 0.0 < warmup_fraction < 1.0:
        raise ValueError("invalid walk-forward fractions")
    development_end = int(length * (1.0 - oos_fraction))
    first = int(length * warmup_fraction)
    if development_end <= first or fold_count < 1:
        raise ValueError("not enough candles for requested walk-forward folds")
    edges = [first + (development_end - first) * i // fold_count for i in range(fold_count + 1)]
    return [(edges[i], edges[i + 1]) for i in range(fold_count) if edges[i + 1] > edges[i]], development_end


def window_folds(start_ms: int, end_ms: int, *, warmup_fraction: float = 0.50,
                 fold_count: int = 3) -> list[tuple[int, int]]:
    """Time-based walk-forward folds inside [start_ms, end_ms).

    Same idea as ``_fold_ranges`` but defined on a common calendar window so that
    every symbol uses identical boundaries (no per-symbol leakage into a holdout).
    """
    if not 0.0 < warmup_fraction < 1.0 or fold_count < 1 or end_ms <= start_ms:
        raise ValueError("invalid window fold arguments")
    first = start_ms + int((end_ms - start_ms) * warmup_fraction)
    first = (first // TF_MS["15m"]) * TF_MS["15m"]
    edges = [first + ((end_ms - first) * i // fold_count // TF_MS["15m"]) * TF_MS["15m"]
             for i in range(fold_count + 1)]
    edges[-1] = end_ms
    return [(edges[i], edges[i + 1]) for i in range(fold_count) if edges[i + 1] > edges[i]]


# --------------------------------------------------------------------- simulation
def _trade_exit(direction: str, candle: Mapping[str, Any], position: Mapping[str, float]):
    high, low = float(candle["high"]), float(candle["low"])
    if direction == "LONG":
        sl_hit, tp_hit = low <= position["stop"], high >= position["target"]
    else:
        sl_hit, tp_hit = high >= position["stop"], low <= position["target"]
    if sl_hit:
        return position["stop"], "SL"
    if tp_hit:
        return position["target"], "TP"
    return None


def event_timeframe(genome: StrategyGenome) -> str:
    """Finest timeframe required by the selected families."""
    families = set(genome.signal_families)
    if families & {"EMA_CROSS_PRO", "SWING_OB"}:
        return "15m"
    if families == {"SMC_LIQUIDITY_SWEEP"}:
        return "4h"
    return "1h"


def collect_signals_multi(
    symbol: str, raw_frames: Mapping[str, pd.DataFrame], btc_frames: Mapping[str, pd.DataFrame] | None,
    jobs: Mapping[Any, tuple[StrategyGenome, str]], start_ms: int, end_ms: int, *,
    swing_signal_provider=None,
) -> dict[Any, dict[int, dict[str, Any]]]:
    """Evaluate many (genome, direction) jobs of ONE symbol decision-by-decision.

    Closed slices, data-health gates and indicator work are shared between jobs, so a
    whole candidate pool costs little more than one candidate. Per job the result is
    identical to evaluating it alone: signals keyed by decision time (ms) in
    [start_ms, end_ms), chronological, NOT de-duplicated.
    """
    fifteen, hourly, four_hour = raw_frames["15m"], raw_frames["1h"], raw_frames["4h"]
    out: dict[Any, dict[int, dict[str, Any]]] = {key: {} for key in jobs}
    info = {}
    todo: dict[int, list[Any]] = {}
    by_tf: dict[str, list[Any]] = {}
    for key, (genome, direction) in jobs.items():
        families = set(genome.signal_families)
        info[key] = (
            genome, direction,
            bool(families & {"EMA_CROSS_PRO", "BTC_ALIGNED_PULLBACK", "SMC_LIQUIDITY_SWEEP"}),
            "BTC_ALIGNED_PULLBACK" in families,
        )
        by_tf.setdefault(event_timeframe(genome), []).append(key)
    for tf, keys in by_tf.items():
        times = raw_frames[tf]["timestamp"].to_numpy(dtype="int64") + TF_MS[tf]
        lo = int(times.searchsorted(int(start_ms), side="left"))
        hi = int(times.searchsorted(int(end_ms), side="left"))
        for decision_ms in times[lo:hi]:
            todo.setdefault(int(decision_ms), []).extend(keys)

    for decision_ms in sorted(todo):
        frames = {
            "15m": _closed_slice(fifteen, "15m", decision_ms),
            "1h": _closed_slice(hourly, "1h", decision_ms),
            "4h": _closed_slice(four_hour, "4h", decision_ms),
        }
        if any(len(frames[tf]) < 40 for tf in ("15m", "1h")):
            continue
        # Do not let a data gap turn an old candle into a fresh signal.
        latest_1h_close = int(frames["1h"].iloc[-1]["timestamp"]) + TF_MS["1h"]
        if latest_1h_close > decision_ms or decision_ms-latest_1h_close >= TF_MS["1h"]:
            continue
        if int(frames["15m"].iloc[-1]["timestamp"]) + TF_MS["15m"] != decision_ms:
            continue
        four_hour_ok = not (
            frames["4h"].empty or
            decision_ms - (int(frames["4h"].iloc[-1]["timestamp"]) + TF_MS["4h"]) >= TF_MS["4h"])
        btc_cache: list = []                      # computed lazily, once per decision
        for key in todo[decision_ms]:
            genome, direction, needs_four_hour, uses_btc = info[key]
            if needs_four_hour and not four_hour_ok:
                continue
            btc = None
            if uses_btc and btc_frames:
                if not btc_cache:
                    cand = {
                        "1h": _closed_slice(btc_frames["1h"], "1h", decision_ms),
                        "4h": _closed_slice(btc_frames["4h"], "4h", decision_ms),
                        "1d": _closed_slice(btc_frames["1d"], "1d", decision_ms),
                    }
                    valid = (not cand["1h"].empty and
                             int(cand["1h"].iloc[-1]["timestamp"]) + TF_MS["1h"] <= decision_ms and
                             decision_ms-(int(cand["1h"].iloc[-1]["timestamp"])+TF_MS["1h"]) < TF_MS["1h"] and
                             not cand["4h"].empty and not cand["1d"].empty and
                             decision_ms-(int(cand["4h"].iloc[-1]["timestamp"])+TF_MS["4h"]) < TF_MS["4h"] and
                             decision_ms-(int(cand["1d"].iloc[-1]["timestamp"])+TF_MS["1d"]) < TF_MS["1d"])
                    btc_cache.append(cand if valid else None)
                btc = btc_cache[0]
            swing_signal = None
            if callable(swing_signal_provider):
                swing_signal = swing_signal_provider(symbol, direction, decision_ms, frames)
            signal = evaluate_genome_signals(frames, genome, direction, btc_frames=btc,
                                             swing_signal=swing_signal)
            if signal:
                out[key][decision_ms] = signal
    return out


def collect_raw_signals(
    symbol: str, raw_frames: Mapping[str, pd.DataFrame], btc_frames: Mapping[str, pd.DataFrame] | None,
    genome: StrategyGenome, direction: str, start_ms: int, end_ms: int, *,
    swing_signal_provider=None,
) -> dict[int, dict[str, Any]]:
    """Single-candidate convenience wrapper around ``collect_signals_multi``."""
    return collect_signals_multi(
        symbol, raw_frames, btc_frames, {0: (genome, direction)}, start_ms, end_ms,
        swing_signal_provider=swing_signal_provider)[0]


def _signature(signal: Mapping[str, Any]) -> tuple:
    return (
        signal.get("method", ""), tuple(sorted(signal.get("family_signal_timestamps", {}).items())),
        tuple(signal.get("families_agreed", ())),
    )


def dedupe_signals(raw: Mapping[int, dict[str, Any]], start_ms: int, end_ms: int) -> dict[int, dict[str, Any]]:
    """Keep the first occurrence of every signal signature inside [start_ms, end_ms)."""
    seen: set[tuple] = set()
    out: dict[int, dict[str, Any]] = {}
    for decision_ms in sorted(raw):
        if decision_ms < start_ms or decision_ms >= end_ms:
            continue
        sig = _signature(raw[decision_ms])
        if sig in seen:
            continue
        seen.add(sig)
        out[decision_ms] = raw[decision_ms]
    return out


def run_trades(
    fifteen: pd.DataFrame, genome: StrategyGenome, direction: str,
    event_signals: Mapping[int, dict[str, Any]], start_ms: int, end_ms: int, *,
    initial_equity: float,
) -> FoldResult:
    """One position at a time; entry at the 15m open of the decision timestamp."""
    signed = 1.0 if direction == "LONG" else -1.0
    state = {"equity": float(initial_equity), "peak": float(initial_equity), "max_dd": 0.0,
             "gp": 0.0, "gl": 0.0, "trades": 0}
    r_values: list[float] = []
    position = None

    def close(exit_price: float, entry: float, quantity: float, risk_amount: float) -> None:
        pnl = (exit_price - entry) * signed * quantity
        state["equity"] += pnl
        state["gp"] += max(0.0, pnl)
        state["gl"] += max(0.0, -pnl)
        r_values.append(pnl / risk_amount if risk_amount else 0.0)
        state["peak"] = max(state["peak"], state["equity"])
        peak = state["peak"]
        state["max_dd"] = max(state["max_dd"], (peak - state["equity"]) / peak * 100 if peak > 0 else 100.0)
        state["trades"] += 1

    last_candle = None
    window = fifteen.loc[(fifteen["timestamp"] >= start_ms) & (fifteen["timestamp"] < end_ms)]
    for row in window.to_dict("records"):
        ts = int(row["timestamp"])
        last_candle = row
        if position is not None and ts > position["opened_at"]:
            exited = _trade_exit(direction, row, position)
            if exited:
                close(exited[0], position["entry"], position["quantity"], position["risk_amount"])
                position = None

        signal = event_signals.get(ts)
        if position is None and signal is not None and state["equity"] > 0:
            risk_distance = float(signal["risk_distance"])
            reward_distance = float(signal["reward_distance"])
            entry = float(row["open"])
            if risk_distance > 0 and reward_distance > 0 and entry > 0:
                stop = entry - risk_distance if direction == "LONG" else entry + risk_distance
                target = entry + reward_distance if direction == "LONG" else entry - reward_distance
                if stop > 0 and target > 0:
                    risk_amount = state["equity"] * genome.risk_percent / 100.0
                    quantity = risk_amount / risk_distance
                    position = {"entry": entry, "stop": stop, "target": target,
                                "risk_amount": risk_amount, "quantity": quantity, "opened_at": ts}
                    # Entry is at this 15m open; same-candle collision is SL-first.
                    exited = _trade_exit(direction, row, position)
                    if exited:
                        close(exited[0], entry, quantity, risk_amount)
                        position = None

    if position is not None and last_candle is not None:
        close(float(last_candle["close"]), position["entry"], position["quantity"], position["risk_amount"])

    gp, gl = state["gp"], state["gl"]
    pf = gp / gl if gl > 0 else (10.0 if gp > 0 else 0.0)
    return FoldResult(
        start_ms=int(start_ms), end_ms=int(end_ms), trade_count=state["trades"],
        net_pnl=state["equity"] - initial_equity,
        average_r=sum(r_values) / len(r_values) if r_values else 0.0,
        profit_factor=min(float(pf), 10.0), max_drawdown_pct=state["max_dd"],
        gross_profit=gp, gross_loss=gl,
    )


def _simulate_fold(
    symbol: str, raw_frames: Mapping[str, pd.DataFrame], btc_frames: Mapping[str, pd.DataFrame] | None,
    genome: StrategyGenome, direction: str, start_ms: int, end_ms: int, *,
    initial_equity: float, swing_signal_provider=None,
) -> FoldResult:
    raw = collect_raw_signals(symbol, raw_frames, btc_frames, genome, direction, start_ms, end_ms,
                              swing_signal_provider=swing_signal_provider)
    signals = dedupe_signals(raw, start_ms, end_ms)
    return run_trades(raw_frames["15m"], genome, direction, signals, start_ms, end_ms,
                      initial_equity=initial_equity)


def simulate_segments(
    symbol: str, raw_frames: Mapping[str, pd.DataFrame], btc_frames: Mapping[str, pd.DataFrame] | None,
    genome: StrategyGenome, direction: str, segments: Mapping[str, tuple[int, int]], *,
    initial_equity: float, swing_signal_provider=None,
) -> dict[str, FoldResult]:
    """Evaluate signals once over the union window, then trade each segment separately."""
    lo = min(s for s, _ in segments.values())
    hi = max(e for _, e in segments.values())
    raw = collect_raw_signals(symbol, raw_frames, btc_frames, genome, direction, lo, hi,
                              swing_signal_provider=swing_signal_provider)
    out = {}
    for name, (start_ms, end_ms) in segments.items():
        signals = dedupe_signals(raw, start_ms, end_ms)
        out[name] = run_trades(raw_frames["15m"], genome, direction, signals, start_ms, end_ms,
                               initial_equity=initial_equity)
    return out


def simulate_segments_multi(
    symbol: str, raw_frames: Mapping[str, pd.DataFrame], btc_frames: Mapping[str, pd.DataFrame] | None,
    jobs: Mapping[Any, tuple[StrategyGenome, str]], segments: Mapping[str, tuple[int, int]], *,
    initial_equity: float,
) -> dict[Any, dict[str, FoldResult]]:
    """``simulate_segments`` for many (genome, direction) jobs of one symbol at once."""
    lo = min(s for s, _ in segments.values())
    hi = max(e for _, e in segments.values())
    raw = collect_signals_multi(symbol, raw_frames, btc_frames, jobs, lo, hi)
    out: dict[Any, dict[str, FoldResult]] = {}
    for key, (genome, direction) in jobs.items():
        out[key] = {
            name: run_trades(raw_frames["15m"], genome, direction, dedupe_signals(raw[key], s, e), s, e,
                             initial_equity=initial_equity)
            for name, (s, e) in segments.items()}
    return out


# ------------------------------------------------------------------------ metrics
def aggregate_fold_metrics(
    fold_results: Sequence[FoldResult], genome: StrategyGenome, initial_equity: float = 100.0,
) -> dict[str, float | int]:
    """Collapse validation folds into the HistoricalMetrics fields."""
    trades = sum(f.trade_count for f in fold_results)
    total_profit = sum(f.gross_profit for f in fold_results)
    total_loss = sum(f.gross_loss for f in fold_results)
    pf = total_profit/total_loss if total_loss > 0 else (10.0 if total_profit > 0 else 0.0)
    r_weighted = sum(f.average_r*f.trade_count for f in fold_results)/trades if trades else 0.0
    positive_ratio = sum(f.net_pnl > 0 for f in fold_results)/len(fold_results) if fold_results else 0.0
    fold_scores = []
    for f in fold_results:
        return_component = (math.tanh(f.net_pnl/max(initial_equity, 1.0)*5.0)+1.0)/2.0
        pf_component = min(f.profit_factor/2.0, 1.0)
        dd_component = max(0.0, 1.0-f.max_drawdown_pct/30.0)
        fold_scores.append(0.45*return_component+0.30*pf_component+0.25*dd_component)
    validation_score = sum(fold_scores)/len(fold_scores) if fold_scores else 0.0
    complexity_penalty = min(1.0, 0.04*len(genome.signal_families) +
                             (0.03 if genome.family_combination in ("ALL", "VOTE") else 0.0))
    return {
        "average_r": float(r_weighted),
        "max_drawdown_pct": float(max((f.max_drawdown_pct for f in fold_results), default=0.0)),
        "profit_factor": float(min(pf, 10.0)),
        "positive_window_ratio": float(positive_ratio),
        "trade_count": int(trades),
        "validation_score": float(validation_score),
        "complexity_penalty": float(complexity_penalty),
    }


def evaluate_candidate_walk_forward(
    all_symbol_data: Mapping[str, Any], symbol: str, genome: StrategyGenome, direction: str, *,
    initial_equity: float = 100.0, oos_fraction: float = 0.15,
    warmup_fraction: float = 0.50, fold_count: int = 3,
    swing_signal_provider=None,
) -> dict[str, float | int]:
    """Historical fitness metrics from rolling validation folds only.

    The final oos_fraction of 15m candles is excluded from all fold ranges.
    """
    genome.validate()
    direction = direction.upper()
    if direction not in ("LONG", "SHORT"):
        raise ValueError("direction must be LONG or SHORT")
    symbol = symbol.strip().upper()
    if symbol not in all_symbol_data:
        raise KeyError(f"symbol not found in dataset: {symbol}")
    raw = _frames_of(all_symbol_data[symbol])
    if not all(tf in raw for tf in SYMBOL_TIMEFRAMES):
        raise ValueError("symbol data must contain 15m, 1h and 4h frames")
    normalized = {tf: _normalize(raw[tf]) for tf in SYMBOL_TIMEFRAMES}
    folds, _ = _fold_ranges(
        len(normalized["15m"]), oos_fraction=oos_fraction,
        warmup_fraction=warmup_fraction, fold_count=fold_count,
    )
    btc = _find_btc_frames(all_symbol_data) if "BTC_ALIGNED_PULLBACK" in genome.signal_families else None
    fold_results = []
    for start_idx, end_idx in folds:
        start_ms = int(normalized["15m"].iloc[start_idx]["timestamp"])
        end_ms = int(normalized["15m"].iloc[end_idx]["timestamp"])
        fold_results.append(_simulate_fold(
            symbol, normalized, btc, genome, direction, start_ms, end_ms,
            initial_equity=initial_equity, swing_signal_provider=swing_signal_provider,
        ))
    return aggregate_fold_metrics(fold_results, genome, initial_equity)
