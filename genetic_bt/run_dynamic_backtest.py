"""Cycle-by-cycle dynamic signal-selection backtest with a shared portfolio.

Leak-free design
  * Historical metrics of every candidate come ONLY from walk-forward folds inside the
    development window (first 85% of the common window).
  * Selection is then simulated bar by bar on the final 15% (the holdout), using only
    closed candles. No metric ever sees the holdout.
  * One portfolio (start $100): max positions / per direction / per symbol from config.
  * ALL costs are zero (fees, slippage, funding).

Selectors (compare them!):  score  = dynamic_selector ranking (+ LONG/SHORT conflict margin)
                            random = random eligible candidate (null baseline)

Resumable: the heavy part (signals for every candidate) is split into small work units
(symbol x time chunk). Each finished unit is written to ``<output-dir>/_cache`` immediately.
If the run is interrupted, run the SAME command again and it continues from the last finished
unit. ``--selector both`` (or running score then random) reuses the same cache, so the second
selector only costs the quick portfolio simulation. Progress is printed as done/remaining %.

    python -m genetic_bt.run_dynamic_backtest --data-root data/okx --manifest frozen_manifest.json --selector both --workers 6
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random

from . import config
from .candidate_pool import build_symbol_candidate_pool
from .data_loader import load_okx_data
from .dynamic_selector import HistoricalMetrics, ScoringPolicy, SignalCandidate, select_cycle_signals
from .genome import StrategyGenome
from .historical_evaluator import (
    _find_btc_frames, _trade_exit, aggregate_fold_metrics, collect_signals_multi,
    dedupe_signals, run_trades, window_folds,
)
from .checkpoint import CheckpointMismatch, Progress, UnitStore, code_hash
from .historical_evaluator import DAY_MS
from .manifest import load_frozen_manifest
from .parallel import iter_jobs
from .window import prepare_window

BTC_SYMBOL = "BTC-USDT-SWAP"
_G: dict = {}
SIGNAL_KEYS = ("risk_distance", "reward_distance", "setup_quality", "method", "signal_timestamp",
               "family_signal_timestamps", "families_agreed")
CODE_FILES = ("family_signals.py", "historical_evaluator.py", "genome.py", "config.py",
              "candidate_pool.py", "window.py")


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


# ----------------------------------------------------------------- work units
def _init_worker(normalized, btc_frames, pool_by_symbol):
    _G.update(normalized=normalized, btc=btc_frames, pool=pool_by_symbol)


def _compute_unit(job):
    """Raw signals of every candidate of one symbol for decisions in [a, b)."""
    uid, symbol, a, b = job
    jobs = {sp["candidate_id"]: (StrategyGenome.from_dict(sp["genome"]), sp["direction"])
            for sp in _G["pool"][symbol]}
    raw = collect_signals_multi(symbol, _G["normalized"][symbol], _G["btc"], jobs, a, b)
    payload = {cid: [[ms, {k: sig[k] for k in SIGNAL_KEYS}] for ms, sig in sigs.items()]
               for cid, sigs in raw.items() if sigs}
    return uid, payload


def _chunk_edges(first: int, end: int, chunk_days: float) -> list[tuple[int, int]]:
    step = max(1, int(chunk_days * DAY_MS // 900_000)) * 900_000
    edges, cur = [], first
    while cur < end:
        edges.append((cur, min(cur + step, end)))
        cur += step
    return edges


def _finalize_symbol(symbol, specs, raw, frames, folds, split, end):
    """Dev-fold metrics + holdout signal streams from the merged raw signals."""
    out = []
    for sp in specs:
        cid = sp["candidate_id"]
        genome = StrategyGenome.from_dict(sp["genome"])
        direction = sp["direction"]
        sigs = raw.get(cid, {})
        fold_results = [
            run_trades(frames["15m"], genome, direction, dedupe_signals(sigs, s, e), s, e,
                       initial_equity=100.0)
            for s, e in folds]
        metrics = aggregate_fold_metrics(fold_results, genome, 100.0)
        stream = {
            ms: {"q": float(sig["setup_quality"]), "risk": float(sig["risk_distance"]),
                 "reward": float(sig["reward_distance"]), "method": sig["method"],
                 "ts": str(sig["signal_timestamp"])}
            for ms, sig in dedupe_signals(sigs, split, end).items()}
        out.append((cid, metrics, stream))
    return out


# --------------------------------------------------------------- portfolio engine
def simulate_portfolio(*, candles, specs, metrics, streams, split, end, policy, selector,
                       seed, conflict_margin, initial_equity=100.0, on_progress=None):
    rng = random.Random(seed)
    by_ts: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for cid, stream in streams.items():
        symbol = specs[cid]["symbol"]
        for ms, sig in stream.items():
            by_ts[ms][symbol].append((cid, sig))

    timeline = sorted({ts for sym in candles.values() for ts in sym if split <= ts < end})
    equity, peak, max_dd = float(initial_equity), float(initial_equity), 0.0
    open_pos: list[dict] = []
    trades: list[dict] = []
    curve: list[tuple[int, float]] = [(split, equity)]
    rejections: Counter = Counter()
    stats = Counter()

    def close_position(pos, price, ts, reason):
        nonlocal equity, peak, max_dd
        signed = 1.0 if pos["direction"] == "LONG" else -1.0
        pnl = (price - pos["entry"]) * signed * pos["quantity"]
        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100 if peak > 0 else 100.0)
        trades.append({
            "symbol": pos["symbol"], "direction": pos["direction"], "method": pos["method"],
            "candidate_id": pos["cid"], "score": round(pos["score"], 6),
            "opened_utc": _iso(pos["opened_at"]), "closed_utc": _iso(ts), "entry": pos["entry"],
            "exit": price, "reason": reason, "pnl": pnl,
            "r": pnl / pos["risk_amount"] if pos["risk_amount"] else 0.0, "equity_after": equity})
        curve.append((ts, equity))

    for n_ts, ts in enumerate(timeline):
        if on_progress is not None and n_ts % 500 == 0:
            on_progress(n_ts / max(1, len(timeline)))
        # 1) exits on this candle for positions opened earlier
        for pos in list(open_pos):
            row = candles[pos["symbol"]].get(ts)
            if row is None or ts <= pos["opened_at"]:
                continue
            hit = _trade_exit(pos["direction"], row, pos)
            if hit:
                close_position(pos, hit[0], ts, hit[1])
                open_pos.remove(pos)

        # 2) selection among signals decided at this timestamp
        groups = by_ts.get(ts)
        if not groups:
            continue
        signal_candidates, lookup = [], {}
        for symbol, items in groups.items():
            for cid, sig in items:
                spec = specs[cid]
                signal_candidates.append(SignalCandidate(
                    symbol=symbol, candidate_id=cid, method=sig["method"], direction=spec["direction"],
                    signal_timestamp=sig["ts"], historical=HistoricalMetrics(**metrics[cid]),
                    current_setup_quality=sig["q"], has_current_signal=True, candle_closed=True))
                lookup[cid] = sig
        stats["signals"] += len(signal_candidates)
        decisions = select_cycle_signals(signal_candidates, policy)
        winners = []
        for symbol in sorted(decisions):
            dec = decisions[symbol]
            for item in dec.ranked_candidates:
                for reason in item.rejection_reasons:
                    rejections[reason] += 1
            eligible = [r for r in dec.ranked_candidates if r.eligible]
            stats["eligible"] += len(eligible)
            if not eligible:
                continue
            if selector == "random":
                winners.append(rng.choice(eligible))
                continue
            win = eligible[0]
            rival = next((r for r in eligible if r.direction != win.direction), None)
            if rival is not None and win.final_score - rival.final_score < conflict_margin:
                stats["conflict_skipped"] += 1
                continue
            winners.append(win)
        if selector == "random":
            rng.shuffle(winners)
        else:
            winners.sort(key=lambda w: (-w.final_score, w.symbol))

        # 3) open positions subject to portfolio caps (entry = this candle's open)
        for w in winners:
            if (len(open_pos) >= config.MAX_OPEN_POSITIONS
                    or sum(p["direction"] == w.direction for p in open_pos) >= config.MAX_OPEN_PER_DIRECTION
                    or sum(p["symbol"] == w.symbol for p in open_pos) >= config.MAX_OPEN_PER_SYMBOL):
                stats["cap_skipped"] += 1
                continue
            row = candles[w.symbol].get(ts)
            sig = lookup[w.candidate_id]
            if row is None or equity <= 0 or sig["risk"] <= 0 or sig["reward"] <= 0:
                continue
            entry = float(row["open"])
            if entry <= 0:
                continue
            long_ = w.direction == "LONG"
            stop = entry - sig["risk"] if long_ else entry + sig["risk"]
            target = entry + sig["reward"] if long_ else entry - sig["reward"]
            if stop <= 0 or target <= 0:
                continue
            risk_pct = specs[w.candidate_id]["risk_percent"]
            risk_amount = equity * risk_pct / 100.0
            pos = {"symbol": w.symbol, "direction": w.direction, "method": w.method,
                   "cid": w.candidate_id, "score": w.final_score, "entry": entry, "stop": stop,
                   "target": target, "risk_amount": risk_amount, "quantity": risk_amount / sig["risk"],
                   "opened_at": ts}
            stats["opened"] += 1
            hit = _trade_exit(w.direction, row, pos)      # same-candle collision, SL first
            if hit:
                close_position(pos, hit[0], ts, hit[1])
            else:
                open_pos.append(pos)

    for pos in open_pos:                                   # force-close at the final close
        last_ts = max(t for t in candles[pos["symbol"]] if t < end)
        close_position(pos, float(candles[pos["symbol"]][last_ts]["close"]), last_ts, "END")
    return trades, curve, max_dd, rejections, stats


def summarize(trades, equity0, max_dd):
    def block(items):
        n = len(items)
        gp = sum(t["pnl"] for t in items if t["pnl"] > 0)
        gl = -sum(t["pnl"] for t in items if t["pnl"] < 0)
        return {"trades": n, "net_pnl": sum(t["pnl"] for t in items),
                "win_rate": sum(t["pnl"] > 0 for t in items) / n if n else 0.0,
                "average_r": sum(t["r"] for t in items) / n if n else 0.0,
                "profit_factor": min(gp / gl, 10.0) if gl > 0 else (10.0 if gp > 0 else 0.0)}
    out = block(trades)
    out["return_pct"] = out["net_pnl"] / equity0 * 100
    out["max_drawdown_pct_closed_trades"] = max_dd
    for key in ("symbol", "direction", "method"):
        groups: dict[str, list] = defaultdict(list)
        for t in trades:
            groups[t[key]].append(t)
        out[f"by_{key}"] = {k: block(v) for k, v in sorted(groups.items())}
    return out


# ------------------------------------------------------------------------ driver
def prepare_pool_data(data, manifest_path, *, symbols=None, min_history_days=700.0, smoke_days=None,
                      dev_fraction=0.85, workers=1, chunk_days=14.0, cache_dir=None, fresh=False):
    """Heavy, resumable phase: candidate metrics (dev folds) + holdout signal streams."""
    manifest, manifest_bytes = load_frozen_manifest(manifest_path)
    seeds: dict[str, list[StrategyGenome]] = defaultdict(list)
    for c in manifest["candidates"]:
        seeds[c["symbol"]].append(StrategyGenome.from_dict(c["genome"]))
    symbols = sorted(symbols or seeds)
    required = set(symbols) | {BTC_SYMBOL}
    normalized, start, split, end, span_days = prepare_window(
        data, required, min_history_days=min_history_days, smoke_days=smoke_days, dev_fraction=dev_fraction)
    btc_frames = _find_btc_frames(data)
    if btc_frames is None:
        raise ValueError("need exactly one BTC-USDT-SWAP series with 1h/4h data")
    folds = window_folds(start, split)

    specs, pool_by_symbol = {}, {}
    for symbol in symbols:
        pool_by_symbol[symbol] = build_symbol_candidate_pool(symbol, seeds.get(symbol, []))
        for spec in pool_by_symbol[symbol]:
            specs[spec["candidate_id"]] = {**spec, "symbol": symbol,
                                           "risk_percent": spec["genome"]["risk_percent"]}
    chunks = _chunk_edges(folds[0][0], end, chunk_days)
    units = [(f"{sym}_{i:03d}", sym, a, b) for sym in symbols for i, (a, b) in enumerate(chunks)]

    fingerprint = {
        "format": 1, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "symbols": symbols, "candidate_ids_sha256": hashlib.sha256(
            "\n".join(sorted(specs)).encode()).hexdigest(),
        "window": [start, split, end], "folds": folds, "chunk_days": chunk_days,
        "code": code_hash(Path(__file__).parent / f for f in CODE_FILES),
        "data": {s: {tf: [len(normalized[s][tf]), int(normalized[s][tf]["timestamp"].iloc[0]),
                          int(normalized[s][tf]["timestamp"].iloc[-1])] for tf in ("15m", "1h", "4h")}
                 for s in sorted(required)},
    }
    store = UnitStore(cache_dir, fingerprint, fresh=fresh)
    done = store.done_ids() & {u[0] for u in units}
    pending = [u for u in units if u[0] not in done]
    print(f"[POOL] symbols={len(symbols)} candidates={len(specs)} chunks/symbol={len(chunks)} "
          f"units={len(units)} (already done: {len(done)})", flush=True)
    progress = Progress("signals", len(units), len(done),
                        mirror=(store.dir / "progress.json") if store.dir else None)
    needed = {s: normalized[s] for s in symbols}
    for uid, payload in iter_jobs(_compute_unit, pending, workers=workers, initializer=_init_worker,
                                  initargs=(needed, btc_frames, pool_by_symbol)):
        store.save(uid, payload)               # persisted immediately => safe to interrupt
        progress.tick()

    merged: dict[str, dict[str, dict[int, dict]]] = {s: defaultdict(dict) for s in symbols}
    for uid, sym, _, _ in units:
        for cid, items in store.load(uid).items():
            merged[sym][cid].update({int(ms): sig for ms, sig in items})
    metrics, streams = {}, {}
    for sym in symbols:
        for cid, m, st in _finalize_symbol(sym, pool_by_symbol[sym], merged[sym], normalized[sym],
                                           folds, split, end):
            metrics[cid], streams[cid] = m, st
    print(f"[SIGNALS] holdout signals total={sum(len(x) for x in streams.values())}", flush=True)
    candles = {s: {int(r["timestamp"]): r for r in normalized[s]["15m"].to_dict("records")
                   if split <= int(r["timestamp"]) < end} for s in symbols}
    return {"specs": specs, "metrics": metrics, "streams": streams, "candles": candles,
            "start": start, "split": split, "end": end, "span_days": span_days, "folds": folds,
            "symbols": symbols, "smoke_days": smoke_days}


def backtest_selector(pool, *, selector="score", seed=1, conflict_margin=0.03, policy=ScoringPolicy()):
    """Quick phase: bar-by-bar selection + shared-portfolio simulation on the holdout."""
    last = [-1.0]

    def on_progress(frac):
        if frac - last[0] >= 0.05:
            last[0] = frac
            print(f"[PROGRESS] selection simulation ({selector}): done {frac * 100:5.1f}% | "
                  f"remaining {100 - frac * 100:5.1f}%", flush=True)

    specs, metrics, streams = pool["specs"], pool["metrics"], pool["streams"]
    start, split, end, folds = pool["start"], pool["split"], pool["end"], pool["folds"]
    trades, curve, max_dd, rejections, stats = simulate_portfolio(
        candles=pool["candles"], specs=specs, metrics=metrics, streams=streams, split=split, end=end,
        policy=policy, selector=selector, seed=seed, conflict_margin=conflict_margin,
        on_progress=on_progress)
    print(f"[PROGRESS] selection simulation ({selector}): done 100.0% | remaining   0.0%", flush=True)
    report = {
        "schema_version": 1, "selector": selector, "seed": seed,
        "costs": {"fee_rate": 0.0, "slippage_rate": 0.0, "funding": 0.0},
        "window": {"measured_start_utc": _iso(start), "metrics_end_exclusive_utc": _iso(split),
                   "holdout_start_utc": _iso(split), "holdout_end_exclusive_utc": _iso(end),
                   "common_history_days": round(pool["span_days"], 3), "smoke_days": pool["smoke_days"]},
        "metric_folds_utc": [[_iso(a), _iso(b)] for a, b in folds],
        "policy": policy.__dict__, "conflict_margin": conflict_margin,
        "caps": {"max_open": config.MAX_OPEN_POSITIONS, "per_direction": config.MAX_OPEN_PER_DIRECTION,
                 "per_symbol": config.MAX_OPEN_PER_SYMBOL},
        "symbols": pool["symbols"], "candidate_count": len(specs),
        "selection_stats": dict(stats), "rejection_reason_counts": dict(rejections),
        "summary": summarize(trades, 100.0, max_dd),
        "notes": ["Metrics use development folds only; selection runs on the untouched holdout.",
                  "Frozen seed genomes were chosen on development history by the source run.",
                  "SWING_OB genomes are excluded (no SwingEngine provider).",
                  "Closed-trade equity; entry at the next 15m open; SL-first; all costs zero."],
    }
    cand_rows = [{"candidate_id": cid, "symbol": specs[cid]["symbol"], "direction": specs[cid]["direction"],
                  "families": "+".join(specs[cid]["signal_families"]),
                  "combination": specs[cid]["family_combination"],
                  "holdout_signals": len(streams[cid]), **metrics[cid]} for cid in specs]
    return report, trades, curve, cand_rows


def run_dynamic_backtest(data, manifest_path, *, selector="score", seed=1, conflict_margin=0.03,
                         policy=ScoringPolicy(), **prepare_kwargs):
    """Convenience wrapper: prepare (resumable) + one selector run."""
    pool = prepare_pool_data(data, manifest_path, **prepare_kwargs)
    return backtest_selector(pool, selector=selector, seed=seed, conflict_margin=conflict_margin,
                             policy=policy)


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="data/okx")
    p.add_argument("--manifest", required=True, help="used for the seed genomes and default symbols")
    p.add_argument("--output-dir", default="results/dynamic_backtest")
    p.add_argument("--symbols", nargs="*", default=None)
    p.add_argument("--expected-symbols", type=int, default=None)
    p.add_argument("--min-history-days", type=float, default=700.0)
    p.add_argument("--smoke-days", type=float, default=None)
    p.add_argument("--dev-fraction", type=float, default=0.85)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--selector", choices=("score", "random", "both"), default="both")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--conflict-margin", type=float, default=0.03)
    p.add_argument("--min-trades", type=int, default=ScoringPolicy().min_trades,
                   help="min dev-fold trades for a candidate to be eligible")
    p.add_argument("--chunk-days", type=float, default=14.0,
                   help="size of one resumable work unit (smaller = finer resume + smoother progress)")
    p.add_argument("--fresh", action="store_true", help="discard the cache and start from zero")
    a = p.parse_args(argv)

    print("[LOAD] loading and validating OKX data", flush=True)
    data = load_okx_data(a.data_root, expected_symbols=a.expected_symbols)
    base = Path(a.output_dir)
    try:
        pool = prepare_pool_data(
            data, a.manifest, symbols=a.symbols, min_history_days=a.min_history_days,
            smoke_days=a.smoke_days, dev_fraction=a.dev_fraction, workers=a.workers,
            chunk_days=a.chunk_days, cache_dir=base / "_cache", fresh=a.fresh)
    except CheckpointMismatch as exc:
        raise SystemExit(f"[ERROR] {exc}")
    except KeyboardInterrupt:
        raise SystemExit("\n[INTERRUPTED] progress is saved. Run the same command again to continue.")

    policy = ScoringPolicy(min_trades=a.min_trades)
    for selector in (("score", "random") if a.selector == "both" else (a.selector,)):
        report, trades, curve, cand_rows = backtest_selector(
            pool, selector=selector, seed=a.seed, conflict_margin=a.conflict_margin, policy=policy)
        out = base / selector
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_csv(out / "trades.csv", trades)
        _write_csv(out / "candidate_metrics.csv", cand_rows)
        _write_csv(out / "equity_curve.csv", [{"utc": _iso(t), "equity": e} for t, e in curve])
        s = report["summary"]
        print(f"[COMPLETE] selector={selector} trades={s['trades']} net_pnl={s['net_pnl']:.2f} "
              f"PF={s['profit_factor']:.2f} avgR={s['average_r']:.3f} "
              f"maxDD={s['max_drawdown_pct_closed_trades']:.2f}%", flush=True)
        print(f"[REPORT] {out}", flush=True)


if __name__ == "__main__":
    main()
