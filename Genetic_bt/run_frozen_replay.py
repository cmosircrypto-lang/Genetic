"""Replay immutable genetic champions over the full history with a final holdout.

The first 85% (development) is descriptive / in-sample because the champions were
selected on development history. The final 15% is reported separately and must not
be used to alter, rank or re-select the frozen genomes. All costs are ZERO.

    python -m genetic_bt.run_frozen_replay --data-root data/okx --manifest frozen_manifest.json
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .data_loader import load_okx_data
from .genome import StrategyGenome
from .historical_evaluator import _find_btc_frames, simulate_segments_multi
from .manifest import load_frozen_manifest
from .parallel import run_jobs
from .window import prepare_window

BTC_SYMBOL = "BTC-USDT-SWAP"
SEGMENTS = ("development_85pct", "untouched_holdout_15pct", "combined_two_year_descriptive_only")
_G: dict = {}


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def _metrics_dict(result) -> dict:
    return {
        "start_utc": _iso(result.start_ms), "end_exclusive_utc": _iso(result.end_ms),
        "trade_count": int(result.trade_count), "net_pnl": float(result.net_pnl),
        "average_r": float(result.average_r), "profit_factor": float(result.profit_factor),
        "max_drawdown_pct": float(result.max_drawdown_pct),
        "gross_profit": float(result.gross_profit), "gross_loss": float(result.gross_loss),
    }


def _init_worker(normalized, btc_frames, bounds, total):
    _G.update(normalized=normalized, btc=btc_frames, bounds=bounds, total=total)


def _run_symbol(job):
    """All frozen candidates of one symbol share closed slices and indicator work."""
    symbol, items = job
    start, split, end = _G["bounds"]
    genomes = {c["candidate_id"]: (StrategyGenome.from_dict(c["genome"]), c["direction"]) for _, c in items}
    print(f"[SYMBOL] {symbol}: {len(items)} candidate(s)", flush=True)
    segs = simulate_segments_multi(
        symbol, _G["normalized"][symbol], _G["btc"], genomes,
        {SEGMENTS[0]: (start, split), SEGMENTS[1]: (split, end), SEGMENTS[2]: (start, end)},
        initial_equity=100.0)
    rows = []
    for index, candidate in items:
        genome = genomes[candidate["candidate_id"]][0]
        cs = segs[candidate["candidate_id"]]
        rows.append((index, {
            "candidate_id": candidate["candidate_id"], "symbol": symbol, "direction": candidate["direction"],
            "signal_families": list(genome.signal_families),
            "family_combination": genome.family_combination,
            "genome_sha256": hashlib.sha256(
                json.dumps(candidate["genome"], sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "source_historical_reference_metrics": candidate["historical_reference_metrics"],
            **{name: _metrics_dict(cs[name]) for name in SEGMENTS},
        }))
    return rows


def run_frozen_replay(data, manifest_path, *, min_history_days=700.0, smoke_days=None,
                      dev_fraction=0.85, workers=1) -> dict:
    manifest, manifest_bytes = load_frozen_manifest(manifest_path)
    candidates = manifest["candidates"]
    candidate_symbols = sorted({c["symbol"] for c in candidates})
    needs_btc = any("BTC_ALIGNED_PULLBACK" in c["genome"]["signal_families"] for c in candidates)
    required = set(candidate_symbols) | ({BTC_SYMBOL} if needs_btc else set())

    normalized, start, split, end, span_days = prepare_window(
        data, required, min_history_days=min_history_days, smoke_days=smoke_days,
        dev_fraction=dev_fraction)
    btc_frames = _find_btc_frames(data) if needs_btc else None
    if needs_btc and btc_frames is None:
        raise ValueError("BTC-aligned candidate(s) need exactly one BTC-USDT-SWAP series with 1h/4h data")

    needed = {s: normalized[s] for s in candidate_symbols}
    by_symbol: dict[str, list] = {}
    for index, candidate in enumerate(candidates):
        by_symbol.setdefault(candidate["symbol"], []).append((index, candidate))
    parts = run_jobs(_run_symbol, sorted(by_symbol.items()), workers=workers, initializer=_init_worker,
                     initargs=(needed, btc_frames, (start, split, end), len(candidates)))
    results = [row for _, row in sorted((pair for part in parts for pair in part), key=lambda x: x[0])]

    swing = [r["candidate_id"] for r in results if "SWING_OB" in r["signal_families"]]
    return {
        "schema_version": 2, "status": "COMPLETE", "source": manifest["source"],
        "dataset": {
            "exchange": "OKX", "symbols_loaded": len(data), "candidate_symbols": candidate_symbols,
            "common_measured_start_utc": _iso(start), "common_end_exclusive_utc": _iso(end),
            "common_history_days": round(span_days, 3), "min_history_days": min_history_days,
            "smoke_days": smoke_days,
            "split_rule": f"chronological first {dev_fraction:.0%} development / final "
                          f"{1 - dev_fraction:.0%} untouched holdout",
            "development_end_exclusive_utc": _iso(split), "holdout_start_utc": _iso(split),
        },
        "costs": {"fee_rate": 0.0, "slippage_rate": 0.0, "funding": 0.0},
        "candidate_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "candidate_count": len(results), "excluded_candidates": manifest["excluded"],
        "candidates_using_unsupported_swing_ob": swing,
        "evolution_ran": False, "candidate_parameters_changed": False,
        "holdout_used_for_selection": False, "portfolio_simulation": False,
        "results": results,
        "methodology_notes": [
            "Champion genomes are frozen; no evolution, mutation, retraining or reselection is performed.",
            "All candidates use the same common chronological window.",
            "The development segment is not independent: champions were selected on development history.",
            "The holdout is evaluated separately and is not used for ranking, tuning or selection.",
            "The combined metric is descriptive only.",
            "Each symbol-direction candidate is simulated independently from $100 (no portfolio).",
            "Causal semantics: closed candles, entry at the next 15m open, SL-first on same-candle collisions.",
            "ALL costs are zero (fees, slippage, funding); live results will be lower.",
            "SWING_OB needs an external SwingEngine provider that is not available here; genomes "
            "containing it cannot fire that family (see candidates_using_unsupported_swing_ob).",
            "Max drawdown is closed-trade equity drawdown, not intrabar.",
        ],
    }


def write_reports(report: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "frozen_replay.json"
    csv_path = output_dir / "frozen_replay.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    columns = ["candidate_id", "symbol", "direction", "segment", "start_utc", "end_exclusive_utc",
               "trade_count", "net_pnl", "average_r", "profit_factor", "max_drawdown_pct"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in report["results"]:
            for segment in SEGMENTS:
                row = {k: v for k, v in item[segment].items() if k in columns}
                writer.writerow({"candidate_id": item["candidate_id"], "symbol": item["symbol"],
                                 "direction": item["direction"], "segment": segment, **row})
    return json_path, csv_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="data/okx")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", default="results/frozen_replay")
    p.add_argument("--expected-symbols", type=int, default=None,
                   help="optional strict check of the number of symbols in the dataset")
    p.add_argument("--min-history-days", type=float, default=700.0)
    p.add_argument("--smoke-days", type=float, default=None, help="quick test on the last N days only")
    p.add_argument("--dev-fraction", type=float, default=0.85)
    p.add_argument("--workers", type=int, default=1, help="parallel processes (one per symbol)")
    a = p.parse_args(argv)

    print("[LOAD] loading and validating OKX data", flush=True)
    data = load_okx_data(a.data_root, expected_symbols=a.expected_symbols)
    print(f"[RUN] frozen replay; symbols_in_dataset={len(data)}", flush=True)
    report = run_frozen_replay(data, a.manifest, min_history_days=a.min_history_days,
                               smoke_days=a.smoke_days, dev_fraction=a.dev_fraction, workers=a.workers)
    json_path, csv_path = write_reports(report, Path(a.output_dir))
    print(f"[COMPLETE] candidates={report['candidate_count']}; "
          f"days={report['dataset']['common_history_days']}; "
          f"holdout_start={report['dataset']['holdout_start_utc']}", flush=True)
    if report["candidates_using_unsupported_swing_ob"]:
        print("[WARN] candidates containing SWING_OB (that family never fires here): "
              f"{report['candidates_using_unsupported_swing_ob']}", flush=True)
    print(f"[REPORT] {json_path}\n[REPORT] {csv_path}", flush=True)


if __name__ == "__main__":
    main()
