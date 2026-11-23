"""Per-symbol candidate pool: seeds + every single family + all pairwise VOTE hybrids."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from itertools import combinations
from typing import Any, Iterable

from . import config
from .genome import StrategyGenome

DIRECTIONS = ("LONG", "SHORT")


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_symbol_candidate_pool(
    symbol: str, genomes: Iterable[StrategyGenome], *, include_swing_ob: bool = False,
) -> list[dict[str, Any]]:
    """Candidates for one symbol, both directions, de-duplicated and deterministic.

    ``SWING_OB`` needs an external SwingEngine signal provider that this research
    package does not ship, so by default every genome containing it is excluded
    (it could never produce a signal and would only waste CPU time).
    """
    symbol = symbol.strip().upper()
    if not symbol:
        raise ValueError("symbol is required")
    baseline = StrategyGenome.baseline()
    seeds = list(genomes)
    for family in config.STRATEGY_FAMILIES:
        seeds.append(replace(baseline, signal_families=(family,), family_combination="SINGLE"))
    for left, right in combinations(config.STRATEGY_FAMILIES, 2):
        seeds.append(replace(baseline, signal_families=(left, right),
                             family_combination="VOTE", min_family_agreement=2))
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    for genome in seeds:
        genome.validate()
        has_swing = "SWING_OB" in genome.signal_families
        if has_swing and not include_swing_ob:
            continue
        methods = genome.entry_methods if has_swing else (genome.entry_methods[0],)
        for method in methods:
            one = replace(genome, entry_methods=(method,), symbols=(symbol,))
            payload = one.to_dict()
            for direction in DIRECTIONS:
                candidate_id = canonical_hash({"genome": payload, "direction": direction})[:20]
                if candidate_id in seen:
                    continue
                seen.add(candidate_id)
                pool.append({
                    "candidate_id": candidate_id,
                    "method": method if has_swing else "FAMILY_NATIVE",
                    "signal_families": list(one.signal_families),
                    "family_combination": one.family_combination,
                    "direction": direction, "genome": payload,
                })
    return sorted(pool, key=lambda item: (
        tuple(item["signal_families"]), item["family_combination"],
        item["method"], item["direction"], item["candidate_id"]))
