"""Load and validate the frozen champion manifest (JSON written by freeze_candidates)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .genome import StrategyGenome

DIRECTIONS = ("LONG", "SHORT")


def load_frozen_manifest(path: str | Path) -> tuple[dict[str, Any], bytes]:
    """Return (manifest, raw_bytes). Raw bytes are returned so callers can hash them."""
    raw = Path(path).read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    if manifest.get("schema_version") != 1 or manifest.get("frozen") is not True:
        raise ValueError("manifest must have schema_version=1 and frozen=true")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("manifest has no candidates")
    seen: set[str] = set()
    for item in candidates:
        for key in ("candidate_id", "symbol", "direction", "genome"):
            if key not in item:
                raise ValueError(f"candidate missing '{key}': {item.get('candidate_id')}")
        if item["candidate_id"] in seen:
            raise ValueError(f"duplicate candidate_id: {item['candidate_id']}")
        seen.add(item["candidate_id"])
        if item["direction"] not in DIRECTIONS:
            raise ValueError(f"bad direction for {item['candidate_id']}: {item['direction']}")
        genome = StrategyGenome.from_dict(item["genome"])
        if genome.symbols != (item["symbol"],):
            raise ValueError(f"genome symbol scope mismatch for {item['candidate_id']}")
        item.setdefault("historical_reference_metrics", {})
    manifest.setdefault("excluded", [])
    manifest.setdefault("source", {})
    return manifest, raw
