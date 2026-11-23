"""Offline smoke tests on synthetic data:  python -m unittest discover -s tests -t ."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from genetic_bt import run_dynamic_backtest as dyn
from genetic_bt import run_frozen_replay as rep
from genetic_bt.data_loader import load_okx_data
from genetic_bt.genome import StrategyGenome
from genetic_bt.historical_evaluator import _simulate_fold, simulate_segments
from genetic_bt.window import prepare_window
from tests.synthetic_data import SYMBOLS, make_dataset, make_manifest


class SmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        make_dataset(cls.root / "okx", days=30)
        make_manifest(cls.root / "manifest.json")
        cls.data = load_okx_data(cls.root / "okx", expected_symbols=len(SYMBOLS))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_segments_equal_separate_folds(self):
        manifest = json.loads((self.root / "manifest.json").read_text())
        norm, start, split, end, _ = prepare_window(self.data, SYMBOLS, min_history_days=15)
        for cand in manifest["candidates"]:
            genome = StrategyGenome.from_dict(cand["genome"])
            sym, d = cand["symbol"], cand["direction"]
            segs = {"a": (start, split), "b": (split, end), "c": (start, end)}
            fast = simulate_segments(sym, norm[sym], None, genome, d, segs, initial_equity=100.0)
            for name, (s, e) in segs.items():
                slow = _simulate_fold(sym, norm[sym], None, genome, d, s, e, initial_equity=100.0)
                self.assertEqual(fast[name], slow, f"{sym} {d} {name}")

    def test_frozen_replay(self):
        report = rep.run_frozen_replay(self.data, self.root / "manifest.json", min_history_days=15)
        self.assertEqual(report["candidate_count"], 6)
        out = rep.write_reports(report, self.root / "out_replay")
        self.assertTrue(all(p.exists() for p in out))
        self.assertGreater(sum(r["combined_two_year_descriptive_only"]["trade_count"]
                               for r in report["results"]), 0)

    def test_dynamic_backtest_runs(self):
        from genetic_bt.dynamic_selector import ScoringPolicy
        report, trades, curve, rows = dyn.run_dynamic_backtest(
            self.data, self.root / "manifest.json", min_history_days=15, workers=3,
            policy=ScoringPolicy(min_trades=1))
        self.assertEqual(report["candidate_count"], len(rows))
        self.assertLessEqual(len(trades), report["selection_stats"].get("opened", 0))
        self.assertIn("summary", report)


if __name__ == "__main__":
    unittest.main()
