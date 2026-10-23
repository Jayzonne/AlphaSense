import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

# The root CLI script is not part of the AlphaSense package, so load it from
# the repo root through importlib. This file lives at
# src/tests/TestBacktestCli_test.py -> parents[2] is the repo root.
ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("alphasense_cli_backtest", ROOT / "backtest.py")
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


def _symbol_df(include_benchmark: bool, include_metrics: bool = False) -> pd.DataFrame:
    """Two thresholds x two fees for a single long direction."""
    metric_values = {
        (1, 0.0): (12.5, 1.23),
        (1, 0.5): (13.5, 1.10),
        (2, 0.0): (None, None),  # missing metric for one cell -> "n/a"
        (2, 0.5): (14.5, 0.90),
    }
    rows = []
    for threshold in (1, 2):
        for fee_pct, benchmark in ((0.0, 20.0), (0.5, 19.0)):
            row = {
                "symbol": "AAPL",
                "threshold": threshold,
                "fee_pct": fee_pct,
                "direction": "long",
                "final_value": 1200.0,
                "return_pct": 15.0 + threshold,
                "num_trades": 3,
                "win": True,
                "error": None,
            }
            if include_benchmark:
                row["benchmark_return_pct"] = benchmark
                row["beats_benchmark"] = benchmark < 15.0 + threshold
            if include_metrics:
                max_drawdown_pct, sharpe = metric_values[(threshold, fee_pct)]
                row["sharpe"] = sharpe
                row["sortino"] = 1.5 if sharpe is not None else None
                row["max_drawdown_pct"] = max_drawdown_pct
                row["calmar"] = 0.25 if sharpe is not None else None
            rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_df(include_benchmark: bool, include_metrics: bool = False) -> pd.DataFrame:
    """One group of four symbols, two of which beat the benchmark."""
    rows = []
    for symbol, beats in (("AAA", True), ("BBB", True), ("CCC", False), ("DDD", False)):
        row = {
            "symbol": symbol,
            "threshold": 1,
            "fee_pct": 0.0,
            "direction": "long",
            "final_value": 1100.0,
            "return_pct": 10.0,
            "num_trades": 2,
            "win": True,
            "error": None,
        }
        if include_benchmark:
            row["benchmark_return_pct"] = 10.0
            row["beats_benchmark"] = beats
        if include_metrics:
            row["sharpe"] = 1.0 if beats else 0.5
            row["sortino"] = 1.5 if beats else 0.75
            row["max_drawdown_pct"] = 10.0 if beats else 20.0
            row["calmar"] = 0.2 if beats else 0.1
        rows.append(row)
    return pd.DataFrame(rows)


def _capture(fn, *args) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        fn(*args)
    return out.getvalue()


class TestPrintSymbolTable(unittest.TestCase):
    def test_prints_buy_and_hold_when_benchmark_available(self):
        df = _symbol_df(include_benchmark=True)
        out = _capture(cli._print_symbol_table, "AAPL", df)
        self.assertIn("buy & hold:", out)
        self.assertIn("Fee 0% = +20.00%", out)
        self.assertIn("Fee 0.5% = +19.00%", out)
        # The existing WIN/LOSS + return + trades layout must be untouched.
        self.assertIn("WIN  +16.00% (3t)", out)
        self.assertIn("WIN  +17.00% (3t)", out)

    def test_prints_risk_metrics_in_widened_cells(self):
        df = _symbol_df(include_benchmark=True, include_metrics=True)
        out = _capture(cli._print_symbol_table, "AAPL", df)
        self.assertIn("DD 12.5% SR 1.23", out)
        self.assertIn("DD 13.5% SR 1.10", out)
        self.assertIn("DD 14.5% SR 0.90", out)
        # A missing metric for one cell is shown as n/a, not "nan" or a crash.
        self.assertIn("DD n/a SR n/a", out)
        self.assertNotIn(" nan", out.lower())

    def test_backwards_safe_without_benchmark_columns(self):
        df = _symbol_df(include_benchmark=False)
        out = _capture(cli._print_symbol_table, "AAPL", df)
        self.assertNotIn("buy & hold:", out)
        self.assertNotIn("nan", out.lower())
        # Existing table still prints.
        self.assertIn("--- LONG ---", out)
        self.assertIn("Threshold", out)
        self.assertIn("WIN  +16.00% (3t)", out)

    def test_old_cell_format_is_kept_when_metrics_columns_are_absent(self):
        df = _symbol_df(include_benchmark=True)
        out = _capture(cli._print_symbol_table, "AAPL", df)
        self.assertNotIn(" DD ", out)
        self.assertNotIn(" SR ", out)
        self.assertIn("WIN  +16.00% (3t)", out)

    def test_error_row_is_skipped_without_benchmark_line(self):
        df = pd.DataFrame([{
            "symbol": "BAD",
            "threshold": None,
            "fee_pct": None,
            "direction": None,
            "final_value": None,
            "return_pct": None,
            "benchmark_return_pct": None,
            "beats_benchmark": None,
            "num_trades": None,
            "win": None,
            "error": "boom",
        }])
        out = _capture(cli._print_symbol_table, "BAD", df)
        self.assertIn("(skipped: boom)", out)
        self.assertNotIn("buy & hold:", out)


class TestPrintAggregateSummary(unittest.TestCase):
    def test_benchmark_average_is_no_longer_in_the_main_table(self):
        df = _aggregate_df(include_benchmark=True)
        out = _capture(cli._print_aggregate_summary, df)
        # The benchmark return moved to its own dedicated table.
        self.assertNotIn("avg_benchmark_pct", out)
        # ...while the beat rate stays in the main table.
        self.assertIn("beat_rate_pct", out)
        self.assertIn("50.0", out)
        self.assertIn("symbols_tested", out)

    def test_keeps_risk_metric_averages_when_available(self):
        df = _aggregate_df(include_benchmark=True, include_metrics=True)
        out = _capture(cli._print_aggregate_summary, df)
        for name in ("avg_sharpe", "avg_sortino", "avg_maxdd_pct", "avg_calmar"):
            self.assertIn(name, out)
        self.assertIn("beat_rate_pct", out)
        self.assertNotIn("avg_benchmark_pct", out)

    def test_adds_risk_metric_averages_in_order(self):
        df = _aggregate_df(include_benchmark=True, include_metrics=True)
        out = _capture(cli._print_aggregate_summary, df)
        for name in ("avg_sharpe", "avg_sortino", "avg_maxdd_pct", "avg_calmar"):
            self.assertIn(name, out)
        # Averages of the four symbols: sharpe 0.75, sortino 1.125 -> 1.12,
        # maxdd 15.0, calmar 0.15.
        self.assertIn("0.75", out)
        self.assertIn("1.12", out)
        self.assertIn("15.0", out)
        self.assertIn("0.15", out)
        self.assertLess(out.index("avg_sharpe"), out.index("avg_sortino"))
        self.assertLess(out.index("avg_sortino"), out.index("avg_maxdd_pct"))
        self.assertLess(out.index("avg_maxdd_pct"), out.index("avg_calmar"))

    def test_risk_metric_averages_are_skipped_when_all_nan(self):
        df = _aggregate_df(include_benchmark=True)
        for column in ("sharpe", "sortino", "max_drawdown_pct", "calmar"):
            df[column] = float("nan")
        out = _capture(cli._print_aggregate_summary, df)
        for name in ("avg_sharpe", "avg_sortino", "avg_maxdd_pct", "avg_calmar"):
            self.assertNotIn(name, out)

    def test_backwards_safe_without_benchmark_columns(self):
        df = _aggregate_df(include_benchmark=False)
        out = _capture(cli._print_aggregate_summary, df)
        self.assertNotIn("avg_benchmark_pct", out)
        self.assertNotIn("beat_rate_pct", out)
        # Existing summary intact.
        self.assertIn("win_rate_pct", out)
        self.assertIn("avg_return_pct", out)
        self.assertIn("symbols_tested", out)
        self.assertNotIn("avg_sharpe", out)
        self.assertNotIn("avg_sortino", out)
        self.assertNotIn("avg_maxdd_pct", out)
        self.assertNotIn("avg_calmar", out)


def _benchmark_df(include_metrics: bool = True) -> pd.DataFrame:
    """Two symbols x two fees, duplicated over directions/thresholds."""
    rows = []
    for symbol, bench_return in (("AAA", 10.0), ("BBB", 20.0)):
        for fee_pct, fee_offset in ((0.0, 0.0), (0.5, -1.0)):
            for threshold in (1, 2):
                for direction in ("long", "short"):
                    row = {
                        "symbol": symbol,
                        "threshold": threshold,
                        "fee_pct": fee_pct,
                        "direction": direction,
                        "final_value": 1100.0,
                        "return_pct": 10.0,
                        "benchmark_return_pct": bench_return + fee_offset,
                        "num_trades": 2,
                        "win": True,
                        "error": None,
                    }
                    if include_metrics:
                        row["benchmark_sharpe"] = 1.0
                        row["benchmark_sortino"] = 1.5
                        row["benchmark_max_drawdown_pct"] = 12.0
                        row["benchmark_calmar"] = 0.3
                    rows.append(row)
    return pd.DataFrame(rows)


class TestPrintBenchmarkSummary(unittest.TestCase):
    def test_prints_header_and_all_benchmark_columns(self):
        out = _capture(cli._print_benchmark_summary, _benchmark_df())
        self.assertIn("=== Benchmark (buy & hold) across all symbols ===", out)
        for name in (
            "avg_benchmark_pct",
            "avg_benchmark_sharpe",
            "avg_benchmark_sortino",
            "avg_benchmark_maxdd_pct",
            "avg_benchmark_calmar",
            "symbols_tested",
        ):
            self.assertIn(name, out)
        # Fee 0%: mean of 10 and 20 = 15.0; fee 0.5%: mean of 9 and 19 = 14.0.
        self.assertIn("15.0", out)
        self.assertIn("14.0", out)

    def test_benchmark_is_counted_once_per_symbol_and_fee(self):
        # AAA appears 3x for fee 0 and BBB only once: a non-deduplicated mean
        # would be (10*3 + 20)/4 = 12.5 instead of the correct (10+20)/2 = 15.
        rows = [
            {
                "symbol": "AAA", "threshold": threshold, "fee_pct": 0.0,
                "direction": "long", "benchmark_return_pct": 10.0, "error": None,
            }
            for threshold in (1, 2, 3)
        ]
        rows.append({
            "symbol": "BBB", "threshold": 1, "fee_pct": 0.0,
            "direction": "long", "benchmark_return_pct": 20.0, "error": None,
        })
        out = _capture(cli._print_benchmark_summary, pd.DataFrame(rows))
        self.assertIn("15.0", out)
        self.assertNotIn("12.5", out)
        self.assertIn("2", out)  # symbols_tested

    def test_skips_silently_when_benchmark_columns_are_absent(self):
        df = _aggregate_df(include_benchmark=False)
        out = _capture(cli._print_benchmark_summary, df)
        self.assertEqual(out, "")

    def test_reduced_table_when_only_the_benchmark_return_is_present(self):
        out = _capture(cli._print_benchmark_summary, _benchmark_df(include_metrics=False))
        self.assertIn("=== Benchmark (buy & hold) across all symbols ===", out)
        self.assertIn("avg_benchmark_pct", out)
        for name in (
            "avg_benchmark_sharpe",
            "avg_benchmark_sortino",
            "avg_benchmark_maxdd_pct",
            "avg_benchmark_calmar",
        ):
            self.assertNotIn(name, out)

    def test_error_rows_are_ignored_without_crashing(self):
        rows = _benchmark_df().to_dict("records")
        rows.append({
            "symbol": "BAD", "threshold": None, "fee_pct": None,
            "direction": None, "benchmark_return_pct": None, "error": "boom",
        })
        df = pd.DataFrame(rows)
        out = _capture(cli._print_benchmark_summary, df)
        self.assertIn("=== Benchmark (buy & hold) across all symbols ===", out)
        self.assertNotIn("BAD", out)


class TestVolatilityTargetCli(unittest.TestCase):
    def _run_main(self, argv: list[str]):
        captured = {}

        def fake_run_backtest(**kwargs):
            captured.update(kwargs)
            return _symbol_df(include_benchmark=False)

        with patch.object(cli, "run_backtest", side_effect=fake_run_backtest), \
             patch.object(sys, "argv", ["backtest.py", *argv]):
            out = _capture(cli.main)
        return out, captured

    def test_target_vol_and_lookback_are_accepted_and_announced(self):
        out, kwargs = self._run_main(["--target-vol", "10", "--vol-lookback", "30"])
        self.assertIn("target_vol=10.0% vol_lookback=30", out)
        # The CLI percent is converted to a fraction for the engine.
        self.assertAlmostEqual(kwargs["target_vol"], 0.10)
        self.assertEqual(kwargs["vol_lookback"], 30)

    def test_header_is_unchanged_without_target_vol(self):
        out, kwargs = self._run_main([])
        self.assertIn("Backtesting", out)
        self.assertNotIn("target_vol=", out)
        self.assertNotIn("vol_lookback=", out)
        self.assertIsNone(kwargs["target_vol"])
        self.assertEqual(kwargs["vol_lookback"], 20)

    def test_hysteresis_thresholds_passed_and_announced(self):
        out, kwargs = self._run_main(["--entry-threshold", "2", "--exit-threshold", "1"])
        self.assertIn("entry_threshold=2, exit_threshold=1", out)
        self.assertEqual(kwargs["entry_threshold"], 2)
        self.assertEqual(kwargs["exit_threshold"], 1)

    def test_hysteresis_single_entry_threshold(self):
        out, kwargs = self._run_main(["--entry-threshold", "2"])
        self.assertIn("entry_threshold=2, exit_threshold=None", out)
        self.assertEqual(kwargs["entry_threshold"], 2)
        self.assertIsNone(kwargs["exit_threshold"])

    def test_header_unchanged_without_hysteresis(self):
        out, kwargs = self._run_main([])
        self.assertNotIn("entry_threshold", out)
        self.assertIsNone(kwargs["entry_threshold"])
        self.assertIsNone(kwargs["exit_threshold"])

    def test_fee_fixed_passed_and_announced(self):
        out, kwargs = self._run_main(["--fee-fixed", "1.0"])
        self.assertIn("fee_fixed=$1.00", out)
        self.assertEqual(kwargs["fee_fixed"], 1.0)

    def test_header_unchanged_without_fee_fixed(self):
        out, kwargs = self._run_main([])
        self.assertNotIn("fee_fixed", out)
        self.assertEqual(kwargs["fee_fixed"], 0.0)


class TestUniverseCli(unittest.TestCase):
    def _run_main(self, argv: list[str]):
        captured = {}

        def fake_run_backtest(**kwargs):
            captured.update(kwargs)
            return _symbol_df(include_benchmark=False)

        with patch.object(cli, "run_backtest", side_effect=fake_run_backtest), \
             patch.object(sys, "argv", ["backtest.py", *argv]):
            out = _capture(cli.main)
        return out, captured

    def test_universe_us_selects_us_symbols_and_is_announced(self):
        from AlphaSense.analysis.backtest import DEFAULT_SYMBOLS_US

        out, kwargs = self._run_main(["--universe", "us"])
        self.assertEqual(kwargs["symbols"], DEFAULT_SYMBOLS_US)
        self.assertIn("universe=us", out)

    def test_universe_eu_selects_eu_symbols_and_is_announced(self):
        from AlphaSense.analysis.backtest import DEFAULT_SYMBOLS_EU

        out, kwargs = self._run_main(["--universe", "eu"])
        self.assertEqual(kwargs["symbols"], DEFAULT_SYMBOLS_EU)
        self.assertIn("universe=eu", out)

    def test_default_universe_is_all_and_not_announced(self):
        from AlphaSense.analysis.backtest import DEFAULT_SYMBOLS

        out, kwargs = self._run_main([])
        self.assertEqual(kwargs["symbols"], DEFAULT_SYMBOLS)
        self.assertNotIn("universe=", out)

    def test_explicit_symbols_override_universe(self):
        out, kwargs = self._run_main(["--symbols", "AAPL", "--universe", "eu"])
        self.assertEqual(kwargs["symbols"], ["AAPL"])
        self.assertNotIn("universe=", out)


if __name__ == "__main__":
    unittest.main()
