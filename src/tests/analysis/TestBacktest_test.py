import unittest
from datetime import datetime, timedelta

import pandas as pd

from AlphaSense.analysis.backtest import simulate_wallet


def _candles(prices: list[float]) -> pd.DataFrame:
    start = datetime(2026, 1, 1, 9, 30)
    idx = [start + timedelta(hours=i) for i in range(len(prices))]
    return pd.DataFrame({
        "open": prices, "high": [p + 1 for p in prices], "low": [p - 1 for p in prices], "close": prices,
    }, index=pd.DatetimeIndex(idx, name="time"))


def _confluence_at(df: pd.DataFrame, index: int, bullish: int = 0, bearish: int = 0) -> dict:
    ts_iso = pd.Timestamp(df.index[index]).isoformat()
    return {ts_iso: {"bullish_count": bullish, "bearish_count": bearish, "bullish_sources": [], "bearish_sources": []}}


class TestSimulateWallet(unittest.TestCase):
    def test_no_signals_leaves_wallet_untouched(self):
        df = _candles([100, 101, 102, 103, 104])
        final, trades = simulate_wallet(df, {}, threshold=1, fee_rate=0.0, starting_cash=1000.0)
        self.assertEqual(final, 1000.0)
        self.assertEqual(trades, [])

    def test_single_round_trip_zero_fee(self):
        # Buy signal at candle 0 (executes at candle 1's open = 100),
        # sell signal at candle 2 (executes at candle 3's open = 120).
        df = _candles([99, 100, 99, 120, 121])
        confluence = _confluence_at(df, 0, bullish=3)
        confluence.update(_confluence_at(df, 2, bearish=3))
        final, trades = simulate_wallet(df, confluence, threshold=2, fee_rate=0.0, starting_cash=1000.0)
        # 1000 / 100 = 10 shares, sold at 120 -> 1200
        self.assertAlmostEqual(final, 1200.0)
        self.assertEqual([t.action for t in trades], ["BUY", "SELL"])
        self.assertAlmostEqual(trades[0].price, 100.0)
        self.assertAlmostEqual(trades[1].price, 120.0)

    def test_single_round_trip_with_fee(self):
        df = _candles([99, 100, 99, 120, 121])
        confluence = _confluence_at(df, 0, bullish=3)
        confluence.update(_confluence_at(df, 2, bearish=3))
        final, trades = simulate_wallet(df, confluence, threshold=2, fee_rate=0.01, starting_cash=1000.0)
        # Buy: fee = 10, shares = 990 / 100 = 9.9
        # Sell: proceeds = 9.9 * 120 = 1188, fee = 11.88, cash = 1176.12
        self.assertAlmostEqual(final, 1176.12, places=2)
        self.assertAlmostEqual(trades[0].fee_paid, 10.0, places=2)
        self.assertAlmostEqual(trades[1].fee_paid, 11.88, places=2)

    def test_higher_fee_always_reduces_final_value_for_a_profitable_run(self):
        df = _candles([99, 100, 99, 120, 121])
        confluence = _confluence_at(df, 0, bullish=3)
        confluence.update(_confluence_at(df, 2, bearish=3))
        no_fee, _ = simulate_wallet(df, confluence, threshold=2, fee_rate=0.0, starting_cash=1000.0)
        low_fee, _ = simulate_wallet(df, confluence, threshold=2, fee_rate=0.005, starting_cash=1000.0)
        high_fee, _ = simulate_wallet(df, confluence, threshold=2, fee_rate=0.015, starting_cash=1000.0)
        self.assertGreater(no_fee, low_fee)
        self.assertGreater(low_fee, high_fee)

    def test_open_position_is_liquidated_at_final_close(self):
        # Bullish signal at candle 0 (buys at candle 1's open = 100); no
        # sell signal ever fires, so the position must be closed at the
        # last candle's close (130) for the run to end in cash.
        df = _candles([99, 100, 105, 110, 130])
        confluence = _confluence_at(df, 0, bullish=3)
        final, trades = simulate_wallet(df, confluence, threshold=2, fee_rate=0.0, starting_cash=1000.0)
        self.assertAlmostEqual(final, 1300.0)  # 10 shares * 130
        self.assertEqual(trades[-1].action, "SELL (liquidated at end)")

    def test_does_not_buy_twice_while_already_holding(self):
        df = _candles([99, 100, 101, 102, 103, 104])
        confluence = _confluence_at(df, 0, bullish=3)
        confluence.update(_confluence_at(df, 1, bullish=3))  # a second bullish signal while already holding
        final, trades = simulate_wallet(df, confluence, threshold=2, fee_rate=0.0, starting_cash=1000.0)
        buys = [t for t in trades if t.action == "BUY"]
        self.assertEqual(len(buys), 1, "should not buy again while already holding a position")

    def test_does_not_sell_when_flat(self):
        df = _candles([99, 100, 101, 102])
        confluence = _confluence_at(df, 0, bearish=3)  # bearish signal but never bought anything
        final, trades = simulate_wallet(df, confluence, threshold=2, fee_rate=0.0, starting_cash=1000.0)
        self.assertEqual(trades, [])
        self.assertEqual(final, 1000.0)

    def test_higher_threshold_never_trades_more_than_a_lower_one(self):
        df = _candles([100 + i for i in range(20)])
        confluence = {}
        for i in [2, 5, 9, 13]:
            confluence.update(_confluence_at(df, i, bullish=(i % 4) + 1))
        for i in [3, 7, 11, 16]:
            confluence.update(_confluence_at(df, i, bearish=(i % 4) + 1))
        _, trades_low = simulate_wallet(df, confluence, threshold=1, fee_rate=0.0)
        _, trades_high = simulate_wallet(df, confluence, threshold=4, fee_rate=0.0)
        self.assertLessEqual(len(trades_high), len(trades_low))

    def test_win_loss_is_strict_greater_than(self):
        df = _candles([100, 100, 100])
        final, _ = simulate_wallet(df, {}, threshold=1, fee_rate=0.0, starting_cash=1000.0)
        self.assertEqual(final, 1000.0)  # exactly break-even (no trades) - must count as a loss, not a win


class TestRunBacktestErrorHandling(unittest.TestCase):
    def test_a_failing_symbol_is_recorded_as_an_error_row_not_a_crash(self):
        from unittest.mock import patch
        from AlphaSense.analysis.backtest import run_backtest

        with patch("AlphaSense.analysis.backtest.get_price_candles_dataframe", side_effect=RuntimeError("boom")):
            df = run_backtest(symbols=["BROKEN"], thresholds=[1], fee_rates=[0.0])
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["symbol"], "BROKEN")
        self.assertIn("boom", df.iloc[0]["error"])

    def test_empty_price_data_is_recorded_as_no_data_not_a_crash(self):
        from unittest.mock import patch
        from AlphaSense.analysis.backtest import run_backtest

        with patch("AlphaSense.analysis.backtest.get_price_candles_dataframe", return_value=pd.DataFrame()):
            df = run_backtest(symbols=["EMPTY"], thresholds=[1], fee_rates=[0.0])
        self.assertEqual(len(df), 1)
        self.assertIn("no data", df.iloc[0]["error"])


if __name__ == "__main__":
    unittest.main()
