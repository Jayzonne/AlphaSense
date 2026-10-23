import unittest
from datetime import datetime, timedelta

import pandas as pd

from AlphaSense.analysis.backtest import BacktestResult, simulate_wallet
from AlphaSense.analysis.confluence import compute_confluence
from AlphaSense.analysis.signals import (
    ConfluenceProvider,
    TrendProvider,
    available_providers,
    get_provider,
    register_provider,
)


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

    # --- Short direction tests ---

    def test_short_simple_profit_when_price_drops(self):
        # Sell short at candle 0 (executes at candle 1's open = 100),
        # cover at candle 2 (executes at candle 3's open = 80).
        # Price drops from 100 to 80 -> short profit.
        df = _candles([99, 100, 99, 80, 81])
        confluence = _confluence_at(df, 0, bearish=3)
        confluence.update(_confluence_at(df, 2, bullish=3))
        final, trades = simulate_wallet(df, confluence, threshold=2, fee_rate=0.0, starting_cash=1000.0, direction="short")
        # Short 10 shares at 100 -> receive 1000, cover at 80 -> pay 800, profit = 200
        self.assertAlmostEqual(final, 1200.0)
        self.assertEqual([t.action for t in trades], ["SELL (short)", "BUY (cover)"])
        self.assertAlmostEqual(trades[0].price, 100.0)
        self.assertAlmostEqual(trades[1].price, 80.0)

    def test_short_with_fees(self):
        df = _candles([99, 100, 99, 80, 81])
        confluence = _confluence_at(df, 0, bearish=3)
        confluence.update(_confluence_at(df, 2, bullish=3))
        final, trades = simulate_wallet(df, confluence, threshold=2, fee_rate=0.01, starting_cash=1000.0, direction="short")
        # Short: fee = 10, cash = 1000 + 1000 - 10 = 1990, shares = -10
        # Cover: cost = 10 * 80 = 800, fee = 8, cash = 1990 - 800 - 8 = 1182
        self.assertAlmostEqual(final, 1182.0, places=2)
        self.assertAlmostEqual(trades[0].fee_paid, 10.0, places=2)
        self.assertAlmostEqual(trades[1].fee_paid, 8.0, places=2)

    def test_short_liquidated_at_final_close(self):
        # Short at candle 0 (executes at candle 1's open = 100), no cover signal,
        # position covered at last candle's close (130) -> loss since price rose.
        df = _candles([99, 100, 105, 110, 130])
        confluence = _confluence_at(df, 0, bearish=3)
        final, trades = simulate_wallet(df, confluence, threshold=2, fee_rate=0.0, starting_cash=1000.0, direction="short")
        # Short 10 shares at 100 -> receive 1000, cash = 2000, shares = -10
        # Cover at 130 -> cost = 1300, cash = 2000 - 1300 = 700
        self.assertAlmostEqual(final, 700.0)
        self.assertEqual(trades[-1].action, "BUY (cover liquidated at end)")

    def test_short_does_not_cover_when_flat(self):
        df = _candles([99, 100, 101, 102])
        confluence = _confluence_at(df, 0, bullish=3)  # bullish signal but never shorted
        final, trades = simulate_wallet(df, confluence, threshold=2, fee_rate=0.0, starting_cash=1000.0, direction="short")
        self.assertEqual(trades, [])
        self.assertEqual(final, 1000.0)

    def test_short_higher_fee_always_reduces_final_value_for_a_profitable_run(self):
        df = _candles([99, 100, 99, 80, 81])
        confluence = _confluence_at(df, 0, bearish=3)
        confluence.update(_confluence_at(df, 2, bullish=3))
        no_fee, _ = simulate_wallet(df, confluence, threshold=2, fee_rate=0.0, starting_cash=1000.0, direction="short")
        low_fee, _ = simulate_wallet(df, confluence, threshold=2, fee_rate=0.005, starting_cash=1000.0, direction="short")
        high_fee, _ = simulate_wallet(df, confluence, threshold=2, fee_rate=0.015, starting_cash=1000.0, direction="short")
        self.assertGreater(no_fee, low_fee)
        self.assertGreater(low_fee, high_fee)


class TestSimulateWalletReverse(unittest.TestCase):
    # --- Reverse direction tests: flips directly between long and short ---

    def test_reverse_long_to_short_flip(self):
        # Bullish signal at candle 0 buys at candle 1's open (100). Bearish
        # signal at candle 2 flips the wallet at candle 3's open (120): the
        # long is sold and the proceeds immediately sold short, both legs at
        # the same price. The short is covered at the last close (121).
        df = _candles([99, 100, 99, 120, 121])
        confluence = _confluence_at(df, 0, bullish=3)
        confluence.update(_confluence_at(df, 2, bearish=3))
        final, trades = simulate_wallet(
            df, confluence, threshold=2, fee_rate=0.0, starting_cash=1000.0, direction="reverse"
        )
        # BUY 10 shares @100 -> cash 0. Flip @120: SELL 10 sh -> cash 1200;
        # SELL SHORT 1200 notional -> shares=-10, cash=2400.
        # Liquidate at 121: cover 10 sh costs 1210 -> 2400-1210 = 1190.
        self.assertAlmostEqual(final, 1190.0)
        self.assertEqual(
            [t.action for t in trades],
            ["BUY", "SELL", "SELL (short)", "BUY (cover liquidated at end)"],
        )
        self.assertAlmostEqual(trades[0].price, 100.0)
        self.assertAlmostEqual(trades[1].price, 120.0)
        self.assertAlmostEqual(trades[2].price, 120.0)

    def test_reverse_short_to_long_flip(self):
        # Bearish signal at candle 0 shorts at candle 1's open (100). Bullish
        # signal at candle 2 flips the wallet at candle 3's open (80): the
        # short is covered and the cash immediately buys long, both legs at
        # the same price. The long is liquidated at the last close (81).
        df = _candles([99, 100, 99, 80, 81])
        confluence = _confluence_at(df, 0, bearish=3)
        confluence.update(_confluence_at(df, 2, bullish=3))
        final, trades = simulate_wallet(
            df, confluence, threshold=2, fee_rate=0.0, starting_cash=1000.0, direction="reverse"
        )
        # SELL SHORT 1000 notional @100 -> shares=-10, cash=2000.
        # Flip @80: cover 10 sh costs 800 -> cash=1200; BUY 15 sh @80 ->
        # cash=0, shares=15. Liquidate at 81: proceeds 15*81 = 1215.
        self.assertAlmostEqual(final, 1215.0)
        self.assertEqual(
            [t.action for t in trades],
            ["SELL (short)", "BUY (cover)", "BUY", "SELL (liquidated at end)"],
        )
        self.assertAlmostEqual(trades[0].price, 100.0)
        self.assertAlmostEqual(trades[1].price, 80.0)
        self.assertAlmostEqual(trades[2].price, 80.0)

    def test_reverse_flip_trades_execute_at_the_same_price(self):
        # The two legs of a flip are one decision executed at one price/time.
        df = _candles([99, 100, 99, 120, 121])
        confluence = _confluence_at(df, 0, bullish=3)
        confluence.update(_confluence_at(df, 2, bearish=3))
        _, trades = simulate_wallet(
            df, confluence, threshold=2, fee_rate=0.0, starting_cash=1000.0, direction="reverse"
        )
        self.assertEqual(trades[1].price, trades[2].price)
        self.assertEqual(trades[1].time, trades[2].time)

    def test_reverse_flat_bullish_goes_long(self):
        df = _candles([99, 100, 101, 102, 103])
        confluence = _confluence_at(df, 0, bullish=3)
        final, trades = simulate_wallet(
            df, confluence, threshold=2, fee_rate=0.0, starting_cash=1000.0, direction="reverse"
        )
        # BUY 10 shares @100, liquidated at the last close 103 -> 10*103 = 1030.
        self.assertAlmostEqual(final, 1030.0)
        self.assertEqual([t.action for t in trades], ["BUY", "SELL (liquidated at end)"])

    def test_reverse_flat_bearish_goes_short(self):
        df = _candles([99, 100, 101, 102, 103])
        confluence = _confluence_at(df, 0, bearish=3)
        final, trades = simulate_wallet(
            df, confluence, threshold=2, fee_rate=0.0, starting_cash=1000.0, direction="reverse"
        )
        # SELL SHORT 1000 notional @100 -> cash = 1000 + 1000 = 2000, shares=-10.
        # Liquidate at the last close 103: cover 10 sh costs 1030 -> 970.
        self.assertAlmostEqual(final, 970.0)
        self.assertEqual([t.action for t in trades], ["SELL (short)", "BUY (cover liquidated at end)"])

    def test_reverse_flip_with_fees_charged_on_both_legs(self):
        df = _candles([99, 100, 99, 120, 121])
        confluence = _confluence_at(df, 0, bullish=3)
        confluence.update(_confluence_at(df, 2, bearish=3))
        final, trades = simulate_wallet(
            df, confluence, threshold=2, fee_rate=0.01, starting_cash=1000.0, direction="reverse"
        )
        # BUY: fee = 1000 * 0.01 = 10; shares = (1000-10)/100 = 9.9; cash = 0.
        # SELL @120: proceeds = 9.9*120 = 1188; fee = 11.88; cash = 1176.12.
        # SELL (short) @120: notional = 1176.12; fee = 1176.12*0.01 = 11.7612;
        #   shares = -1176.12/120 = -9.801; cash = 1176.12+1176.12-11.7612
        #   = 2340.4788.
        # Liquidate @121: cover 9.801 sh costs 1185.921; fee = 11.85921;
        #   cash = 2340.4788-1185.921-11.85921 = 1142.69859.
        self.assertAlmostEqual(final, 1142.69859, places=5)
        self.assertAlmostEqual(trades[0].fee_paid, 10.0, places=2)
        self.assertAlmostEqual(trades[1].fee_paid, 11.88, places=2)
        self.assertAlmostEqual(trades[2].fee_paid, 11.7612, places=4)

    def test_reverse_higher_fee_never_improves_the_result(self):
        df = _candles([99, 100, 99, 120, 121])
        confluence = _confluence_at(df, 0, bullish=3)
        confluence.update(_confluence_at(df, 2, bearish=3))
        no_fee, _ = simulate_wallet(
            df, confluence, threshold=2, fee_rate=0.0, starting_cash=1000.0, direction="reverse"
        )
        low_fee, _ = simulate_wallet(
            df, confluence, threshold=2, fee_rate=0.005, starting_cash=1000.0, direction="reverse"
        )
        high_fee, _ = simulate_wallet(
            df, confluence, threshold=2, fee_rate=0.015, starting_cash=1000.0, direction="reverse"
        )
        self.assertGreater(no_fee, low_fee)
        self.assertGreater(low_fee, high_fee)

    def test_reverse_num_round_trips_counts_exits_not_entries(self):
        # The flip's SELL closes the long (an exit), the following
        # "SELL (short)" opens the short (an entry, not counted), and the end
        # of run cover closes the short (an exit). -> exactly 2 round trips.
        df = _candles([99, 100, 99, 120, 121])
        confluence = _confluence_at(df, 0, bullish=3)
        confluence.update(_confluence_at(df, 2, bearish=3))
        final, trades = simulate_wallet(
            df, confluence, threshold=2, fee_rate=0.0, starting_cash=1000.0, direction="reverse"
        )
        result = BacktestResult(
            symbol="TEST",
            threshold=2,
            fee_rate=0.0,
            starting_cash=1000.0,
            final_value=final,
            direction="reverse",
            trades=trades,
        )
        self.assertEqual(result.num_round_trips, 2)

    def test_reverse_does_not_trade_without_signals(self):
        df = _candles([100, 101, 102, 103, 104])
        final, trades = simulate_wallet(
            df, {}, threshold=1, fee_rate=0.0, starting_cash=1000.0, direction="reverse"
        )
        self.assertEqual(trades, [])
        self.assertEqual(final, 1000.0)


class TestRunBacktestReverseDirection(unittest.TestCase):
    def test_backtest_symbol_accepts_reverse_direction(self):
        from unittest.mock import patch
        from AlphaSense.analysis.backtest import run_backtest

        with patch("AlphaSense.analysis.backtest.get_price_candles_dataframe") as mock_df, \
             patch("AlphaSense.analysis.backtest.compute_confluence") as mock_conf:
            mock_df.return_value = pd.DataFrame({
                "open": [100, 101, 102, 103, 104],
                "high": [101, 102, 103, 104, 105],
                "low": [99, 100, 101, 102, 103],
                "close": [100, 101, 102, 103, 104],
            }, index=pd.date_range("2026-01-01", periods=5, freq="h"))
            mock_conf.return_value = {"confluence": {}}

            df = run_backtest(symbols=["TEST"], thresholds=[1], fee_rates=[0.0], direction="reverse")
        # Unlike "both", "reverse" is a single simulation -> exactly one row.
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["direction"], "reverse")
        self.assertTrue(pd.isna(df.iloc[0]["error"]))


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

    def test_direction_both_produces_long_and_short_rows(self):
        from unittest.mock import patch
        from AlphaSense.analysis.backtest import run_backtest

        with patch("AlphaSense.analysis.backtest.get_price_candles_dataframe") as mock_df, \
             patch("AlphaSense.analysis.backtest.compute_confluence") as mock_conf:
            mock_df.return_value = pd.DataFrame({
                "open": [100, 101, 102, 103, 104],
                "high": [101, 102, 103, 104, 105],
                "low": [99, 100, 101, 102, 103],
                "close": [100, 101, 102, 103, 104],
            }, index=pd.date_range("2026-01-01", periods=5, freq="h"))
            mock_conf.return_value = {"confluence": {}}

            df = run_backtest(symbols=["TEST"], thresholds=[1], fee_rates=[0.0], direction="both")
        self.assertEqual(len(df), 2)
        directions = set(df["direction"].dropna())
        self.assertEqual(directions, {"long", "short"})
        self.assertTrue(all(df["error"].isna()))


class TestSignalProviders(unittest.TestCase):
    def test_registered_provider_can_be_injected_without_core_edits(self):
        from unittest.mock import patch
        from AlphaSense.analysis.backtest import backtest_symbol

        df = pd.DataFrame({
            "open": [99, 100, 99, 120, 121],
            "high": [100, 101, 100, 121, 122],
            "low": [98, 99, 98, 119, 120],
            "close": [99, 100, 99, 120, 121],
        }, index=pd.date_range("2026-01-01", periods=5, freq="h"))
        fixed = _confluence_at(df, 0, bullish=3)
        fixed.update(_confluence_at(df, 2, bearish=3))

        class FixedSignalProvider:
            def build(self, price_df: pd.DataFrame) -> dict:
                return fixed

        register_provider("test_fixed", lambda **kwargs: FixedSignalProvider())

        with patch("AlphaSense.analysis.backtest.get_price_candles_dataframe", return_value=df), \
             patch("AlphaSense.analysis.backtest.compute_confluence", return_value={"confluence": {}}):
            injected = backtest_symbol(
                "TEST", df.index.min(), df.index.max(), "1h",
                thresholds=[2], fee_rates=[0.0], signal_provider="test_fixed",
            )
            default = backtest_symbol(
                "TEST", df.index.min(), df.index.max(), "1h",
                thresholds=[2], fee_rates=[0.0],
            )

        # The injected provider's fixed signals drive the simulation...
        self.assertEqual(len(injected), 1)
        self.assertAlmostEqual(injected[0].final_value, 1200.0)
        self.assertEqual([t.action for t in injected[0].trades], ["BUY", "SELL"])
        # ...while the default (patched-empty) confluence produces no trades,
        # proving the backtest engine actually consulted the selected provider.
        self.assertEqual(default[0].trades, [])
        self.assertEqual(default[0].final_value, 1000.0)

    def test_confluence_provider_matches_compute_confluence(self):
        df = _candles([100 + i for i in range(10)])
        provider = ConfluenceProvider("TEST", df.index.min(), df.index.max(), "1h")

        expected = compute_confluence("TEST", df.index.min(), df.index.max(), "1h", df)["confluence"]

        self.assertEqual(provider.build(df), expected)

    def test_default_signal_provider_name_resolves_to_confluence(self):
        start = datetime(2026, 1, 1)
        end = datetime(2026, 1, 2)

        self.assertIn("confluence", available_providers())
        self.assertIsInstance(get_provider("confluence", "TEST", start, end, "1h"), ConfluenceProvider)


class TestTrendProvider(unittest.TestCase):
    def _provider(self, lookbacks=(1,), interval: str = "1h") -> TrendProvider:
        start = datetime(2026, 1, 1)
        end = datetime(2026, 1, 2)
        return TrendProvider("TEST", start, end, interval, lookbacks=lookbacks)

    def test_build_emits_the_standard_shape_with_correct_counts(self):
        up_df = _candles([100, 101, 102, 103])
        up = self._provider().build(up_df)
        self.assertEqual(set(up), {pd.Timestamp(ts).isoformat() for ts in up_df.index})
        self.assertEqual(
            up[pd.Timestamp(up_df.index[2]).isoformat()],
            {"bullish_count": 1, "bearish_count": 0,
             "bullish_sources": ["tsmom"], "bearish_sources": []},
        )

        down_df = _candles([103, 102, 101, 100])
        down = self._provider().build(down_df)
        self.assertEqual(
            down[pd.Timestamp(down_df.index[2]).isoformat()],
            {"bullish_count": 0, "bearish_count": 1,
             "bullish_sources": [], "bearish_sources": ["tsmom"]},
        )

        # A flat return is neither bullish nor bearish.
        flat_df = _candles([100, 100, 100, 100])
        flat = self._provider().build(flat_df)
        self.assertEqual(
            flat[pd.Timestamp(flat_df.index[2]).isoformat()],
            {"bullish_count": 0, "bearish_count": 0,
             "bullish_sources": [], "bearish_sources": []},
        )

    def test_warm_up_bars_before_the_first_lookback_are_flat(self):
        df = _candles([100 + i for i in range(12)])
        signals = self._provider(lookbacks=(5,)).build(df)

        flat = {"bullish_count": 0, "bearish_count": 0,
                "bullish_sources": [], "bearish_sources": []}
        for i in range(5):
            self.assertEqual(signals[pd.Timestamp(df.index[i]).isoformat()], flat)
        # Bar 5 is the first with a full lookback available.
        self.assertEqual(
            signals[pd.Timestamp(df.index[5]).isoformat()],
            {"bullish_count": 1, "bearish_count": 0,
             "bullish_sources": ["tsmom"], "bearish_sources": []},
        )

    def test_no_lookahead_uses_close_i_not_close_i_plus_one(self):
        # With L=2, bar 2's true return is close[2]/close[0] = 150/100 (+50%).
        # A future-shifted implementation reading close[3]/close[1] = 100/200
        # (-50%) would flip the vote to bearish, so this catches lookahead.
        df = _candles([100, 200, 150, 100, 90])
        signals = self._provider(lookbacks=(2,)).build(df)

        self.assertEqual(
            signals[pd.Timestamp(df.index[2]).isoformat()],
            {"bullish_count": 1, "bearish_count": 0,
             "bullish_sources": ["tsmom"], "bearish_sources": []},
        )
        # Bar 3 correctly uses close[3]/close[1] = 100/200 (-50%), not
        # close[4]/close[2] or close[3]/close[0].
        self.assertEqual(
            signals[pd.Timestamp(df.index[3]).isoformat()],
            {"bullish_count": 0, "bearish_count": 1,
             "bullish_sources": [], "bearish_sources": ["tsmom"]},
        )

    def test_ensemble_two_up_one_down_is_bullish(self):
        # At bar 3: L1 60/40 up, L2 60/50 up, L3 60/100 down -> majority up.
        df = _candles([100, 50, 40, 60])
        signals = self._provider(lookbacks=(1, 2, 3)).build(df)
        self.assertEqual(
            signals[pd.Timestamp(df.index[3]).isoformat()],
            {"bullish_count": 1, "bearish_count": 0,
             "bullish_sources": ["tsmom"], "bearish_sources": []},
        )

    def test_ensemble_one_up_two_down_is_bearish(self):
        # At bar 3: L1 61/60 up, L2 61/90 down, L3 61/100 down -> majority down.
        df = _candles([100, 90, 60, 61])
        signals = self._provider(lookbacks=(1, 2, 3)).build(df)
        self.assertEqual(
            signals[pd.Timestamp(df.index[3]).isoformat()],
            {"bullish_count": 0, "bearish_count": 1,
             "bullish_sources": [], "bearish_sources": ["tsmom"]},
        )

    def test_ensemble_tie_is_flat(self):
        # At bar 2: L1 60/50 up, L2 60/100 down -> 1-1 tie means flat.
        df = _candles([100, 50, 60])
        signals = self._provider(lookbacks=(1, 2)).build(df)
        self.assertEqual(
            signals[pd.Timestamp(df.index[2]).isoformat()],
            {"bullish_count": 0, "bearish_count": 0,
             "bullish_sources": [], "bearish_sources": []},
        )

    def test_tsmom_is_registered_and_resolves_to_trend_provider(self):
        start = datetime(2026, 1, 1)
        end = datetime(2026, 1, 2)

        self.assertIn("tsmom", available_providers())
        provider = get_provider("tsmom", "TEST", start, end, "1d")
        self.assertIsInstance(provider, TrendProvider)
        # Defaults are exactly 1/3/12 months in daily bars.
        self.assertEqual(provider.lookbacks, (21, 63, 252))

        # Explicit lookbacks are forwarded by get_provider and stay literal.
        custom = get_provider("tsmom", "TEST", start, end, "1h", lookbacks=(5, 10))
        self.assertEqual(custom.lookbacks, (5, 10))

    def test_default_lookbacks_scale_with_the_bar_interval(self):
        start = datetime(2026, 1, 1)
        end = datetime(2026, 1, 2)

        daily = TrendProvider("TEST", start, end, "1d")
        weekly = TrendProvider("TEST", start, end, "1wk")
        self.assertEqual(daily.lookbacks, (21, 63, 252))
        # 21 daily bars / 5 weekly bars per month = 3 bars per month.
        self.assertEqual(weekly.lookbacks, (3, 9, 36))

    def test_backtest_symbol_tsmom_direction_both_produces_long_and_short_rows(self):
        from unittest.mock import patch
        from AlphaSense.analysis.backtest import backtest_symbol

        # 50 bars up, then 50 bars down: the up leg produces bullish TSMOM
        # signals and the down leg bearish ones, so both directions trade.
        prices = [100 + i for i in range(50)] + [149 - i for i in range(1, 51)]
        df = _candles(prices)

        with patch("AlphaSense.analysis.backtest.get_price_candles_dataframe", return_value=df):
            results = backtest_symbol(
                "TEST", df.index.min(), df.index.max(), "1h",
                thresholds=[1], fee_rates=[0.0], direction="both",
                signal_provider="tsmom", provider_kwargs={"lookbacks": [21, 63]},
            )

        self.assertEqual({r.direction for r in results}, {"long", "short"})
        by_direction = {r.direction: r for r in results}
        self.assertTrue(by_direction["long"].trades)
        self.assertTrue(by_direction["short"].trades)


if __name__ == "__main__":
    unittest.main()
