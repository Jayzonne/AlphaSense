import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from AlphaSense.analysis.backtest import (
    BacktestResult,
    _vol_target_fraction_series,
    backtest_symbol,
    buy_and_hold_equity_curve,
    buy_and_hold_value,
    run_backtest,
    simulate_wallet,
    simulate_wallet_with_equity,
)
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


class TestVolatilityTargetedSizing(unittest.TestCase):
    @staticmethod
    def _volatile_candles(n: int = 80, seed: int = 7, daily_vol: float = 0.02) -> pd.DataFrame:
        """Deterministic pseudo-random walk with roughly `daily_vol` log returns."""
        rng = np.random.default_rng(seed)
        prices = [100.0]
        for _ in range(n - 1):
            prices.append(prices[-1] * float(np.exp(rng.normal(0.0, daily_vol))))
        return _candles(prices)

    def test_identical_calls_are_deterministic(self):
        df = self._volatile_candles()
        confluence = _confluence_at(df, 0, bullish=3)
        confluence.update(_confluence_at(df, 30, bearish=3))
        kwargs = dict(
            threshold=1, fee_rate=0.01, starting_cash=1000.0, direction="reverse",
            target_vol=0.10, vol_lookback=10, bars_per_year=252,
        )
        cash_a, trades_a, equity_a = simulate_wallet_with_equity(df, confluence, **kwargs)
        cash_b, trades_b, equity_b = simulate_wallet_with_equity(df, confluence, **kwargs)

        self.assertEqual(cash_a, cash_b)
        self.assertEqual(
            [(t.action, t.time, t.price, t.fee_paid) for t in trades_a],
            [(t.action, t.time, t.price, t.fee_paid) for t in trades_b],
        )
        pd.testing.assert_series_equal(equity_a, equity_b)

    def test_fraction_is_clamped_between_zero_and_one(self):
        df = self._volatile_candles()
        for target_vol in (0.0, 0.001, 0.10, 1000.0):
            with self.subTest(target_vol=target_vol):
                fractions = _vol_target_fraction_series(df, target_vol, 20, 252)
                self.assertEqual(len(fractions), len(df))
                self.assertTrue((fractions >= 0.0).all())
                self.assertTrue((fractions <= 1.0).all())

    def test_incomplete_window_falls_back_to_one(self):
        df = self._volatile_candles()
        lookback = 20
        fractions = _vol_target_fraction_series(df, 0.10, lookback, 252)
        self.assertTrue((fractions.iloc[:lookback] == 1.0).all())

    def test_fraction_has_no_lookahead(self):
        # Identical through bar k, divergent afterwards: the fractions up to
        # and including k must be identical, and truncating after k must not
        # change fraction.iloc[k].
        prices = [100.0 * (1.0 + 0.01 * ((-1) ** i)) for i in range(40)]
        k = 25
        prices_a = prices
        prices_b = prices[: k + 1] + [prices[k] * (1.0 + 0.05 * i) for i in range(1, len(prices) - k)]
        df_a, df_b = _candles(prices_a), _candles(prices_b)

        fractions_a = _vol_target_fraction_series(df_a, 0.10, 5, 252)
        fractions_b = _vol_target_fraction_series(df_b, 0.10, 5, 252)
        pd.testing.assert_series_equal(fractions_a.iloc[: k + 1], fractions_b.iloc[: k + 1])

        truncated = _vol_target_fraction_series(df_a.iloc[: k + 1], 0.10, 5, 252)
        self.assertEqual(truncated.iloc[k], fractions_a.iloc[k])

    def test_very_high_target_matches_disabled_sizing_exactly(self):
        df = self._volatile_candles()
        confluence = _confluence_at(df, 0, bullish=3)
        confluence.update(_confluence_at(df, 30, bearish=3))

        base = simulate_wallet_with_equity(
            df, confluence, threshold=1, fee_rate=0.01, starting_cash=1000.0,
            direction="reverse", target_vol=None,
        )
        high = simulate_wallet_with_equity(
            df, confluence, threshold=1, fee_rate=0.01, starting_cash=1000.0,
            direction="reverse", target_vol=1000.0, vol_lookback=20, bars_per_year=252,
        )

        self.assertEqual(base[0], high[0])
        self.assertEqual(
            [(t.action, t.time, t.price, t.fee_paid) for t in base[1]],
            [(t.action, t.time, t.price, t.fee_paid) for t in high[1]],
        )
        pd.testing.assert_series_equal(base[2], high[2])

    def test_low_target_reduces_exposure_and_keeps_cash(self):
        df = self._volatile_candles()
        confluence = _confluence_at(df, 25, bullish=3)
        starting_cash = 1000.0

        cash, trades, _ = simulate_wallet_with_equity(
            df, confluence, threshold=1, fee_rate=0.0, starting_cash=starting_cash,
            target_vol=0.001, vol_lookback=20, bars_per_year=252,
        )

        fraction = float(_vol_target_fraction_series(df, 0.001, 20, 252).iloc[25])
        self.assertLess(fraction, 1.0)
        self.assertGreater(fraction, 0.0)
        notional = fraction * starting_cash
        self.assertLess(notional, starting_cash)
        self.assertGreater(cash, 0.0)
        self.assertEqual([t.action for t in trades], ["BUY", "SELL (liquidated at end)"])

    def test_fees_are_charged_on_the_deployed_notional(self):
        df = self._volatile_candles()
        confluence = _confluence_at(df, 25, bullish=3)
        starting_cash = 1000.0
        fee_rate = 0.01

        # A different open than close at the execution bar makes the share
        # count observable through the mark-to-market equity point.
        df = df.copy()
        execution_bar = 26
        df.iloc[execution_bar, df.columns.get_loc("open")] = 150.0

        _, trades, equity = simulate_wallet_with_equity(
            df, confluence, threshold=1, fee_rate=fee_rate, starting_cash=starting_cash,
            target_vol=0.05, vol_lookback=20, bars_per_year=252,
        )

        fraction = float(_vol_target_fraction_series(df, 0.05, 20, 252).iloc[25])
        notional = fraction * starting_cash
        expected_fee = notional * fee_rate
        expected_shares = (notional - expected_fee) / 150.0
        expected_equity = (starting_cash - notional) + expected_shares * float(
            df["close"].iloc[execution_bar]
        )

        self.assertAlmostEqual(trades[0].fee_paid, expected_fee, places=10)
        self.assertAlmostEqual(trades[0].price, 150.0)
        self.assertAlmostEqual(equity.iloc[execution_bar], expected_equity, places=8)

    def test_residual_cash_is_preserved_when_a_long_is_sold(self):
        df = self._volatile_candles()
        confluence = _confluence_at(df, 25, bullish=3)
        confluence.update(_confluence_at(df, 40, bearish=3))
        starting_cash = 1000.0

        cash, _, _ = simulate_wallet_with_equity(
            df, confluence, threshold=1, fee_rate=0.0, starting_cash=starting_cash,
            target_vol=0.05, vol_lookback=20, bars_per_year=252,
        )

        entry_fraction = float(_vol_target_fraction_series(df, 0.05, 20, 252).iloc[25])
        entry_notional = entry_fraction * starting_cash
        entry_price = float(df["open"].iloc[26])
        exit_price = float(df["open"].iloc[41])
        shares = entry_notional / entry_price
        residual_cash = starting_cash - entry_notional

        # The uninvested residual stays in the wallet (earning 0%); only the
        # shares' P&L is added on the exit.
        self.assertGreater(residual_cash, 0.0)
        self.assertAlmostEqual(cash, residual_cash + shares * exit_price, places=8)

    def test_target_vol_requires_bars_per_year(self):
        df = self._volatile_candles()
        with self.assertRaises(ValueError):
            _vol_target_fraction_series(df, 0.10, 20, None)

    def test_zero_fraction_does_not_open_a_position(self):
        df = self._volatile_candles()
        confluence = _confluence_at(df, 25, bullish=3)

        cash, trades, equity = simulate_wallet_with_equity(
            df, confluence, threshold=1, fee_rate=0.0, starting_cash=1000.0,
            target_vol=0.0, vol_lookback=20, bars_per_year=252,
        )

        self.assertEqual(trades, [])
        self.assertEqual(cash, 1000.0)
        self.assertTrue((equity == 1000.0).all())

    def test_simulate_wallet_keeps_its_two_tuple_with_target_vol_disabled(self):
        df = self._volatile_candles()
        confluence = _confluence_at(df, 0, bullish=3)
        confluence.update(_confluence_at(df, 30, bearish=3))

        result = simulate_wallet(
            df, confluence, threshold=1, fee_rate=0.01, starting_cash=1000.0,
            target_vol=None, bars_per_year=252,
        )
        full = simulate_wallet_with_equity(
            df, confluence, threshold=1, fee_rate=0.01, starting_cash=1000.0,
            target_vol=None, bars_per_year=252,
        )

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], full[0])
        self.assertEqual(result[1], full[1])


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


class TestBuyAndHoldBenchmark(unittest.TestCase):
    def test_zero_fee_benchmark_return_matches_last_close_over_first_open(self):
        df = _candles([100, 50, 120])
        value = buy_and_hold_value(df, fee_rate=0.0, starting_cash=1000.0)
        # First open = 100, last close = 120 -> +20%.
        self.assertAlmostEqual(value, 1200.0)
        self.assertAlmostEqual(value / 1000.0 - 1, 120 / 100 - 1)

    def test_fees_are_applied_on_both_legs(self):
        df = _candles([100, 50, 120])
        value = buy_and_hold_value(df, fee_rate=0.01, starting_cash=1000.0)
        # entry_fee = 10, shares = 990/100 = 9.9, gross = 1188.0,
        # exit_fee = 11.88, value = 1176.12.
        self.assertAlmostEqual(value, 1176.12, places=2)

    def test_beats_benchmark_is_strict_greater_than(self):
        tie = BacktestResult(
            symbol="TEST", threshold=1, fee_rate=0.0, starting_cash=1000.0,
            final_value=1100.0, benchmark_final_value=1100.0,
        )
        self.assertFalse(tie.beats_benchmark)
        self.assertAlmostEqual(tie.benchmark_return_pct, 10.0)

        ahead = BacktestResult(
            symbol="TEST", threshold=1, fee_rate=0.0, starting_cash=1000.0,
            final_value=1100.01, benchmark_final_value=1100.0,
        )
        self.assertTrue(ahead.beats_benchmark)

        no_benchmark = BacktestResult(
            symbol="TEST", threshold=1, fee_rate=0.0, starting_cash=1000.0, final_value=1100.0,
        )
        self.assertIsNone(no_benchmark.beats_benchmark)
        self.assertIsNone(no_benchmark.benchmark_return_pct)

    def test_backtest_symbol_attaches_the_same_benchmark_to_long_and_short(self):
        from unittest.mock import patch

        df = _candles([99, 100, 99, 120, 121])
        fixed = _confluence_at(df, 0, bullish=3)
        fixed.update(_confluence_at(df, 2, bearish=3))

        class FixedSignalProvider:
            def build(self, price_df: pd.DataFrame) -> dict:
                return fixed

        register_provider("test_benchmark_fixed", lambda **kwargs: FixedSignalProvider())

        with patch("AlphaSense.analysis.backtest.get_price_candles_dataframe", return_value=df):
            results = backtest_symbol(
                "TEST", df.index.min(), df.index.max(), "1h",
                thresholds=[2], fee_rates=[0.01], direction="both",
                signal_provider="test_benchmark_fixed",
            )

        by_direction = {r.direction: r for r in results}
        expected = buy_and_hold_value(df, 0.01, 1000.0)
        self.assertEqual(set(by_direction), {"long", "short"})
        self.assertAlmostEqual(by_direction["long"].benchmark_final_value, expected)
        self.assertAlmostEqual(by_direction["short"].benchmark_final_value, expected)

    def test_run_backtest_exposes_benchmark_columns_on_normal_rows(self):
        from unittest.mock import patch

        with patch("AlphaSense.analysis.backtest.get_price_candles_dataframe") as mock_df, \
             patch("AlphaSense.analysis.backtest.compute_confluence") as mock_conf:
            mock_df.return_value = pd.DataFrame({
                "open": [100, 101, 102, 103, 104],
                "high": [101, 102, 103, 104, 105],
                "low": [99, 100, 101, 102, 103],
                "close": [100, 101, 102, 103, 104],
            }, index=pd.date_range("2026-01-01", periods=5, freq="h"))
            mock_conf.return_value = {"confluence": {}}

            df = run_backtest(symbols=["TEST"], thresholds=[1], fee_rates=[0.0])

        self.assertIn("benchmark_return_pct", df.columns)
        self.assertIn("beats_benchmark", df.columns)
        self.assertTrue(df["benchmark_return_pct"].notna().all())
        self.assertTrue(df["beats_benchmark"].notna().all())
        # Open 100 -> close 104 with no fees: +4%.
        self.assertAlmostEqual(df.iloc[0]["benchmark_return_pct"], 4.0)


class TestBenchmarkEquityAndMetrics(unittest.TestCase):
    def test_curve_has_one_point_per_bar_and_ends_at_buy_and_hold_value(self):
        df = _candles([100, 50, 120])
        for fee_rate in (0.0, 0.01):
            with self.subTest(fee_rate=fee_rate):
                curve = buy_and_hold_equity_curve(df, fee_rate, 1000.0)
                self.assertEqual(len(curve), len(df))
                self.assertEqual(list(curve.index), list(df.index))
                self.assertAlmostEqual(
                    curve.iloc[-1], buy_and_hold_value(df, fee_rate, 1000.0), places=9
                )

    def test_curve_applies_fees_on_both_legs(self):
        df = _candles([100, 50, 120])
        curve = buy_and_hold_equity_curve(df, 0.01, 1000.0)
        # Entry fee is baked into the share count: 990 / 100 = 9.9 shares,
        # so the first mark-to-market point is 9.9 * 100 = 990 (the fee does
        # not appear again as a return).
        self.assertAlmostEqual(curve.iloc[0], 990.0, places=2)
        self.assertAlmostEqual(curve.iloc[1], 495.0, places=2)  # 9.9 * 50
        # Exit fee on the final close: 9.9 * 120 = 1188 gross, * 0.99 = 1176.12.
        self.assertAlmostEqual(curve.iloc[-1], 1176.12, places=2)

    def test_empty_frame_returns_an_empty_float_series(self):
        curve = buy_and_hold_equity_curve(_candles([]), 0.01, 1000.0)
        self.assertTrue(curve.empty)
        self.assertEqual(curve.dtype, float)

    def test_backtest_symbol_reuses_the_same_benchmark_metrics_for_all_rows(self):
        from unittest.mock import patch

        df = _candles([99, 100, 99, 80, 90, 120, 121])
        fixed = _confluence_at(df, 0, bullish=3)
        fixed.update(_confluence_at(df, 2, bearish=3))

        class FixedSignalProvider:
            def build(self, price_df: pd.DataFrame) -> dict:
                return fixed

        register_provider("test_benchmark_metrics_fixed", lambda **kwargs: FixedSignalProvider())

        with patch("AlphaSense.analysis.backtest.get_price_candles_dataframe", return_value=df):
            results = backtest_symbol(
                "TEST", df.index.min(), df.index.max(), "1h",
                thresholds=[1, 2], fee_rates=[0.0], direction="both",
                signal_provider="test_benchmark_metrics_fixed",
            )

        # 2 directions x 2 thresholds = 4 rows, all sharing the fee-0 benchmark.
        self.assertEqual(len(results), 4)
        tuples = {
            (
                r.benchmark_sharpe,
                r.benchmark_sortino,
                r.benchmark_max_drawdown_pct,
                r.benchmark_calmar,
            )
            for r in results
        }
        self.assertEqual(len(tuples), 1)
        sharpe, sortino, maxdd, calmar = next(iter(tuples))
        for value in (sharpe, sortino, maxdd, calmar):
            self.assertIsNotNone(value)

    def test_run_backtest_exposes_benchmark_metric_columns_on_normal_rows(self):
        from unittest.mock import patch

        with patch("AlphaSense.analysis.backtest.get_price_candles_dataframe") as mock_df, \
             patch("AlphaSense.analysis.backtest.compute_confluence") as mock_conf:
            mock_df.return_value = pd.DataFrame({
                "open": [100, 101, 99, 103, 104],
                "high": [101, 102, 100, 104, 105],
                "low": [99, 100, 98, 102, 103],
                # A dip keeps the downside deviation non-zero (Sortino defined).
                "close": [100, 101, 99, 103, 104],
            }, index=pd.date_range("2026-01-01", periods=5, freq="h"))
            mock_conf.return_value = {"confluence": {}}

            df = run_backtest(symbols=["TEST"], thresholds=[1], fee_rates=[0.0])

        for column in (
            "benchmark_sharpe",
            "benchmark_sortino",
            "benchmark_max_drawdown_pct",
            "benchmark_calmar",
        ):
            self.assertIn(column, df.columns)
            self.assertTrue(df[column].notna().all(), column)

    def test_unknown_interval_leaves_only_the_drawdown_defined(self):
        from unittest.mock import patch

        df = _candles([99, 100, 99, 80, 90, 120, 121])
        fixed = _confluence_at(df, 0, bullish=3)
        fixed.update(_confluence_at(df, 2, bearish=3))

        class FixedSignalProvider:
            def build(self, price_df: pd.DataFrame) -> dict:
                return fixed

        register_provider("test_benchmark_unknown_interval", lambda **kwargs: FixedSignalProvider())

        with patch("AlphaSense.analysis.backtest.get_price_candles_dataframe", return_value=df):
            results = backtest_symbol(
                "TEST", df.index.min(), df.index.max(), "7m",
                thresholds=[2], fee_rates=[0.0], direction="long",
                signal_provider="test_benchmark_unknown_interval",
            )

        result = results[0]
        self.assertIsNone(result.benchmark_sharpe)
        self.assertIsNone(result.benchmark_sortino)
        self.assertIsNone(result.benchmark_calmar)
        self.assertIsNotNone(result.benchmark_max_drawdown_pct)


class TestEquityCurveAndMetrics(unittest.TestCase):
    def test_curve_has_one_point_per_bar_and_starts_at_starting_cash(self):
        df = _candles([100, 101, 102, 103, 104])
        _, _, equity = simulate_wallet_with_equity(df, {}, threshold=1, fee_rate=0.0, starting_cash=1000.0)
        self.assertEqual(len(equity), len(df))
        self.assertEqual(list(equity.index), list(df.index))
        self.assertEqual(equity.iloc[0], 1000.0)

    def test_empty_and_single_bar_frames_do_not_crash(self):
        for prices in ([], [100]):
            with self.subTest(prices=prices):
                df = _candles(prices)
                cash, trades, equity = simulate_wallet_with_equity(
                    df, {}, threshold=1, fee_rate=0.0, starting_cash=1000.0
                )
                self.assertEqual(len(equity), len(df))
                self.assertEqual(trades, [])
                self.assertEqual(cash, 1000.0)
                if len(df):
                    self.assertEqual(equity.iloc[0], 1000.0)

    def test_last_equity_point_equals_final_value_for_every_direction(self):
        cases = [
            # Long, position closed before the end.
            ([99, 100, 99, 120, 121], "long", [(0, "bullish"), (2, "bearish")], 0.0),
            # Long, still open at the end -> liquidated at the last close.
            ([99, 100, 105, 110, 130], "long", [(0, "bullish")], 0.01),
            # Short, position covered before the end.
            ([99, 100, 99, 80, 81], "short", [(0, "bearish"), (2, "bullish")], 0.0),
            # Short, still open at the end -> covered at the last close.
            ([99, 100, 105, 110, 130], "short", [(0, "bearish")], 0.01),
            # Reverse, flips long -> short and ends short (liquidated).
            ([99, 100, 99, 120, 121], "reverse", [(0, "bullish"), (2, "bearish")], 0.0),
            # Reverse, no signal at all -> stays flat.
            ([99, 100, 101, 102, 103], "reverse", [], 0.0),
        ]
        for prices, direction, signals, fee_rate in cases:
            with self.subTest(direction=direction, prices=prices, fee_rate=fee_rate):
                df = _candles(prices)
                confluence = {}
                for index, kind in signals:
                    confluence.update(_confluence_at(df, index, **{kind: 3}))
                final, _, equity = simulate_wallet_with_equity(
                    df, confluence, threshold=2, fee_rate=fee_rate,
                    starting_cash=1000.0, direction=direction,
                )
                self.assertEqual(len(equity), len(df))
                self.assertAlmostEqual(equity.iloc[-1], final, places=10)

    def test_simulate_wallet_keeps_its_two_tuple_contract(self):
        df = _candles([99, 100, 99, 120, 121])
        confluence = _confluence_at(df, 0, bullish=3)
        confluence.update(_confluence_at(df, 2, bearish=3))

        result = simulate_wallet(df, confluence, threshold=2, fee_rate=0.01, starting_cash=1000.0)
        full = simulate_wallet_with_equity(df, confluence, threshold=2, fee_rate=0.01, starting_cash=1000.0)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], full[0])
        self.assertEqual(result[1], full[1])

    def test_equity_does_not_look_ahead_of_the_current_bar(self):
        # Both frames are identical through bar 4 and diverge afterwards; the
        # second one also carries a future-only signal at bar 5. Equity up to
        # and including bar 4 must be identical.
        prices_a = [100, 101, 102, 103, 104, 105, 106, 107]
        prices_b = [100, 101, 102, 103, 104, 50, 40, 30]
        df_a, df_b = _candles(prices_a), _candles(prices_b)

        confluence_a = _confluence_at(df_a, 0, bullish=3)
        confluence_a.update(_confluence_at(df_a, 3, bearish=3))
        confluence_b = _confluence_at(df_b, 0, bullish=3)
        confluence_b.update(_confluence_at(df_b, 3, bearish=3))
        confluence_b.update(_confluence_at(df_b, 5, bullish=3))  # only in the future

        _, _, equity_a = simulate_wallet_with_equity(df_a, confluence_a, 2, 0.01, 1000.0)
        _, _, equity_b = simulate_wallet_with_equity(df_b, confluence_b, 2, 0.01, 1000.0)

        k = 4
        pd.testing.assert_series_equal(equity_a.iloc[: k + 1], equity_b.iloc[: k + 1])

    def test_backtest_result_exposes_risk_metrics_when_equity_is_available(self):
        equity = pd.Series(
            [1000.0, 1100.0, 990.0, 1089.0],
            index=pd.date_range("2026-01-01", periods=4, freq="D"),
        )
        result = BacktestResult(
            symbol="TEST", threshold=1, fee_rate=0.0, starting_cash=1000.0,
            final_value=1089.0, equity_curve=equity, bars_per_year=252,
        )
        self.assertIsNotNone(result.sharpe)
        self.assertIsNotNone(result.sortino)
        self.assertIsNotNone(result.calmar)
        self.assertAlmostEqual(result.max_drawdown_pct, 10.0)

    def test_backtest_result_metrics_are_none_without_equity_or_annualization(self):
        bare = BacktestResult(
            symbol="TEST", threshold=1, fee_rate=0.0, starting_cash=1000.0, final_value=1000.0,
        )
        for name in ("sharpe", "sortino", "max_drawdown_pct", "calmar"):
            with self.subTest(name=name):
                self.assertIsNone(getattr(bare, name))

        equity = pd.Series([1000.0, 1010.0, 1005.0], index=pd.date_range("2026-01-01", periods=3, freq="D"))
        no_annualization = BacktestResult(
            symbol="TEST", threshold=1, fee_rate=0.0, starting_cash=1000.0,
            final_value=1005.0, equity_curve=equity,
        )
        for name in ("sharpe", "sortino", "calmar"):
            with self.subTest(name=name):
                self.assertIsNone(getattr(no_annualization, name))
        self.assertIsNotNone(no_annualization.max_drawdown_pct)

    def test_backtest_symbol_attaches_equity_and_bars_per_year(self):
        from unittest.mock import patch

        df = _candles([99, 100, 99, 80, 90, 120, 121])
        fixed = _confluence_at(df, 0, bullish=3)
        fixed.update(_confluence_at(df, 2, bearish=3))

        class FixedSignalProvider:
            def build(self, price_df: pd.DataFrame) -> dict:
                return fixed

        register_provider("test_equity_fixed", lambda **kwargs: FixedSignalProvider())

        with patch("AlphaSense.analysis.backtest.get_price_candles_dataframe", return_value=df):
            results = backtest_symbol(
                "TEST", df.index.min(), df.index.max(), "1h",
                thresholds=[2], fee_rates=[0.0], direction="long",
                signal_provider="test_equity_fixed",
            )

        result = results[0]
        self.assertEqual(result.bars_per_year, 1638.0)
        self.assertIsNotNone(result.equity_curve)
        self.assertEqual(len(result.equity_curve), len(df))
        self.assertAlmostEqual(result.equity_curve.iloc[-1], result.final_value)
        self.assertIsNotNone(result.sharpe)
        self.assertIsNotNone(result.max_drawdown_pct)

    def test_backtest_symbol_unknown_interval_disables_annualized_metrics_only(self):
        from unittest.mock import patch

        df = _candles([99, 100, 99, 80, 90, 120, 121])
        fixed = _confluence_at(df, 0, bullish=3)
        fixed.update(_confluence_at(df, 2, bearish=3))

        class FixedSignalProvider:
            def build(self, price_df: pd.DataFrame) -> dict:
                return fixed

        register_provider("test_unknown_interval_fixed", lambda **kwargs: FixedSignalProvider())

        with patch("AlphaSense.analysis.backtest.get_price_candles_dataframe", return_value=df):
            results = backtest_symbol(
                "TEST", df.index.min(), df.index.max(), "7m",
                thresholds=[2], fee_rates=[0.0], direction="long",
                signal_provider="test_unknown_interval_fixed",
            )

        result = results[0]
        self.assertIsNone(result.bars_per_year)
        self.assertIsNone(result.sharpe)
        self.assertIsNone(result.calmar)
        self.assertIsNotNone(result.equity_curve)

    def test_run_backtest_fills_equity_out_and_keeps_the_dataframe_flat(self):
        from unittest.mock import patch

        eq_out = {}
        with patch("AlphaSense.analysis.backtest.get_price_candles_dataframe") as mock_df, \
             patch("AlphaSense.analysis.backtest.compute_confluence") as mock_conf:
            mock_df.return_value = pd.DataFrame({
                "open": [100, 101, 102, 103, 104],
                "high": [101, 102, 103, 104, 105],
                "low": [99, 100, 101, 102, 103],
                "close": [100, 101, 102, 103, 104],
            }, index=pd.date_range("2026-01-01", periods=5, freq="h"))
            mock_conf.return_value = {"confluence": {}}

            df = run_backtest(
                symbols=["TEST"], thresholds=[1], fee_rates=[0.0], interval="1h",
                equity_out=eq_out,
            )

        self.assertIn(("TEST", "long", 1, 0.0), eq_out)
        self.assertEqual(len(eq_out[("TEST", "long", 1, 0.0)]), 5)
        self.assertNotIn("equity_curve", df.columns)
        for column in ("sharpe", "sortino", "max_drawdown_pct", "calmar"):
            self.assertIn(column, df.columns)


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
