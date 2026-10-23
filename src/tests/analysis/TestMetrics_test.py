import unittest

import numpy as np
import pandas as pd
import pytest

from AlphaSense.analysis import metrics


class TestAnnualizationFactor(unittest.TestCase):
    def test_every_known_interval(self):
        expected = {
            "5m": 19656,
            "15m": 6552,
            "30m": 3276,
            "1h": 1638,
            "4h": 409.5,
            "1d": 252,
            "1wk": 52,
            "1mo": 12,
            "3mo": 4,
        }
        for interval, factor in expected.items():
            with self.subTest(interval=interval):
                self.assertEqual(metrics.annualization_factor(interval), factor)

    def test_unknown_interval_raises_value_error(self):
        with self.assertRaises(ValueError):
            metrics.annualization_factor("7m")


class TestSharpeRatio(unittest.TestCase):
    def test_known_value(self):
        returns = [0.1, -0.1, 0.1]
        value = metrics.sharpe_ratio(returns, periods_per_year=1)
        self.assertEqual(value, pytest.approx(0.2887, abs=1e-4))

    def test_risk_free_is_annualized(self):
        # 12% annual risk-free over 12 periods = 1% per period. Returns of
        # exactly 1% then have zero excess and therefore a zero std -> None,
        # while a flat 2% excess gives a finite positive ratio.
        self.assertIsNone(metrics.sharpe_ratio([0.01, 0.01], 12, risk_free=0.12))
        self.assertIsNotNone(metrics.sharpe_ratio([0.02, 0.03], 12, risk_free=0.12))

    def test_drops_nan_and_ignores_non_finite(self):
        returns = pd.Series([0.1, np.nan, np.inf, -0.1, 0.1, -np.inf])
        value = metrics.sharpe_ratio(returns, periods_per_year=1)
        self.assertEqual(value, pytest.approx(0.2887, abs=1e-4))

    def test_degenerate_inputs_return_none(self):
        self.assertIsNone(metrics.sharpe_ratio([], 1))
        self.assertIsNone(metrics.sharpe_ratio([0.1], 1))
        self.assertIsNone(metrics.sharpe_ratio([0.1, 0.1], 1))  # zero std
        self.assertIsNone(metrics.sharpe_ratio([np.nan, np.nan], 1))


class TestSortinoRatio(unittest.TestCase):
    def test_known_value(self):
        returns = [0.1, -0.1, 0.1]
        value = metrics.sortino_ratio(returns, periods_per_year=1)
        self.assertEqual(value, pytest.approx(0.5774, abs=1e-4))

    def test_downside_deviation_divides_by_total_observations(self):
        # [0.1, -0.1, 0.1]: downside RMS = sqrt((0 + 0.01 + 0) / 3), not / 1.
        expected = (0.1 / 3) / np.sqrt(0.01 / 3)
        self.assertEqual(
            metrics.sortino_ratio([0.1, -0.1, 0.1], 1),
            pytest.approx(expected, abs=1e-12),
        )

    def test_all_positive_returns_have_no_downside(self):
        self.assertIsNone(metrics.sortino_ratio([0.01, 0.02, 0.03], 1))

    def test_degenerate_inputs_return_none(self):
        self.assertIsNone(metrics.sortino_ratio([], 1))
        self.assertIsNone(metrics.sortino_ratio([np.nan], 1))


class TestMaxDrawdown(unittest.TestCase):
    def test_known_value_is_positive_magnitude(self):
        self.assertEqual(metrics.max_drawdown([100, 120, 90, 110]), 0.25)

    def test_monotonic_gains_have_zero_drawdown(self):
        self.assertEqual(metrics.max_drawdown([100, 110, 120]), 0.0)

    def test_empty_curve_returns_none(self):
        self.assertIsNone(metrics.max_drawdown([]))
        self.assertIsNone(metrics.max_drawdown(pd.Series([], dtype=float)))

    def test_non_positive_running_peak_returns_none(self):
        self.assertIsNone(metrics.max_drawdown([0, 10, 5]))
        self.assertIsNone(metrics.max_drawdown([-10, -5]))


class TestCalmarRatio(unittest.TestCase):
    def test_known_value(self):
        # First/last = 100/108.9 over 3 years, max drawdown = 10%.
        value = metrics.calmar_ratio([100, 110, 99, 108.9], periods_per_year=1)
        self.assertEqual(value, pytest.approx(0.2883, abs=1e-4))

    def test_too_short_returns_none(self):
        self.assertIsNone(metrics.calmar_ratio([100], 252))
        self.assertIsNone(metrics.calmar_ratio([], 252))

    def test_no_drawdown_returns_none(self):
        self.assertIsNone(metrics.calmar_ratio([100, 110, 120], 252))

    def test_non_positive_growth_returns_none(self):
        self.assertIsNone(metrics.calmar_ratio([100, 120, 0], 1))
        self.assertIsNone(metrics.calmar_ratio([100, 120, -10], 1))

    def test_drops_nan_before_computing(self):
        value = metrics.calmar_ratio(pd.Series([100, np.nan, 110, 99, 108.9]), 1)
        self.assertEqual(value, pytest.approx(0.2883, abs=1e-4))


if __name__ == "__main__":
    unittest.main()
