import unittest
from datetime import datetime, timedelta

from AlphaSense.requests.candle_patterns import RequestHeadAndShoulders, RequestInverseHeadAndShoulders


def _build_candles(high_path: list[float], low_path: list[float]) -> list[dict]:
    """ Turns two parallel high/low paths into a synthetic candle list, evenly spaced 5 minutes apart. """
    start = datetime(2026, 1, 1, 9, 30)
    candles = []
    for i, (high, low) in enumerate(zip(high_path, low_path)):
        mid = (high + low) / 2
        candles.append({
            "time": start + timedelta(minutes=5 * i), "symbol": "TEST",
            "open": mid, "close": mid, "high": high, "low": low, "volume": 1000,
        })
    return candles


def _flat_padding(n: int, high: float = 100, low: float = 90) -> tuple[list[float], list[float]]:
    return [high] * n, [low] * n


class TestHeadAndShoulders(unittest.TestCase):
    def test_detects_a_clean_head_and_shoulders(self):
        pad_h, pad_l = _flat_padding(4)
        # left shoulder (110) -> neckline (95) -> head (130) -> neckline (96) -> right shoulder (111) -> decline
        shape_high = [100, 105, 110, 105, 100, 95, 100, 115, 130, 115, 100, 96, 100, 105, 111, 105, 100]
        shape_low = [d - 10 for d in shape_high]
        highs = pad_h + shape_high + pad_h
        lows = pad_l + shape_low + pad_l
        candles = _build_candles(highs, lows)

        matches = RequestHeadAndShoulders(
            "TEST", datetime.now() - timedelta(days=1), datetime.now(), "5m", price_data=candles,
        ).get_pattern()

        self.assertTrue(len(matches) >= 1, "expected at least one Head & Shoulders match")
        # the right shoulder's high (111) should be among the reported candles
        self.assertTrue(any(c["high"] == 111 for c in matches))

    def test_no_false_positive_on_a_flat_series(self):
        highs, lows = _flat_padding(30)
        candles = _build_candles(highs, lows)
        matches = RequestHeadAndShoulders(
            "TEST", datetime.now() - timedelta(days=1), datetime.now(), "5m", price_data=candles,
        ).get_pattern()
        self.assertEqual(matches, [])

    def test_no_false_positive_on_a_steady_uptrend(self):
        # Three consecutive higher highs is NOT a head and shoulders (the head must
        # be higher than BOTH shoulders, which are roughly level with each other).
        highs = [100 + i for i in range(30)]
        lows = [h - 10 for h in highs]
        candles = _build_candles(highs, lows)
        matches = RequestHeadAndShoulders(
            "TEST", datetime.now() - timedelta(days=1), datetime.now(), "5m", price_data=candles,
        ).get_pattern()
        self.assertEqual(matches, [])


class TestInverseHeadAndShoulders(unittest.TestCase):
    def test_detects_a_clean_inverse_head_and_shoulders(self):
        pad_h, pad_l = _flat_padding(4)
        shape_low = [90, 85, 80, 85, 90, 95, 90, 75, 60, 75, 90, 94, 90, 85, 79, 85, 90]
        shape_high = [d + 10 for d in shape_low]
        highs = pad_h + shape_high + pad_h
        lows = pad_l + shape_low + pad_l
        candles = _build_candles(highs, lows)

        matches = RequestInverseHeadAndShoulders(
            "TEST", datetime.now() - timedelta(days=1), datetime.now(), "5m", price_data=candles,
        ).get_pattern()

        self.assertTrue(len(matches) >= 1, "expected at least one Inverse Head & Shoulders match")
        self.assertTrue(any(c["low"] == 79 for c in matches))

    def test_no_false_positive_on_a_flat_series(self):
        highs, lows = _flat_padding(30)
        candles = _build_candles(highs, lows)
        matches = RequestInverseHeadAndShoulders(
            "TEST", datetime.now() - timedelta(days=1), datetime.now(), "5m", price_data=candles,
        ).get_pattern()
        self.assertEqual(matches, [])


if __name__ == "__main__":
    unittest.main()
