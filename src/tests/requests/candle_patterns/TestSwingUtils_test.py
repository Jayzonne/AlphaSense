import unittest
from AlphaSense.requests.candle_patterns.swing_utils import (
    find_pivots, is_head_and_shoulders, is_inverse_head_and_shoulders,
)


def _candle(high=10, low=5, open_=7, close=7):
    return {"high": high, "low": low, "open": open_, "close": close}


class TestFindPivots(unittest.TestCase):
    def test_finds_a_single_peak_and_trough(self):
        candles = [
            _candle(high=10, low=8), _candle(high=12, low=9), _candle(high=20, low=15),
            _candle(high=12, low=9), _candle(high=10, low=3), _candle(high=12, low=9),
            _candle(high=20, low=15),
        ]
        peaks, troughs = find_pivots(candles, order=2)
        self.assertIn(2, peaks)
        self.assertIn(4, troughs)

    def test_ignores_the_order_wide_edges(self):
        # A candle within `order` of either edge can never be evaluated (no full window).
        candles = [_candle(high=100)] + [_candle(high=10) for _ in range(5)]
        peaks, _ = find_pivots(candles, order=2)
        self.assertNotIn(0, peaks)


class TestIsHeadAndShoulders(unittest.TestCase):
    def test_valid_head_and_shoulders(self):
        left_shoulder = _candle(high=110)
        head = _candle(high=130)
        right_shoulder = _candle(high=112)
        neckline_1 = _candle(low=95)
        neckline_2 = _candle(low=97)
        self.assertTrue(is_head_and_shoulders(left_shoulder, head, right_shoulder, neckline_1, neckline_2))

    def test_rejects_when_head_is_not_the_highest(self):
        left_shoulder = _candle(high=130)
        head = _candle(high=120)
        right_shoulder = _candle(high=112)
        neckline_1 = _candle(low=95)
        neckline_2 = _candle(low=97)
        self.assertFalse(is_head_and_shoulders(left_shoulder, head, right_shoulder, neckline_1, neckline_2))

    def test_rejects_asymmetric_shoulders(self):
        left_shoulder = _candle(high=100)
        head = _candle(high=130)
        right_shoulder = _candle(high=125)  # way closer to the head than to the left shoulder
        neckline_1 = _candle(low=90)
        neckline_2 = _candle(low=91)
        self.assertFalse(is_head_and_shoulders(left_shoulder, head, right_shoulder, neckline_1, neckline_2))

    def test_rejects_sloped_neckline(self):
        left_shoulder = _candle(high=110)
        head = _candle(high=130)
        right_shoulder = _candle(high=112)
        neckline_1 = _candle(low=95)
        neckline_2 = _candle(low=60)  # way lower than the first trough
        self.assertFalse(is_head_and_shoulders(left_shoulder, head, right_shoulder, neckline_1, neckline_2))


class TestIsInverseHeadAndShoulders(unittest.TestCase):
    def test_valid_inverse_head_and_shoulders(self):
        left_shoulder = _candle(low=90)
        head = _candle(low=70)
        right_shoulder = _candle(low=88)
        neckline_1 = _candle(high=105)
        neckline_2 = _candle(high=103)
        self.assertTrue(
            is_inverse_head_and_shoulders(left_shoulder, head, right_shoulder, neckline_1, neckline_2)
        )

    def test_rejects_when_head_is_not_the_lowest(self):
        left_shoulder = _candle(low=70)
        head = _candle(low=80)
        right_shoulder = _candle(low=88)
        neckline_1 = _candle(high=105)
        neckline_2 = _candle(high=103)
        self.assertFalse(
            is_inverse_head_and_shoulders(left_shoulder, head, right_shoulder, neckline_1, neckline_2)
        )


if __name__ == "__main__":
    unittest.main()
