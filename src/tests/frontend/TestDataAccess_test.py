import unittest
from datetime import datetime

from AlphaSense.frontend.data_access import resolve_range_preset, RANGE_PRESETS


class TestResolveRangePreset(unittest.TestCase):
    def test_unknown_preset_returns_none(self):
        self.assertIsNone(resolve_range_preset("NOT-A-PRESET"))

    def test_every_registered_preset_resolves(self):
        now = datetime(2026, 9, 12)
        for key in RANGE_PRESETS:
            resolved = resolve_range_preset(key, now=now)
            self.assertIsNotNone(resolved, f"{key} should resolve")
            start, end, interval = resolved
            self.assertLess(start, end)
            self.assertIsInstance(interval, str)

    def test_ytd_starts_on_january_first_of_the_current_year(self):
        now = datetime(2026, 9, 12)
        start, end, interval = resolve_range_preset("YTD", now=now)
        self.assertEqual(start, datetime(2026, 1, 1))
        self.assertEqual(end, now)
        self.assertEqual(interval, "1d")

    def test_1d_preset_is_a_one_day_span_at_fine_granularity(self):
        now = datetime(2026, 9, 12)
        start, end, interval = resolve_range_preset("1D", now=now)
        self.assertEqual((end - start).days, 1)
        self.assertEqual(interval, "5m")

    def test_5y_and_max_use_the_newly_fixed_wk_mo_intervals(self):
        _, _, interval_5y = resolve_range_preset("5Y", now=datetime(2026, 9, 12))
        _, _, interval_max = resolve_range_preset("MAX", now=datetime(2026, 9, 12))
        self.assertEqual(interval_5y, "1wk")
        self.assertEqual(interval_max, "1mo")


if __name__ == "__main__":
    unittest.main()
