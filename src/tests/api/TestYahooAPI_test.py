import unittest
from datetime import datetime

from AlphaSense.api.YahooAPI import YahooAPI


class TestFormatYahooDate(unittest.TestCase):
    def test_no_offset(self):
        self.assertEqual(YahooAPI._format_yahoo_date(datetime(2026, 3, 15)), "2026-03-15")

    def test_offset_within_month(self):
        self.assertEqual(YahooAPI._format_yahoo_date(datetime(2026, 3, 15), offset_days=1), "2026-03-16")

    def test_offset_across_month_boundary(self):
        # The original inline version reused the pre-offset month/year with
        # only the day advanced, producing "2026-01-01" here instead of the
        # correct "2026-02-01".
        self.assertEqual(YahooAPI._format_yahoo_date(datetime(2026, 1, 31), offset_days=1), "2026-02-01")

    def test_offset_across_year_boundary(self):
        self.assertEqual(YahooAPI._format_yahoo_date(datetime(2026, 12, 31), offset_days=1), "2027-01-01")

    def test_offset_across_leap_day(self):
        self.assertEqual(YahooAPI._format_yahoo_date(datetime(2028, 2, 28), offset_days=1), "2028-02-29")

    def test_zero_padding(self):
        self.assertEqual(YahooAPI._format_yahoo_date(datetime(2026, 1, 5)), "2026-01-05")


if __name__ == "__main__":
    unittest.main()
