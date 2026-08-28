import unittest
from AlphaSense.requests.candle_patterns import *
from datetime import datetime, timedelta


class TestCandleRequestsShootingStar(unittest.TestCase):
    def test_is_shooting_star(self):
        request_hammer = RequestShootingStar("AAPL",
                                             datetime.now()-timedelta(days=7),
                                             datetime.now(),
                                             '5m'
                                             )
        self.assertEqual(request_hammer._is_pattern(
            {'high': 25, 'low': 10, 'open': 10, 'close': 12}), True
        )

    def test_is_not_shooting_star_lower_wick_to_big(self):
        request_hammer = RequestShootingStar("AAPL",
                                             datetime.now()-timedelta(days=7),
                                             datetime.now(),
                                             '5m'
                                             )
        self.assertEqual(request_hammer._is_pattern(
            {'high': 25, 'low': 7, 'open': 10, 'close': 12}), False
        )

    def test_is_not_shooting_star_upper_wick_to_small(self):
        request_hammer = RequestShootingStar("AAPL",
                                             datetime.now()-timedelta(days=7),
                                             datetime.now(),
                                             '5m'
                                             )
        self.assertEqual(request_hammer._is_pattern(
            {'high': 25, 'low': 7, 'open': 10, 'close': 19}), False
        )

    def test_is_not_shooting_star_zero_body(self):
        request_hammer = RequestShootingStar("AAPL",
                                             datetime.now()-timedelta(days=7),
                                             datetime.now(),
                                             '5m'
                                             )
        self.assertEqual(request_hammer._is_pattern(
            {'high': 25, 'low': 25, 'open': 25, 'close': 25}), False
        )




