import unittest
from AlphaSense.requests.candle_patterns import *
from datetime import datetime, timedelta


class TestCandleRequestsHammer(unittest.TestCase):
    def test_is_hammer(self):
        request_hammer = RequestHammers("AAPL",
                                        datetime.now()-timedelta(days=7),
                                        datetime.now(),
                                        '5m'
                                        )
        self.assertEqual(request_hammer._is_pattern(
            {'high': 25, 'low': 10, 'open': 25, 'close': 24}), True
        )

    def test_is_not_hammer_upper_wick_to_big(self):
        request_hammer = RequestHammers("AAPL",
                                        datetime.now()-timedelta(days=7),
                                        datetime.now(),
                                        '5m'
                                        )
        self.assertEqual(request_hammer._is_pattern(
            {'high': 28, 'low': 10, 'open': 25, 'close': 24}), False
        )

    def test_is_not_hammer_lower_wick_to_small(self):
        request_hammer = RequestHammers("AAPL",
                                        datetime.now()-timedelta(days=7),
                                        datetime.now(),
                                        '5m'
                                        )
        self.assertEqual(request_hammer._is_pattern(
            {'high': 25, 'low': 23, 'open': 25, 'close': 24}), False
        )

    def test_is_not_hammer_zero_body(self):
        request_hammer = RequestHammers("AAPL",
                                        datetime.now()-timedelta(days=7),
                                        datetime.now(),
                                        '5m'
                                        )
        self.assertEqual(request_hammer._is_pattern(
            {'high': 25, 'low': 25, 'open': 25, 'close': 25}), False
        )




