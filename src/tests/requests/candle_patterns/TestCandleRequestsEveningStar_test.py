import unittest
from AlphaSense.requests.candle_patterns import *
from datetime import datetime, timedelta


class TestCandleEveningStar(unittest.TestCase):
    def test_is_evening_star(self):
        request_evening_star = RequestEveningStar("AAPL",
                                                  datetime.now()-timedelta(days=7),
                                                  datetime.now(),
                                                  '5m'
                                                  )
        self.assertEqual(request_evening_star._is_complex_pattern(
            [
                {'high': 25, 'low': 10, 'open': 10, 'close': 25},
                {'high': 26, 'low': 25, 'open': 25, 'close': 26},
                {'high': 26, 'low': 11, 'open': 26, 'close': 11}
            ]), True
        )

    def test_is_not_evening_star_first_candle_is_bear(self):
        request_evening_star = RequestEveningStar("AAPL",
                                                  datetime.now()-timedelta(days=7),
                                                  datetime.now(),
                                                  '5m'
                                                  )
        self.assertEqual(request_evening_star._is_complex_pattern(
            [
                {'high': 25, 'low': 10, 'open': 25, 'close': 10},
                {'high': 26, 'low': 25, 'open': 25, 'close': 26},
                {'high': 26, 'low': 11, 'open': 26, 'close': 11}
            ]), False
        )

    def test_is_not_evening_star_third_candle_is_bull(self):
        request_evening_star = RequestEveningStar("AAPL",
                                                  datetime.now()-timedelta(days=7),
                                                  datetime.now(),
                                                  '5m'
                                                  )
        self.assertEqual(request_evening_star._is_complex_pattern(
            [
                {'high': 25, 'low': 10, 'open': 10, 'close': 25},
                {'high': 26, 'low': 25, 'open': 25, 'close': 26},
                {'high': 26, 'low': 11, 'open': 11, 'close': 26}
            ]), False
        )

    def test_is_not_evening_star_second_candle_is_too_big(self):
        request_evening_star = RequestEveningStar("AAPL",
                                                  datetime.now()-timedelta(days=7),
                                                  datetime.now(),
                                                  '5m'
                                                  )
        self.assertEqual(request_evening_star._is_complex_pattern(
            [
                {'high': 25, 'low': 10, 'open': 10, 'close': 25},
                {'high': 30, 'low': 25, 'open': 25, 'close': 30},
                {'high': 26, 'low': 11, 'open': 26, 'close': 11}
            ]), False
        )

    def test_is_not_evening_star_last_candle_close_too_high(self):
        request_evening_star = RequestEveningStar("AAPL",
                                                  datetime.now()-timedelta(days=7),
                                                  datetime.now(),
                                                  '5m'
                                                  )
        self.assertEqual(request_evening_star._is_complex_pattern(
            [
                {'high': 25, 'low': 10, 'open': 10, 'close': 25},
                {'high': 26, 'low': 25, 'open': 25, 'close': 26},
                {'high': 26, 'low': 11, 'open': 26, 'close': 20}
            ]), False
        )



