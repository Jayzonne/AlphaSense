import unittest
from AlphaSense.requests.candle_patterns import *
from datetime import datetime, timedelta


class TestCandleMorningStar(unittest.TestCase):
    def test_is_morning_star(self):
        request_morning_star = RequestMorningStar("AAPL",
                                                  datetime.now()-timedelta(days=7),
                                                  datetime.now(),
                                                  '5m'
                                                  )
        self.assertEqual(request_morning_star._is_complex_pattern(
            [
                {'high': 25, 'low': 10, 'open': 25, 'close': 10},
                {'high': 10, 'low': 9, 'open': 10, 'close': 9},
                {'high': 20, 'low': 10, 'open': 10, 'close': 20}
            ]), True
        )

    def test_is_not_morning_star_first_candle_is_bull(self):
        request_morning_star = RequestMorningStar("AAPL",
                                                  datetime.now()-timedelta(days=7),
                                                  datetime.now(),
                                                  '5m'
                                                  )
        self.assertEqual(request_morning_star._is_complex_pattern(
            [
                {'high': 25, 'low': 10, 'open': 10, 'close': 25},
                {'high': 10, 'low': 9, 'open': 10, 'close': 9},
                {'high': 20, 'low': 10, 'open': 10, 'close': 20}
            ]), False
        )

    def test_is_not_morning_star_third_candle_is_bear(self):
        request_morning_star = RequestMorningStar("AAPL",
                                                  datetime.now()-timedelta(days=7),
                                                  datetime.now(),
                                                  '5m'
                                                  )
        self.assertEqual(request_morning_star._is_complex_pattern(
            [
                {'high': 25, 'low': 10, 'open': 25, 'close': 10},
                {'high': 10, 'low': 9, 'open': 10, 'close': 9},
                {'high': 20, 'low': 10, 'open': 20, 'close': 10}
            ]), False
        )

    def test_is_not_morning_star_second_candle_is_too_big(self):
        request_morning_star = RequestMorningStar("AAPL",
                                                  datetime.now()-timedelta(days=7),
                                                  datetime.now(),
                                                  '5m'
                                                  )
        self.assertEqual(request_morning_star._is_complex_pattern(
            [
                {'high': 25, 'low': 10, 'open': 25, 'close': 10},
                {'high': 10, 'low': 2, 'open': 10, 'close': 2},
                {'high': 20, 'low': 10, 'open': 10, 'close': 20}
            ]), False
        )

    def test_is_not_morning_star_last_candle_close_too_low(self):
        request_morning_star = RequestMorningStar("AAPL",
                                                  datetime.now()-timedelta(days=7),
                                                  datetime.now(),
                                                  '5m'
                                                  )
        self.assertEqual(request_morning_star._is_complex_pattern(
            [
                {'high': 25, 'low': 10, 'open': 25, 'close': 10},
                {'high': 10, 'low': 9, 'open': 10, 'close': 9},
                {'high': 13, 'low': 10, 'open': 10, 'close': 13}
            ]), False
        )



