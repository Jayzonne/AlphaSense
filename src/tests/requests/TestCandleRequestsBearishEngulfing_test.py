import unittest
from AlphaSense.requests.candle_patterns import *
from datetime import datetime, timedelta


class TestCandleBearishEngulfing(unittest.TestCase):
    def test_is_bearish_engulfing(self):
        request_bearish_engulfing = RequestBearishEngulfing("AAPL",
                                                            datetime.now()-timedelta(days=7),
                                                            datetime.now(),
                                                            '5m'
                                                            )
        self.assertEqual(request_bearish_engulfing._is_complex_pattern(
            [
                {'high': 25, 'low': 10, 'open': 10, 'close': 25},
                {'high': 30, 'low': 8, 'open': 30, 'close': 8},
                {}
            ]), True
        )

    def test_is_not_bearish_engulfing_first_candle_is_bear(self):
        request_bearish_engulfing = RequestBearishEngulfing("AAPL",
                                                            datetime.now()-timedelta(days=7),
                                                            datetime.now(),
                                                            '5m'
                                                            )
        self.assertEqual(request_bearish_engulfing._is_complex_pattern(
            [
                {'high': 25, 'low': 10, 'open': 25, 'close': 10},
                {'high': 30, 'low': 8, 'open': 30, 'close': 8},
                {}
            ]), False
        )

    def test_is_not_bearish_engulfing_second_candle_is_bull(self):
        request_bearish_engulfing = RequestBearishEngulfing("AAPL",
                                                            datetime.now()-timedelta(days=7),
                                                            datetime.now(),
                                                            '5m'
                                                            )
        self.assertEqual(request_bearish_engulfing._is_complex_pattern(
            [
                {'high': 25, 'low': 10, 'open': 10, 'close': 25},
                {'high': 30, 'low': 8, 'open': 8, 'close': 30},
                {}
            ]), False
        )

    def test_is_not_bearish_engulfing_last_close_higher_than_first_open(self):
        request_bearish_engulfing = RequestBearishEngulfing("AAPL",
                                                            datetime.now()-timedelta(days=7),
                                                            datetime.now(),
                                                            '5m'
                                                            )
        self.assertEqual(request_bearish_engulfing._is_complex_pattern(
            [
                {'high': 25, 'low': 10, 'open': 10, 'close': 25},
                {'high': 30, 'low': 8, 'open': 30, 'close': 12},
                {}
            ]), False
        )

    def test_is_not_bearish_engulfing_last_open_lower_than_first_close(self):
        request_bearish_engulfing = RequestBearishEngulfing("AAPL",
                                                            datetime.now()-timedelta(days=7),
                                                            datetime.now(),
                                                            '5m'
                                                            )
        self.assertEqual(request_bearish_engulfing._is_complex_pattern(
            [
                {'high': 25, 'low': 10, 'open': 10, 'close': 25},
                {'high': 30, 'low': 8, 'open': 24, 'close': 8},
                {}
            ]), False
        )



