import unittest
from AlphaSense.requests.candle_patterns import *
from datetime import datetime, timedelta


class TestCandleBullishEngulfing(unittest.TestCase):
    def test_is_bullish_engulfing(self):
        request_bullish_engulfing = RequestBullishEngulfing("AAPL",
                                                            datetime.now()-timedelta(days=7),
                                                            datetime.now(),
                                                            '5m'
                                                            )
        self.assertEqual(request_bullish_engulfing._is_complex_pattern(
            [
                {'high': 25, 'low': 10, 'open': 25, 'close': 10},
                {'high': 30, 'low': 8, 'open': 8, 'close': 30},
                {}
            ]), True
        )

    def test_is_not_bullish_engulfing_first_candle_is_bull(self):
        request_bullish_engulfing = RequestBullishEngulfing("AAPL",
                                                            datetime.now()-timedelta(days=7),
                                                            datetime.now(),
                                                            '5m'
                                                            )
        self.assertEqual(request_bullish_engulfing._is_complex_pattern(
            [
                {'high': 25, 'low': 10, 'open': 10, 'close': 25},
                {'high': 30, 'low': 8, 'open': 8, 'close': 30},
                {}
            ]), False
        )

    def test_is_not_bullish_engulfing_second_candle_is_bear(self):
        request_bullish_engulfing = RequestBullishEngulfing("AAPL",
                                                            datetime.now()-timedelta(days=7),
                                                            datetime.now(),
                                                            '5m'
                                                            )
        self.assertEqual(request_bullish_engulfing._is_complex_pattern(
            [
                {'high': 25, 'low': 10, 'open': 25, 'close': 10},
                {'high': 30, 'low': 8, 'open': 30, 'close': 8},
                {}
            ]), False
        )

    def test_is_not_bullish_engulfing_last_close_lower_than_first_open(self):
        request_bullish_engulfing = RequestBullishEngulfing("AAPL",
                                                            datetime.now()-timedelta(days=7),
                                                            datetime.now(),
                                                            '5m'
                                                            )
        self.assertEqual(request_bullish_engulfing._is_complex_pattern(
            [
                {'high': 25, 'low': 10, 'open': 25, 'close': 10},
                {'high': 30, 'low': 8, 'open': 8, 'close': 24},
                {}
            ]), False
        )

    def test_is_not_bullish_engulfing_last_open_higher_than_first_close(self):
        request_bullish_engulfing = RequestBullishEngulfing("AAPL",
                                                            datetime.now()-timedelta(days=7),
                                                            datetime.now(),
                                                            '5m'
                                                            )
        self.assertEqual(request_bullish_engulfing._is_complex_pattern(
            [
                {'high': 25, 'low': 10, 'open': 25, 'close': 10},
                {'high': 30, 'low': 8, 'open': 12, 'close': 30},
                {}
            ]), False
        )



