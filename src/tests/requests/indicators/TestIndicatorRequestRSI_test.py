
# tests/requests/TestIndicatorRequestsRSI_test.py
import unittest
import pandas as pd
from AlphaSense.requests.indicators import *
from datetime import datetime, timedelta


class TestIndicatorRequestsRSI(unittest.TestCase):
    def test_rsi_gains_only(self):
        request_rsi = RequestRSI("AAPL",
                                 datetime.now()-timedelta(days=7),
                                 datetime.now(),
                                 '5m',
                                 period=3
                                 )
        result = request_rsi._compute(pd.DataFrame({'close': [10, 11, 12, 13, 14, 15, 16]}))
        self.assertAlmostEqual(result['rsi'].iloc[-1], 100.0)

    def test_rsi_losses_only(self):
        request_rsi = RequestRSI("AAPL",
                                 datetime.now()-timedelta(days=7),
                                 datetime.now(),
                                 '5m',
                                 period=3
                                 )
        result = request_rsi._compute(pd.DataFrame({'close': [20, 19, 18, 17, 16, 15, 14]}))
        self.assertAlmostEqual(result['rsi'].iloc[-1], 0.0)

    def test_rsi_flat_prices_is_nan(self):
        request_rsi = RequestRSI("AAPL",
                                 datetime.now()-timedelta(days=7),
                                 datetime.now(),
                                 '5m',
                                 period=3
                                 )
        result = request_rsi._compute(pd.DataFrame({'close': [10, 10, 10, 10, 10]}))
        self.assertTrue(pd.isna(result['rsi'].iloc[-1]))

    def test_rsi_mixed_gains_and_losses(self):
        request_rsi = RequestRSI("AAPL",
                                 datetime.now()-timedelta(days=7),
                                 datetime.now(),
                                 '5m',
                                 period=3
                                 )
        result = request_rsi._compute(pd.DataFrame({'close': [10, 11, 9, 12, 8, 13, 10]}))
        self.assertAlmostEqual(result['rsi'].iloc[-1], 41.66666666666667)
