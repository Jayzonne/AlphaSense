# tests/requests/TestIndicatorRequestsMACD_test.py
import unittest
import pandas as pd
from AlphaSense.requests.indicators import *
from datetime import datetime, timedelta


class TestIndicatorRequestsMACD(unittest.TestCase):
    def test_macd_line_value(self):
        request_macd = RequestMACD("AAPL",
                                   datetime.now()-timedelta(days=7),
                                   datetime.now(),
                                   '5m',
                                   fast=2, slow=4, signal=2
                                   )
        result = request_macd._compute(pd.DataFrame({'close': [10, 11, 12, 11, 13, 12]}))
        self.assertAlmostEqual(result['macd'].iloc[-1], 0.15354664132859597)

    def test_macd_signal_value(self):
        request_macd = RequestMACD("AAPL",
                                   datetime.now()-timedelta(days=7),
                                   datetime.now(),
                                   '5m',
                                   fast=2, slow=4, signal=2
                                   )
        result = request_macd._compute(pd.DataFrame({'close': [10, 11, 12, 11, 13, 12]}))
        self.assertAlmostEqual(result['signal'].iloc[-1], 0.20979500609309729)

    def test_macd_histogram_value(self):
        request_macd = RequestMACD("AAPL",
                                   datetime.now()-timedelta(days=7),
                                   datetime.now(),
                                   '5m',
                                   fast=2, slow=4, signal=2
                                   )
        result = request_macd._compute(pd.DataFrame({'close': [10, 11, 12, 11, 13, 12]}))
        self.assertAlmostEqual(result['histogram'].iloc[-1], -0.05624836476450132)
