# tests/requests/TestIndicatorRequestsStochastic_test.py
import unittest
import pandas as pd
from AlphaSense.requests.indicators import *
from datetime import datetime, timedelta


class TestIndicatorRequestsStochastic(unittest.TestCase):
    def test_stochastic_percent_k_and_d_values(self):
        request_stochastic = RequestStochastic("AAPL",
                                                datetime.now()-timedelta(days=7),
                                                datetime.now(),
                                                '5m',
                                                period=3, smooth_d=2
                                                )
        result = request_stochastic._compute(pd.DataFrame({
            'high':  [12, 14, 13, 15, 16, 14],
            'low':   [10, 11, 10, 12, 13, 11],
            'close': [11, 13, 12, 14.5, 15, 12.5]
        }))
        self.assertAlmostEqual(result['percent_k'].iloc[-1], 30.0)
        self.assertAlmostEqual(result['percent_d'].iloc[-1], 56.666666666666664)

    def test_stochastic_close_at_top_of_range(self):
        request_stochastic = RequestStochastic("AAPL",
                                                datetime.now()-timedelta(days=7),
                                                datetime.now(),
                                                '5m',
                                                period=3, smooth_d=2
                                                )
        result = request_stochastic._compute(pd.DataFrame({
            'high':  [12, 12, 12],
            'low':   [10, 10, 10],
            'close': [10, 11, 12]
        }))
        self.assertAlmostEqual(result['percent_k'].iloc[-1], 100.0)
