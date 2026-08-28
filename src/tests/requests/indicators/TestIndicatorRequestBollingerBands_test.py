import unittest
import pandas as pd
from AlphaSense.requests.indicators import *
from datetime import datetime, timedelta


class TestIndicatorRequestsBollingerBands(unittest.TestCase):
    def test_bollinger_bands_values(self):
        request_bbands = RequestBollingerBands("AAPL",
                                                datetime.now()-timedelta(days=7),
                                                datetime.now(),
                                                '5m',
                                                period=3, std_dev=2
                                                )
        result = request_bbands._compute(pd.DataFrame({'close': [10, 12, 11, 14, 13, 15]}))
        self.assertAlmostEqual(result['upper_band'].iloc[-1], 16.0)
        self.assertAlmostEqual(result['middle_band'].iloc[-1], 14.0)
        self.assertAlmostEqual(result['lower_band'].iloc[-1], 12.0)

    def test_bollinger_bands_flat_prices_bands_collapse(self):
        request_bbands = RequestBollingerBands("AAPL",
                                                datetime.now()-timedelta(days=7),
                                                datetime.now(),
                                                '5m',
                                                period=3, std_dev=2
                                                )
        result = request_bbands._compute(pd.DataFrame({'close': [10, 10, 10, 10]}))
        self.assertAlmostEqual(result['upper_band'].iloc[-1], 10.0)
        self.assertAlmostEqual(result['middle_band'].iloc[-1], 10.0)
        self.assertAlmostEqual(result['lower_band'].iloc[-1], 10.0)
