# AlphaSense/requests/indicators/RequestStochastic.py
import pandas as pd
from AlphaSense.requests.indicators.RequestIndicator import RequestIndicator


class RequestStochastic(RequestIndicator):
    def __init__(self, *args, period: int = 14, smooth_d: int = 3, **kwargs):
        self._period = period
        self._smooth_d = smooth_d
        super().__init__(*args, **kwargs)

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        lowest_low = df["low"].rolling(self._period).min()
        highest_high = df["high"].rolling(self._period).max()
        percent_k = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low)
        percent_d = percent_k.rolling(self._smooth_d).mean()
        return pd.DataFrame({
            "percent_k": percent_k,
            "percent_d": percent_d
        })

    def _signal(self, df: pd.DataFrame, computed: pd.DataFrame) -> pd.Series:
        return computed["percent_k"].apply(
            lambda v: "Bullish" if v < 20 else ("Bearish" if v > 80 else "Neutral")
        )
