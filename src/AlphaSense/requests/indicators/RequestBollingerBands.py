# AlphaSense/requests/indicators/RequestBollingerBands.py
import pandas as pd
from AlphaSense.requests.indicators.RequestIndicator import RequestIndicator


class RequestBollingerBands(RequestIndicator):
    def __init__(self, *args, period: int = 20, std_dev: int = 2, **kwargs):
        self._period = period
        self._std_dev = std_dev
        super().__init__(*args, **kwargs)

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        middle = df["close"].rolling(self._period).mean()
        std = df["close"].rolling(self._period).std()
        upper = middle + self._std_dev * std
        lower = middle - self._std_dev * std
        return pd.DataFrame({
            "upper_band": upper,
            "middle_band": middle,
            "lower_band": lower
        })
    
    def _signal(self, df: pd.DataFrame, computed: pd.DataFrame) -> pd.Series:
        def label(close_value, lower, upper):
            if close_value <= lower:
                return "Bullish"
            if close_value >= upper:
                return "Bearish"
            return "Neutral"
        return pd.Series(
            [label(c, l, u) for c, l, u in zip(df["close"], computed["lower_band"], computed["upper_band"])],
            index=computed.index
        )
