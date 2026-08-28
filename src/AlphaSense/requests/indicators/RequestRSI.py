import pandas as pd
from AlphaSense.requests.indicators.RequestIndicator import RequestIndicator


# RequestRSI.py
class RequestRSI(RequestIndicator):
    def __init__(self, *args, period: int = 14, **kwargs):
        self._period = period
        super().__init__(*args, **kwargs)

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        deltas = df["close"].diff()
        gains = deltas.clip(lower=0).rolling(self._period).mean()
        losses = (-deltas.clip(upper=0)).rolling(self._period).mean()
        rs = gains / losses
        return pd.DataFrame({"rsi": 100 - (100 / (1 + rs))})

    def _signal(self, df: pd.DataFrame, computed: pd.DataFrame) -> pd.Series:
        return computed["rsi"].apply(
            lambda v: "Bullish" if v < 30 else ("Bearish" if v > 70 else "Neutral")
        )
