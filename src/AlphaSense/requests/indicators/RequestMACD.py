import pandas as pd
from AlphaSense.requests.indicators.RequestIndicator import RequestIndicator


# RequestMACD.py
class RequestMACD(RequestIndicator):
    def __init__(self, *args, fast=12, slow=26, signal=9, **kwargs):
        self._fast, self._slow, self._signal_value = fast, slow, signal
        super().__init__(*args, **kwargs)

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        ema_fast = df["close"].ewm(span=self._fast).mean()
        ema_slow = df["close"].ewm(span=self._slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self._signal_value).mean()
        return pd.DataFrame({
            "macd": macd_line,
            "signal": signal_line,
            "histogram": macd_line - signal_line
        })

    def _signal(self, df: pd.DataFrame, computed: pd.DataFrame) -> pd.Series:
        return computed["histogram"].apply(
            lambda v: "Bullish" if v > 0 else ("Bearish" if v < 0 else "Neutral")
        )
