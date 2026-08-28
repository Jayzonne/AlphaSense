from datetime import datetime
import pandas as pd
from AlphaSense.requests.indicators import INDICATOR_REGISTRY
from AlphaSense.requests.candle_patterns import PATTERN_REGISTRY


def compute_confluence(symbol: str, start: datetime, end: datetime, interval: str, price_data: pd.DataFrame) -> dict:
    """
    For every candle in [start, end], counts how many indicators AND candle
    patterns signal Bullish vs Bearish. Always evaluates every registered
    indicator/pattern, independent of what's toggled visible in the UI.
    """
    confluence: dict[str, dict] = {}

    for entry in INDICATOR_REGISTRY.values():
        instance = entry["class"](symbol, start, end, interval, price_data=price_data)
        if instance._df.empty:
            continue
        computed = instance._compute(instance._df)
        signal = instance._signal(instance._df, computed)
        for ts, label in signal.items():
            if label == "Neutral":
                continue
            record = confluence.setdefault(pd.Timestamp(ts).isoformat(), {"bullish": [], "bearish": []})
            record["bullish" if label == "Bullish" else "bearish"].append(entry["label"])

    price_candles_list = price_data.reset_index().to_dict("records")  # matches get_price_candles() shape

    for entry in PATTERN_REGISTRY.values():
        try:
            instances = entry["class"](symbol, start, end, interval, price_data=price_candles_list).get_pattern()
        except Exception:
            continue
        for candle in instances:
            if "time" not in candle:
                continue
            ts = pd.Timestamp(candle["time"]).isoformat()
            record = confluence.setdefault(ts, {"bullish": [], "bearish": []})
            record["bullish" if entry["direction"] == "Bullish" else "bearish"].append(entry["label"])

    return {
        ts: {
            "bullish_count": len(r["bullish"]), "bearish_count": len(r["bearish"]),
            "bullish_sources": r["bullish"], "bearish_sources": r["bearish"],
        }
        for ts, r in confluence.items()
    }
