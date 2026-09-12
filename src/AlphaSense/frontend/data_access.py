import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from AlphaSense.requests.RequestData import RequestData, interval_to_minutes


def get_price_candles_dataframe(
    symbol: str,
    start: datetime,
    end: datetime,
    interval: str,
    data_source: str = "YAHOO"
) -> pd.DataFrame:
    """
    Fetches OHLCV data as a DataFrame indexed by time.

    A fresh RequestData is created per call rather than reusing one shared,
    mutable instance across callbacks. Dash can invoke callbacks concurrently,
    and mutating one shared instance's symbol/date/interval fields from two
    callbacks racing on the same inputs is a real source of stale or
    incorrect reads - creating a new instance sidesteps the problem entirely.
    """
    return RequestData(symbol, start, end, interval, data_source).get_price_candles_dataframe()


def get_authorized_intervals(data_source: str = "YAHOO") -> list[str]:
    """
    Used once at startup to populate the interval dropdown. Needs a throwaway
    RequestData instance since the authorized values live on the underlying
    API class (YahooAPI._interval_authorized_values), not as a standalone constant.
    """
    probe = RequestData("AAPL", datetime.now(), datetime.now(), "1d", data_source)
    return probe.get_authorized_intervals()


def dataframe_to_records(df: pd.DataFrame) -> list[dict]:
    """
    Serializes a time-indexed DataFrame into plain dicts so it can be handed
    to a dcc.Store (which round-trips through JSON in the browser). NaN is
    swapped for None first: Python's json encoder emits NaN as a bare `NaN`
    token, which is not valid JSON and makes the browser's JSON.parse throw.
    """
    if df.empty:
        return []
    safe = df.replace({np.nan: None})
    records = safe.reset_index().to_dict("records")
    for r in records:
        r["time"] = pd.Timestamp(r["time"]).isoformat()
    return records


def dataframe_from_records(records: list[dict] | None) -> pd.DataFrame:
    """ Inverse of dataframe_to_records(). """
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    return df


def to_minutes(interval: str) -> float:
    """
    Converts a Yahoo-style interval string ('1m', '5m', '1h', '1d', ...) into
    minutes. Delegates to RequestData.interval_to_minutes(), which also
    handles '1wk'/'1mo'/'3mo' - those used to raise inside pd.Timedelta
    (invalid unit abbreviation), silently turned into an error banner by the
    calling callback's try/except rather than a hard crash. Now fixed at the
    source instead of just contained here.
    """
    return interval_to_minutes(interval)


# (span, interval) per named time-range preset. Longer spans step down to
# coarser intervals so a chart (or a backtest) doesn't try to work with e.g.
# a year of 5-minute candles. "1wk" and "1mo" only became safe to use here
# after fixing interval_to_minutes() above - they used to crash
# _data_exists()/_query_price_candle(). Shared between the Dash UI's preset
# buttons and the backtest CLI's --preset flag, so "YTD" means exactly the
# same date range and interval in both places.
RANGE_PRESETS = {
    "1D": (timedelta(days=1), "5m"),
    "1W": (timedelta(days=7), "30m"),
    "1M": (timedelta(days=30), "1h"),
    "YTD": (None, "1d"),  # start of the current calendar year - handled specially below
    "1Y": (timedelta(days=365), "1d"),
    "5Y": (timedelta(days=5 * 365), "1wk"),
    "MAX": (timedelta(days=20 * 365), "1mo"),  # RequestData has no "earliest available" concept, so this is a generous proxy for it rather than a true max
}


def resolve_range_preset(preset_key: str, now: datetime = None) -> tuple[datetime, datetime, str] | None:
    """ Resolves a RANGE_PRESETS key into (start, end, interval), or None if the key isn't recognized. """
    if preset_key not in RANGE_PRESETS:
        return None
    span, interval = RANGE_PRESETS[preset_key]
    end = now or datetime.now()
    start = datetime(end.year, 1, 1) if span is None else end - span
    return start, end, interval
