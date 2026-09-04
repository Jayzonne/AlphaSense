import numpy as np
import pandas as pd
from datetime import datetime
from AlphaSense.requests.RequestData import RequestData


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
    Converts a Yahoo-style interval string ('1m', '5m', '1h', '1d', ...) into minutes.
    Long-form intervals ('1wk', '1mo', '3mo') aren't a fixed duration - pandas
    raises on those, which the calling callback's try/except turns into an
    error banner rather than a hard crash.
    """
    return pd.Timedelta(interval).total_seconds() / 60
