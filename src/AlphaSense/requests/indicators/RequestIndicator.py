from datetime import datetime
import pandas as pd
from AlphaSense.requests.RequestData import RequestData


class RequestIndicator:
    """
    RequestIndicator
    Class to easily request indicator values given an action, a start_date an end_date and interval
    Different indicators can be requested 
    And this class can be extended to return more indicators
    action_symbol -> symbol of the stock action you want the data from
    start_date -> date you want the requested data to start from, datetime format
    end_date -> date you want the requested data to end, datetime format
    interval -> interval between two stock point (string), avaiable:
    Depends on the data_source
    data_source -> One of the data source avaiable, current data sources are:
        YAHOO, ALPHAVANTAGE
    price_data -> optional pre-fetched candles (pandas DataFrame, same shape as
        RequestData.get_price_candles_dataframe()). When provided, skips the DB/API
        fetch entirely and reuses this data instead.
    """
    def __init__(
        self,
        action_symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str,
        data_source: str = "YAHOO",
        price_data: pd.DataFrame | None=None
    ):
        if price_data is not None:
            self._df = price_data
        else:
            self._request_data = RequestData(action_symbol, start_date, end_date, interval, data_source)
            self._df = self._request_data.get_price_candles_dataframe()
   
    def _signal(self, df: pd.DataFrame, computed: pd.DataFrame) -> pd.Series:
        """ Subclasses return a Series aligned to `computed`'s index: 'Bullish', 'Bearish', or 'Neutral'. """
        raise NotImplementedError
    
    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Subclasses implement the math here. Returns one or more named columns, same index as df. """
        raise NotImplementedError

    def get_indicator_values(self) -> list[dict]:
        if self._df.empty:
            return []
        result = self._compute(self._df).dropna(how="all")
        records = result.reset_index().to_dict("records")
        for r in records:
            timestamp = pd.Timestamp(r["time"])
            if not isinstance(timestamp, pd.Timestamp):
                raise ValueError(f"Invalid time value: {r['time']}")
            r["time"] = timestamp.isoformat()
        return records
