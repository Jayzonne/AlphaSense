from .GenericAPI import GenericAPI
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Dict
import json

class YahooAPI(GenericAPI):
    """
    YahooAPI Class
    Class for requesting Yahoo Stock market
    start_date -> datetime format, month from which the data start
    end_date -> datetime format, month from which the data end (EXCLUDED)
    interval -> interval between two stock point (string), avaiable:
        1min, 5min, 15min, 30min, 60min
    action_symbol -> symbol of the stock action you want the data from
    """

    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        interval: str,
        action_symbol: str,

    ):
        self._interval_authorized_values = ["1m",
                                            "2m",
                                            "5m",
                                            "15m",
                                            "30m",
                                            "60m",
                                            "90m",
                                            "1h",
                                            "1d",
                                            "5d",
                                            "1wk",
                                            "1mo",
                                            "3mo",
                                            ]
        self._start_date = start_date
        self._end_date = end_date
        self.interval = interval
        self.action_symbol = action_symbol

    def get_json_api(self) -> Dict:
        data = yf.download(self.action_symbol,
                           interval=self.interval,
                           start=f"{str(self._start_date.year)}-\
{str(self._start_date.month).zfill(2)}-\
{str(self._start_date.day).zfill(2)}",
                           end=f"{str(self._end_date.year)}-\
{str(self._end_date.month).zfill(2)}-\
{str(self._end_date.day).zfill(2)}"
                           )
        if data is None:
            raise ValueError("Error when requesting yahoo API, \
                    resulting data is empty")
        data.index = pd.to_datetime(data.index).strftime("%Y-%m-%d %H:%M:%S")
        return json.loads(
                data.to_json(orient="index") or "{}")

    def get_standard_json(self) -> Dict:
        return self.get_json_api()
