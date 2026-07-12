from .GenericAPI import GenericAPI
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Dict
import json
import ast


class YahooAPI(GenericAPI):
    """
    YahooAPI Class
    Class for requesting Yahoo Stock market
    start_date -> datetime format, month from which the data start
    end_date -> datetime format, month from which the data end (EXCLUDED)
    interval -> interval between two stock point (string), avaiable:
        1m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
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
        self._interval = interval
        self._action_symbol = action_symbol

    def get_json_api(self) -> Dict:
        data = yf.download(self._action_symbol,
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
        json_api_data = self.get_json_api()
        json_api_data_formated = {}
        for date, data in json_api_data.items():
            new_data = {}
            for tag, value in data.items():
                tag_tuple = ast.literal_eval(tag)
                new_data[tag_tuple[0].lower()] = value
            json_api_data_formated[date] = new_data
        json_api_data_with_symbol = {}
        json_api_data_with_symbol["symbol"] = self._action_symbol
        json_api_data_with_symbol["data"] = json_api_data_formated

        return json_api_data_with_symbol
