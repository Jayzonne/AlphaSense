#!/usr/bin/env python
import requests
from datetime import datetime
from dateutil import rrule
from dotenv import load_dotenv, find_dotenv
import os
from typing import Dict


class AlphavantageAPI:
    """
    AlphavantageAPI class
    Allow requesting alphavantage API to get intraday historical information
    between two months
    We are not responsible for API limits, if issues are present, please by an
    API key from alphavantage
    Class initialization take the following arguments
    start_date -> datetime format, month from which the data start
    end_date -> datetime format, month from which the data end
    interval -> interval between two stock point (string), avaiable:
        1min, 5min, 15min, 30min, 60min
    action_symbol -> symbol of the stock action you want the data from
    """

    _url = "https://www.alphavantage.co"
    _start_date = None
    _end_date = None
    _interval = ""
    _action_symbol = None
    _api_token = None
    __interval_authorized_values = ["1min", "5min", "15min", "30min", "60min"]

    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        interval: str,
        action_symbol: str,
    ):
        # Start date is a month in format YYYY-MM, idem for end_date
        self._start_date = start_date
        self._end_date = end_date
        self.interval = interval
        self._action_symbol = action_symbol
        load_dotenv(find_dotenv())
        self._api_token = os.getenv("ALPHAVANTAGE_API_TOKEN")

    @property
    def interval(self) -> str:
        return self._interval

    @interval.setter
    def interval(self, interval: str) -> None:
        if interval not in self.__interval_authorized_values:
            raise ValueError(
                f"Interval must have one of the following value:\
{self.__interval_authorized_values}"
            )
        self._interval = interval

    def get_json_api(self) -> Dict:
        """
        This function will request the alphavantage API
        It will return a json containing all months requested informations
        under (python Dict format)
        """
        full_data = {}
        for month_to_request in rrule.rrule(
            rrule.MONTHLY, dtstart=self._start_date, until=self._end_date
        ):
            requested_month = f"{str(month_to_request.year)}-\
{str(month_to_request.month).zfill(2)}"
            request_url = f"{self._url}/query?function=TIME_SERIES_INTRADAY\
&symbol={self._action_symbol}\
&interval={self._interval}\
&month={requested_month}\
&outputsize=full\
&adjusted=false\
&apikey={self._api_token}"
            request_result = requests.get(request_url)
            month_data = request_result.json()
            full_data[requested_month] = month_data
        return full_data
