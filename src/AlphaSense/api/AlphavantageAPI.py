#!/usr/bin/env python
import requests
from datetime import datetime
from dateutil import rrule
from dotenv import load_dotenv, find_dotenv
import os
from .GenericAPI import GenericAPI
from typing import Dict
from zoneinfo import ZoneInfo


class AlphavantageAPI(GenericAPI):
    """
    AlphavantageAPI class (Now unmaintained, since they only do TimeSeries with
    Premium subscription)
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
        self._interval_authorized_values = ["1min", "5min", "15min", "30min", "60min"]

        self.interval = interval
        self._action_symbol = action_symbol
        load_dotenv(find_dotenv())
        self._api_token = os.getenv("ALPHAVANTAGE_API_TOKEN")
        self._url = "https://www.alphavantage.co"

    def get_json_api(self) -> Dict:
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
&extended_hours=false\
&apikey={self._api_token}"
            request_result = requests.get(request_url)
            month_data = request_result.json()
            full_data[requested_month] = month_data

        return full_data

    def get_standard_json(self) -> Dict:
        json_api_data = self.get_json_api()
        request_month_list = []
        full_data_list = {}

        # Get list of month requested in order to agregate the data in
        # a single dictionnary
        for month_to_request in rrule.rrule(
            rrule.MONTHLY, dtstart=self._start_date, until=self._end_date
        ):
            request_month_list.append(
                f"{str(month_to_request.year)}-\
{str(month_to_request.month).zfill(2)}"
            )
        # Get Time Series string to extract data from this dictionnary entry
        time_series_string = f"Time Series ({self.interval})"
        key_eastern_timezone_list = []
        # Get origin timezone for UTC conversion
        try:
            origin_timezone = json_api_data[request_month_list[0]]["Meta Data"][
                "6. Time Zone"
            ]
        except Exception:
            raise Exception("Json data is invalid, verify your dates and verify that you API key is valid")
        for month in request_month_list:
            full_data_list.update(json_api_data[month][time_series_string])
        # Replace old dictionnary keys with new one
        for key in full_data_list:
            key_eastern_timezone_list.append(key)
            full_data_list[key]["Open"] = full_data_list[
                key
            ].pop("1. open")
            full_data_list[key]["Close"] = full_data_list[
                key
            ].pop("4. close")
            full_data_list[key]["High"] = full_data_list[
                key
            ].pop("2. high")
            full_data_list[key]["Low"] = full_data_list[
                key
            ].pop("3. low")
            full_data_list[key]["Volume"] = (
                full_data_list[key].pop("5. volume")
            )
        # Convert hours into UTC
        for key_date in key_eastern_timezone_list:
            converted_key = (
                datetime.fromisoformat(key_date)
                .replace(tzinfo=ZoneInfo(origin_timezone))
                .astimezone(ZoneInfo("UTC"))
            )
            full_data_list[converted_key.strftime("%Y-%m-%d %H:%M:%S")] = (
                full_data_list.pop(key_date)
            )
        full_data_list_with_symbol = {}
        full_data_list_with_symbol["symbol"] = self._action_symbol
        full_data_list_with_symbol["data"] = full_data_list
        return full_data_list_with_symbol
