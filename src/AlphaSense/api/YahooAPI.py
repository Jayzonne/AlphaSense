from .GenericAPI import GenericAPI
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
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

    @staticmethod
    def _format_yahoo_date(date: datetime, offset_days: int = 0) -> str:
        """
        Formats a date as YYYY-MM-DD for yfinance, optionally shifted by
        offset_days first (used for the end date: yfinance's `end` is
        EXCLUSIVE, so it needs +1 day to actually include it).

        Shifting via timedelta() BEFORE formatting is the part that matters:
        the previous inline version built the string from the original
        date's year/month but a separately-offset day, e.g.
        f"{end.year}-{end.month}-{(end + timedelta(days=1)).day}" - which is
        wrong (and can even be an invalid date) whenever end_date falls on
        the last day of a month, since only the day component ever actually
        advanced. Example: end_date = 2026-01-31 produced "2026-01-01"
        instead of "2026-02-01".
        """
        shifted = date + timedelta(days=offset_days)
        return f"{shifted.year}-{str(shifted.month).zfill(2)}-{str(shifted.day).zfill(2)}"
    
    def _get_daily_closes(self) -> Dict[str, float]:
        """
        Fetches Yahoo's own daily-interval close prices for the requested range,
        keyed by date string "YYYY-MM-DD". Used to correct the last intraday
        candle of each day, since Yahoo's intraday feed can report a slightly
        different value for that final bar than the officially settled close.
        """
        daily_data = yf.download(self._action_symbol,
                                 interval="1d",
                                 start=self._format_yahoo_date(self._start_date),
                                 end=self._format_yahoo_date(self._end_date),
                                 auto_adjust=False
                                 )
        if daily_data is None or daily_data.empty:
            return {}

        daily_data.index = pd.to_datetime(daily_data.index).strftime("%Y-%m-%d")
        json_daily = json.loads(daily_data.to_json(orient="index") or "{}")

        daily_closes = {}
        for date, fields in json_daily.items():
            for tag, value in fields.items():
                tag_tuple = ast.literal_eval(tag)
                if tag_tuple[0].lower() == "close":
                    daily_closes[date] = value
        return daily_closes

    def _fix_last_candle_close(self, candles: Dict) -> Dict:
        """
        Returns a new candles dict where the close of the last candle of each day
        is replaced with Yahoo's own daily close value. Only applies to intraday
        intervals (minutes/hours) - daily and longer intervals already ARE the
        daily bar, so there's nothing to correct. High/low are widened if needed
        so the corrected candle stays internally consistent (close within [low, high]).
        """
        intraday_intervals = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}
        if self._interval not in intraday_intervals or not candles:
            return candles

        daily_closes = self._get_daily_closes()
        if not daily_closes:
            return candles

        last_timestamp_per_day: Dict[str, str] = {}
        for timestamp in candles:
            day = timestamp[:10]  # "YYYY-MM-DD HH:MM:SS" -> "YYYY-MM-DD"
            if day not in last_timestamp_per_day or timestamp > last_timestamp_per_day[day]:
                last_timestamp_per_day[day] = timestamp

        fixed_candles = dict(candles)
        for day, timestamp in last_timestamp_per_day.items():
            if day not in daily_closes:
                continue
            real_close = daily_closes[day]
            candle = dict(fixed_candles[timestamp])
            candle["close"] = real_close
            candle["high"] = max(candle["high"], real_close)
            candle["low"] = min(candle["low"], real_close)
            fixed_candles[timestamp] = candle

        return fixed_candles
    
    def get_json_api(self) -> Dict:
        data = yf.download(self._action_symbol,
                           interval=self.interval,
                           start=self._format_yahoo_date(self._start_date),
                           end=self._format_yahoo_date(self._end_date, offset_days=1)
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

        json_api_data_formated=self._fix_last_candle_close(json_api_data_formated)
        json_api_data_with_symbol = {}
        json_api_data_with_symbol["symbol"] = self._action_symbol
        json_api_data_with_symbol["data"] = json_api_data_formated
        return json_api_data_with_symbol
