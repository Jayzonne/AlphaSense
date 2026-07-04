from datetime import datetime

from psycopg2.extras import execute_values
from AlphaSense.api import *
from dotenv import load_dotenv
import psycopg2
import os
import pandas as pd


class RequestData():
    """
    RequestData
    Class to Easily request data from any of the datasource with a single entry point
    This class also add every requested data to the TimeScaleDB database
    action_symbol -> symbol of the stock action you want the data from
    start_date -> date you want the requested data to start from, datetime format
    end_date -> date you want the requested data to end, datetime format
    interval -> interval between two stock point (string), avaiable:
    Depends on the data_source
    data_source -> One of the data source avaiable, current data sources are:
        YAHOO, ALPHAVANTAGE
    """
    _action_symbol = ""
    _start_date = datetime(1970, 1, 1)
    _end_date = datetime(1970, 1, 1)
    _interval = ""
    _data_source = ""
    _database_connection: psycopg2.extensions.connection
    _api_data_class: GenericAPI

    def __init__(
            self,
            action_symbol: str,
            start_date: datetime,
            end_date: datetime,
            interval: str,
            data_source="YAHOO"
    ):
        self._action_symbol = action_symbol
        self._start_date = start_date
        self._end_date = end_date
        self._interval = interval
        self._data_source = data_source
        load_dotenv()
        self._update_dataclass()
        self._database_connection = psycopg2.connect(
                host="localhost",
                port=5432,
                dbname=os.getenv("POSTGRES_DB"),
                user="postgres",
                password=os.getenv("POSTGRES_PASSWORD")
        )

    def _update_dataclass(self) -> None:
        interval_to_request = self._interval
        interval_in_min = pd.Timedelta(self._interval).total_seconds() / 60
        # Since requested interval is 60 min for somme API, if the requested interval is more than thatn
        # Insert data with an interval of 60 min
        if interval_in_min > 60:
            interval_to_request = "60min"
        match self._data_source:
            case "YAHOO":
                self._api_data_class = YahooAPI(
                        self._start_date,
                        self._end_date,
                        interval_to_request,
                        self._action_symbol
                )
            case "ALPHAVANTAGE":
                self._api_data_class = AlphavantageAPI(
                        self._start_date,
                        self._end_date,
                        interval_to_request,
                        self._action_symbol
                )
            case _:
                raise TypeError(
                        "Data source does not exist, please see avaiable data sources")

    def get_authorized_intervals(self) -> list[str]:
        return self._api_data_class.interval_authorized_values

    def set_action_symbol(self, action_symbol: str) -> None:
        self._action_symbol = action_symbol
        self._update_dataclass()

    def set_start_date(self, start_date: datetime) -> None:
        self._start_date = start_date
        self._update_dataclass()

    def set_end_date(self, end_date: datetime) -> None:
        self._end_date = end_date
        self._update_dataclass()

    def set_interval(self, interval: str) -> None:
        self._interval = interval
        self._update_dataclass()

    def set_data_source(self, data_source: str) -> None:
        self._data_source = data_source
        self._update_dataclass()

    def get_price_candles(self) -> list[dict]:
        """
        Return price_candles under a list format for the specified symbol.
        This function will automatically identified if data is present in database or not.
        If not, data will be fetch from source and put inside the database
        """
        if not self._data_exists():
            json_data = self._request_data_from_api()
            self._insert_price_candles(json_data)
        return self._query_price_candle()

    def get_price_candles_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.get_price_candles())
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
        return df

    def _data_exists(self) -> bool:
        """
        Check if data exist with the wanted interval inside the database
        """
        with self._database_connection.cursor() as cur:
            interval_in_min = pd.Timedelta(self._interval).total_seconds() / 60
            cur.execute("""
                SELECT MODE() WITHIN GROUP (ORDER BY diff) FROM (
                    SELECT EXTRACT(EPOCH FROM (time - LAG(time) OVER (ORDER BY time))) / 60 as diff
                    FROM price_candles
                    WHERE symbol = %s
                    AND time >= %s
                    AND time <= %s
                ) diffs
                WHERE diff IS NOT NULL
            """, (self._action_symbol, self._start_date, self._end_date))
            result = cur.fetchone()
            if result is None:
                return False
            return result[0] is not None and round(result[0]) <= interval_in_min

    def _query_price_candle(self) -> list[dict]:
        """
        Query price from timescaleDB, aggregating it to the request interval
        """
        with self._database_connection.cursor() as cur:
            interval_in_min = pd.Timedelta(self._interval).total_seconds() / 60
            cur.execute("""
                SELECT
                    time_bucket(%s, time) AS bucket,
                    symbol,
                    FIRST(open, time) AS open,
                    MAX(high)         AS high,
                    MIN(low)          AS low,
                    LAST(close, time) AS close,
                    SUM(volume)       AS volume
                FROM price_candles
                WHERE symbol = %s
                AND time BETWEEN %s and %s
                GROUP BY bucket, symbol
                ORDER BY bucket ASC
            """, (f"{interval_in_min} minutes",
                  self._action_symbol,
                  self._start_date,
                  self._end_date
                  )
            )
            columns = ["time",
                       "symbol",
                       "open",
                       "high",
                       "low",
                       "close",
                       "volume"]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    def _request_data_from_api(self):
        """
        Request data from the selected data_source
        if data does not exist in database
        """
        self._update_dataclass()
        json_data = self._api_data_class.get_standard_json()
        return json_data

    def _insert_price_candles(self, data: dict) -> None:
        symbol = data["symbol"]
        rows = [
                (
                    timestamp,
                    symbol,
                    candle["open"],
                    candle["high"],
                    candle["low"],
                    candle["close"],
                    candle["volume"]
                ) for timestamp, candle in data["data"].items()
               ]
        with self._database_connection.cursor() as cur:
            execute_values(cur, """
                            INSERT INTO price_candles (time, symbol, open, high, low, close, volume)
                            VALUES %s
                            ON CONFLICT (time, symbol) DO UPDATE SET
                                open   = EXCLUDED.open,
                                high   = EXCLUDED.high,
                                low    = EXCLUDED.low,
                                close  = EXCLUDED.close,
                                volume = EXCLUDED.volume
                           """, rows)
        self._database_connection.commit()
