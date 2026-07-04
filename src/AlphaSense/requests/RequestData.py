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
    start_date -> date you want the requested data to start from, datetime format
    end_date -> date you want the requested data to end, datetime format
    interval -> interval between two stock point (string), avaiable:
        1min, 5min, 15min, 30min, 60min
    data_source -> One of the data source avaiable, current data sources are:
        YAHOO, ALPHAVANTAGE
    """
    _action_symbol = ""
    _start_date = datetime(1970, 1, 1)
    _end_date = datetime(1970, 1, 1)
    _interval = ""
    _data_source = ""
    _database_connection: psycopg2.extensions.connection

    def __init__(
            self,
            action_symbol: str,
            start_date: datetime,
            end_date: datetime,
            interval: str,
            data_source="YAHOO"
    ):
        self._action_symbol=action_symbol
        self._start_date = start_date
        self._end_date = end_date
        self._interval = interval
        self._data_source = data_source
        load_dotenv()
        self._database_connection = psycopg2.connect(
                host="localhost",
                port=5432,
                dbname=os.getenv("POSTGRES_DB"),
                user="postgres",
                password=os.getenv("POSTGRES_PASSWORD")
        )

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

    def _data_exists(self) -> bool:
        """
        Check if data exist with the wanted interval inside the database
        """
        with self._database_connection.cursor() as cur:
            interval_in_min = pd.Timedelta(self._interval).total_seconds() / 60
            cur.execute("""
                SELECT AVG(diff) FROM (
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
        api_data_class: GenericAPI
        match self._data_source:
            case "YAHOO":
                api_data_class = YahooAPI(
                        self._start_date,
                        self._end_date,
                        self._interval,
                        self._action_symbol
                )
            case "ALPHAVANTAGE":
                api_data_class = AlphavantageAPI(
                        self._start_date,
                        self._end_date,
                        self._interval,
                        self._action_symbol
                )
            case _:
                raise TypeError(
                        "Data source does not exist, please see avaiable data sources")
        json_data = api_data_class.get_standard_json()
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
