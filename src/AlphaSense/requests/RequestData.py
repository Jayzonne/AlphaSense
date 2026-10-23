from datetime import datetime, time, timedelta
import threading

from psycopg2.extras import execute_values
from psycopg2.pool import ThreadedConnectionPool
from AlphaSense.api import *
from dotenv import load_dotenv
import psycopg2
import os
import pandas as pd


_connection_pool: ThreadedConnectionPool | None = None
_connection_pool_lock = threading.Lock()

# pd.Timedelta has no fixed-duration unit for "week"/"month" (a month is
# 28-31 days, so pandas rejects "wk"/"mo" suffixes outright rather than
# guessing) - yet "1wk", "1mo", and "3mo" are all listed as YahooAPI's own
# authorized interval values. Left as-is, selecting any of them crashed
# _data_exists()/_query_price_candle() with "invalid unit abbreviation".
# These are fixed-duration approximations (consistent with how every other
# interval here, including the existing "5d" case, is already treated as a
# fixed elapsed-time bucket rather than a calendar-aware one) - "1mo"/"3mo"
# won't line up exactly with calendar month boundaries, which is worth
# knowing if that precision ever matters.
_INTERVAL_MINUTES_OVERRIDE = {
    "1wk": 7 * 24 * 60,
    "1mo": 30 * 24 * 60,
    "3mo": 90 * 24 * 60,
}

# price_candles has no granularity column, so a stored range must hold exactly
# ONE base resolution. That base must be aggregatable UP to every interval the
# caller can later request:
#   <= 60m -> the requested interval itself
#   4h     -> 60m (except providers that aggregate locally from 1m)
#   1d     -> 60m when the provider can serve the range, otherwise native 1d
#   1wk+   -> 1d (never native weekly/monthly, because those cannot be
#            aggregated DOWN into daily or intraday queries)
# This is why _update_dataclass() maps the REQUESTED interval to a BASE interval
# instead of passing native resolutions through: a native 1wk row would poison a
# later 1d query, and a native 1d row would look like a fake 00:00 hourly bar.
# _data_exists()/_query_price_candle() filter those coarse midnight markers out
# of finer requests.


def interval_to_minutes(interval: str) -> float:
    """ Single source of truth for turning an interval string into minutes, used for TimescaleDB bucketing. """
    if interval in _INTERVAL_MINUTES_OVERRIDE:
        return _INTERVAL_MINUTES_OVERRIDE[interval]
    return pd.Timedelta(interval).total_seconds() / 60


def _get_connection_pool() -> ThreadedConnectionPool:
    """
    Lazily builds one shared pool of Postgres connections for every RequestData
    instance in the process, instead of each instance paying for its own
    psycopg2.connect() (fresh TCP handshake + auth round trip). Dash renders
    a single symbol/interval/date-range change through several callbacks
    (chart, confluence, pattern table), each of which used to spin up and
    never explicitly close its own connection - this pool lets them borrow
    and return connections instead, bounding how many sockets are ever open.
    ThreadedConnectionPool because Dash can invoke callbacks concurrently
    across threads.
    """
    global _connection_pool
    if _connection_pool is None:
        with _connection_pool_lock:
            if _connection_pool is None:
                load_dotenv()
                _connection_pool = ThreadedConnectionPool(
                        1, 10,
                        host="localhost",
                        port=5432,
                        dbname=os.getenv("POSTGRES_DB"),
                        user="postgres",
                        password=os.getenv("POSTGRES_PASSWORD"),
                        # Pin EVERY pooled connection to UTC at connect time.
                        # price_candles.time is TIMESTAMPTZ: the canonical UTC
                        # strings we insert/read must be interpreted as UTC
                        # regardless of the host or server default (both have
                        # been non-UTC in practice). `options` is applied by
                        # libpq on connection creation, so it also covers
                        # connections reused from the pool and costs no extra
                        # round trip per borrow (unlike a SET on every getconn).
                        options="-c timezone=UTC",
                )
    return _connection_pool


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
        YAHOO, ALPHAVANTAGE, EULERPOOL, HFDATALIBRARY, ALPACA
    """
    _action_symbol = ""
    _start_date = datetime(1970, 1, 1)
    _end_date = datetime(1970, 1, 1)
    _interval = ""
    _data_source = ""
    _database_connection: psycopg2.extensions.connection | None
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
        # No connection is opened here. Some callers (get_authorized_intervals())
        # only ever touch _api_data_class and never hit the database at all -
        # borrowing lazily means that path costs zero DB connections instead of one.
        self._database_connection = None

    def _end_boundary(self) -> datetime:
        """
        Upper bound for the half-open [start, boundary) scans, normalising the
        two conventions callers actually use.

        A date-only `--end YYYY-MM-DD` is parsed by backtest.py into a midnight
        datetime, and that means "up to and including the whole day". Every
        other caller (Dash's datetime.now(), resolve_range_preset) passes a real
        instant whose exclusive semantics must be preserved. Treating the
        midnight case as an instant scanned `time < day 00:00`, dropping every
        candle of that day: `_data_exists()` then saw the range as incomplete
        (endless re-downloads) and `_query_price_candle()` silently truncated
        the backtest by one day. For non-midnight instants the bound is
        unchanged, so those callers keep their exclusive-instant behaviour.
        """
        if self._end_date.time() == time.min:
            return self._end_date + timedelta(days=1)
        return self._end_date

    @property
    def _connection(self) -> psycopg2.extensions.connection:
        """ Borrows a connection from the shared pool on first use, memoized on this instance. """
        if self._database_connection is None:
            self._database_connection = _get_connection_pool().getconn()
        return self._database_connection

    def close(self) -> None:
        """
        Returns this instance's borrowed connection to the shared pool, if one
        was ever borrowed. Safe to call multiple times / on instances that
        never touched the database.
        """
        if self._database_connection is not None:
            _get_connection_pool().putconn(self._database_connection)
            self._database_connection = None

    def __enter__(self) -> "RequestData":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _update_dataclass(self) -> None:
        # Provider-aware BASE interval mapping. The DB stores whatever the API
        # class returns and aggregates it UP to self._interval at query time, so
        # the selected interval must be the finest resolution the provider can
        # serve for this range - never a coarser native weekly/monthly bar that
        # could not be broken back down into daily/intraday candles.
        interval_in_min = interval_to_minutes(self._interval)
        match self._data_source:
            case "YAHOO":
                if interval_in_min <= 60:
                    interval_to_request = self._interval
                elif self._interval == "1d" and (
                        (self._end_date - self._start_date).days <= 700
                ):
                    # yfinance's 60m feed reaches back ~730 days. Store hourly
                    # bars when the whole range fits so daily requests can be
                    # aggregated from true intraday data.
                    interval_to_request = "60m"
                elif interval_in_min < 1440:
                    # 4h/90m: fetch hourly, aggregate up in TimescaleDB.
                    interval_to_request = "60m"
                else:
                    # 1d beyond the 60m cap, and every coarser interval
                    # (5d/1wk/1mo/3mo): store native daily bars, which are
                    # still aggregatable up to weekly/monthly.
                    interval_to_request = "1d"
                self._api_data_class = YahooAPI(
                        self._start_date,
                        self._end_date,
                        interval_to_request,
                        self._action_symbol
                )
            case "EULERPOOL":
                if interval_in_min <= 60:
                    interval_to_request = self._interval
                elif interval_in_min < 1440:
                    # 4h: fetch 60m and aggregate up in TimescaleDB.
                    interval_to_request = "60m"
                else:
                    # 1d/1wk/1mo: never store native weekly/monthly bars.
                    interval_to_request = "1d"
                self._api_data_class = EulerpoolAPI(
                        self._start_date,
                        self._end_date,
                        interval_to_request,
                        self._action_symbol
                )
            case "HFDATALIBRARY":
                # HF Data Library ships 1-minute files and aggregates locally.
                # 4h is kept as the base (it is already a safe local
                # aggregation), while weekly/monthly requests fall back to 1d
                # so they remain aggregatable to every coarser interval.
                if interval_in_min <= 60:
                    interval_to_request = self._interval
                elif self._interval == "4h":
                    interval_to_request = "4h"
                elif interval_in_min < 1440:
                    interval_to_request = "60m"
                else:
                    interval_to_request = "1d"
                self._api_data_class = HFDataLibraryAPI(
                        self._start_date,
                        self._end_date,
                        interval_to_request,
                        self._action_symbol
                )
            case "ALPACA":
                # Alpaca serves US stocks/ETFs only (history since 2016).
                # BUCKET-GRID CAVEAT: Alpaca's NATIVE 1h bars are stamped on the
                # top of the hour (09:00, 10:00 ET), while Yahoo's 1h bars are
                # stamped on the market open (09:30, 10:30 ET). The two grids
                # share zero timestamps, so storing native Alpaca 1h would make
                # 1h requests incoherent with every other source. Alpaca's 30m
                # grid DOES match Yahoo's 30m grid exactly, so 30m is used as the
                # intraday base and TimescaleDB aggregates it up to 1h/4h on the
                # Yahoo-aligned grid (verified 100% within 0.5%).
                if interval_in_min <= 30:
                    # 1m/5m/15m/30m: identical grids, store natively.
                    interval_to_request = self._interval
                elif interval_in_min < 1440:
                    # 60m/1h/4h: store 30m so the 1h bucket lands on the market
                    # open (Yahoo grid) instead of the top of the hour.
                    interval_to_request = "30m"
                else:
                    # 1d and every coarser interval (1wk/1mo/3mo): native daily
                    # bars, still aggregatable up to weekly/monthly.
                    interval_to_request = "1d"
                self._api_data_class = AlpacaAPI(
                        self._start_date,
                        self._end_date,
                        interval_to_request,
                        self._action_symbol
                )
            case "ALPHAVANTAGE":
                # Alphavantage only exposes intraday resolutions: keep its
                # historical >60min -> 60min collapse unless the class ever
                # declares a native daily/weekly interval.
                interval_to_request = self._interval
                if interval_to_minutes(self._interval) > 60:
                    interval_to_request = "60m"
                if interval_to_request == "60m":
                    interval_to_request = "60min"
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
        If not, data will be fetch from source and put inside the database.
        The borrowed connection is returned to the pool as soon as this call
        finishes, so it's immediately available for the next RequestData
        instance instead of sitting open for the lifetime of this object.
        """
        try:
            if not self._data_exists():
                json_data = self._request_data_from_api()
                self._insert_price_candles(json_data)
                self._mark_closed_days(json_data)
            return self._query_price_candle()
        finally:
            self.close()

    def get_price_candles_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.get_price_candles())
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
        return df

    def _data_exists(self) -> bool:
        """
        Check if data exist with the wanted interval inside the database.

        Day grouping below uses `date_trunc('day', time)` under the connection's
        UTC session (see _get_connection_pool), so a "day" is a UTC calendar day.
        That is exact for US/EU sessions, which end well before UTC midnight.
        Exchanges whose session crosses UTC midnight (parts of Asia/Australia)
        would have one session split across two UTC days; that limitation is
        accepted for the current US/EU universe (see _mark_closed_days).
        """
        with self._connection.cursor() as cur:
            interval_in_min = interval_to_minutes(self._interval)
            end_boundary = self._end_boundary()
            cur.execute("""
                SELECT day, COUNT(*) AS candle_count,
                       MODE() WITHIN GROUP (ORDER BY diff) AS modal_diff,
                       MIN(time) AS first_time
                FROM (
                    SELECT 
                        date_trunc('day', time) AS day,
                        time,
                        EXTRACT(EPOCH FROM (
                                time - LAG(time) OVER (PARTITION BY date_trunc('day',time) ORDER BY time)
                        )) / 60 AS diff
                    FROM price_candles
                    WHERE symbol = %s
                    AND time >= %s AND time < %s
                ) diffs
                GROUP BY day
            """, (self._action_symbol, self._start_date, end_boundary))
            result = cur.fetchall()
            if not result:
                return False
            # A day holding exactly ONE candle has no LAG diff at all (every
            # diff is NULL), so filtering on "diff IS NOT NULL" made it vanish
            # and look missing forever when a request starts/ends mid-session.
            # Day presence is therefore count-based; stored resolution is only
            # measurable - and only checked - when the day has a real gap.
            #
            # A lone candle at exactly 00:00:00 UTC is a native daily/weekly
            # DATE MARKER, not an intraday bar. It only proves that a request at
            # 1d or coarser is cached; an intraday request must still download
            # real 60m bars. A lone non-midnight candle is a genuine partial
            # session and stays acceptable.
            def day_is_present(candle_count, modal_diff, first_time) -> bool:
                if candle_count == 1:
                    is_midnight = (
                            first_time is not None
                            and first_time.time() == time.min
                    )
                    return not is_midnight or interval_in_min >= 1440
                return modal_diff is not None and modal_diff <= interval_in_min

            if not all(
                    day_is_present(candle_count, modal_diff, first_time)
                    for _, candle_count, modal_diff, first_time in result
            ):
                return False
            existing_days = {day.date() for day, _, _, _ in result}
            closed_days = self._get_closed_days()
            # Express the same half-open [start, end_boundary) window in day
            # units so expected_days and existing_days agree on which days must
            # be present - with a midnight end this now includes that whole day.
            last_day = (end_boundary - timedelta(microseconds=1)).date()
            expected_days = {
                    d.date() for d in pd.bdate_range(self._start_date, last_day)
            } - closed_days
            return expected_days.issubset(existing_days)

    def _query_price_candle(self) -> list[dict]:
        """
        Query price from timescaleDB, aggregating it to the request interval
        """
        with self._connection.cursor() as cur:
            interval_in_min = interval_to_minutes(self._interval)
            # Mixed-granularity guard. price_candles can legitimately contain
            # both native daily markers (00:00:00 UTC) and real intraday rows
            # for the same day. Intraday queries must never see the daily
            # marker (it would become a fake 00:00 hourly bar), and daily+
            # queries on mixed days must prefer the intraday rows so volume is
            # not counted twice (daily volume + sum of hourly volume).
            cur.execute("""
                WITH filtered AS (
                    SELECT
                        time, symbol, open, high, low, close, volume,
                        (time::time = TIME '00:00:00') AS is_midnight
                    FROM price_candles
                    WHERE symbol = %s
                    AND time >= %s and time < %s
                ),
                intraday_days AS (
                    SELECT DISTINCT date_trunc('day', time) AS day
                    FROM filtered
                    WHERE NOT is_midnight
                ),
                base AS (
                    SELECT *
                    FROM filtered
                    WHERE CASE
                        WHEN %s < 1440 THEN NOT is_midnight
                        ELSE NOT (
                            is_midnight
                            AND date_trunc('day', time) IN (
                                SELECT day FROM intraday_days
                            )
                        )
                    END
                )
                SELECT
                    time_bucket(%s, time) AS bucket,
                    symbol,
                    FIRST(open, time) AS open,
                    MAX(high)         AS high,
                    MIN(low)          AS low,
                    LAST(close, time) AS close,
                    SUM(volume)       AS volume
                FROM base
                GROUP BY bucket, symbol
                ORDER BY bucket ASC
            """, (self._action_symbol,
                  self._start_date,
                  # Same inclusive-whole-day boundary as _data_exists(), so a
                  # date-only end actually returns the end day's candles
                  # instead of silently truncating the backtest one day early.
                  self._end_boundary(),
                  interval_in_min,
                  f"{interval_in_min} minutes")
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

    def _mark_closed_days(self, json_data: dict) -> None:
        """
        Record genuine market holidays for this symbol so future requests stop
        expecting data there.

        Day identity is the UTC calendar date (canonical timestamps are UTC).
        LIMITATION: sessions crossing UTC midnight (parts of Asia/Australia)
        can be split over two UTC dates, so a session's first/last bars may be
        attributed to different "days" here. US/EU megacaps are unaffected; no
        code change is needed until such a symbol is added.

        Only INTERIOR gaps (business days strictly between the first and last
        day the API actually returned) are always trustworthy. Leading/trailing
        gaps are marked only when the response covers >= 90% of the expected
        business days: a capped feed (Eulerpool 60m returns ~8 sessions for a
        10-month request, Yahoo 60m caps at ~730 days) would otherwise label
        months of real trading days as holidays and poison `_data_exists()`.
        """
        returned_days = {pd.Timestamp(ts).date() for ts in json_data["data"].keys()}
        if not returned_days:
            return # empty - response - Could be an error or every requested days are open

        today = datetime.now().date()
        expected_days = {
                d.date() for d in pd.bdate_range(self._start_date, self._end_date)
                if d.date() < today # Do not mark today / future day as closed market
        }
        if not expected_days:
            return

        # Coverage counts only days the request actually expected: an
        # occasional weekend bar from the source must not inflate it.
        covered_expected_days = returned_days & expected_days
        if not covered_expected_days:
            return

        first_returned = min(covered_expected_days)
        last_returned = max(covered_expected_days)

        closed_days = {
                day for day in expected_days
                if first_returned < day < last_returned
                and day not in returned_days
        }

        coverage_ratio = len(covered_expected_days) / len(expected_days)
        if coverage_ratio >= 0.9:
            # Normal feed: a gap at the very start/end is a boundary holiday,
            # not missing history.
            closed_days |= {
                    day for day in expected_days
                    if (day < first_returned or day > last_returned)
                    and day not in returned_days
            }

        if not closed_days:
            return
        
        with self._connection.cursor() as cur:
            execute_values(cur, """
                INSERT INTO closed_market_days (symbol, day)
                VALUES %s
                ON CONFLICT DO NOTHING
            """, [(self._action_symbol, day) for day in closed_days])
            self._connection.commit()

    def _get_closed_days(self) -> set:
        # closed_market_days is day-granular, so its range is inclusive of the
        # end date. The old `< end.date()` cutoff disagreed with the
        # day-granular expected_days above and dropped an end-day holiday.
        with self._connection.cursor() as cur:
            cur.execute("""
                SELECT day FROM closed_market_days
                WHERE symbol = %s AND day >= %s AND day <= %s
            """, (self._action_symbol, self._start_date.date(), self._end_date.date()))
            return {row[0] for row in cur.fetchall()}
    
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
        with self._connection.cursor() as cur:
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
        self._connection.commit()
