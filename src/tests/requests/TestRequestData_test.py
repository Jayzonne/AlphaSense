import os
import unittest
from datetime import date, datetime, timezone
from unittest.mock import PropertyMock, patch

import pandas as pd

from AlphaSense.requests.RequestData import RequestData


class _FakeCursor:
    """
    Minimal cursor double: every execute() is recorded (normalised query +
    params) and fetchall() pops one pre-canned result set, in call order. That
    is enough to exercise _data_exists() + _get_closed_days() without a real
    Postgres connection.
    """

    def __init__(self, result_sets):
        self._result_sets = list(result_sets)
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self, query, params=None):
        self.executed.append((" ".join(query.split()), params))

    def fetchall(self):
        return self._result_sets.pop(0)


class _FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.committed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True


def _patched_connection(cursor):
    return patch.object(
        RequestData,
        "_connection",
        new_callable=PropertyMock,
        return_value=_FakeConnection(cursor),
    )


class TestConnectionPoolTimezone(unittest.TestCase):
    def test_pool_connections_are_pinned_to_utc(self):
        # price_candles.time is TIMESTAMPTZ and the canonical Yahoo strings are
        # UTC. The session timezone must not depend on the host/server default
        # (this host already reported Europe/Paris at one point). `options` is
        # a libpq connect parameter, so every pooled/reused connection gets it.
        import AlphaSense.requests.RequestData as request_data

        with patch.object(request_data, "_connection_pool", None):
            with patch.object(request_data, "ThreadedConnectionPool") as pool_cls:
                pool = request_data._get_connection_pool()

        self.assertIs(pool, pool_cls.return_value)
        self.assertEqual(pool_cls.call_args.kwargs["options"], "-c timezone=UTC")


class TestUtcBucketing(unittest.TestCase):
    def test_query_uses_utc_time_bucket_and_preserves_instants(self):
        # The connection session is pinned to UTC, so time_bucket() boundaries
        # are UTC and a tz-aware UTC timestamp must come back unchanged.
        ts = datetime(2026, 10, 21, 13, 30, tzinfo=timezone.utc)
        cursor = _FakeCursor([[(ts, "AAPL", 1.0, 2.0, 0.5, 1.5, 100)]])
        request = RequestData("AAPL", datetime(2026, 10, 21), datetime(2026, 10, 21), "60m")

        with _patched_connection(cursor):
            rows = request._query_price_candle()

        self.assertEqual(rows[0]["time"], ts)
        query, params = cursor.executed[0]
        self.assertIn("time_bucket(%s, time)", query)
        # The CTE placeholders now precede the bucket interval, so it is the
        # last parameter rather than the first.
        self.assertEqual(params[-1], "60.0 minutes")
        self.assertEqual(params[0], "AAPL")


class TestEndBoundary(unittest.TestCase):
    def _boundary(self, end: datetime) -> datetime:
        return RequestData("AAPL", datetime(2026, 1, 1), end, "1D")._end_boundary()

    def test_midnight_end_is_extended_to_include_the_whole_day(self):
        self.assertEqual(self._boundary(datetime(2026, 10, 21)), datetime(2026, 10, 22))

    def test_non_midnight_end_keeps_exclusive_instant_semantics(self):
        end = datetime(2026, 10, 21, 12, 34, 56)
        self.assertEqual(self._boundary(end), end)

    def test_midnight_month_boundary(self):
        self.assertEqual(self._boundary(datetime(2026, 1, 31)), datetime(2026, 2, 1))

    def test_midnight_year_boundary(self):
        self.assertEqual(self._boundary(datetime(2026, 12, 31)), datetime(2027, 1, 1))

    def test_midnight_before_leap_day(self):
        self.assertEqual(self._boundary(datetime(2028, 2, 28)), datetime(2028, 2, 29))


class TestDataExistsEndBoundary(unittest.TestCase):
    START = datetime(2026, 10, 21)
    END = datetime(2026, 10, 21)  # date-only end, as parsed by backtest.py

    def _data_exists(self, rows, closed_rows=(), start=None, end=None, interval="60m"):
        cursor = _FakeCursor([rows, list(closed_rows)])
        request = RequestData("AAPL", start or self.START, end or self.END, interval)
        with _patched_connection(cursor):
            result = request._data_exists()
        return result, cursor

    @staticmethod
    def _price_query(cursor):
        return next(q for q in cursor.executed if "price_candles" in q[0])

    @staticmethod
    def _closed_days_query(cursor):
        return next(q for q in cursor.executed if "closed_market_days" in q[0])

    def test_partial_end_day_is_cached_for_midnight_end(self):
        # 7 hourly candles = a partial (mid-session) Oct 21, the shape the
        # old exclusive bound excluded entirely.
        rows = [(
            pd.Timestamp("2026-10-21"),
            7,
            60.0,
            pd.Timestamp("2026-10-21 13:30:00"),
        )]

        exists, cursor = self._data_exists(rows)

        self.assertTrue(exists)
        # The SQL upper bound must be the next day, not midnight of Oct 21.
        self.assertEqual(
            self._price_query(cursor)[1],
            ("AAPL", datetime(2026, 10, 21), datetime(2026, 10, 22)),
        )
        # closed_market_days is day-granular: inclusive end date.
        self.assertEqual(
            self._closed_days_query(cursor)[1],
            ("AAPL", date(2026, 10, 21), date(2026, 10, 21)),
        )

    def test_single_non_midnight_candle_counts_as_present(self):
        # One candle means every LAG diff is NULL; a "diff IS NOT NULL" filter
        # would drop this partial day and re-download it forever. It is NOT a
        # 00:00 native marker, so it still satisfies an intraday request.
        rows = [(
            pd.Timestamp("2026-10-21"),
            1,
            None,
            pd.Timestamp("2026-10-21 13:30:00"),
        )]

        exists, _ = self._data_exists(rows)

        self.assertTrue(exists)

    def test_non_midnight_end_keeps_exclusive_sql_bound(self):
        end = datetime(2026, 10, 21, 12, 0)
        rows = [(
            pd.Timestamp("2026-10-21"),
            3,
            60.0,
            pd.Timestamp("2026-10-21 09:30:00"),
        )]

        exists, cursor = self._data_exists(rows, start=datetime(2026, 10, 21), end=end)

        self.assertTrue(exists)
        self.assertEqual(
            self._price_query(cursor)[1],
            ("AAPL", datetime(2026, 10, 21), end),
        )

    def test_missing_expected_day_is_not_cached(self):
        # Sanity check that the new logic still reports a genuine gap.
        rows = [(
            pd.Timestamp("2026-10-19"),
            7,
            60.0,
            pd.Timestamp("2026-10-19 13:30:00"),
        )]

        exists, _ = self._data_exists(
            rows,
            start=datetime(2026, 10, 19),
            end=datetime(2026, 10, 21),
        )

        self.assertFalse(exists)


class TestDataExistsGranularity(unittest.TestCase):
    """
    BUG-007 AC2/AC4: a native daily row is a single 00:00:00 UTC date marker.
    It must not make an intraday request look cached, while a lone non-midnight
    (partial-session) candle must keep counting as present.
    """

    DAY = datetime(2026, 10, 21)

    def _data_exists(self, rows, interval, closed_rows=()):
        cursor = _FakeCursor([rows, list(closed_rows)])
        request = RequestData("AAPL", self.DAY, self.DAY, interval)
        with _patched_connection(cursor):
            return request._data_exists(), cursor

    @staticmethod
    def _daily_marker():
        return (
            pd.Timestamp("2026-10-21"),
            1,
            None,
            pd.Timestamp("2026-10-21 00:00:00"),
        )

    def test_daily_only_midnight_marker_does_not_satisfy_intraday_request(self):
        exists, _ = self._data_exists([self._daily_marker()], "1h")

        self.assertFalse(exists)

    def test_daily_only_midnight_marker_satisfies_daily_request(self):
        exists, _ = self._data_exists([self._daily_marker()], "1d")

        self.assertTrue(exists)

    def test_single_non_midnight_candle_satisfies_intraday_request(self):
        rows = [(
            pd.Timestamp("2026-10-21"),
            1,
            None,
            pd.Timestamp("2026-10-21 13:30:00"),
        )]

        exists, _ = self._data_exists(rows, "1h")

        self.assertTrue(exists)

    def test_multi_candle_day_still_uses_modal_diff(self):
        rows = [(
            pd.Timestamp("2026-10-21"),
            7,
            60.0,
            pd.Timestamp("2026-10-21 13:30:00"),
        )]

        exists, _ = self._data_exists(rows, "1h")

        self.assertTrue(exists)


class TestQueryPriceCandleGranularity(unittest.TestCase):
    """
    BUG-007 AC3/AC4: the query must filter coarse 00:00 daily markers out of
    intraday requests and prefer intraday rows on mixed days at 1d+, so volume
    is never counted twice. A live TimescaleDB is not required: the mocked
    cursor records the exact SQL/params and returns synthetic aggregated rows.
    """

    START = datetime(2026, 10, 21)
    END = datetime(2026, 10, 21)

    def _query(self, interval, rows=None):
        cursor = _FakeCursor([rows or []])
        request = RequestData("AAPL", self.START, self.END, interval)
        with _patched_connection(cursor):
            result = request._query_price_candle()
        return result, cursor

    def test_intraday_query_excludes_midnight_markers(self):
        _, cursor = self._query("1h")

        query, params = cursor.executed[0]
        self.assertIn("is_midnight", query)
        self.assertIn("intraday_days", query)
        self.assertIn("WHEN %s < 1440 THEN NOT is_midnight", query)
        self.assertEqual(params[3], 60.0)
        self.assertEqual(params[-1], "60.0 minutes")

    def test_daily_query_excludes_midnight_when_the_day_has_intraday_rows(self):
        _, cursor = self._query("1d")

        query, params = cursor.executed[0]
        self.assertIn("ELSE NOT (", query)
        self.assertIn(
            "is_midnight AND date_trunc('day', time) IN",
            query,
        )
        self.assertIn("SELECT day FROM intraday_days", query)
        self.assertEqual(params[3], 1440.0)
        self.assertEqual(params[-1], "1440.0 minutes")

    def test_mixed_day_daily_query_returns_hourly_aggregate_only(self):
        # One 00:00 daily row + four 60m rows: the midnight filter leaves only
        # the hourly rows, so SUM(volume) is the intraday sum (4000), not
        # 1000 + 4000. OPEN/CLOSE come from the intraday FIRST/LAST.
        bucket = datetime(2026, 10, 21, tzinfo=timezone.utc)
        rows = [(bucket, "AAPL", 101.0, 110.0, 99.0, 108.0, 4000.0)]

        result, cursor = self._query("1d", rows)

        self.assertEqual(result[0]["volume"], 4000.0)
        self.assertEqual(result[0]["open"], 101.0)
        self.assertEqual(result[0]["close"], 108.0)
        query, _ = cursor.executed[0]
        self.assertIn("SUM(volume)", query)

    def test_daily_only_day_is_still_returned_for_daily_query(self):
        bucket = datetime(2026, 10, 21, tzinfo=timezone.utc)
        rows = [(bucket, "AAPL", 100.0, 110.0, 99.0, 108.0, 1000.0)]

        result, _ = self._query("1d", rows)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["volume"], 1000.0)
        self.assertEqual(result[0]["close"], 108.0)


class TestProviderAwareIntervalMapping(unittest.TestCase):
    """
    BUG-006 AC1/AC2 mapped requested intervals to provider-native ones.
    BUG-007 AC1/AC5 replaces that with an aggregatable BASE interval: the DB
    stores one resolution per range and aggregates UP at query time, so
    weekly/monthly requests must be stored as daily (never native).
    """

    START = datetime(2026, 1, 1)
    END = datetime(2026, 10, 21)
    LONG_END = datetime(2028, 10, 21)  # > 700 days (~1024)

    def _requested_interval(self, data_source, interval, start=None, end=None) -> str:
        request = RequestData(
            "AAPL", start or self.START, end or self.END, interval, data_source
        )
        return request._api_data_class._interval

    def test_yahoo_short_range_daily_uses_hourly_base(self):
        self.assertEqual(self._requested_interval("YAHOO", "1d"), "60m")

    def test_yahoo_long_range_daily_falls_back_to_native_daily(self):
        # yfinance's 60m feed is capped at ~730 days.
        self.assertEqual(
            self._requested_interval(
                "YAHOO", "1d", start=self.START, end=self.LONG_END
            ),
            "1d",
        )

    def test_yahoo_coarse_intervals_use_daily_base(self):
        self.assertEqual(self._requested_interval("YAHOO", "1wk"), "1d")
        self.assertEqual(self._requested_interval("YAHOO", "1mo"), "1d")
        self.assertEqual(self._requested_interval("YAHOO", "3mo"), "1d")

    def test_yahoo_falls_back_to_60m_for_non_native_4h(self):
        self.assertEqual(self._requested_interval("YAHOO", "4h"), "60m")

    def test_eulerpool_uses_daily_base_for_daily_and_coarser(self):
        self.assertEqual(self._requested_interval("EULERPOOL", "1d"), "1d")
        self.assertEqual(self._requested_interval("EULERPOOL", "1wk"), "1d")
        self.assertEqual(self._requested_interval("EULERPOOL", "1mo"), "1d")

    def test_eulerpool_falls_back_to_60m_for_non_native_4h(self):
        self.assertEqual(self._requested_interval("EULERPOOL", "4h"), "60m")

    def test_hf_data_library_uses_aggregatable_base_intervals(self):
        # The constructor requires an API key but performs no network I/O.
        with patch.dict(os.environ, {"HFDATALIBRARY_API_KEY": "test-key"}):
            self.assertEqual(self._requested_interval("HFDATALIBRARY", "1d"), "1d")
            self.assertEqual(self._requested_interval("HFDATALIBRARY", "4h"), "4h")
            self.assertEqual(self._requested_interval("HFDATALIBRARY", "1wk"), "1d")

    def test_alphavantage_keeps_its_60min_behaviour(self):
        self.assertEqual(self._requested_interval("ALPHAVANTAGE", "4h"), "60min")
        self.assertEqual(self._requested_interval("ALPHAVANTAGE", "60m"), "60min")

    def test_alpaca_uses_30m_base_for_hourly_and_4h(self):
        # Alpaca's native 1h grid is top-of-the-hour while Yahoo's is the market
        # open; 30m shares Yahoo's grid, so 1h/4h must be stored as 30m and
        # aggregated up in TimescaleDB to stay coherent with every other source.
        with patch.dict(
            os.environ,
            {"ALPACA_API_KEY": "test-key", "ALPACA_SECRET_KEY": "test-secret"},
        ):
            self.assertEqual(self._requested_interval("ALPACA", "60m"), "30m")
            self.assertEqual(self._requested_interval("ALPACA", "1h"), "30m")
            self.assertEqual(self._requested_interval("ALPACA", "4h"), "30m")

    def test_alpaca_keeps_fine_intraday_intervals_native(self):
        with patch.dict(
            os.environ,
            {"ALPACA_API_KEY": "test-key", "ALPACA_SECRET_KEY": "test-secret"},
        ):
            self.assertEqual(self._requested_interval("ALPACA", "1m"), "1m")
            self.assertEqual(self._requested_interval("ALPACA", "5m"), "5m")
            self.assertEqual(self._requested_interval("ALPACA", "15m"), "15m")
            self.assertEqual(self._requested_interval("ALPACA", "30m"), "30m")

    def test_alpaca_uses_daily_base_for_daily_and_coarser(self):
        with patch.dict(
            os.environ,
            {"ALPACA_API_KEY": "test-key", "ALPACA_SECRET_KEY": "test-secret"},
        ):
            self.assertEqual(self._requested_interval("ALPACA", "1d"), "1d")
            self.assertEqual(self._requested_interval("ALPACA", "1wk"), "1d")
            self.assertEqual(self._requested_interval("ALPACA", "1mo"), "1d")


class TestMarkClosedDays(unittest.TestCase):
    """
    BUG-006 AC4: only genuine holiday gaps may be recorded. A capped feed that
    returns a handful of sessions for a long range must never fill
    closed_market_days with the missing months of real trading days.
    """

    START = datetime(2026, 1, 1)
    END = datetime(2026, 10, 21)

    @classmethod
    def setUpClass(cls):
        cls.expected = [d.date() for d in pd.bdate_range(cls.START, cls.END)]

    def _mark(self, returned_days):
        json_data = {
            "symbol": "AAPL",
            "data": {
                f"{day.isoformat()} 00:00:00": {} for day in returned_days
            },
        }
        cursor = _FakeCursor([])
        connection = _FakeConnection(cursor)
        request = RequestData("AAPL", self.START, self.END, "1d")
        with _patched_connection(cursor), patch(
            "AlphaSense.requests.RequestData.execute_values"
        ) as execute:
            request._mark_closed_days(json_data)
        return execute, connection

    @staticmethod
    def _inserted_days(execute):
        rows = execute.call_args.args[2]
        return {day for _, day in rows}

    def test_capped_feed_inserts_no_closed_days(self):
        # Eulerpool 60m returned ~8 sessions for a ~210-business-day request:
        # all missing days are leading/trailing coverage, not holidays.
        execute, connection = self._mark(self.expected[:8])

        execute.assert_not_called()
        self.assertFalse(connection.committed)

    def test_normal_feed_marks_only_the_interior_holiday(self):
        missing = self.expected[len(self.expected) // 2]
        returned = [day for day in self.expected if day != missing]

        execute, _ = self._mark(returned)

        self.assertEqual(self._inserted_days(execute), {missing})

    def test_normal_feed_marks_a_leading_boundary_holiday(self):
        missing = self.expected[0]
        returned = self.expected[1:]

        execute, _ = self._mark(returned)

        self.assertEqual(self._inserted_days(execute), {missing})

    def test_normal_feed_marks_a_trailing_boundary_holiday(self):
        missing = self.expected[-1]
        returned = self.expected[:-1]

        execute, _ = self._mark(returned)

        self.assertEqual(self._inserted_days(execute), {missing})


if __name__ == "__main__":
    unittest.main()
