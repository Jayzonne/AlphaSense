import calendar
import os
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd
import requests

from AlphaSense.api.EulerpoolAPI import EulerpoolAPI
from AlphaSense.api.YahooAPI import YahooAPI


def _epoch(*args) -> int:
    """Unix seconds for a naive datetime interpreted as UTC."""
    return calendar.timegm(datetime(*args).timetuple())


def _payload(timestamps, opens, highs, lows, closes, volumes):
    return {"t": timestamps, "o": opens, "h": highs, "l": lows, "c": closes, "v": volumes}


class _FakeResponse:
    def __init__(self, payload=None, status_code=200):
        self._payload = payload if payload is not None else {}
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} HTTP error")


def _yahoo_frame():
    columns = pd.MultiIndex.from_tuples(
        [(name, "AAPL") for name in ("Open", "High", "Low", "Close", "Adj Close", "Volume")]
    )
    return pd.DataFrame(
        [[100.0, 101.0, 99.0, 100.5, 100.4, 1000]],
        index=pd.to_datetime(["2026-10-21"]),
        columns=columns,
    )


class TestParallelArrayParsing(unittest.TestCase):
    def test_arrays_become_the_canonical_schema(self):
        payload = _payload(
            [_epoch(2026, 1, 2, 14, 30), _epoch(2026, 1, 2, 15, 30)],
            [100.25, 101.5],
            [102.0, 103.0],
            [99.5, 100.5],
            [101.0, 102.5],
            [12345, 0],
        )
        with patch(
            "AlphaSense.api.EulerpoolAPI.requests.get",
            return_value=_FakeResponse(payload),
        ):
            result = EulerpoolAPI(
                datetime(2026, 1, 1), datetime(2026, 1, 2), "60m", "AAPL"
            ).get_standard_json()

        self.assertEqual(list(result.keys()), ["symbol", "data"])
        self.assertEqual(result["symbol"], "AAPL")
        self.assertEqual(
            result["data"]["2026-01-02 14:30:00"],
            {"open": 100.25, "high": 102.0, "low": 99.5, "close": 101.0, "volume": 12345},
        )
        # A source 0 volume is preserved (not dropped, not turned into None).
        self.assertEqual(result["data"]["2026-01-02 15:30:00"]["volume"], 0)

    def test_records_are_plain_python_numbers(self):
        payload = _payload(
            [_epoch(2026, 1, 2, 14, 30)],
            [100], [101], [99], [100], [1000],
        )
        with patch(
            "AlphaSense.api.EulerpoolAPI.requests.get",
            return_value=_FakeResponse(payload),
        ):
            record = EulerpoolAPI(
                datetime(2026, 1, 1), datetime(2026, 1, 2), "60m", "AAPL"
            ).get_json_api()["2026-01-02 14:30:00"]

        for value in record.values():
            self.assertIsInstance(value, (int, float))
            self.assertNotIsInstance(value, bool)

    def test_incomplete_rows_are_skipped(self):
        # The volume array is shorter than t and one OHLC value is missing:
        # the incomplete OHLC row is skipped, the complete row keeps volume 0.
        payload = {
            "t": [_epoch(2026, 1, 2, 14, 0), _epoch(2026, 1, 2, 15, 0)],
            "o": [100, None],
            "h": [101, 101],
            "l": [99, 99],
            "c": [100, 100],
            "v": [],
        }
        with patch(
            "AlphaSense.api.EulerpoolAPI.requests.get",
            return_value=_FakeResponse(payload),
        ):
            data = EulerpoolAPI(
                datetime(2026, 1, 1), datetime(2026, 1, 2), "60m", "AAPL"
            ).get_json_api()

        self.assertEqual(list(data), ["2026-01-02 14:00:00"])
        self.assertEqual(data["2026-01-02 14:00:00"]["volume"], 0)


class TestUtcConversion(unittest.TestCase):
    def _fetch_at(self, timestamp, interval="60m", start=None, end=None):
        start = start or datetime(2026, 1, 1)
        end = end or datetime(2026, 1, 2)
        payload = _payload([timestamp], [100], [101], [99], [100], [1000])
        with patch(
            "AlphaSense.api.EulerpoolAPI.requests.get",
            return_value=_FakeResponse(payload),
        ):
            return EulerpoolAPI(start, end, interval, "AAPL").get_json_api()

    def test_est_winter_open_is_14_30_utc(self):
        # 09:30 EST (UTC-5) on 2026-01-02 -> 14:30 UTC.
        data = self._fetch_at(_epoch(2026, 1, 2, 14, 30))
        self.assertIn("2026-01-02 14:30:00", data)

    def test_edt_summer_open_is_13_30_utc(self):
        # 09:30 EDT (UTC-4) on 2026-07-01 -> 13:30 UTC.
        data = self._fetch_at(
            _epoch(2026, 7, 1, 13, 30),
            start=datetime(2026, 7, 1),
            end=datetime(2026, 7, 2),
        )
        self.assertIn("2026-07-01 13:30:00", data)

    def test_daily_bar_is_emitted_as_a_midnight_date_marker(self):
        # The source stamps AAPL daily bars at 21:00 UTC (US close); the
        # canonical daily output must stay a date marker at 00:00:00 UTC.
        data = self._fetch_at(_epoch(2026, 1, 2, 21, 0), interval="1d")
        self.assertEqual(list(data), ["2026-01-02 00:00:00"])

    def test_milliseconds_epochs_are_guarded(self):
        # AC8 seconds-vs-milliseconds guard: the documented unit is seconds,
        # but a milliseconds value must not become year 56000.
        milliseconds = _epoch(2026, 1, 2, 14, 30) * 1000
        data = self._fetch_at(milliseconds)
        self.assertEqual(list(data), ["2026-01-02 14:30:00"])


class TestResolutionMapping(unittest.TestCase):
    def _resolution_for(self, interval):
        with patch(
            "AlphaSense.api.EulerpoolAPI.requests.get",
            return_value=_FakeResponse(_payload([], [], [], [], [], [])),
        ) as get:
            EulerpoolAPI(
                datetime(2026, 1, 1), datetime(2026, 1, 2), interval, "AAPL"
            ).get_json_api()
        return get.call_args.kwargs["params"]["resolution"]

    def test_minute_and_hour_mapping(self):
        expected = {
            "1m": "1",
            "5m": "5",
            "15m": "15",
            "30m": "30",
            "1h": "60",
            "60m": "60",
        }
        for interval, resolution in expected.items():
            with self.subTest(interval=interval):
                self.assertEqual(self._resolution_for(interval), resolution)

    def test_4h_is_fetched_as_60m_for_the_database_to_aggregate(self):
        self.assertEqual(self._resolution_for("4h"), "60")

    def test_daily_and_weekly_mapping(self):
        self.assertEqual(self._resolution_for("1d"), "D")
        self.assertEqual(self._resolution_for("1w"), "W")
        self.assertEqual(self._resolution_for("1wk"), "W")


class TestRangeWindow(unittest.TestCase):
    def test_from_to_are_unix_seconds_and_midnight_end_includes_the_whole_day(self):
        # RequestData semantics: end 2026-10-21 00:00 -> boundary 2026-10-22
        # 00:00, and `to` must be one second before that boundary.
        with patch(
            "AlphaSense.api.EulerpoolAPI.requests.get",
            return_value=_FakeResponse(_payload([], [], [], [], [], [])),
        ) as get:
            EulerpoolAPI(
                datetime(2026, 1, 1), datetime(2026, 10, 21), "1d", "AAPL"
            ).get_json_api()

        params = get.call_args.kwargs["params"]
        self.assertEqual(params["from"], _epoch(2026, 1, 1))
        self.assertEqual(params["to"], _epoch(2026, 10, 22) - 1)

    def test_non_midnight_end_keeps_exclusive_instant_semantics(self):
        with patch(
            "AlphaSense.api.EulerpoolAPI.requests.get",
            return_value=_FakeResponse(_payload([], [], [], [], [], [])),
        ) as get:
            EulerpoolAPI(
                datetime(2026, 1, 1), datetime(2026, 1, 2, 12, 0), "60m", "AAPL"
            ).get_json_api()

        params = get.call_args.kwargs["params"]
        self.assertEqual(params["to"], _epoch(2026, 1, 2, 12, 0) - 1)

    def test_out_of_range_bars_are_dropped(self):
        payload = _payload(
            [
                _epoch(2026, 1, 1, 23, 30),  # before start
                _epoch(2026, 1, 2, 0, 0),    # exactly start -> kept
                _epoch(2026, 1, 2, 6, 0),    # inside -> kept
                _epoch(2026, 1, 2, 12, 0),   # exactly end boundary -> dropped
                _epoch(2026, 1, 2, 12, 30),  # after boundary -> dropped
            ],
            [100] * 5, [101] * 5, [99] * 5, [100] * 5, [1000] * 5,
        )
        with patch(
            "AlphaSense.api.EulerpoolAPI.requests.get",
            return_value=_FakeResponse(payload),
        ):
            data = EulerpoolAPI(
                datetime(2026, 1, 2, 0, 0),
                datetime(2026, 1, 2, 12, 0),
                "60m",
                "AAPL",
            ).get_json_api()

        self.assertEqual(
            list(data), ["2026-01-02 00:00:00", "2026-01-02 06:00:00"]
        )

    def test_daily_duplicates_deduplicate_on_the_utc_date_marker(self):
        # Real Eulerpool D responses sometimes stamp the same trading day
        # twice; flooring to the date marker collapses them (last wins).
        payload = _payload(
            [_epoch(2026, 1, 2, 20, 0), _epoch(2026, 1, 2, 23, 0)],
            [100, 200], [101, 201], [99, 199], [100.5, 200.5], [10, 20],
        )
        with patch(
            "AlphaSense.api.EulerpoolAPI.requests.get",
            return_value=_FakeResponse(payload),
        ):
            data = EulerpoolAPI(
                datetime(2026, 1, 1), datetime(2026, 1, 2), "1d", "AAPL"
            ).get_json_api()

        self.assertEqual(list(data), ["2026-01-02 00:00:00"])
        self.assertEqual(data["2026-01-02 00:00:00"]["close"], 200.5)
        self.assertEqual(data["2026-01-02 00:00:00"]["volume"], 20)


class TestErrorsAndEmptyResponses(unittest.TestCase):
    def test_empty_arrays_return_empty_data(self):
        with patch(
            "AlphaSense.api.EulerpoolAPI.requests.get",
            return_value=_FakeResponse(_payload([], [], [], [], [], [])),
        ):
            result = EulerpoolAPI(
                datetime(2026, 1, 1), datetime(2026, 1, 2), "1d", "AAPL"
            ).get_standard_json()

        self.assertEqual(result, {"symbol": "AAPL", "data": {}})

    def test_missing_arrays_return_empty_data(self):
        with patch(
            "AlphaSense.api.EulerpoolAPI.requests.get",
            return_value=_FakeResponse({}),
        ):
            data = EulerpoolAPI(
                datetime(2026, 1, 1), datetime(2026, 1, 2), "1d", "AAPL"
            ).get_json_api()

        self.assertEqual(data, {})

    def test_http_errors_propagate_without_daily_fallback(self):
        for status in (404, 500):
            with self.subTest(status=status):
                with patch(
                    "AlphaSense.api.EulerpoolAPI.requests.get",
                    return_value=_FakeResponse({}, status_code=status),
                ) as get:
                    with self.assertRaises(requests.HTTPError):
                        EulerpoolAPI(
                            datetime(2026, 1, 1),
                            datetime(2026, 1, 2),
                            "60m",
                            "AAPL",
                        ).get_json_api()

                # Exactly one request: no silent fallback to another endpoint.
                self.assertEqual(get.call_count, 1)
                self.assertIn("/api/1/charting/ohlcv/AAPL", get.call_args.args[0])
                self.assertNotIn("/v1/equities/", get.call_args.args[0])

    def test_token_is_sent_as_a_query_parameter(self):
        with patch.dict(os.environ, {"EULERPOOL_API_KEY": "test-token"}):
            with patch(
                "AlphaSense.api.EulerpoolAPI.requests.get",
                return_value=_FakeResponse(_payload([], [], [], [], [], [])),
            ) as get:
                EulerpoolAPI(
                    datetime(2026, 1, 1), datetime(2026, 1, 2), "1d", "AAPL"
                ).get_json_api()

        params = get.call_args.kwargs["params"]
        self.assertEqual(params["token"], "test-token")
        self.assertNotIn("headers", get.call_args.kwargs)


class TestCanonicalSchemaMatchesYahoo(unittest.TestCase):
    def test_record_keys_equal_yahoo_get_standard_json_keys(self):
        euler_payload = _payload(
            [_epoch(2026, 10, 21, 19, 30)],
            [100.0], [101.0], [99.0], [100.5], [1000],
        )
        with patch(
            "AlphaSense.api.EulerpoolAPI.requests.get",
            return_value=_FakeResponse(euler_payload),
        ):
            euler = EulerpoolAPI(
                datetime(2026, 10, 20), datetime(2026, 10, 21), "60m", "AAPL"
            ).get_standard_json()

        with patch("AlphaSense.api.YahooAPI.yf.download", return_value=_yahoo_frame()):
            with patch.object(YahooAPI, "_get_daily_closes", return_value={}):
                yahoo = YahooAPI(
                    datetime(2026, 10, 20), datetime(2026, 10, 21), "60m", "AAPL"
                ).get_standard_json()

        euler_record = next(iter(euler["data"].values()))
        yahoo_record = next(iter(yahoo["data"].values()))
        self.assertEqual(
            list(euler.keys()),
            list(yahoo.keys()),
        )
        self.assertEqual(
            sorted(euler_record.keys()),
            sorted(yahoo_record.keys()),
        )
        self.assertEqual(
            sorted(euler_record.keys()),
            ["close", "high", "low", "open", "volume"],
        )

    def test_raw_non_adjusted_values_are_not_transformed(self):
        # Regression guard for AC6: the charting/ohlcv feed is raw and must
        # pass through untouched (no split/dividend adjustment whatsoever).
        raw = (271.01, 277.84, 269.0, 271.01, 37838100)
        payload = _payload(
            [_epoch(2026, 1, 2, 21, 0)],
            [raw[0]], [raw[1]], [raw[2]], [raw[3]], [raw[4]],
        )
        with patch(
            "AlphaSense.api.EulerpoolAPI.requests.get",
            return_value=_FakeResponse(payload),
        ):
            record = EulerpoolAPI(
                datetime(2026, 1, 1), datetime(2026, 1, 2), "1d", "AAPL"
            ).get_json_api()["2026-01-02 00:00:00"]

        self.assertEqual(
            (record["open"], record["high"], record["low"], record["close"], record["volume"]),
            raw,
        )


if __name__ == "__main__":
    unittest.main()
