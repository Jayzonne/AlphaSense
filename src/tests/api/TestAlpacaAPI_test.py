import unittest
from datetime import datetime
from unittest.mock import patch

from AlphaSense.api.AlpacaAPI import AlpacaAPI, AlpacaAuthError, AlpacaError


def _bar(
    timestamp,
    open_=100.0,
    high=101.0,
    low=99.0,
    close=100.5,
    volume=1000,
):
    """One Alpaca bar object, including the extra fields we must drop."""
    return {
        "t": timestamp,
        "o": open_,
        "h": high,
        "l": low,
        "c": close,
        "v": volume,
        "n": 42,
        "vw": 100.25,
    }


def _payload(bars, symbol="AAPL", next_page_token=None):
    """Builds an Alpaca bars response; the token is omitted when None."""
    payload = {"bars": bars, "symbol": symbol}
    if next_page_token is not None:
        payload["next_page_token"] = next_page_token
    return payload


class _FakeResponse:
    """Minimal requests.Response stand-in; no network is ever touched."""

    def __init__(self, status_code=200, payload=None, headers=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._payload


def _api(
    interval="1d",
    start=datetime(2026, 1, 1),
    end=datetime(2026, 1, 3),
    symbol="AAPL",
    **kwargs,
):
    return AlpacaAPI(
        start,
        end,
        interval,
        symbol,
        api_key="test-key",
        api_secret="test-secret",
        **kwargs,
    )


def _fetch(payload, interval="1d", start=datetime(2026, 1, 1),
           end=datetime(2026, 1, 3), **kwargs):
    with patch(
        "AlphaSense.api.AlpacaAPI.requests.get",
        return_value=_FakeResponse(200, payload),
    ):
        return _api(interval, start, end, **kwargs).get_json_api()


class TestCanonicalSchema(unittest.TestCase):
    def test_record_schema_is_locked_and_json_native(self):
        payload = _payload(
            [_bar("2026-01-02T14:30:00Z", 100, 101.5, 99.25, 100.75, 12345)]
        )
        result = _fetch(payload, interval="1m", start=datetime(2026, 1, 2),
                        end=datetime(2026, 1, 2))

        self.assertEqual(list(result), ["2026-01-02 14:30:00"])
        record = result["2026-01-02 14:30:00"]
        self.assertEqual(
            sorted(record), ["close", "high", "low", "open", "volume"]
        )
        # Extra Alpaca fields (n, vw, t) must never leak into the record.
        for field in ("t", "n", "vw"):
            self.assertNotIn(field, record)

        # Prices are plain floats even when the source sent an int, volume is
        # a plain int, and no bool/numpy type is allowed.
        for field in ("open", "high", "low", "close"):
            self.assertIsInstance(record[field], float)
        self.assertIsInstance(record["volume"], int)
        self.assertNotIsInstance(record["volume"], bool)
        self.assertEqual(
            record,
            {
                "open": 100.0,
                "high": 101.5,
                "low": 99.25,
                "close": 100.75,
                "volume": 12345,
            },
        )

    def test_timestamps_use_the_exact_utc_string_format(self):
        payload = _payload([_bar("2026-01-02T14:30:00Z")])
        result = _fetch(payload, interval="1m", start=datetime(2026, 1, 2),
                        end=datetime(2026, 1, 2))

        timestamp = next(iter(result))
        # Raises ValueError if the key does not match the canonical shape.
        parsed = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        self.assertEqual(parsed, datetime(2026, 1, 2, 14, 30, 0))

    def test_get_standard_json_wraps_the_symbol(self):
        payload = _payload([_bar("2026-01-02T14:30:00Z")])
        with patch(
            "AlphaSense.api.AlpacaAPI.requests.get",
            return_value=_FakeResponse(200, payload),
        ):
            result = _api(
                "1m", datetime(2026, 1, 2), datetime(2026, 1, 2), "MSFT"
            ).get_standard_json()

        self.assertEqual(list(result), ["symbol", "data"])
        self.assertEqual(result["symbol"], "MSFT")
        self.assertIn("2026-01-02 14:30:00", result["data"])


class TestIntervalMapping(unittest.TestCase):
    def test_every_authorized_interval_maps_to_an_alpaca_timeframe(self):
        expected = {
            "1m": "1Min",
            "5m": "5Min",
            "15m": "15Min",
            "30m": "30Min",
            "60m": "1Hour",
            "1h": "1Hour",
            "4h": "4Hour",
            "1d": "1Day",
            "1wk": "1Week",
            "1w": "1Week",
        }
        for interval, timeframe in expected.items():
            with self.subTest(interval=interval):
                with patch(
                    "AlphaSense.api.AlpacaAPI.requests.get",
                    return_value=_FakeResponse(
                        200, _payload([_bar("2026-01-02T15:00:00Z")])
                    ),
                ) as get:
                    _api(interval).get_json_api()

                self.assertEqual(
                    get.call_args.kwargs["params"]["timeframe"], timeframe
                )

    def test_authorized_values_are_exposed(self):
        self.assertEqual(
            _api().interval_authorized_values,
            ["1m", "5m", "15m", "30m", "60m", "1h", "4h", "1d", "1wk", "1w"],
        )

    def test_unsupported_interval_is_rejected(self):
        with self.assertRaises(ValueError):
            _api(interval="2m")


class TestRequestConstruction(unittest.TestCase):
    def test_raw_adjustment_and_iex_feed_are_sent_by_default(self):
        with patch(
            "AlphaSense.api.AlpacaAPI.requests.get",
            return_value=_FakeResponse(
                200, _payload([_bar("2026-01-02T15:00:00Z")])
            ),
        ) as get:
            _api(
                "1m",
                start=datetime(2026, 1, 1),
                end=datetime(2026, 1, 2),  # midnight -> boundary is Jan 3
            ).get_json_api()

        params = get.call_args.kwargs["params"]
        self.assertEqual(params["adjustment"], "raw")
        self.assertEqual(params["feed"], "iex")
        self.assertEqual(params["limit"], 10000)
        self.assertEqual(params["start"], "2026-01-01T00:00:00Z")
        self.assertEqual(params["end"], "2026-01-03T00:00:00Z")

    def test_credentials_are_sent_in_the_apca_headers(self):
        with patch(
            "AlphaSense.api.AlpacaAPI.requests.get",
            return_value=_FakeResponse(
                200, _payload([_bar("2026-01-02T15:00:00Z")])
            ),
        ) as get:
            _api("1m").get_json_api()

        headers = get.call_args.kwargs["headers"]
        self.assertEqual(headers["APCA-API-KEY-ID"], "test-key")
        self.assertEqual(headers["APCA-API-SECRET-KEY"], "test-secret")
        self.assertIn("/v2/stocks/AAPL/bars", get.call_args.args[0])


class TestTimestampHandling(unittest.TestCase):
    def test_daily_bar_is_floored_to_a_midnight_utc_date_marker(self):
        # Alpaca stamps US daily bars at the session open (05:00Z here); the
        # canonical output must be the date marker 00:00:00 UTC.
        payload = _payload([_bar("2022-02-28T05:00:00Z")])
        result = _fetch(
            payload,
            interval="1d",
            start=datetime(2022, 2, 28),
            end=datetime(2022, 2, 28),
        )

        self.assertEqual(list(result), ["2022-02-28 00:00:00"])
        self.assertNotIn("2022-02-28 05:00:00", result)

    def test_weekly_bar_is_also_floored_to_midnight(self):
        payload = _payload([_bar("2022-02-28T05:00:00Z")])
        result = _fetch(
            payload,
            interval="1wk",
            start=datetime(2022, 2, 28),
            end=datetime(2022, 3, 6),
        )

        self.assertEqual(list(result), ["2022-02-28 00:00:00"])

    def test_intraday_bar_keeps_its_true_utc_instant(self):
        payload = _payload([_bar("2022-02-28T14:30:00Z")])
        result = _fetch(
            payload,
            interval="1m",
            start=datetime(2022, 2, 28),
            end=datetime(2022, 2, 28),
        )

        self.assertEqual(list(result), ["2022-02-28 14:30:00"])


class TestEndBoundary(unittest.TestCase):
    def test_midnight_end_includes_the_whole_end_day(self):
        api = _api(end=datetime(2026, 1, 2))  # midnight
        self.assertEqual(api._end_boundary(), datetime(2026, 1, 3))

    def test_non_midnight_end_stays_exclusive(self):
        end = datetime(2026, 1, 2, 12, 0)
        self.assertEqual(_api(end=end)._end_boundary(), end)

    def test_request_end_uses_the_computed_boundary(self):
        with patch(
            "AlphaSense.api.AlpacaAPI.requests.get",
            return_value=_FakeResponse(
                200,
                _payload(
                    [
                        # Exactly at the boundary -> must be dropped.
                        _bar("2026-01-03T00:00:00Z"),
                        _bar("2026-01-02T23:59:59Z"),
                    ]
                ),
            ),
        ) as get:
            result = _api(
                "1m",
                start=datetime(2026, 1, 2),
                end=datetime(2026, 1, 2),  # midnight -> +1 day
            ).get_json_api()

        self.assertEqual(
            get.call_args.kwargs["params"]["end"], "2026-01-03T00:00:00Z"
        )
        self.assertEqual(list(result), ["2026-01-02 23:59:59"])

    def test_non_midnight_end_is_honoured_in_the_request(self):
        with patch(
            "AlphaSense.api.AlpacaAPI.requests.get",
            return_value=_FakeResponse(
                200, _payload([_bar("2026-01-02T11:59:59Z")])
            ),
        ) as get:
            result = _api(
                "1m",
                start=datetime(2026, 1, 2),
                end=datetime(2026, 1, 2, 12, 0),
            ).get_json_api()

        self.assertEqual(
            get.call_args.kwargs["params"]["end"], "2026-01-02T12:00:00Z"
        )
        self.assertEqual(list(result), ["2026-01-02 11:59:59"])


class TestPagination(unittest.TestCase):
    def test_two_pages_are_concatenated_and_stop_on_null_token(self):
        first = _FakeResponse(
            200,
            _payload(
                [_bar("2026-01-02T14:30:00Z")],
                next_page_token="page-2",
            ),
        )
        second = _FakeResponse(
            200,
            _payload(
                [_bar("2026-01-02T14:31:00Z")],
                next_page_token=None,
            ),
        )

        with patch(
            "AlphaSense.api.AlpacaAPI.requests.get",
            side_effect=[first, second],
        ) as get:
            result = _api(
                "1m", start=datetime(2026, 1, 2), end=datetime(2026, 1, 2)
            ).get_json_api()

        self.assertEqual(
            list(result),
            ["2026-01-02 14:30:00", "2026-01-02 14:31:00"],
        )
        self.assertEqual(get.call_count, 2)
        self.assertEqual(
            get.call_args_list[0].kwargs["params"].get("page_token"), None
        )
        self.assertEqual(
            get.call_args_list[1].kwargs["params"]["page_token"], "page-2"
        )

    def test_repeated_token_stops_instead_of_looping_forever(self):
        pages = [
            _FakeResponse(
                200,
                _payload(
                    [_bar("2026-01-02T14:30:00Z")],
                    next_page_token="stuck",
                ),
            ),
            _FakeResponse(
                200,
                _payload(
                    [_bar("2026-01-02T14:31:00Z")],
                    next_page_token="stuck",
                ),
            ),
        ]

        with patch(
            "AlphaSense.api.AlpacaAPI.requests.get", side_effect=pages
        ) as get:
            result = _api(
                "1m", start=datetime(2026, 1, 2), end=datetime(2026, 1, 2)
            ).get_json_api()

        self.assertEqual(get.call_count, 2)
        self.assertEqual(len(result), 2)


class TestCredentials(unittest.TestCase):
    def test_missing_key_raises_an_actionable_error(self):
        with self.assertRaises(ValueError) as ctx:
            AlpacaAPI(
                datetime(2026, 1, 2),
                datetime(2026, 1, 2),
                "1d",
                "AAPL",
                api_key="",
                api_secret="",
            )

        message = str(ctx.exception)
        self.assertIn("ALPACA_API_KEY", message)
        self.assertIn("ALPACA_SECRET_KEY", message)
        self.assertIn("APCA_API_KEY_ID", message)
        self.assertIn("APCA_API_SECRET_KEY", message)
        self.assertIn("https://alpaca.markets", message)

    def test_constructor_arguments_override_the_environment(self):
        import os

        with patch.dict(
            os.environ,
            {
                "ALPACA_API_KEY": "env-key",
                "ALPACA_SECRET_KEY": "env-secret",
            },
        ):
            api = AlpacaAPI(
                datetime(2026, 1, 2),
                datetime(2026, 1, 2),
                "1d",
                "AAPL",
                api_key="arg-key",
                api_secret="arg-secret",
            )

        self.assertEqual(api._api_key, "arg-key")
        self.assertEqual(api._api_secret, "arg-secret")

    def test_legacy_apca_environment_names_are_accepted(self):
        import os

        with patch.dict(
            os.environ,
            {
                # Empty (not absent) ALPACA_* values guarantee the legacy
                # fallback is exercised even if a developer's .env has keys.
                "ALPACA_API_KEY": "",
                "ALPACA_SECRET_KEY": "",
                "APCA_API_KEY_ID": "legacy-key",
                "APCA_API_SECRET_KEY": "legacy-secret",
            },
            clear=False,
        ):
            api = AlpacaAPI(
                datetime(2026, 1, 2), datetime(2026, 1, 2), "1d", "AAPL"
            )

        self.assertEqual(api._api_key, "legacy-key")
        self.assertEqual(api._api_secret, "legacy-secret")


class TestHttpErrors(unittest.TestCase):
    def test_401_raises_auth_error_without_retrying(self):
        with patch(
            "AlphaSense.api.AlpacaAPI.requests.get",
            return_value=_FakeResponse(401),
        ) as get:
            with self.assertRaises(AlpacaAuthError) as ctx:
                _api("1m").get_json_api()

        self.assertIsInstance(ctx.exception, AlpacaError)
        self.assertIn("401", str(ctx.exception))
        self.assertEqual(get.call_count, 1)

    def test_403_raises_auth_error_without_retrying(self):
        with patch(
            "AlphaSense.api.AlpacaAPI.requests.get",
            return_value=_FakeResponse(403),
        ) as get:
            with self.assertRaises(AlpacaAuthError) as ctx:
                _api("1m").get_json_api()

        self.assertIn("403", str(ctx.exception))
        self.assertIn("plan", str(ctx.exception))
        self.assertEqual(get.call_count, 1)

    def test_429_is_retried_honouring_retry_after(self):
        responses = [
            _FakeResponse(429, headers={"Retry-After": "2"}),
            _FakeResponse(200, _payload([_bar("2026-01-02T14:30:00Z")])),
        ]
        with patch(
            "AlphaSense.api.AlpacaAPI.requests.get", side_effect=responses
        ) as get, patch(
            "AlphaSense.api.AlpacaAPI.time_module.sleep"
        ) as sleep:
            result = _api(
                "1m", start=datetime(2026, 1, 2), end=datetime(2026, 1, 2)
            ).get_json_api()

        self.assertEqual(get.call_count, 2)
        sleep.assert_called_once_with(2.0)
        self.assertIn("2026-01-02 14:30:00", result)

    def test_5xx_is_retried_with_backoff(self):
        responses = [
            _FakeResponse(503),
            _FakeResponse(200, _payload([_bar("2026-01-02T14:30:00Z")])),
        ]
        with patch(
            "AlphaSense.api.AlpacaAPI.requests.get", side_effect=responses
        ) as get, patch(
            "AlphaSense.api.AlpacaAPI.time_module.sleep"
        ) as sleep:
            result = _api(
                "1m", start=datetime(2026, 1, 2), end=datetime(2026, 1, 2)
            ).get_json_api()

        self.assertEqual(get.call_count, 2)
        sleep.assert_called_once_with(1.5)
        self.assertIn("2026-01-02 14:30:00", result)

    def test_persistent_5xx_raises_alpaca_error(self):
        with patch(
            "AlphaSense.api.AlpacaAPI.requests.get",
            return_value=_FakeResponse(500),
        ), patch("AlphaSense.api.AlpacaAPI.time_module.sleep"):
            with self.assertRaises(AlpacaError):
                _api("1m", max_retries=2).get_json_api()

    def test_other_4xx_raises_alpaca_error(self):
        with patch(
            "AlphaSense.api.AlpacaAPI.requests.get",
            return_value=_FakeResponse(404, text="not found"),
        ) as get:
            with self.assertRaises(AlpacaError) as ctx:
                _api("1m").get_json_api()

        self.assertIn("404", str(ctx.exception))
        self.assertEqual(get.call_count, 1)


class TestEmptyResults(unittest.TestCase):
    def test_empty_bars_raise_a_clear_value_error(self):
        payload = _payload([], symbol="AAPL")
        with self.assertRaises(ValueError) as ctx:
            _fetch(
                payload,
                interval="1d",
                start=datetime(2026, 1, 1),
                end=datetime(2026, 1, 2),
            )

        message = str(ctx.exception)
        self.assertIn("AAPL", message)
        self.assertIn("1d", message)
        self.assertIn("2026-01-01", message)
        self.assertIn("2026-01-03", message)

    def test_missing_bars_key_also_raises(self):
        with self.assertRaises(ValueError):
            _fetch({"symbol": "AAPL"}, interval="1d",
                   start=datetime(2026, 1, 1), end=datetime(2026, 1, 2))


if __name__ == "__main__":
    unittest.main()
