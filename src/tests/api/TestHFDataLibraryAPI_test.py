import os
import tempfile
import unittest
import warnings
from datetime import datetime
from unittest.mock import patch

import pandas as pd

from AlphaSense.api.HFDataLibraryAPI import (
    HFDataLibraryAPI,
    HFDataLibraryAuthError,
    HFDataLibraryError,
    SplitAdjustedPricesError,
    SplitAdjustedPricesWarning,
)

EASTERN = "America/New_York"


def _frame(periods=90, start="2026-01-02 09:30:00"):
    """
    Fabricated 1-minute bars with deterministic OHLCV so every aggregation can
    be checked arithmetically (open=100+i, high=102+i, low=98+i,
    close=101+i, volume=1).
    """
    timestamps = pd.date_range(start, periods=periods, freq="min")
    return pd.DataFrame(
        {
            "datetime": timestamps,
            "open": [100.0 + i for i in range(periods)],
            "high": [102.0 + i for i in range(periods)],
            "low": [98.0 + i for i in range(periods)],
            "close": [101.0 + i for i in range(periods)],
            "volume": [1 for _ in range(periods)],
        }
    )


def _write_parquet(frame, directory, name="fixture.parquet"):
    path = os.path.join(directory, name)
    frame.to_parquet(path, engine="pyarrow")
    return path


def _api(interval="1m", start=datetime(2026, 1, 2), end=datetime(2026, 1, 2),
         version="raw", cache_dir=None, allow_split_adjusted=True):
    # Parsing/aggregation tests deliberately accept split-adjusted data; the
    # default hard gate is exercised explicitly in TestNonAdjustedPolicy.
    return HFDataLibraryAPI(
        start,
        end,
        interval,
        "AAPL",
        version=version,
        api_key="test-key",
        cache_dir=cache_dir,
        allow_split_adjusted=allow_split_adjusted,
    )


def _run(path, interval="1m", start=datetime(2026, 1, 2),
         end=datetime(2026, 1, 2), version="raw"):
    """Run the public path with the network replaced by a local fixture."""
    with patch.object(HFDataLibraryAPI, "_download_parquet", return_value=path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SplitAdjustedPricesWarning)
            return _api(interval, start, end, version).get_json_api()


class _FakeResponse:
    def __init__(self, status_code, body=b"", headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        self.text = body.decode("utf-8", errors="replace")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        pass

    def iter_content(self, chunk_size=1024):
        if self._body:
            yield self._body


class TestColumnNormalisation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_capitalised_columns_parse(self):
        frame = _frame(periods=5)
        path = _write_parquet(
            frame.rename(
                columns={
                    "datetime": "datetime",
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                }
            ),
            self.tmp.name,
        )
        result = _run(path)
        self.assertEqual(len(result), 5)
        self.assertEqual(
            result["2026-01-02 14:30:00"],
            {"open": 100.0, "high": 102.0, "low": 98.0, "close": 101.0, "volume": 1},
        )

    def test_lowercase_columns_parse(self):
        frame = _frame(periods=5)
        path = _write_parquet(frame, self.tmp.name)
        result = _run(path)
        self.assertEqual(result["2026-01-02 14:30:00"]["close"], 101.0)

    def test_extra_source_and_adj_close_columns_are_ignored(self):
        frame = _frame(periods=2).rename(
            columns={"open": "Open", "high": "High", "low": "Low",
                     "close": "Close", "volume": "Volume"}
        )
        frame["source"] = "iex"
        frame["Adj Close"] = [1.0, 1.0]
        path = _write_parquet(frame, self.tmp.name)
        result = _run(path)
        self.assertEqual(
            sorted(result["2026-01-02 14:30:00"]),
            ["close", "high", "low", "open", "volume"],
        )

    def test_datetime_as_index_parses(self):
        frame = _frame(periods=5).set_index("datetime")
        path = _write_parquet(frame, self.tmp.name)
        result = _run(path)
        self.assertEqual(len(result), 5)
        self.assertIn("2026-01-02 14:30:00", result)

    def test_datetime_as_unnamed_index_parses(self):
        # pyarrow/pandas can round-trip an unnamed index as "__index_level_0__".
        frame = _frame(periods=3).set_index("datetime")
        frame.index.name = None
        path = _write_parquet(frame, self.tmp.name)
        result = _run(path)
        self.assertEqual(len(result), 3)

    def test_date_and_time_split_parses(self):
        frame = _frame(periods=2)
        frame["Date"] = frame["datetime"].dt.strftime("%m/%d/%Y")
        frame["Time"] = frame["datetime"].dt.strftime("%H%M")
        frame = frame.drop(columns=["datetime"])
        path = _write_parquet(frame, self.tmp.name)
        result = _run(path)
        self.assertIn("2026-01-02 14:30:00", result)
        self.assertIn("2026-01-02 14:31:00", result)


class TestTimestampConversion(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_est_bar_0930_becomes_1430_utc(self):
        path = _write_parquet(_frame(periods=1), self.tmp.name)
        result = _run(path)
        self.assertIn("2026-01-02 14:30:00", result)
        self.assertNotIn("2026-01-02 09:30:00", result)

    def test_edt_bar_0930_becomes_1330_utc(self):
        frame = _frame(periods=1, start="2026-07-01 09:30:00")
        path = _write_parquet(frame, self.tmp.name)
        result = _run(
            path,
            start=datetime(2026, 7, 1),
            end=datetime(2026, 7, 1),
        )
        self.assertIn("2026-07-01 13:30:00", result)
        self.assertNotIn("2026-07-01 09:30:00", result)


class TestEndBoundary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_midnight_end_includes_the_whole_end_day(self):
        path = _write_parquet(_frame(periods=6), self.tmp.name)
        result = _run(
            path,
            start=datetime(2026, 1, 1),
            end=datetime(2026, 1, 2),  # midnight
        )
        self.assertEqual(len(result), 6)
        self.assertIn("2026-01-02 14:35:00", result)

    def test_non_midnight_end_stays_exclusive(self):
        path = _write_parquet(_frame(periods=6), self.tmp.name)
        result = _run(
            path,
            start=datetime(2026, 1, 2),
            # Naive datetimes are UTC. 14:35 UTC == 09:35 ET, so only
            # 09:30-09:34 ET (14:30-14:34 UTC) survive.
            end=datetime(2026, 1, 2, 14, 35),
        )
        self.assertEqual(len(result), 5)
        self.assertIn("2026-01-02 14:34:00", result)
        self.assertNotIn("2026-01-02 14:35:00", result)


class TestAggregation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = _write_parquet(_frame(periods=90), cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _first(self, interval):
        result = _run(self.path, interval=interval)
        first_key = next(iter(result))
        return first_key, result[first_key], result

    def test_1m_passes_every_minute_through(self):
        key, record, result = self._first("1m")
        self.assertEqual(key, "2026-01-02 14:30:00")
        self.assertEqual(len(result), 90)
        self.assertEqual(record["open"], 100.0)
        self.assertEqual(record["volume"], 1)
        self.assertIn("2026-01-02 15:59:00", result)

    def test_5m_ohlcv(self):
        key, record, _ = self._first("5m")
        self.assertEqual(key, "2026-01-02 14:30:00")
        self.assertEqual(record["open"], 100.0)
        self.assertEqual(record["high"], 106.0)
        self.assertEqual(record["low"], 98.0)
        self.assertEqual(record["close"], 105.0)
        self.assertEqual(record["volume"], 5)

    def test_15m_ohlcv(self):
        key, record, _ = self._first("15m")
        self.assertEqual(key, "2026-01-02 14:30:00")
        self.assertEqual(record["open"], 100.0)
        self.assertEqual(record["high"], 116.0)
        self.assertEqual(record["low"], 98.0)
        self.assertEqual(record["close"], 115.0)
        self.assertEqual(record["volume"], 15)

    def test_30m_ohlcv(self):
        key, record, _ = self._first("30m")
        self.assertEqual(key, "2026-01-02 14:30:00")
        self.assertEqual(record["open"], 100.0)
        self.assertEqual(record["high"], 131.0)
        self.assertEqual(record["low"], 98.0)
        self.assertEqual(record["close"], 130.0)
        self.assertEqual(record["volume"], 30)

    def test_60m_and_1h_ohlcv(self):
        for interval in ("60m", "1h"):
            with self.subTest(interval=interval):
                # resample(..., label="left", closed="left") anchors 60-minute
                # buckets to whole UTC hours. 09:30 ET == 14:30 UTC therefore
                # lands in the [14:00, 15:00) bucket and that first bucket only
                # contains 09:30-09:59 ET.
                key, record, _ = self._first(interval)
                self.assertEqual(key, "2026-01-02 14:00:00")
                self.assertEqual(record["open"], 100.0)
                self.assertEqual(record["high"], 131.0)
                self.assertEqual(record["low"], 98.0)
                self.assertEqual(record["close"], 130.0)
                self.assertEqual(record["volume"], 30)

    def test_4h_ohlcv(self):
        key, record, _ = self._first("4h")
        # 09:30-10:59 ET == 14:30-15:59 UTC, inside the 12:00-16:00 UTC bucket.
        self.assertEqual(key, "2026-01-02 12:00:00")
        self.assertEqual(record["open"], 100.0)
        self.assertEqual(record["high"], 191.0)
        self.assertEqual(record["low"], 98.0)
        self.assertEqual(record["close"], 190.0)
        self.assertEqual(record["volume"], 90)

    def test_1d_ohlcv_is_a_midnight_utc_date_marker(self):
        key, record, _ = self._first("1d")
        self.assertEqual(key, "2026-01-02 00:00:00")
        self.assertEqual(record["open"], 100.0)
        self.assertEqual(record["high"], 191.0)
        self.assertEqual(record["low"], 98.0)
        self.assertEqual(record["close"], 190.0)
        self.assertEqual(record["volume"], 90)

    def test_1wk_is_monday_anchored(self):
        # 2026-01-02 is a Friday; its Monday-anchored week starts 2025-12-29.
        key, record, _ = self._first("1wk")
        self.assertEqual(key, "2025-12-29 00:00:00")
        self.assertEqual(record["open"], 100.0)
        self.assertEqual(record["high"], 191.0)
        self.assertEqual(record["low"], 98.0)
        self.assertEqual(record["close"], 190.0)
        self.assertEqual(record["volume"], 90)


class TestCanonicalSchema(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_record_schema_is_locked_and_json_native(self):
        path = _write_parquet(_frame(periods=3), self.tmp.name)
        with patch.object(HFDataLibraryAPI, "_download_parquet", return_value=path):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SplitAdjustedPricesWarning)
                result = _api().get_standard_json()

        self.assertEqual(list(result.keys()), ["symbol", "data"])
        self.assertEqual(result["symbol"], "AAPL")
        for record in result["data"].values():
            self.assertEqual(
                sorted(record.keys()),
                ["close", "high", "low", "open", "volume"],
            )
            for value in record.values():
                self.assertIsInstance(value, (int, float))
                self.assertNotIsInstance(value, bool)


class TestMissingKey(unittest.TestCase):
    def test_missing_key_raises_actionable_error(self):
        with self.assertRaises(ValueError) as ctx:
            HFDataLibraryAPI(
                datetime(2026, 1, 2),
                datetime(2026, 1, 2),
                "1m",
                "AAPL",
                api_key="",
            )
        message = str(ctx.exception)
        self.assertIn("HFDATALIBRARY_API_KEY", message)
        self.assertIn("hfdatalibrary.com/pages/account", message)


class TestNonAdjustedPolicy(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = _write_parquet(_frame(periods=3), self.tmp.name)

    def test_source_flags_that_prices_are_split_adjusted(self):
        self.assertTrue(HFDataLibraryAPI.PRICES_ARE_SPLIT_ADJUSTED)
        self.assertFalse(HFDataLibraryAPI.SUPPORTS_NON_ADJUSTED_PRICES)
        self.assertFalse(HFDataLibraryAPI.VOLUME_IS_CONSOLIDATED)

    def test_default_version_is_the_least_processed_raw_variant(self):
        self.assertEqual(HFDataLibraryAPI.DEFAULT_VERSION, "raw")
        with patch.object(
            HFDataLibraryAPI, "_download_parquet", return_value=self.path
        ) as download:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SplitAdjustedPricesWarning)
                _api().get_json_api()
        download.assert_called_once_with("AAPL", "raw")

    def test_version_is_configurable_and_forwarded_to_the_download(self):
        with patch.object(
            HFDataLibraryAPI, "_download_parquet", return_value=self.path
        ) as download:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SplitAdjustedPricesWarning)
                _api(version="clean").get_json_api()
        download.assert_called_once_with("AAPL", "clean")

    def test_invalid_version_is_rejected(self):
        with self.assertRaises(ValueError):
            _api(version="adjusted")

    def test_allow_split_adjusted_defaults_to_false(self):
        api = HFDataLibraryAPI(
            datetime(2026, 1, 2),
            datetime(2026, 1, 2),
            "1m",
            "AAPL",
            api_key="test-key",
            cache_dir=self.tmp.name,
        )
        self.assertFalse(api.allow_split_adjusted)

    def test_default_raises_before_any_download_or_cache_read(self):
        # If the gate were after materialisation, this side effect would make
        # the test fail with AssertionError instead of the policy error.
        with patch.object(
            HFDataLibraryAPI,
            "_download_parquet",
            side_effect=AssertionError("no download/cache read allowed"),
        ):
            with self.assertRaises(SplitAdjustedPricesError) as ctx:
                _api(allow_split_adjusted=False).get_json_api()

        self.assertIsInstance(ctx.exception, ValueError)
        message = str(ctx.exception)
        self.assertIn("NON-ADJUSTED", message)
        self.assertIn("clean", message)
        self.assertIn("allow_split_adjusted=True", message)

    def test_override_allows_data_and_still_warns(self):
        with patch.object(HFDataLibraryAPI, "_download_parquet", return_value=self.path):
            with self.assertWarns(SplitAdjustedPricesWarning) as ctx:
                result = _api(allow_split_adjusted=True).get_json_api()

        self.assertIn("SPLIT-ADJUSTED", str(ctx.warning))
        self.assertEqual(len(result), 3)


class TestDownloadAndCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = _write_parquet(_frame(periods=3), self.tmp.name)
        with open(self.fixture, "rb") as handle:
            self.parquet_bytes = handle.read()

    def test_401_is_explicit(self):
        api = _api(cache_dir=os.path.join(self.tmp.name, "cache"))
        with patch(
            "AlphaSense.api.HFDataLibraryAPI.requests.get",
            return_value=_FakeResponse(401),
        ):
            with self.assertRaises(HFDataLibraryAuthError) as ctx:
                api._download_parquet("AAPL", "raw")
        self.assertIn("API key", str(ctx.exception))

    def test_403_explains_the_profile_requirement(self):
        api = _api(cache_dir=os.path.join(self.tmp.name, "cache"))
        with patch(
            "AlphaSense.api.HFDataLibraryAPI.requests.get",
            return_value=_FakeResponse(403),
        ):
            with self.assertRaises(HFDataLibraryAuthError) as ctx:
                api._download_parquet("AAPL", "raw")
        self.assertIn("403", str(ctx.exception))
        self.assertIn("profile", str(ctx.exception))
        self.assertIn("institution, country, role", str(ctx.exception))

    def test_429_is_retried_and_honours_retry_after(self):
        cache_dir = os.path.join(self.tmp.name, "cache")
        api = _api(cache_dir=cache_dir)
        responses = [
            _FakeResponse(429, headers={"Retry-After": "0"}),
            _FakeResponse(200, body=self.parquet_bytes),
        ]
        with patch(
            "AlphaSense.api.HFDataLibraryAPI.requests.get",
            side_effect=responses,
        ), patch("AlphaSense.api.HFDataLibraryAPI.time_module.sleep") as sleep:
            path = api._download_parquet("AAPL", "raw")

        self.assertTrue(os.path.exists(path))
        sleep.assert_called_once()
        self.assertEqual(
            [name for name in os.listdir(cache_dir) if name.endswith(".tmp")],
            [],
        )

    def test_fresh_cache_avoids_the_network(self):
        cache_dir = os.path.join(self.tmp.name, "cache")
        os.makedirs(cache_dir)
        api = _api(cache_dir=cache_dir)
        cache_file = api._cache_path("AAPL", "raw")
        with open(cache_file, "wb") as handle:
            handle.write(self.parquet_bytes)

        with patch(
            "AlphaSense.api.HFDataLibraryAPI.requests.get",
            side_effect=AssertionError("network must not be used"),
        ):
            path = api._download_parquet("AAPL", "raw")

        self.assertEqual(path, cache_file)
        self.assertTrue(api.last_cache_hit)

    def test_download_is_published_atomically(self):
        cache_dir = os.path.join(self.tmp.name, "cache")
        api = _api(cache_dir=cache_dir)
        with patch(
            "AlphaSense.api.HFDataLibraryAPI.requests.get",
            return_value=_FakeResponse(200, body=self.parquet_bytes),
        ):
            path = api._download_parquet("AAPL", "raw")

        self.assertTrue(os.path.exists(path))
        self.assertEqual(
            [name for name in os.listdir(cache_dir) if name.endswith(".tmp")],
            [],
        )

    def test_invalid_parquet_body_is_not_cached(self):
        cache_dir = os.path.join(self.tmp.name, "cache")
        api = _api(cache_dir=cache_dir)
        with patch(
            "AlphaSense.api.HFDataLibraryAPI.requests.get",
            return_value=_FakeResponse(200, body=b"<html>not parquet</html>"),
        ):
            with self.assertRaises(HFDataLibraryError):
                api._download_parquet("AAPL", "raw")

        self.assertFalse(os.path.exists(api._cache_path("AAPL", "raw")))


class TestLiveSmoke(unittest.TestCase):
    """
    AC9 live smoke: one real AAPL 1-day fetch against the provider.

    Skipped automatically when HFDATALIBRARY_API_KEY is absent (CI, fresh
    clones). It uses a throwaway cache directory so it never touches the
    user's real ~/.cache/hfdatalibrary, and it opts into split-adjusted data
    explicitly because that is the only basis the source offers.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # Load the repo .env the same way the API class does.
        from dotenv import load_dotenv

        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        load_dotenv(os.path.join(repo_root, ".env"))
        self.key = os.getenv("HFDATALIBRARY_API_KEY", "").strip()
        if not self.key:
            self.skipTest("HFDATALIBRARY_API_KEY not set; live smoke skipped.")

    def test_live_aapl_one_day_matches_canonical_schema(self):
        api = HFDataLibraryAPI(
            datetime(2026, 10, 20),
            datetime(2026, 10, 21),
            "1h",
            "AAPL",
            version="raw",
            api_key=self.key,
            cache_dir=os.path.join(self.tmp.name, "cache"),
            allow_split_adjusted=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SplitAdjustedPricesWarning)
            payload = api.get_standard_json()

        self.assertEqual(payload["symbol"], "AAPL")
        data = payload["data"]
        self.assertGreater(len(data), 0, "live fetch returned no records")

        for timestamp, record in data.items():
            # Canonical timestamp shape and exact record keys.
            datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            self.assertEqual(
                set(record.keys()),
                {"open", "high", "low", "close", "volume"},
            )
            for value in record.values():
                self.assertIsInstance(value, (int, float))
                self.assertNotIsInstance(value, bool)
            # OHLC sanity: high >= low and high >= open/close.
            self.assertGreaterEqual(record["high"], record["low"])
            self.assertGreaterEqual(record["high"], record["open"])
            self.assertGreaterEqual(record["high"], record["close"])

        # The window is [2026-10-20, 2026-10-22) UTC; every bar must fall in it.
        for timestamp in data:
            self.assertGreaterEqual(timestamp, "2026-10-20 00:00:00")
            self.assertLess(timestamp, "2026-10-22 00:00:00")


if __name__ == "__main__":
    unittest.main()
