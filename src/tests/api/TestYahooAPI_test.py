import unittest
from datetime import datetime
from unittest.mock import patch

import pandas as pd

from AlphaSense.api.YahooAPI import YahooAPI

NY = "America/New_York"


def _yahoo_frame(closes=(100.0, 200.0), index=None):
    """
    Mimics yfinance's MultiIndex-columns frame closely enough for to_json().

    "Adj Close" is included on purpose: with auto_adjust=False real yfinance
    returns it, and get_standard_json() must drop it to keep the canonical
    record schema at exactly OHLCV.
    """
    if index is None:
        index = pd.to_datetime(["2026-10-20 15:30:00", "2026-10-21 15:30:00"])
    columns = pd.MultiIndex.from_tuples(
        [(name, "AAPL") for name in ("Open", "High", "Low", "Close", "Adj Close", "Volume")]
    )
    rows = [
        [close, close, close, close, close * 0.9, 1000]
        for close in closes
    ]
    return pd.DataFrame(rows, index=index, columns=columns)


def _ny_index(timestamps):
    """ Exchange-local New York index, as returned by yfinance for intraday. """
    return pd.DatetimeIndex(timestamps).tz_localize(NY)


class TestFormatYahooDate(unittest.TestCase):
    def test_no_offset(self):
        self.assertEqual(YahooAPI._format_yahoo_date(datetime(2026, 3, 15)), "2026-03-15")

    def test_offset_within_month(self):
        self.assertEqual(YahooAPI._format_yahoo_date(datetime(2026, 3, 15), offset_days=1), "2026-03-16")

    def test_offset_across_month_boundary(self):
        # The original inline version reused the pre-offset month/year with
        # only the day advanced, producing "2026-01-01" here instead of the
        # correct "2026-02-01".
        self.assertEqual(YahooAPI._format_yahoo_date(datetime(2026, 1, 31), offset_days=1), "2026-02-01")

    def test_offset_across_year_boundary(self):
        self.assertEqual(YahooAPI._format_yahoo_date(datetime(2026, 12, 31), offset_days=1), "2027-01-01")

    def test_offset_across_leap_day(self):
        self.assertEqual(YahooAPI._format_yahoo_date(datetime(2028, 2, 28), offset_days=1), "2028-02-29")

    def test_zero_padding(self):
        self.assertEqual(YahooAPI._format_yahoo_date(datetime(2026, 1, 5)), "2026-01-05")


class TestNonAdjustedPolicy(unittest.TestCase):
    START = datetime(2026, 10, 20)
    END = datetime(2026, 10, 21)

    def test_daily_download_is_explicitly_non_adjusted(self):
        with patch(
            "AlphaSense.api.YahooAPI.yf.download", return_value=_yahoo_frame()
        ) as download:
            YahooAPI(self.START, self.END, "1d", "AAPL").get_json_api()

        self.assertIs(download.call_args.kwargs["auto_adjust"], False)

    def test_both_intraday_and_daily_downloads_are_non_adjusted(self):
        # The main fetch used to rely on yfinance's default (auto_adjust=True)
        # while the daily-close correction used False. Both must be False so
        # the correction cannot introduce an adjusted/unadjusted jump.
        intraday = _yahoo_frame(closes=(100.0, 200.0))
        daily = _yahoo_frame(closes=(100.0, 250.0))
        with patch(
            "AlphaSense.api.YahooAPI.yf.download",
            side_effect=[intraday, daily],
        ) as download:
            YahooAPI(self.START, self.END, "60m", "AAPL").get_standard_json()

        self.assertEqual(download.call_count, 2)
        for call in download.call_args_list:
            self.assertIs(call.kwargs["auto_adjust"], False)


class TestUtcTimestamps(unittest.TestCase):
    START = datetime(2026, 1, 1)
    END = datetime(2026, 1, 2)

    def test_winter_intraday_bar_is_converted_to_utc(self):
        # 09:30 EST (UTC-5) -> 14:30 UTC.
        frame = _yahoo_frame(
            closes=(100.0,),
            index=_ny_index(["2026-01-02 09:30:00"]),
        )
        with patch("AlphaSense.api.YahooAPI.yf.download", return_value=frame):
            result = YahooAPI(self.START, self.END, "60m", "AAPL").get_json_api()

        self.assertIn("2026-01-02 14:30:00", result)

    def test_summer_intraday_bar_is_converted_to_utc(self):
        # 09:30 EDT (UTC-4) -> 13:30 UTC.
        frame = _yahoo_frame(
            closes=(100.0,),
            index=_ny_index(["2026-07-01 09:30:00"]),
        )
        with patch("AlphaSense.api.YahooAPI.yf.download", return_value=frame):
            result = YahooAPI(datetime(2026, 7, 1), datetime(2026, 7, 2), "60m", "AAPL").get_json_api()

        self.assertIn("2026-07-01 13:30:00", result)

    def test_native_daily_bar_stays_a_midnight_date_marker(self):
        # Daily bars are dates, not instants: no UTC shift is applied.
        frame = _yahoo_frame(
            closes=(100.0,),
            index=pd.to_datetime(["2026-10-21"]),
        )
        with patch("AlphaSense.api.YahooAPI.yf.download", return_value=frame):
            result = YahooAPI(datetime(2026, 10, 20), datetime(2026, 10, 21), "1d", "AAPL").get_json_api()

        self.assertIn("2026-10-21 00:00:00", result)


class TestCanonicalSchema(unittest.TestCase):
    def test_record_schema_is_locked(self):
        frame = _yahoo_frame(
            closes=(100.0, 200.0),
            index=_ny_index(["2026-01-02 09:30:00", "2026-01-02 10:30:00"]),
        )
        with patch("AlphaSense.api.YahooAPI.yf.download", return_value=frame):
            with patch.object(YahooAPI, "_get_daily_closes", return_value={}):
                result = YahooAPI(
                    datetime(2026, 1, 1), datetime(2026, 1, 2), "60m", "AAPL"
                ).get_standard_json()

        self.assertEqual(list(result.keys()), ["symbol", "data"])
        self.assertEqual(result["symbol"], "AAPL")
        for record in result["data"].values():
            self.assertEqual(
                sorted(record.keys()),
                ["close", "high", "low", "open", "volume"],
            )
            self.assertIn("volume", record)
            for value in record.values():
                self.assertIsInstance(value, (int, float))
                self.assertNotIsInstance(value, bool)


class TestDailyCloseCorrectionScope(unittest.TestCase):
    START = datetime(2026, 10, 20)
    END = datetime(2026, 10, 21)

    def test_daily_interval_skips_the_second_download(self):
        # A daily request already IS the daily bar, so correcting its last
        # candle with another daily fetch cannot apply - it must not be issued.
        with patch("AlphaSense.api.YahooAPI.yf.download", return_value=_yahoo_frame()) as download:
            with patch.object(YahooAPI, "_get_daily_closes") as daily_closes:
                YahooAPI(self.START, self.END, "1d", "AAPL").get_standard_json()

        self.assertEqual(download.call_count, 1)
        daily_closes.assert_not_called()

    def test_intraday_interval_keeps_the_daily_close_correction(self):
        # The 60m fetch behind a 1D request still needs the end-day daily
        # close, so the correction (and its 2nd download) is preserved.
        intraday = _yahoo_frame(closes=(100.0, 200.0))
        daily = _yahoo_frame(closes=(100.0, 250.0))
        with patch(
            "AlphaSense.api.YahooAPI.yf.download",
            side_effect=[intraday, daily],
        ) as download:
            result = YahooAPI(self.START, self.END, "60m", "AAPL").get_standard_json()

        self.assertEqual(download.call_count, 2)
        # The end day's last bar was corrected to the daily close.
        self.assertEqual(result["data"]["2026-10-21 15:30:00"]["close"], 250.0)


class TestFixLastCandleClose(unittest.TestCase):
    def test_uses_unadjusted_daily_close_without_artificial_jump(self):
        # Simulate the old failure mode: the intraday stream says 270.27
        # (adjusted-like) while the unadjusted daily close is 271.01.
        # After correction the last bar must equal the UNADJUSTED close
        # within 0.01%, i.e. no adjusted/unadjusted jump is introduced.
        intraday = _yahoo_frame(
            closes=(270.27,),
            index=_ny_index(["2026-10-21 15:30:00"]),
        )
        daily = _yahoo_frame(
            closes=(271.01,),
            index=pd.to_datetime(["2026-10-21"]),
        )
        with patch(
            "AlphaSense.api.YahooAPI.yf.download",
            side_effect=[intraday, daily],
        ):
            result = YahooAPI(
                datetime(2026, 10, 20), datetime(2026, 10, 21), "60m", "AAPL"
            ).get_standard_json()

        # October is still EDT (UTC-4): 15:30 New York -> 19:30 UTC.
        corrected = result["data"]["2026-10-21 19:30:00"]["close"]
        self.assertEqual(corrected, 271.01)
        self.assertLess(abs(corrected - 271.01) / 271.01, 0.0001)

    def test_post_utc_midnight_bar_corrects_the_right_exchange_session(self):
        # New York session 2026-01-01 has an extended-hours bar at 19:30 EST
        # = 2026-01-02 00:30 UTC. Grouping by the UTC date prefix would treat
        # that bar as part of Jan 2 and correct the WRONG bar. The exchange-day
        # mapping must keep it in the Jan 1 session so Jan 1's close (300) goes
        # there, while Jan 2's 14:30 UTC bar gets Jan 2's close (400).
        intraday = _yahoo_frame(
            closes=(100.0, 110.0, 200.0),
            index=_ny_index([
                "2026-01-01 15:30:00",
                "2026-01-01 19:30:00",
                "2026-01-02 09:30:00",
            ]),
        )
        daily = _yahoo_frame(
            closes=(300.0, 400.0),
            index=pd.to_datetime(["2026-01-01", "2026-01-02"]),
        )
        with patch(
            "AlphaSense.api.YahooAPI.yf.download",
            side_effect=[intraday, daily],
        ):
            result = YahooAPI(
                datetime(2026, 1, 1), datetime(2026, 1, 2), "60m", "AAPL"
            ).get_standard_json()

        data = result["data"]
        # The 00:30 UTC bar belongs to the Jan 1 session.
        self.assertEqual(data["2026-01-02 00:30:00"]["close"], 300.0)
        self.assertEqual(data["2026-01-02 00:30:00"]["high"], 300.0)
        # The Jan 2 session's own last bar is corrected independently.
        self.assertEqual(data["2026-01-02 14:30:00"]["close"], 400.0)
        # Earlier Jan 1 bar untouched.
        self.assertEqual(data["2026-01-01 20:30:00"]["close"], 100.0)


class TestMissingCandleValues(unittest.TestCase):
    """
    BUG-008: yfinance returns NaN for the last, still-forming session of some
    European symbols. Once serialised this becomes None, and max(None, float)
    used to raise TypeError inside _fix_last_candle_close().
    """

    START = datetime(2026, 10, 20)
    END = datetime(2026, 10, 21)

    def _assert_no_missing_values(self, result):
        for record in result["data"].values():
            for field, value in record.items():
                self.assertIsNotNone(value, f"{field} is None")
                self.assertFalse(
                    isinstance(value, float) and pd.isna(value),
                    f"{field} is NaN",
                )

    def test_last_bar_and_daily_close_both_missing_does_not_crash(self):
        # The intraday frame's last bar is entirely NaN (as the helper derives
        # every OHLC field from `close`) and the daily frame has no usable
        # close for that day. The incomplete candle must be dropped rather
        # than emitted with None values.
        intraday = _yahoo_frame(
            closes=(100.0, float("nan")),
            index=_ny_index(["2026-10-20 15:30:00", "2026-10-21 15:30:00"]),
        )
        daily = _yahoo_frame(
            closes=(float("nan"),),
            index=pd.to_datetime(["2026-10-21"]),
        )
        with patch(
            "AlphaSense.api.YahooAPI.yf.download",
            side_effect=[intraday, daily],
        ):
            result = YahooAPI(self.START, self.END, "60m", "AAPL").get_standard_json()

        self._assert_no_missing_values(result)
        self.assertEqual(list(result["data"].keys()), ["2026-10-20 19:30:00"])

    def test_valid_intraday_close_with_missing_daily_close_is_uncorrected(self):
        # A missing daily close simply means "no correction target": the
        # valid intraday candle must survive untouched and None must not leak.
        intraday = _yahoo_frame(
            closes=(100.0, 200.0),
            index=_ny_index(["2026-10-20 15:30:00", "2026-10-21 15:30:00"]),
        )
        daily = _yahoo_frame(
            closes=(100.0, float("nan")),
            index=pd.to_datetime(["2026-10-20", "2026-10-21"]),
        )
        with patch(
            "AlphaSense.api.YahooAPI.yf.download",
            side_effect=[intraday, daily],
        ):
            result = YahooAPI(self.START, self.END, "60m", "AAPL").get_standard_json()

        self._assert_no_missing_values(result)
        # Oct 20 was corrected with its valid daily close.
        self.assertEqual(result["data"]["2026-10-20 19:30:00"]["close"], 100.0)
        # Oct 21 has no daily reference and keeps its intraday value.
        self.assertEqual(result["data"]["2026-10-21 19:30:00"]["close"], 200.0)

    def test_missing_high_low_is_replaced_by_the_daily_close(self):
        # Acceptance criterion: a candle with None high/low must not crash and
        # must not emit None; it is completed with the valid daily close.
        intraday = _yahoo_frame(
            closes=(100.0,),
            index=_ny_index(["2026-10-21 15:30:00"]),
        )
        intraday.loc[intraday.index[-1], ("High", "AAPL")] = None
        intraday.loc[intraday.index[-1], ("Low", "AAPL")] = None
        daily = _yahoo_frame(
            closes=(250.0,),
            index=pd.to_datetime(["2026-10-21"]),
        )
        with patch(
            "AlphaSense.api.YahooAPI.yf.download",
            side_effect=[intraday, daily],
        ):
            result = YahooAPI(self.START, self.END, "60m", "AAPL").get_standard_json()

        self._assert_no_missing_values(result)
        candle = result["data"]["2026-10-21 19:30:00"]
        self.assertEqual(candle["close"], 250.0)
        self.assertEqual(candle["high"], 250.0)
        self.assertEqual(candle["low"], 250.0)

    def test_native_daily_missing_close_candle_is_dropped(self):
        # A NATIVE DAILY request ("1d") makes a single download and
        # _fix_last_candle_close() returns early for it, so the daily path used
        # to bypass the incomplete-candle cleanup and leak a None close.
        # yfinance reports the still-forming European session as NaN even for
        # "1d"; that incomplete last candle must be dropped instead.
        frame = _yahoo_frame(
            closes=(100.0, float("nan")),
            index=pd.to_datetime(["2026-10-20", "2026-10-21"]),
        )
        with patch(
            "AlphaSense.api.YahooAPI.yf.download", return_value=frame
        ) as download:
            result = YahooAPI(self.START, self.END, "1d", "AAPL").get_standard_json()

        self.assertEqual(download.call_count, 1)
        self._assert_no_missing_values(result)
        self.assertEqual(list(result["data"].keys()), ["2026-10-20 00:00:00"])


class TestEmptyResponseFailsLoudly(unittest.TestCase):
    """
    BUG-006 AC3: yfinance returns an EMPTY DataFrame (not None) for unsupported
    range/interval combinations. That used to produce {"data": {}} silently and
    let a backtest run on zero rows.
    """

    START = datetime(2021, 1, 1)
    END = datetime(2026, 1, 1)

    def _raises(self, return_value):
        with patch(
            "AlphaSense.api.YahooAPI.yf.download", return_value=return_value
        ):
            with self.assertRaises(ValueError) as ctx:
                YahooAPI(self.START, self.END, "1wk", "AAPL").get_json_api()
        return str(ctx.exception)

    def _assert_message_context(self, message):
        self.assertIn("AAPL", message)
        self.assertIn("1wk", message)
        self.assertIn("2021-01-01", message)
        self.assertIn("2026-01-01", message)

    def test_empty_dataframe_raises_with_symbol_interval_and_range(self):
        self._assert_message_context(self._raises(pd.DataFrame()))

    def test_none_raises_with_symbol_interval_and_range(self):
        self._assert_message_context(self._raises(None))


if __name__ == "__main__":
    unittest.main()
