from .GenericAPI import GenericAPI
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict
import json
import ast
import math


class YahooAPI(GenericAPI):
    """
    YahooAPI Class
    Class for requesting Yahoo Stock market.

    OUTPUT CONVENTIONS (canonical contract shared with every other source):
    - Prices are NON-ADJUSTED (raw). `auto_adjust=False` is passed explicitly
      on every yfinance call: cross-source and cross-time coherence requires
      the same raw prices Eulerpool / HF Data Library return, so splits and
      dividends are never baked into stored candles.
    - Timestamps are UTC strings "YYYY-MM-DD HH:MM:SS". Intraday bars are true
      UTC instants (09:30 New York -> 14:30 UTC in winter, 13:30 UTC in
      summer). Native daily bars stay date markers at "YYYY-MM-DD 00:00:00"
      (they are dates, not instants), which is why they are not tz-converted.
    - `end_date` is inclusive as a calendar day. yfinance's `end` is
      EXCLUSIVE, so both fetches pass `end_date + 1 day`; a midnight end then
      includes that whole day instead of silently dropping it.

    start_date -> datetime format, month from which the data start
    end_date -> datetime format, month from which the data end (INCLUDED; +1 day
                is applied internally for yfinance's exclusive boundary)
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
        # Filled by get_json_api(): {utc_timestamp_string: exchange_local_day}.
        # The exchange day MUST come from the original (pre-UTC) index because
        # US extended-hours bars can cross UTC midnight while still belonging to
        # the previous New York session (e.g. 2026-01-02 00:30 UTC is still the
        # 2026-01-01 session). _fix_last_candle_close() needs that mapping to
        # pick the right "last bar of the day" after timestamps become UTC.
        self._exchange_day_by_timestamp: Dict[str, str] = {}

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

    @staticmethod
    def _is_missing_number(value) -> bool:
        """
        True for None/NaN, i.e. values that cannot participate in max()/min()
        and must never reach the canonical output. `pd.NA`-like objects are not
        produced by json.loads() (which yields plain None or float NaN), hence
        the narrow math.isnan() check instead of a heavier pandas path.
        """
        if value is None:
            return True
        try:
            return math.isnan(value)
        except (TypeError, ValueError):
            return False

    def _get_daily_closes(self) -> Dict[str, float]:
        """
        Fetches Yahoo's own daily-interval close prices for the requested range,
        keyed by EXCHANGE-LOCAL date string "YYYY-MM-DD" (the daily index is a
        date marker, not an instant). Used to correct the last intraday candle
        of each exchange session, since Yahoo's intraday feed can report a
        slightly different value for that final bar than the officially settled
        close.

        `auto_adjust=False` here is deliberate and mirrors the main fetch:
        non-adjusted prices are a project-wide requirement (see class docstring).
        Mixing an adjusted intraday stream with an unadjusted daily close is
        exactly the artificial-jump bug this correction used to introduce.
        """
        daily_data = yf.download(self._action_symbol,
                                 interval="1d",
                                 start=self._format_yahoo_date(self._start_date),
                                 # yfinance's `end` is EXCLUSIVE, exactly like the
                                 # main fetch in get_json_api() (already offset by
                                 # 1). Without the offset the daily close of the
                                 # requested end day is missing - which is
                                 # precisely the last intraday bar this method
                                 # exists to correct.
                                 end=self._format_yahoo_date(self._end_date, offset_days=1),
                                 # NON-ADJUSTED on purpose: the canonical store is
                                 # raw prices, coherent with every other source.
                                 auto_adjust=False
                                 )
        if daily_data is None or daily_data.empty:
            return {}

        # Daily bars are date markers. If yfinance ever returns a tz-aware
        # index here, strftime already yields the exchange-local date; a
        # tz-naive index is the usual case and is used as-is. Either way the
        # key matches the exchange days computed in get_json_api().
        daily_data.index = pd.to_datetime(daily_data.index).strftime("%Y-%m-%d")
        json_daily = json.loads(daily_data.to_json(orient="index") or "{}")

        daily_closes = {}
        for date, fields in json_daily.items():
            for tag, value in fields.items():
                tag_tuple = ast.literal_eval(tag)
                if tag_tuple[0].lower() == "close":
                    # yfinance serialises a still-forming/incomplete session as
                    # NaN (-> None after json.loads). Such a value cannot be
                    # used as a correction target and must not be stored.
                    if self._is_missing_number(value):
                        continue
                    daily_closes[date] = value
        return daily_closes

    def _fix_last_candle_close(
            self,
            candles: Dict,
            exchange_day_by_timestamp: Dict[str, str] | None = None,
    ) -> Dict:
        """
        Returns a new candles dict where the close of the last candle of each
        EXCHANGE-LOCAL day is replaced with Yahoo's own UNADJUSTED daily close.
        Only applies to intraday intervals (minutes/hours) - daily and longer
        intervals already ARE the daily bar, so there's nothing to correct.
        High/low are widened if needed so the corrected candle stays internally
        consistent (close within [low, high]).

        Missing values (None/NaN) are tolerated: yfinance reports the last,
        still-forming session of some European symbols with a NaN close, and
        max(None, float) used to raise TypeError. A missing daily close simply
        leaves that day uncorrected, missing high/low are replaced by the valid
        daily close, and any candle still holding an incomplete OHLC value is
        dropped so the canonical output only ever contains JSON numbers.

        `exchange_day_by_timestamp` is the {utc_string: exchange_day} mapping
        produced in get_json_api() from the index BEFORE UTC conversion. It is
        required for correctness: grouping by the UTC date prefix would assign
        a 00:30 UTC extended-hours bar to the wrong session, correcting the
        wrong bar and leaving the real session's close untouched. The fallback
        (UTC prefix) only exists for direct callers that never went through
        get_json_api() - it is correct for naive daily-like strings.
        """
        intraday_intervals = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}
        if self._interval not in intraday_intervals or not candles:
            return candles

        daily_closes = self._get_daily_closes()

        if exchange_day_by_timestamp is None:
            exchange_day_by_timestamp = {ts: ts[:10] for ts in candles}

        fixed_candles = dict(candles)
        if daily_closes:
            last_timestamp_per_day: Dict[str, str] = {}
            for timestamp in candles:
                day = exchange_day_by_timestamp.get(timestamp, timestamp[:10])
                if day not in last_timestamp_per_day or timestamp > last_timestamp_per_day[day]:
                    last_timestamp_per_day[day] = timestamp

            for day, timestamp in last_timestamp_per_day.items():
                if day not in daily_closes:
                    continue
                real_close = daily_closes[day]
                if self._is_missing_number(real_close):
                    # No trustworthy daily reference for this session: leave
                    # the candle untouched (and let the cleanup below drop it
                    # if the intraday stream is incomplete as well).
                    continue
                candle = dict(fixed_candles[timestamp])
                candle["close"] = real_close
                # A None/NaN high/low cannot be compared with max()/min();
                # fall back to the valid daily close so the candle is complete.
                if self._is_missing_number(candle.get("high")):
                    candle["high"] = real_close
                else:
                    candle["high"] = max(candle["high"], real_close)
                if self._is_missing_number(candle.get("low")):
                    candle["low"] = real_close
                else:
                    candle["low"] = min(candle["low"], real_close)
                fixed_candles[timestamp] = candle

        return self._drop_incomplete_candles(fixed_candles)

    def _drop_incomplete_candles(self, candles: Dict) -> Dict:
        """
        Removes candles whose OHLC is not fully numeric, and normalises a
        missing volume to 0. This mirrors the other providers (Alpaca,
        Eulerpool, HF Data Library): an incomplete row is skipped rather than
        serialised with a None, which would violate the canonical contract and
        crash downstream numeric consumers.
        """
        complete_candles = {}
        for timestamp, candle in candles.items():
            if any(
                self._is_missing_number(candle.get(field))
                for field in ("open", "high", "low", "close")
            ):
                continue
            clean_candle = dict(candle)
            if self._is_missing_number(clean_candle.get("volume")):
                clean_candle["volume"] = 0
            complete_candles[timestamp] = clean_candle
        return complete_candles

    @staticmethod
    def _utc_strings_and_exchange_days(index: pd.Index) -> tuple[list, Dict[str, str]]:
        """
        Converts a yfinance index to the canonical UTC timestamp strings and,
        simultaneously, to exchange-local day strings.

        Both must be derived from the SAME original index: the UTC string is
        what gets stored, while the exchange day is what identifies a trading
        session. Deriving the day from the UTC string afterwards would be wrong
        for US extended-hours bars (a 00:30 UTC bar belongs to the previous New
        York day).

        tz-aware intraday indexes are converted with tz_convert("UTC") - never
        datetime.fromtimestamp(), which would silently use the machine's local
        timezone. tz-naive indexes are native daily bars: they are date markers,
        so they stay at 00:00:00 and are NOT shifted.
        """
        index = pd.DatetimeIndex(pd.to_datetime(index))
        exchange_days = index.strftime("%Y-%m-%d")
        if index.tz is not None:
            utc_strings = index.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S")
        else:
            utc_strings = index.strftime("%Y-%m-%d %H:%M:%S")
        return list(utc_strings), dict(zip(utc_strings, exchange_days))
    
    def get_json_api(self) -> Dict:
        """
        Raw Yahoo fetch, keyed by canonical UTC strings. `exchange_day_by_timestamp`
        is captured here (from the pre-UTC index) so the daily-close correction
        can group bars by exchange session later.

        `end` is offset by +1 day because yfinance treats it as EXCLUSIVE: a
        midnight end date must include that whole calendar day.

        `auto_adjust=False` is explicit and non-negotiable: the canonical store
        holds raw, non-adjusted prices for cross-source coherence. Relying on
        yfinance's default (auto_adjust=True) is what used to mix adjusted
        intraday bars with the unadjusted daily close below.
        """
        data = yf.download(self._action_symbol,
                           interval=self.interval,
                           start=self._format_yahoo_date(self._start_date),
                           # yfinance `end` is exclusive -> +1 day so a
                           # midnight end_date includes its whole day.
                           end=self._format_yahoo_date(self._end_date, offset_days=1),
                           # NON-ADJUSTED on purpose: project-wide canonical
                           # requirement shared with every other data source.
                           auto_adjust=False
                           )
        if data is None or data.empty:
            # yfinance returns an EMPTY DataFrame (not None) for unsupported
            # range/interval combinations, e.g. 5 years of 60m bars. Previously
            # this silently produced {"symbol": ..., "data": {}} and the
            # backtest ran on zero rows. Fail loudly and say exactly which
            # request was empty so the caller can adjust its range/interval.
            raise ValueError(
                f"Yahoo API returned no data for symbol "
                f"{self._action_symbol!r} at interval {self.interval!r} "
                f"between {self._format_yahoo_date(self._start_date)} and "
                f"{self._format_yahoo_date(self._end_date)} (inclusive). "
                f"This range/interval combination may be unsupported or empty."
            )
        utc_strings, self._exchange_day_by_timestamp = self._utc_strings_and_exchange_days(data.index)
        data.index = utc_strings
        return json.loads(
                data.to_json(orient="index") or "{}")

    def get_standard_json(self) -> Dict:
        json_api_data = self.get_json_api()
        # Canonical record schema is EXACTLY open/high/low/close/volume.
        # With auto_adjust=False yfinance also returns an "Adj Close" column;
        # it is deliberately dropped because the canonical store is raw prices
        # only (keeping it would leak two competing close definitions).
        canonical_fields = {"open", "high", "low", "close", "volume"}
        json_api_data_formated = {}
        for date, data in json_api_data.items():
            new_data = {}
            for tag, value in data.items():
                tag_tuple = ast.literal_eval(tag)
                field = tag_tuple[0].lower()
                if field in canonical_fields:
                    new_data[field] = value
            json_api_data_formated[date] = new_data

        # Group the last bar by EXCHANGE day (mapping built before UTC
        # conversion), never by the UTC date prefix.
        json_api_data_formated = self._fix_last_candle_close(
            json_api_data_formated,
            self._exchange_day_by_timestamp,
        )
        # _fix_last_candle_close() already sanitises intraday output, but it
        # returns EARLY for native daily/longer intervals (there is no daily
        # close to correct) and therefore never drops their incomplete bars.
        # yfinance reports the last, still-forming session of some European
        # symbols with a NaN close even on "1d", which used to reach the DB as
        # a None. Sanitising unconditionally here covers BOTH paths; the helper
        # builds a new dict, so calling it twice is safe.
        json_api_data_formated = self._drop_incomplete_candles(json_api_data_formated)
        json_api_data_with_symbol = {}
        json_api_data_with_symbol["symbol"] = self._action_symbol
        json_api_data_with_symbol["data"] = json_api_data_formated
        return json_api_data_with_symbol
