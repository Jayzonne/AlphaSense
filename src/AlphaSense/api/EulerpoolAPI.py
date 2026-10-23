from .GenericAPI import GenericAPI
from datetime import datetime, time, timedelta, timezone
from typing import Dict
import calendar
import os
import requests


class EulerpoolAPI(GenericAPI):
    """
    EulerpoolAPI Class
    Class for requesting Eulerpool Financial Data API.

    OUTPUT CONVENTIONS (canonical contract shared with YahooAPI):
    - Prices are NON-ADJUSTED (raw). The `charting/ohlcv` endpoint returns the
      raw exchange feed (no split/dividend adjustment); we pass its o/h/l/c
      values through untouched. This is required for cross-source coherence.
    - Timestamps are UTC strings "YYYY-MM-DD HH:MM:SS". Intraday bars are true
      UTC instants (US 09:30 ET -> 13:30 UTC in EDT, 14:30 UTC in EST).
      Native daily/weekly bars are date markers emitted at "00:00:00" UTC,
      exactly like Yahoo's date-only index.
    - `end_date` is inclusive as a calendar day when it is midnight (same
      `_end_boundary()` rule as RequestData): the query `to` becomes
      end + 1 day - 1 second. A non-midnight end keeps exclusive-instant
      semantics. Returned bars are always clipped to [start, end_boundary).
    - Records contain EXACTLY open/high/low/close/volume; volume is always
      present (0 when the source reports 0).

    Source limitation (documented, not worked around): the Eulerpool
    `charting/ohlcv` endpoint only keeps roughly the latest ~100 bars for
    `resolution=60`, so 60-minute history cannot back long intraday
    backtests. Daily history is bounded by the requested from/to range.
    Daily responses can also contain duplicate UTC dates (the source sometimes
    stamps the same trading day at several times) and an occasional Saturday
    bar with zero volume; the dict keyed by UTC timestamp naturally keeps the
    last duplicate and leaves the source's weekend bar untouched rather than
    fabricating or deleting data.

    start_date -> datetime interpreted as UTC when naive
    end_date -> datetime, inclusive whole day when midnight
    interval -> one of 1m, 5m, 15m, 30m, 1h, 60m, 4h, 1d, 1w/1wk
    action_symbol -> symbol / identifier of the stock action
    """

    # Eulerpool resolution codes (not the same vocabulary as our intervals).
    # 4h is fetched as 60m on purpose: the database aggregates 1-hour candles
    # into 4-hour buckets, exactly like Yahoo's 4h -> 60m path.
    _RESOLUTION_BY_INTERVAL = {
        "1m": "1",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "60m": "60",
        "4h": "60",
        "1d": "D",
        "1w": "W",
        "1wk": "W",
    }
    _DATE_MARKER_RESOLUTIONS = {"D", "W"}

    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        interval: str,
        action_symbol: str,
    ):
        # "60m" is what RequestData sends for every interval > 60 minutes
        # (1d/4h/1wk), so it MUST be accepted here. "1wk" is the canonical
        # spelling used by the rest of the project; "1w" is kept for callers
        # that already pass Eulerpool's own spelling.
        self._interval_authorized_values = [
            "1m", "5m", "15m", "30m", "1h", "60m", "4h", "1d", "1w", "1wk",
        ]
        self._start_date = start_date
        self._end_date = end_date
        self._interval = interval
        self._action_symbol = action_symbol
        # Auth is a query parameter on Eulerpool (`?token=...`), not a bearer
        # header. Read it from the environment as required by the project.
        self._api_key = os.getenv("EULERPOOL_API_KEY", "")
        self._base_url = "https://api.eulerpool.com"

    @staticmethod
    def _to_utc_epoch(value: datetime) -> int:
        """
        Converts a datetime to a Unix timestamp, treating NAIVE datetimes as
        UTC. `datetime.timestamp()` on a naive value would silently use the
        machine's local timezone (the old bug); `calendar.timegm()` never does.
        """
        if value.tzinfo is None:
            return int(calendar.timegm(value.timetuple()))
        return int(value.timestamp())

    def _end_boundary(self) -> datetime:
        """
        Same inclusive-day rule as RequestData._end_boundary(): a midnight end
        means "up to and including the whole day", while any other instant is
        kept exclusive.
        """
        if self._end_date.time() == time.min:
            return self._end_date + timedelta(days=1)
        return self._end_date

    def _get_resolution(self) -> str:
        """Maps our interval vocabulary to Eulerpool's resolution code."""
        try:
            return self._RESOLUTION_BY_INTERVAL[self._interval]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported Eulerpool interval: {self._interval}. "
                f"Authorized values: {self._interval_authorized_values}"
            ) from exc

    @staticmethod
    def _normalise_epoch(raw_timestamp) -> float:
        """
        Seconds/milliseconds guard. Eulerpool documents Unix SECONDS, but a
        milliseconds value would land in year 56000+ if interpreted as
        seconds and poison every timestamp; anything past ~year 5138 is
        treated as milliseconds. The threshold is far above current epoch
        seconds (~1.8e9) and below current epoch milliseconds (~1.8e12).
        """
        timestamp = float(raw_timestamp)
        if abs(timestamp) >= 100_000_000_000:
            timestamp /= 1000.0
        return timestamp

    @staticmethod
    def _date_marker_epoch(epoch: float) -> int:
        """
        Floors an epoch to 00:00:00 UTC of its UTC date. Daily/weekly bars are
        date markers, not instants, so they must stay at midnight like Yahoo's
        daily index (e.g. a US close stamped 21:00 UTC still belongs to that
        date's bar).
        """
        moment = datetime.fromtimestamp(epoch, tz=timezone.utc)
        marker = moment.replace(hour=0, minute=0, second=0, microsecond=0)
        return int(marker.timestamp())

    @staticmethod
    def _as_list(value) -> list:
        """Defensive: the endpoint should return arrays, but never iterate None."""
        return value if isinstance(value, list) else []

    def _parse_ohlcv(self, payload) -> list[tuple]:
        """
        Parses the parallel-array response {t:[], o:[], h:[], l:[], c:[], v:[]}
        into (epoch, open, high, low, close, volume) rows.

        Rows with a missing/short/non-numeric OHLC component are skipped
        entirely: emitting None (or a fabricated value) would violate the
        canonical schema. Volume defaults to 0 only when its slot is absent;
        it never becomes None. Values are coerced to plain Python numbers so
        no numpy type can leak into the JSON.
        """
        if not isinstance(payload, dict):
            return []

        # Defensive support for a wrapped payload ({"data": {...arrays...}});
        # the documented shape is top-level, so this only kicks in if present.
        if "t" not in payload and isinstance(payload.get("data"), dict):
            payload = payload["data"]

        timestamps = self._as_list(payload.get("t"))
        opens = self._as_list(payload.get("o"))
        highs = self._as_list(payload.get("h"))
        lows = self._as_list(payload.get("l"))
        closes = self._as_list(payload.get("c"))
        volumes = self._as_list(payload.get("v"))

        rows = []
        for index, raw_timestamp in enumerate(timestamps):
            try:
                epoch = self._normalise_epoch(raw_timestamp)
                open_ = float(opens[index])
                high = float(highs[index])
                low = float(lows[index])
                close = float(closes[index])
            except (IndexError, TypeError, ValueError):
                # Short or non-numeric OHLC row -> skip, never emit None.
                continue

            volume = 0
            if index < len(volumes) and volumes[index] is not None:
                try:
                    volume = float(volumes[index])
                    if volume.is_integer():
                        volume = int(volume)
                except (TypeError, ValueError):
                    volume = 0

            rows.append((epoch, open_, high, low, close, volume))

        return rows

    def get_json_api(self) -> Dict:
        """
        Fetches `GET /api/1/charting/ohlcv/{identifier}` with Unix-second
        `from`/`to`, then converts the parallel arrays into the canonical
        {utc_timestamp: OHLCV} mapping.

        There is NO fallback to the old `/api/1/equity/candles` endpoint:
        that feed had no volume and mostly flat OHLC, and its `range`-based
        window ignored the exact start/end. HTTP errors now propagate.
        """
        start_epoch = self._to_utc_epoch(self._start_date)
        end_boundary_epoch = self._to_utc_epoch(self._end_boundary())
        resolution = self._get_resolution()

        url = f"{self._base_url}/api/1/charting/ohlcv/{self._action_symbol}"
        params = {
            "resolution": resolution,
            "from": start_epoch,
            # `to` is inclusive in the source, so subtract one second to keep
            # the half-open [start, end_boundary) contract.
            "to": end_boundary_epoch - 1,
        }
        if self._api_key:
            params["token"] = self._api_key

        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()

        is_date_marker = resolution in self._DATE_MARKER_RESOLUTIONS
        result: Dict[str, Dict] = {}
        for epoch, open_, high, low, close, volume in self._parse_ohlcv(payload):
            # Daily/weekly bars are date markers, so both filtering and
            # emission use the floored midnight epoch. Intraday bars use the
            # true UTC instant. This is what keeps the end-day daily bar
            # included even when the source stamps it at the market close.
            effective_epoch = (
                self._date_marker_epoch(epoch) if is_date_marker else int(epoch)
            )
            if effective_epoch < start_epoch or effective_epoch >= end_boundary_epoch:
                continue

            timestamp = datetime.fromtimestamp(
                effective_epoch, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S")
            # The dict deduplicates identical UTC timestamps (last bar wins)
            # and the sort below makes output deterministic.
            result[timestamp] = {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }

        return dict(sorted(result.items()))

    def get_standard_json(self) -> Dict:
        """Canonical top-level shape: {"symbol": ..., "data": {timestamp: OHLCV}}."""
        json_api_data_with_symbol = {
            "symbol": self._action_symbol,
            "data": self.get_json_api(),
        }
        return json_api_data_with_symbol
