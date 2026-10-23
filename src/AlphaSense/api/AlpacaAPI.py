from __future__ import annotations

import os
import time as time_module
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

from .GenericAPI import GenericAPI

# The repo `.env` lives at the repository root, three levels above this module:
# src/AlphaSense/api/AlpacaAPI.py -> src/AlphaSense/api -> src -> repo.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOTENV_PATH = _REPO_ROOT / ".env"

_DEFAULT_BACKOFF_SECONDS = 1.5
_DEFAULT_PAGE_LIMIT = 10_000

# `raw` is the non-adjusted basis required by the canonical contract. The other
# values are accepted by the source but must be opted into explicitly.
_SUPPORTED_ADJUSTMENTS = {"raw", "split", "dividend", "all"}
_SUPPORTED_FEEDS = {"iex", "sip", "otc"}


class AlpacaError(RuntimeError):
    """Raised for Alpaca API/parsing failures (not authentication)."""


class AlpacaAuthError(AlpacaError):
    """Raised for 401/403 responses, with an actionable account message."""


class AlpacaAPI(GenericAPI):
    """
    AlpacaAPI Class
    Class for requesting Alpaca Market Data (US stocks & ETFs).

    SETUP
    -----
    Alpaca requires an API key pair. Create one at https://alpaca.markets and
    expose it as `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` in the environment or in
    the repository `.env` (the legacy `APCA_API_KEY_ID` /
    `APCA_API_SECRET_KEY` names are also accepted). Constructor arguments
    override the environment. A missing key/secret raises an actionable
    ValueError before any request; there is no unauthenticated fallback.

    OUTPUT CONVENTIONS (canonical contract shared with every other source)
    ---------------------------------------------------------------------
    - Outer shape is exactly {"symbol": ..., "data": {...}}.
    - Each record has EXACTLY open/high/low/close/volume as JSON-native
      numbers (float prices, int volume). Alpaca's extra `t`, `n` (trade count)
      and `vw` (VWAP) fields are deliberately dropped.
    - Prices are NON-ADJUSTED. `adjustment=raw` is sent explicitly on every
      request; the project-wide store requires the same raw basis as
      Yahoo/Eulerpool. `split`, `dividend` and `all` can be selected explicitly
      but break cross-source coherence and are not the default.
    - Timestamps are UTC strings "YYYY-MM-DD HH:MM:SS". Alpaca returns RFC-3339
      UTC ("2022-02-28T05:00:00Z"), which is parsed and converted to that exact
      format. Intraday bars keep their true UTC instant.
    - Native daily/weekly bars are date markers at "YYYY-MM-DD 00:00:00" UTC,
      like Yahoo/Eulerpool. Alpaca stamps daily bars at the session open (e.g.
      05:00:00Z for US equities), so `1d`/`1wk` are floored to 00:00:00 UTC of
      their UTC date before emission.
    - Range is the half-open interval [start, end_boundary). A midnight
      `end_date` means "include the whole end day" (+1 day), exactly like
      `RequestData._end_boundary()`; any other instant stays exclusive. Bars
      outside the window are clipped locally as well.

    PAGINATION
    ----------
    `GET /v2/stocks/{symbol}/bars` returns at most `limit` (max 10000) bars per
    page plus a `next_page_token`. This class follows the token until it is
    null/absent, accumulating every page, and stops defensively if the source
    ever repeats a token (a broken cursor must not hang a backtest).

    SOURCE LIMITATIONS
    ------------------
    Alpaca only serves US stocks and ETFs (no European equities) and its
    historical data starts in 2016. `feed` selects the upstream venue:
    `iex` (default, available on the free plan), `sip` (full consolidated
    tape, paid plans) or `otc`. IEX-only volume is a fraction of the
    consolidated tape, so it must not be compared with Yahoo/Eulerpool volume.

    start_date -> datetime interpreted as UTC when naive, inclusive
    end_date -> datetime, inclusive whole day when midnight, otherwise exclusive
    interval -> 1m, 5m, 15m, 30m, 60m, 1h, 4h, 1d, 1w, 1wk
    action_symbol -> US ticker requested from the API
    feed -> "iex" (default), "sip" or "otc"
    adjustment -> "raw" (default, NON-ADJUSTED) or split/dividend/all
    """

    _TIMEFRAME_BY_INTERVAL = {
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
    _DATE_MARKER_TIMEFRAMES = {"1Day", "1Week"}

    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        interval: str,
        action_symbol: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        feed: str = "iex",
        adjustment: str = "raw",
        timeout: float = 60,
        max_retries: int = 3,
    ):
        # Canonical project vocabulary plus Alpaca's native spellings (`1w`).
        # RequestData uses "60m"/"1wk"; both are accepted here.
        self._interval_authorized_values = [
            "1m", "5m", "15m", "30m", "60m", "1h", "4h", "1d", "1wk", "1w",
        ]
        if interval not in self._interval_authorized_values:
            raise ValueError(
                f"Interval must have one of the following value: "
                f"{self._interval_authorized_values}"
            )
        if feed not in _SUPPORTED_FEEDS:
            raise ValueError(
                f"feed must be one of {sorted(_SUPPORTED_FEEDS)}, got {feed!r}"
            )
        if adjustment not in _SUPPORTED_ADJUSTMENTS:
            raise ValueError(
                f"adjustment must be one of {sorted(_SUPPORTED_ADJUSTMENTS)}, "
                f"got {adjustment!r}"
            )

        self._start_date = start_date
        self._end_date = end_date
        self._interval = interval
        self._action_symbol = action_symbol
        self._feed = feed
        self._adjustment = adjustment
        self._base_url = "https://data.alpaca.markets"
        self._timeout = timeout
        self._max_retries = max(1, int(max_retries))
        self._backoff_seconds = _DEFAULT_BACKOFF_SECONDS
        self._page_limit = _DEFAULT_PAGE_LIMIT

        # Load the repo `.env` explicitly (Python may be launched from any cwd),
        # then fall back to the usual dotenv search. A real environment variable
        # wins over the file (`override=False`).
        if api_key is None or api_secret is None:
            if _DOTENV_PATH.exists():
                load_dotenv(_DOTENV_PATH)
            load_dotenv()

        if api_key is None:
            api_key = (
                os.getenv("ALPACA_API_KEY")
                or os.getenv("APCA_API_KEY_ID")
                # Alpaca's own docs/CLI export the dashed spelling; accept it
                # too so a `.env` copied from Alpaca works unchanged.
                or os.getenv("APCA-API-KEY-ID")
            )
        if api_secret is None:
            api_secret = (
                os.getenv("ALPACA_SECRET_KEY")
                or os.getenv("APCA_API_SECRET_KEY")
                or os.getenv("APCA-API-SECRET-KEY")
            )

        self._api_key = (api_key or "").strip()
        self._api_secret = (api_secret or "").strip()
        if not self._api_key or not self._api_secret:
            raise ValueError(
                "Alpaca API credentials are missing. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY (or the legacy APCA_API_KEY_ID / "
                "APCA_API_SECRET_KEY) in the environment or the repository "
                ".env. Create a key pair at https://alpaca.markets."
            )

    # ------------------------------------------------------------- conversions
    @staticmethod
    def _to_utc_datetime(value: datetime) -> datetime:
        """
        Interprets NAIVE datetimes as UTC. Using the machine's local timezone
        here would silently shift every requested window; `replace` never does.
        Aware datetimes are converted from their own offset.
        """
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @classmethod
    def _to_utc_epoch(cls, value: datetime) -> int:
        return int(cls._to_utc_datetime(value).timestamp())

    @classmethod
    def _to_rfc3339(cls, value: datetime) -> str:
        """RFC-3339 UTC string understood by the Alpaca start/end parameters."""
        return cls._to_utc_datetime(value).strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _parse_bar_timestamp(raw_timestamp) -> int:
        """
        Parses an Alpaca RFC-3339 timestamp (e.g. "2022-02-28T05:00:00Z") into
        Unix seconds. Python 3.11+ accepts the trailing "Z" directly, but the
        explicit offset keeps the parser unambiguous across versions.
        """
        text = str(raw_timestamp).strip()
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        moment = datetime.fromisoformat(text)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        else:
            moment = moment.astimezone(timezone.utc)
        return int(moment.timestamp())

    @staticmethod
    def _date_marker_epoch(epoch: float) -> int:
        """
        Floors an epoch to 00:00:00 UTC of its UTC date. Daily/weekly bars are
        date markers, not instants, so a US session-open stamp (05:00Z) still
        belongs to that date's bar, exactly like Yahoo/Eulerpool daily output.
        """
        moment = datetime.fromtimestamp(epoch, tz=timezone.utc)
        marker = moment.replace(hour=0, minute=0, second=0, microsecond=0)
        return int(marker.timestamp())

    def _end_boundary(self) -> datetime:
        """
        Same inclusive-day rule as RequestData._end_boundary(): a midnight end
        means "up to and including the whole day", while any other instant is
        kept exclusive.
        """
        if self._end_date.time() == time.min:
            return self._end_date + timedelta(days=1)
        return self._end_date

    # ------------------------------------------------------------- request bits
    def _get_timeframe(self) -> str:
        """Maps the project interval vocabulary to Alpaca's timeframe code."""
        try:
            return self._TIMEFRAME_BY_INTERVAL[self._interval]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported Alpaca interval: {self._interval}. "
                f"Authorized values: {self._interval_authorized_values}"
            ) from exc

    def _headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._api_secret,
            "Accept": "application/json",
        }

    def _build_params(self, page_token: Optional[str] = None) -> Dict[str, object]:
        params: Dict[str, object] = {
            "timeframe": self._get_timeframe(),
            "start": self._to_rfc3339(self._start_date),
            # Alpaca's `end` is exclusive, which matches the half-open
            # [start, end_boundary) contract directly.
            "end": self._to_rfc3339(self._end_boundary()),
            "limit": self._page_limit,
            # NON-ADJUSTED by default: explicit so a future SDK/API default
            # change cannot silently alter the price basis.
            "adjustment": self._adjustment,
            "feed": self._feed,
        }
        if page_token:
            params["page_token"] = page_token
        return params

    @staticmethod
    def _parse_retry_after(response: requests.Response) -> Optional[float]:
        value = response.headers.get("Retry-After") if response.headers else None
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            # HTTP-date Retry-After values are legal but uncommon here; fall
            # back to the normal exponential backoff.
            return None

    def _request_page(self, params: Dict[str, object]) -> Dict:
        """
        Fetches one page with explicit error handling:
          401/403 -> AlpacaAuthError immediately (retrying cannot fix auth)
          429     -> retried, honouring the Retry-After header when present
          408/5xx/network -> retried with exponential backoff
          other   -> AlpacaError
        """
        url = f"{self._base_url}/v2/stocks/{self._action_symbol}/bars"
        headers = self._headers()
        last_error: Optional[BaseException] = None

        for attempt in range(self._max_retries):
            retry_after: Optional[float] = None
            try:
                response = requests.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=self._timeout,
                )
                status = response.status_code

                if status == 200:
                    payload = response.json()
                    if not isinstance(payload, dict):
                        raise AlpacaError(
                            "Alpaca returned a non-object JSON payload for "
                            f"{self._action_symbol!r}."
                        )
                    return payload

                if status == 401:
                    raise AlpacaAuthError(
                        "HTTP 401 from Alpaca: the API key/secret pair is "
                        "missing, invalid or expired. Create new keys at "
                        "https://alpaca.markets and set ALPACA_API_KEY / "
                        "ALPACA_SECRET_KEY."
                    )
                if status == 403:
                    raise AlpacaAuthError(
                        "HTTP 403 from Alpaca: the credentials are valid but "
                        "this account/plan cannot access the requested data "
                        f"(symbol {self._action_symbol!r}, feed "
                        f"{self._feed!r}). Check the subscription at "
                        "https://alpaca.markets."
                    )
                if status == 429:
                    retry_after = self._parse_retry_after(response)
                    last_error = AlpacaError(
                        f"HTTP 429 from Alpaca: rate limit exceeded "
                        f"(attempt {attempt + 1}/{self._max_retries})."
                    )
                elif status == 408:
                    last_error = AlpacaError(
                        f"HTTP 408 (request timeout) from Alpaca "
                        f"(attempt {attempt + 1}/{self._max_retries})."
                    )
                elif 500 <= status < 600:
                    last_error = AlpacaError(
                        f"HTTP {status} from Alpaca "
                        f"(attempt {attempt + 1}/{self._max_retries})."
                    )
                else:
                    raise AlpacaError(
                        f"HTTP {status} from Alpaca: "
                        f"{getattr(response, 'text', '')[:300]}"
                    )

            except requests.RequestException as exc:
                # Connection reset, DNS hiccup, read timeout, etc.
                last_error = exc

            if attempt + 1 >= self._max_retries:
                break
            delay = retry_after
            if delay is None:
                delay = self._backoff_seconds * (2 ** attempt)
            # Keep a hostile/broken Retry-After from stalling the process.
            time_module.sleep(min(max(delay, 0.0), 60.0))

        raise AlpacaError(
            f"Failed to fetch {self._action_symbol!r} bars from Alpaca after "
            f"{self._max_retries} attempts: {last_error}"
        )

    def _fetch_bars(self) -> List[dict]:
        """Follows `next_page_token` until exhausted, accumulating every bar."""
        bars: List[dict] = []
        page_token: Optional[str] = None
        seen_tokens = set()

        while True:
            payload = self._request_page(self._build_params(page_token))
            page_bars = payload.get("bars")
            if isinstance(page_bars, list):
                bars.extend(page_bars)

            next_token = payload.get("next_page_token")
            if not next_token or next_token in seen_tokens:
                # Empty/absent token is the normal end; a repeated token is a
                # broken cursor and must not loop forever.
                break
            seen_tokens.add(next_token)
            page_token = next_token

        return bars

    # ---------------------------------------------------------------- parsing
    def _parse_ohlcv(self, bars) -> List[Tuple[int, float, float, float, float, int]]:
        """
        Parses Alpaca bar objects {t,o,h,l,c,v,n,vw} into canonical rows.
        Alpaca-specific extras (`n`, `vw`) are ignored. Rows with a missing or
        non-numeric timestamp/OHLC component are skipped entirely rather than
        emitting None. Volume defaults to 0 only when absent; prices and volume
        are coerced to plain Python scalars so no numpy/Decimal leaks out.
        """
        if not isinstance(bars, list):
            return []

        rows = []
        for bar in bars:
            if not isinstance(bar, dict):
                continue
            try:
                epoch = self._parse_bar_timestamp(bar["t"])
                open_ = float(bar["o"])
                high = float(bar["h"])
                low = float(bar["l"])
                close = float(bar["c"])
            except (KeyError, TypeError, ValueError):
                continue

            volume = 0
            if bar.get("v") is not None:
                try:
                    volume = int(float(bar["v"]))
                except (TypeError, ValueError):
                    volume = 0

            rows.append((epoch, open_, high, low, close, volume))
        return rows

    # ---------------------------------------------------------------- public
    def get_json_api(self) -> Dict:
        """
        Fetches all pages of `GET /v2/stocks/{symbol}/bars` and returns the
        canonical {utc_timestamp: OHLCV} mapping.

        Daily/weekly bars are floored to midnight UTC before both filtering and
        emission; intraday bars keep their true UTC instant. The result is
        clipped to [start, end_boundary). An empty result raises a ValueError
        naming the request instead of silently returning {}.
        """
        start_epoch = self._to_utc_epoch(self._start_date)
        end_boundary_epoch = self._to_utc_epoch(self._end_boundary())
        timeframe = self._get_timeframe()
        is_date_marker = timeframe in self._DATE_MARKER_TIMEFRAMES

        result: Dict[str, Dict] = {}
        for epoch, open_, high, low, close, volume in self._parse_ohlcv(
            self._fetch_bars()
        ):
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

        if not result:
            raise ValueError(
                f"Alpaca API returned no data for symbol "
                f"{self._action_symbol!r} at interval {self._interval!r} "
                f"between {self._to_rfc3339(self._start_date)} and "
                f"{self._to_rfc3339(self._end_boundary())} (end exclusive). "
                f"Alpaca only serves US stocks/ETFs since 2016; this "
                f"range/interval/feed combination may be unsupported or empty."
            )

        return dict(sorted(result.items()))

    def get_standard_json(self) -> Dict:
        """Canonical top-level shape: {"symbol": ..., "data": {timestamp: OHLCV}}."""
        return {
            "symbol": self._action_symbol,
            "data": self.get_json_api(),
        }
