from __future__ import annotations

import os
import re
import tempfile
import time as time_module
import warnings
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

from .GenericAPI import GenericAPI

# The source publishes US/Eastern wall-clock timestamps with no offset. The
# conversion to UTC is done explicitly below; never via datetime.fromtimestamp()
# (that would silently use the machine's local timezone).
_EASTERN = "America/New_York"
_UTC = "UTC"

_CANONICAL_FIELDS = ("open", "high", "low", "close", "volume")

# The repo `.env` lives at the repository root, three levels above this module:
# src/AlphaSense/api/HFDataLibraryAPI.py -> src/AlphaSense/api -> ... -> repo.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOTENV_PATH = _REPO_ROOT / ".env"

# Requests timeouts are (connect, read) seconds. A whole-ticker parquet file is
# typically 5-11 MB, so the read timeout is deliberately generous.
_DEFAULT_TIMEOUT = (10, 180)
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BACKOFF_SECONDS = 1.5

# Cache entries are keyed by (ticker, version) and considered fresh for one
# day: the provider updates each ticker daily, and re-downloading ~8 MB on every
# backtest would be wasteful. TTL is documented on the class too.
_CACHE_TTL_SECONDS = 24 * 60 * 60

_SPLIT_ADJUSTED_ERROR_MESSAGE = (
    "HF Data Library only provides SPLIT-ADJUSTED prices: its `clean` version "
    "is split-adjusted (and `raw` is also split-adjusted). The canonical "
    "source contract requires NON-ADJUSTED prices, so returning this data "
    "would silently put cross-source backtests on a different price basis. "
    "Pass allow_split_adjusted=True to HFDataLibraryAPI to override this "
    "gate explicitly."
)


class HFDataLibraryError(RuntimeError):
    """Raised for HF Data Library API/parsing failures (not auth)."""


class HFDataLibraryAuthError(HFDataLibraryError):
    """Raised for 401/403 responses, with an actionable account message."""


class SplitAdjustedPricesError(ValueError):
    """
    Raised when split-adjusted prices would be returned without explicit opt-in.

    The canonical cross-source contract requires NON-ADJUSTED prices. HF Data
    Library has no unadjusted variant, so the default behaviour is to refuse to
    materialise data rather than silently return a different price basis.
    Callers that deliberately accept split-adjusted bars must construct the API
    with ``allow_split_adjusted=True``.
    """


class SplitAdjustedPricesWarning(UserWarning):
    """
    Emitted by every get_json_api() call.

    HF Data Library has NO unadjusted variant: BOTH `raw` and `clean` are
    adjusted for stock splits (documented at
    https://hfdatalibrary.com/pages/versions and /pages/dictionary). The
    project-wide canonical contract requires NON-ADJUSTED prices, so this
    source cannot satisfy that contract for tickers that split inside the
    requested range. Returning the data silently would let a backtest mix
    split-adjusted HF bars with raw Yahoo/Eulerpool bars across a split event
    and manufacture a fake return. Callers that only compare within the
    source, or that pre-screen for split-free windows, can suppress this
    warning deliberately.
    """


class HFDataLibraryAPI(GenericAPI):
    """
    HFDataLibraryAPI Class
    Class for requesting HF Data Library (IEX 1-minute OHLCV bars).

    SETUP
    -----
    A free API key is required (https://hfdatalibrary.com/pages/account) and
    must be available as `HFDATALIBRARY_API_KEY` in the environment or in the
    repository `.env`. The constructor raises a ValueError with the signup URL
    when it is missing; there is no silent unauthenticated fallback.

    OUTPUT CONVENTIONS (canonical contract shared with every other source)
    ---------------------------------------------------------------------
    - Outer shape is exactly {"symbol": ..., "data": {...}}.
    - Each record has exactly open/high/low/close/volume as JSON-native
      numbers. Volume is always present.
    - Timestamps are UTC strings "YYYY-MM-DD HH:MM:SS". The provider's
      timestamps are naive US/Eastern wall-clock values, so they are localised
      to America/New_York and converted to UTC. A 09:30 ET bar is therefore
      "14:30:00" in winter (EST) and "13:30:00" in summer (EDT).
    - Daily bars are emitted as date markers at "YYYY-MM-DD 00:00:00" UTC,
      like native daily bars from Yahoo. They are not timezone-shifted.
    - Range is the half-open interval [start, end_boundary). A midnight
      `end_date` means "include the whole end day", exactly like
      RequestData._end_boundary(), so the final day is never silently dropped.
      A non-midnight end remains exclusive.

    ADJUSTMENT POLICY — IMPORTANT
    -----------------------------
    The source only ships SPLIT-ADJUSTED prices, in BOTH versions:
      * `raw`  : IEX bars with no cleaning/outlier removal, split-adjusted.
      * `clean`: raw plus the eight-step cleaning pipeline, split-adjusted.
    There is no unadjusted/`adj_close=False` variant on the API. The project's
    canonical contract requires raw, non-adjusted prices, so this source
    CANNOT be used interchangeably with Yahoo/Eulerpool across a split event.
    The default is therefore a HARD GATE: get_json_api() raises
    SplitAdjustedPricesError (a ValueError) before any download or cache read.
    A caller that deliberately accepts this limitation must opt in explicitly
    with ``allow_split_adjusted=True``; in that case the data is returned and
    SplitAdjustedPricesWarning is still emitted on every call. The class
    attributes below expose the same limitation programmatically for callers
    that need to gate cross-source backtests.

    IEX-ONLY VOLUME CAVEAT
    ----------------------
    Every bar comes from IEX HIST top-of-book data. IEX represents roughly
    2-3% of consolidated US equity volume, so `volume` is NOT the consolidated
    tape volume reported by Yahoo/Eulerpool and must not be mixed with it in a
    liquidity or volume-based backtest. Prices are also IEX-only prints.

    DATA COMPLETENESS
    -----------------
    Gaps are preserved: a minute with no IEX trade has no bar. The `clean`
    version additionally removes outside-hours bars, zero-volume bars, OHLC
    violations and outliers (about 3% of raw). Neither version is gap-filled.

    CACHING / RETRIES
    -----------------
    The API streams the ticker's entire history as one Parquet file. Files are
    cached under ~/.cache/hfdatalibrary/{TICKER}_{version}.parquet with a
    24-hour TTL because the upstream file is refreshed daily. Downloads are
    written to a temporary file in the same directory and published with
    os.replace(), so a concurrent reader never sees a partial file. Transient
    network errors, 5xx and 429 responses are retried with exponential
    backoff; 401/403/404 raise explicit, actionable errors immediately.

    start_date -> datetime, inclusive start of the requested window
    end_date   -> datetime, inclusive when it is midnight, otherwise exclusive
    interval   -> 1m, 5m, 15m, 30m, 60m, 1h, 4h, 1d, 1w, 1wk
    action_symbol -> ticker requested from the API
    version -> "raw" (default, least processed) or "clean"
    allow_split_adjusted -> False by default; True explicitly accepts
                            split-adjusted HF data (and still warns)
    """

    # Kept in sync with the provider's documented /v1/bars?version values.
    SUPPORTED_VERSIONS = ("raw", "clean")
    # `raw` is the closest thing to non-adjusted this source can offer. It is
    # still split-adjusted; see the class docstring.
    DEFAULT_VERSION = "raw"
    PRICES_ARE_SPLIT_ADJUSTED = True
    SUPPORTS_NON_ADJUSTED_PRICES = False
    VOLUME_IS_CONSOLIDATED = False

    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        interval: str,
        action_symbol: str,
        version: str = DEFAULT_VERSION,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        allow_split_adjusted: bool = False,
    ):
        # RequestData rewrites any interval longer than 60 min to "60m", but
        # direct callers may also use "1h", "1w" or "1wk". All are accepted and
        # aggregated locally from the 1-minute base below.
        self._interval_authorized_values = [
            "1m", "5m", "15m", "30m", "60m", "1h", "4h", "1d", "1w", "1wk"
        ]
        if interval not in self._interval_authorized_values:
            raise ValueError(
                f"Interval must have one of the following value: "
                f"{self._interval_authorized_values}"
            )
        if version not in self.SUPPORTED_VERSIONS:
            raise ValueError(
                f"version must be one of {self.SUPPORTED_VERSIONS}, got {version!r}"
            )

        self._start_date = start_date
        self._end_date = end_date
        self._interval = interval
        self._action_symbol = action_symbol
        self._version = version
        self._allow_split_adjusted = bool(allow_split_adjusted)
        self._base_url = "https://api.hfdatalibrary.com/v1"
        self._cache_dir = os.path.expanduser(
            cache_dir or "~/.cache/hfdatalibrary"
        )
        os.makedirs(self._cache_dir, exist_ok=True)

        self._timeout: Tuple[float, float] = _DEFAULT_TIMEOUT
        self._max_retries = _DEFAULT_MAX_RETRIES
        self._backoff_seconds = _DEFAULT_BACKOFF_SECONDS
        self._cache_ttl_seconds = _CACHE_TTL_SECONDS

        # Public diagnostics for callers/scripts reporting download cost.
        self.last_download_bytes = 0
        self.last_download_seconds = 0.0
        self.last_cache_hit = False

        # Load the repo .env explicitly (the user/CI may launch Python from any
        # cwd), then fall back to the usual dotenv search. `override=False` is
        # the default: a real environment variable always wins over the file.
        if api_key is None:
            if _DOTENV_PATH.exists():
                load_dotenv(_DOTENV_PATH)
            load_dotenv()
            api_key = os.getenv("HFDATALIBRARY_API_KEY", "")

        self._api_key = (api_key or "").strip()
        if not self._api_key:
            raise ValueError(
                "HFDATALIBRARY_API_KEY is not set. Register for a free key at "
                "https://hfdatalibrary.com/pages/account and put "
                "HFDATALIBRARY_API_KEY=<key> in the repository .env (or export "
                "it in the environment)."
            )

    @property
    def version(self) -> str:
        return self._version

    @property
    def cache_dir(self) -> str:
        return self._cache_dir

    @property
    def allow_split_adjusted(self) -> bool:
        return self._allow_split_adjusted

    # ------------------------------------------------------------------ cache
    def _cache_path(self, ticker: str, version: str) -> str:
        """
        Cache identity is (ticker, version). The version suffix is essential:
        raw and clean are different files for the same ticker and must never
        overwrite each other. The sanitising keeps an unexpected ticker string
        from escaping the cache directory.
        """
        safe_ticker = re.sub(r"[^A-Za-z0-9_.-]+", "_", ticker.upper())
        return os.path.join(self._cache_dir, f"{safe_ticker}_{version}.parquet")

    def _cache_is_fresh(self, cache_file: str) -> bool:
        if not os.path.exists(cache_file):
            return False
        try:
            size = os.path.getsize(cache_file)
        except OSError:
            return False
        if size <= 8:
            return False
        age = datetime.now().timestamp() - os.path.getmtime(cache_file)
        return age < self._cache_ttl_seconds

    # --------------------------------------------------------------- download
    def _download_parquet(self, ticker: str, version: str) -> str:
        """
        Download (or reuse) one ticker/version Parquet file and return its
        local path.

        Error handling is intentionally explicit:
          401 -> missing/invalid/expired key
          403 -> key accepted but the account cannot download (verified email
                 and complete institution/country/role profile are required)
          404 -> unknown ticker or unsupported version
          429 -> rate limited; honour Retry-After, then retry
          5xx / network errors -> retry with exponential backoff
        """
        cache_file = self._cache_path(ticker, version)
        self.last_download_bytes = 0
        self.last_download_seconds = 0.0
        self.last_cache_hit = False

        if self._cache_is_fresh(cache_file):
            self.last_cache_hit = True
            self.last_download_bytes = os.path.getsize(cache_file)
            return cache_file

        url = f"{self._base_url}/bars/{ticker}"
        headers = {"X-API-Key": self._api_key}
        params = {"version": version}

        last_error: Optional[BaseException] = None
        for attempt in range(self._max_retries):
            retry_after: Optional[float] = None
            try:
                started = time_module.monotonic()
                with requests.get(
                    url,
                    headers=headers,
                    params=params,
                    stream=True,
                    timeout=self._timeout,
                ) as response:
                    status = response.status_code

                    if status == 200:
                        written = self._write_atomically(response, cache_file)
                        self.last_download_bytes = written
                        self.last_download_seconds = time_module.monotonic() - started
                        return cache_file

                    if status == 401:
                        raise HFDataLibraryAuthError(
                            "HTTP 401 from HF Data Library: the API key was "
                            "missing, invalid or expired. Get a fresh key at "
                            "https://hfdatalibrary.com/pages/account and set "
                            "HFDATALIBRARY_API_KEY."
                        )
                    if status == 403:
                        raise HFDataLibraryAuthError(
                            "HTTP 403 from HF Data Library: the API key is "
                            "valid but this account cannot download. The "
                            "provider requires a verified email and a complete "
                            "profile (institution, country, role). Finish the "
                            "profile at https://hfdatalibrary.com/pages/account."
                        )
                    if status == 404:
                        raise HFDataLibraryError(
                            f"HTTP 404 from HF Data Library: ticker {ticker!r} "
                            f"or version {version!r} was not found."
                        )
                    if status == 429:
                        retry_after = self._parse_retry_after(response)
                        last_error = HFDataLibraryError(
                            f"HTTP 429 from HF Data Library: rate limit "
                            f"exceeded (attempt {attempt + 1}/"
                            f"{self._max_retries})."
                        )
                        # Retried below; honour the server's Retry-After.
                    elif 500 <= status < 600:
                        last_error = HFDataLibraryError(
                            f"HTTP {status} from HF Data Library (attempt "
                            f"{attempt + 1}/{self._max_retries})."
                        )
                        # Retried below.
                    elif status == 408:
                        last_error = HFDataLibraryError(
                            f"HTTP 408 (request timeout) from HF Data Library "
                            f"(attempt {attempt + 1}/{self._max_retries})."
                        )
                        # Retried below.
                    else:
                        raise HFDataLibraryError(
                            f"HTTP {status} from HF Data Library: "
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

        raise HFDataLibraryError(
            f"Failed to download {ticker} ({version}) from HF Data Library "
            f"after {self._max_retries} attempts: {last_error}"
        )

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

    def _write_atomically(self, response, cache_file: str) -> int:
        """
        Stream the response to a temporary file in the SAME directory, fsync it,
        validate the Parquet magic bytes, then publish it with os.replace().
        os.replace() is atomic on the same filesystem, so readers never observe
        a half-written cache entry. The temp file is removed on any failure so
        an aborted download cannot leave gigabytes behind.
        """
        directory = os.path.dirname(cache_file) or "."
        tmp_path: Optional[str] = None
        try:
            fd, tmp_path = tempfile.mkstemp(
                prefix=f".{os.path.basename(cache_file)}.",
                suffix=".tmp",
                dir=directory,
            )
            written = 0
            with os.fdopen(fd, "wb") as handle:
                for chunk in response.iter_content(chunk_size=256 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    written += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())

            if written <= 8:
                raise HFDataLibraryError(
                    f"HF Data Library returned an empty/short response for "
                    f"{os.path.basename(cache_file)} ({written} bytes)."
                )
            self._validate_parquet_file(tmp_path)

            os.replace(tmp_path, cache_file)
            tmp_path = None
            return written
        finally:
            if tmp_path is not None and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    @staticmethod
    def _validate_parquet_file(path: str) -> None:
        """
        Parquet files start and end with the 4-byte magic string "PAR1".
        Checking both ends is a cheap guard against caching an HTML/JSON error
        body that happened to arrive with status 200 (e.g. a proxy page).
        """
        with open(path, "rb") as handle:
            head = handle.read(4)
            handle.seek(-4, os.SEEK_END)
            tail = handle.read(4)
        if head != b"PAR1" or tail != b"PAR1":
            raise HFDataLibraryError(
                f"Downloaded file {path!r} is not a valid Parquet file "
                f"(magic bytes head={head!r}, tail={tail!r})."
            )

    # ---------------------------------------------------------------- parsing
    def _load_raw_frame(self) -> pd.DataFrame:
        parquet_file = self._download_parquet(self._action_symbol, self._version)
        try:
            return pd.read_parquet(parquet_file, engine="pyarrow")
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            # A truncated cache entry (e.g. killed process before this fix) can
            # be fresh-by-mtime but unreadable. Delete only files we own inside
            # the cache directory and retry once.
            if Path(parquet_file).resolve().is_relative_to(
                Path(self._cache_dir).resolve()
            ):
                try:
                    os.unlink(parquet_file)
                except OSError:
                    pass
                retry_path = self._download_parquet(
                    self._action_symbol, self._version
                )
                return pd.read_parquet(retry_path, engine="pyarrow")
            raise HFDataLibraryError(
                f"Could not read Parquet file {parquet_file!r}: {exc}"
            ) from exc

    @staticmethod
    def _normalise_name(value: object) -> str:
        """Case/punctuation-insensitive column-name key ("Adj Close" -> adjclose)."""
        return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())

    def _normalise_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise the live schema to lowercase canonical columns.

        Documented files use Open/High/Low/Close/Volume (+ Date, Time, source),
        while some samples/code use lowercase. The provider may also put the
        timestamp in the index instead of a `datetime` column, and some exports
        split it into `Date` (MM/DD/YYYY) + `Time` (HHMM). All four shapes are
        handled here. Extra columns (source, Adj Close, Date, Time) are ignored
        when the canonical fields are selected later.
        """
        working = frame.copy()
        working.columns = [str(column).strip() for column in working.columns]

        def lookup() -> Dict[str, str]:
            return {self._normalise_name(c): c for c in working.columns}

        columns = lookup()

        # Promote a datetime-like index to a column. A RangeIndex must never be
        # promoted: doing so would turn 0,1,2... into bogus timestamps.
        datetime_keys = ("datetime", "timestamp", "datetime64")
        if not any(key in columns for key in datetime_keys):
            index_name = (
                self._normalise_name(working.index.name)
                if working.index.name is not None
                else ""
            )
            if isinstance(working.index, pd.DatetimeIndex) or index_name in (
                "datetime", "timestamp", "datetime64"
            ):
                working = working.reset_index()
                # reset_index() names an unnamed DatetimeIndex "index"; rename
                # the first promoted column to the canonical `datetime`.
                working = working.rename(columns={working.columns[0]: "datetime"})
                columns = lookup()

        # Date + Time split (Date is documented as MM/DD/YYYY, Time as HHMM).
        if "datetime" not in columns:
            date_column = columns.get("date") or columns.get("tradingdate")
            time_column = columns.get("time") or columns.get("bartime")
            if date_column and time_column:
                working["datetime"] = self._combine_date_and_time(
                    working[date_column], working[time_column]
                )
                columns = lookup()

        # Choose exactly one source column per canonical target. First match
        # wins so a file containing both `Date` and `datetime` cannot produce
        # duplicate target columns.
        renames: Dict[str, str] = {}
        claimed = set()
        for column in working.columns:
            key = self._normalise_name(column)
            target = None
            if key in _CANONICAL_FIELDS:
                target = key
            elif key in datetime_keys:
                target = "datetime"
            if target is not None and target not in claimed:
                renames[column] = target
                claimed.add(target)
        working = working.rename(columns=renames)

        missing_prices = [
            field for field in _CANONICAL_FIELDS if field not in working.columns
        ]
        if missing_prices:
            raise HFDataLibraryError(
                "HF Data Library Parquet is missing required column(s) "
                f"{missing_prices}; found {list(frame.columns)!r}."
            )
        if "datetime" not in working.columns:
            raise HFDataLibraryError(
                "HF Data Library Parquet has no datetime column/index and no "
                f"Date+Time pair; found {list(frame.columns)!r}."
            )
        return working

    @staticmethod
    def _combine_date_and_time(
        date_series: pd.Series, time_series: pd.Series
    ) -> pd.Series:
        """
        Combine the documented `Date` (MM/DD/YYYY) and `Time` (HHMM) strings.
        Time is zero-padded to HHMM and expanded to HH:MM:SS so values such as
        "930" and "1600" both parse correctly. Rows that cannot be parsed
        become NaT and are dropped later (never silently assigned a timestamp
        of "now").
        """
        raw_time = (
            time_series.astype(str)
            .str.strip()
            .str.replace(":", "", regex=False)
            .str.replace(".0", "", regex=False)
        )
        normalised_time = raw_time.str.zfill(4)
        time_text = (
            normalised_time.str[:2]
            + ":"
            + normalised_time.str[2:4]
            + ":"
            + normalised_time.str[4:6].where(
                normalised_time.str.len() >= 5, "00"
            )
        )
        combined = date_series.astype(str).str.strip() + " " + time_text
        # format="mixed" lets pandas accept both MM/DD/YYYY (documented) and
        # ISO YYYY-MM-DD exports without guessing a single rigid layout.
        return pd.to_datetime(combined, format="mixed", errors="coerce")

    def _parse_and_convert_timestamps(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Parse the timestamp column, localise it to US/Eastern and convert to a
        tz-aware UTC index.

        `ambiguous="raise"` / `nonexistent="raise"`: the source only publishes
        09:30-16:00 ET bars, which never hit the DST fold/gap, so a raised
        error here means the input is not what the provider documents rather
        than something we should silently guess. If a future export is already
        tz-aware, its own offset is authoritative and it is converted directly
        (never re-localised, which would double-shift it).
        """
        working = frame.copy()
        parsed = pd.to_datetime(working["datetime"], errors="coerce")

        invalid = parsed.isna()
        if invalid.any():
            # Drop unparseable rows; if that leaves nothing, fail loudly.
            working = working.loc[~invalid].copy()
            parsed = parsed.loc[~invalid]
        if parsed.empty:
            raise HFDataLibraryError(
                "HF Data Library returned no parseable datetimes."
            )

        series = parsed
        if isinstance(series.dtype, pd.DatetimeTZDtype) or getattr(
            series.dt, "tz", None
        ) is not None:
            utc_series = series.dt.tz_convert(_UTC)
        else:
            utc_series = (
                series.dt.tz_localize(
                    _EASTERN, ambiguous="raise", nonexistent="raise"
                ).dt.tz_convert(_UTC)
            )

        working.index = pd.DatetimeIndex(utc_series)
        working = working.drop(columns=["datetime"])
        working = working.sort_index()
        # The clean file should already be unique; de-duplicating here makes
        # the resample deterministic even for the raw file.
        working = working[~working.index.duplicated(keep="first")]
        return working

    def _prepare_frame(self) -> pd.DataFrame:
        """Download -> normalise columns -> parse -> ET-to-UTC -> filter."""
        raw = self._load_raw_frame()
        frame = self._normalise_columns(raw)
        frame = self._parse_and_convert_timestamps(frame)
        return self._filter_to_requested_range(frame)

    # --------------------------------------------------------------- boundary
    @staticmethod
    def _as_utc(value: datetime) -> pd.Timestamp:
        """
        Naive datetimes are interpreted as UTC (the canonical store's
        convention); aware datetimes are converted from their own zone.
        """
        if value.tzinfo is None:
            return pd.Timestamp(value, tz=_UTC)
        return pd.Timestamp(value).tz_convert(_UTC)

    def _end_boundary(self) -> datetime:
        """
        A midnight end date means "include the whole end day" (same semantics
        as RequestData._end_boundary()). Any other time is an exclusive
        instant, so Dash's datetime.now() callers keep their old behaviour.
        """
        if self._end_date.time() == time.min:
            return self._end_date + timedelta(days=1)
        return self._end_date

    def _filter_to_requested_range(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the half-open [start, end_boundary) filter AFTER the index has
        been converted to UTC. Filtering the naive Eastern timestamps against
        UTC bounds (the old bug) shifted the window by 4-5 hours and dropped
        the final day for midnight ends.
        """
        start_utc = self._as_utc(self._start_date)
        end_utc = self._as_utc(self._end_boundary())
        return frame[(frame.index >= start_utc) & (frame.index < end_utc)]

    # ------------------------------------------------------------ aggregation
    def _aggregation_rule(self, interval: str) -> Optional[str]:
        """
        Every interval is built locally from the 1-minute base because the
        /bars endpoint only serves full-history 1-minute files. `60m` matters
        in practice: RequestData rewrites 4h/1d/1wk requests to `60m` before
        constructing this class.
        """
        rules = {
            "1m": None,
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "60m": "60min",
            "1h": "60min",
            "4h": "4h",
            "1d": "1D",
            # Monday-anchored weeks. `1W` alone anchors on Sunday, which would
            # disagree with the Monday convention used elsewhere.
            "1w": "1W-MON",
            "1wk": "1W-MON",
        }
        return rules[interval]

    def _aggregate_to_interval(
        self, frame: pd.DataFrame, interval: str
    ) -> pd.DataFrame:
        """
        Resample the UTC-indexed 1-minute frame with left-closed/left-labelled
        buckets: open=first, high=max, low=min, close=last, volume=sum.
        Empty buckets are dropped (the provider preserves gaps; padding them
        with NaN/0 candles would invent trades).
        """
        rule = self._aggregation_rule(interval)
        if rule is None:
            return frame

        aggregated = frame.resample(rule, label="left", closed="left").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        aggregated = aggregated.dropna(
            subset=["open", "high", "low", "close"]
        )
        if interval == "1d":
            # Pandas already labels a UTC "1D" bucket at 00:00:00; normalise
            # explicitly so the contract is obvious and immune to a future
            # frequency/resample change.
            aggregated.index = aggregated.index.normalize()
        return aggregated

    # ----------------------------------------------------------- serialisation
    @staticmethod
    def _serialise(frame: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Emit the exact canonical record schema with JSON-native Python scalars.

        pandas/numpy scalars are not JSON-serialisable, so every price is cast
        to float and volume to a whole-share int. Volume is always present even
        when the provider reports 0.
        """
        result: Dict[str, Dict[str, float]] = {}
        for timestamp, open_, high, low, close, volume in zip(
            frame.index,
            frame["open"].to_numpy(),
            frame["high"].to_numpy(),
            frame["low"].to_numpy(),
            frame["close"].to_numpy(),
            frame["volume"].to_numpy(),
        ):
            volume_value = 0 if pd.isna(volume) else int(round(float(volume)))
            result[timestamp.strftime("%Y-%m-%d %H:%M:%S")] = {
                "open": float(open_),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": volume_value,
            }
        return result

    # ---------------------------------------------------------------- public
    def get_json_api(self) -> Dict:
        """
        Fetch, normalise and aggregate to the canonical timestamp->OHLCV map.

        Raises SplitAdjustedPricesError before any download/cache read unless
        the caller opted in with ``allow_split_adjusted=True``. When opted in,
        SplitAdjustedPricesWarning is still emitted.
        """
        if not self._allow_split_adjusted:
            raise SplitAdjustedPricesError(_SPLIT_ADJUSTED_ERROR_MESSAGE)

        frame = self._prepare_frame()
        frame = self._aggregate_to_interval(frame, self._interval)

        # Never return split-adjusted data silently: see the class docstring.
        warnings.warn(
            "HF Data Library only provides SPLIT-ADJUSTED prices (both 'raw' "
            "and 'clean' versions). This source cannot satisfy the canonical "
            "non-adjusted contract across split events and must not be mixed "
            "with raw Yahoo/Eulerpool candles in the same backtest without "
            "accounting for splits.",
            SplitAdjustedPricesWarning,
            stacklevel=2,
        )
        return self._serialise(frame)

    def get_standard_json(self) -> Dict:
        json_api_data = self.get_json_api()
        return {
            "symbol": self._action_symbol,
            "data": json_api_data,
        }
