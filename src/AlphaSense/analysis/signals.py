"""
Signal-provider abstraction for the backtest engine.

The simulation core (AlphaSense.analysis.backtest.simulate_wallet) only reads
a per-timestamp dict of bullish/bearish counts. Putting that lookup behind a
provider lets alternative signal sources feed the same engine without touching
the simulation loop itself.

Providers are registered by name so callers (backtest orchestration, CLI) can
select one with a plain string, which also makes the choice easy to expose as
a CLI flag.
"""
from datetime import datetime
from typing import Callable, Protocol, Sequence, runtime_checkable

import pandas as pd

from AlphaSense.analysis.confluence import compute_confluence


@runtime_checkable
class SignalProvider(Protocol):
    """A source of per-candle bullish/bearish signals."""

    def build(self, price_df: pd.DataFrame) -> dict[str, dict]:
        """
        Returns {iso_timestamp: {bullish_count, bearish_count,
        bullish_sources, bearish_sources}} - exactly the inner
        ["confluence"] sub-dict that simulate_wallet() consumes.
        """
        ...


class ConfluenceProvider:
    """
    Wraps the existing compute_confluence() registry-based signal.

    compute_fn is injectable purely so callers can point at their own
    reference to compute_confluence (e.g. a module-level symbol they patch);
    production behaviour is unchanged.
    """

    def __init__(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str,
        compute_fn: Callable[..., dict] = compute_confluence,
    ):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.interval = interval
        self._compute = compute_fn

    def build(self, price_df: pd.DataFrame) -> dict[str, dict]:
        return self._compute(self.symbol, self.start, self.end, self.interval, price_df)["confluence"]


# TSMOM defaults are month-based: 1/3/12 months expressed in daily bars.
# 21 trading days per month is the conventional approximation; at the 1d
# interval the scaled result is exactly (21, 63, 252).
DEFAULT_LOOKBACKS = (21, 63, 252)
_DEFAULT_LOOKBACK_MONTHS = (1, 3, 12)
_TRADING_DAYS_PER_MONTH = 21
_MINUTES_PER_DAY = 24 * 60

# pd.Timedelta rejects "wk"/"mo"/"3mo" suffixes, yet those are valid candle
# intervals elsewhere in the app. Mirror RequestData's fixed-duration
# overrides so interval scaling works for every interval the CLI accepts.
_INTERVAL_MINUTES_OVERRIDE = {
    "1wk": 7 * 24 * 60,
    "1mo": 30 * 24 * 60,
    "3mo": 90 * 24 * 60,
}


def _interval_minutes(interval: str) -> float | None:
    """Bar duration in minutes, or None when `interval` is not time-based."""
    if interval in _INTERVAL_MINUTES_OVERRIDE:
        return _INTERVAL_MINUTES_OVERRIDE[interval]
    try:
        return pd.Timedelta(interval).total_seconds() / 60
    except (ValueError, TypeError):
        return None


def _scaled_lookbacks(interval: str) -> tuple[int, ...]:
    """
    Scales the 1/3/12-month default lookbacks to `interval`.

    One month is treated as 21 daily bars, so the conversion is
    months * 21 days / bar_duration. At 1d this returns DEFAULT_LOOKBACKS
    exactly. Intraday bars are treated as a continuous 24h clock rather than
    modelling a trading session: TSMOM's primary target is daily bars, and
    this layer has no exchange-calendar information to do better. An
    unparseable interval falls back to the literal daily bar counts instead
    of inventing a scale factor.

    Scaling happens only for the implicit defaults. An explicit `lookbacks`
    sequence is always taken literally, in bars, so a caller asking for
    lookbacks=(5,) gets 5 bars regardless of interval.
    """
    minutes = _interval_minutes(interval)
    if minutes is None or minutes <= 0:
        return DEFAULT_LOOKBACKS
    bars_per_month = _TRADING_DAYS_PER_MONTH * _MINUTES_PER_DAY / minutes
    return tuple(
        max(1, round(months * bars_per_month)) for months in _DEFAULT_LOOKBACK_MONTHS
    )


class TrendProvider:
    """
    Time-series momentum (TSMOM) signal over close-to-close returns.

    For each bar i and lookback L the vote is sign(close[i] / close[i-L] - 1).
    Only closes up to and including bar i are consulted; a lookback whose
    base bar is not yet in the window abstains during warm-up, and the
    ensemble is the majority vote of the remaining lookbacks. A tie - which
    includes "no lookback available yet" - is flat (0), and a flat signal
    never clears simulate_wallet's threshold (>=1), so warm-up bars are never
    traded on.

    `lookbacks=None` means "month-based defaults scaled to this interval"
    (1/3/12 months; exactly (21, 63, 252) at 1d). Passing an explicit
    sequence overrides that scaling and is interpreted directly in bars.
    """

    def __init__(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str,
        lookbacks: Sequence[int] | None = None,
    ):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.interval = interval
        self.lookbacks = (
            tuple(lookbacks) if lookbacks is not None else _scaled_lookbacks(interval)
        )
        if not self.lookbacks or any(lookback < 1 for lookback in self.lookbacks):
            raise ValueError(
                f"lookbacks must be non-empty positive bar counts, got {lookbacks!r}"
            )

    def build(self, price_df: pd.DataFrame) -> dict[str, dict]:
        if price_df.empty:
            return {}

        # reset_index makes positional i-L indexing unambiguous even if the
        # caller's index is not a clean RangeIndex.
        closes = price_df["close"].astype(float).reset_index(drop=True)
        signals: dict[str, dict] = {}

        for i, ts in enumerate(price_df.index):
            votes: list[int] = []
            for lookback in self.lookbacks:
                base = i - lookback
                if base < 0:
                    continue  # warm-up: this lookback has no return yet
                # close[i] vs close[i-L], never close[i+L]. Reading any future
                # close here would be lookahead bias and make the backtest
                # untradeable in live markets.
                ret = closes.iloc[i] / closes.iloc[base] - 1.0
                if ret > 0:
                    votes.append(1)
                elif ret < 0:
                    votes.append(-1)
                # ret == 0 is an abstention, not a vote.

            net = sum(votes)
            if net > 0:
                record = {
                    "bullish_count": 1,
                    "bearish_count": 0,
                    "bullish_sources": ["tsmom"],
                    "bearish_sources": [],
                }
            elif net < 0:
                record = {
                    "bullish_count": 0,
                    "bearish_count": 1,
                    "bullish_sources": [],
                    "bearish_sources": ["tsmom"],
                }
            else:
                record = {
                    "bullish_count": 0,
                    "bearish_count": 0,
                    "bullish_sources": [],
                    "bearish_sources": [],
                }
            signals[pd.Timestamp(ts).isoformat()] = record

        return signals


ProviderFactory = Callable[..., SignalProvider]
_PROVIDER_REGISTRY: dict[str, ProviderFactory] = {}


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Registers (or replaces) a provider factory under `name`."""
    _PROVIDER_REGISTRY[name] = factory


def available_providers() -> list[str]:
    """Sorted provider names, suitable for argparse choices."""
    return sorted(_PROVIDER_REGISTRY)


def get_provider(
    name: str, symbol: str, start: datetime, end: datetime, interval: str, **kwargs
) -> SignalProvider:
    """
    Builds the provider registered under `name`; raises ValueError if unknown.

    Extra keyword arguments are forwarded verbatim to the factory. Callers are
    responsible for only passing options the selected factory understands
    (e.g. `lookbacks` for "tsmom"); this stays generic instead of growing a
    provider-specific branch here.
    """
    try:
        factory = _PROVIDER_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown signal provider {name!r}; registered: {available_providers()}"
        ) from None
    return factory(symbol=symbol, start=start, end=end, interval=interval, **kwargs)


register_provider("confluence", ConfluenceProvider)
register_provider("tsmom", TrendProvider)
