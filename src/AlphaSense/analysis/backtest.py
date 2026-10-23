"""
Backtests a simple long-only strategy: buy when a candle's confluence
bullish_count clears a threshold, sell when a later candle's bearish_count
clears the same threshold. Reuses the exact same confluence signal
(AlphaSense.analysis.confluence.compute_confluence) that already drives the
chart's highlighting, so the candles this buys/sells on are exactly the ones
that show up highlighted in the app - nothing new to define or keep in sync.

"Different indicators number" is read here as the confluence threshold - how
many agreeing indicators/patterns are required before a candle counts as a
signal, i.e. the same knob as the app's "Highlight candles with at least
this many agreeing signals" control. If what was actually wanted is testing
specific SUBSETS of indicators (RSI alone vs RSI+MACD, say) rather than how
many must agree, that's a different, larger change to compute_confluence()
this doesn't attempt - it's a reasonably contained follow-up on top of the
same wallet-simulation core below, so flag it if that's the case.

Execution timing: a signal computed from candle i is acted on at candle
(i+1)'s open, not candle i's own close - trading on the same candle that
produced the signal is a classic backtest mistake (lookahead bias), since in
live trading you can't act on a candle until after it's closed and its
indicators are known. One caveat this doesn't fully remove: multi-candle
patterns (RequestMultipleCandlePatterns) classify candle i using candle
i+1's data too, so a pattern-driven signal at i can still see slightly into
the future even with the i+1-open execution rule. Indicator-driven signals
aren't affected by this.

A "win" is final wallet value strictly greater than the starting cash;
exactly breaking even counts as a loss, not a win.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from AlphaSense.frontend.data_access import get_price_candles_dataframe
from AlphaSense.analysis import metrics
from AlphaSense.analysis.confluence import compute_confluence
from AlphaSense.analysis.signals import ConfluenceProvider, get_provider, register_provider
from AlphaSense.requests.quotes import US_MEGACAPS, EU_MEGACAPS
from AlphaSense.costs.brokers import resolve_broker_preset

# Region-split default universes. Kept explicit (rather than derived inline)
# so callers and tests can reason about "US", "EU" and "ALL" independently.
DEFAULT_SYMBOLS_US = [symbol for symbol, _ in US_MEGACAPS]
DEFAULT_SYMBOLS_EU = [symbol for symbol, _ in EU_MEGACAPS]
DEFAULT_SYMBOLS = DEFAULT_SYMBOLS_US + DEFAULT_SYMBOLS_EU

UNIVERSES = {
    "all": DEFAULT_SYMBOLS,
    "us": DEFAULT_SYMBOLS_US,
    "eu": DEFAULT_SYMBOLS_EU,
}


def get_universe_symbols(universe: str = "all") -> list[str]:
    """Return the default symbol list for a named universe ('all'|'us'|'eu')."""
    try:
        return list(UNIVERSES[universe])
    except KeyError:
        raise ValueError(
            f"Unknown universe '{universe}'. Available universes: {sorted(UNIVERSES)}"
        ) from None


DEFAULT_THRESHOLDS = [1, 2, 3, 4]
DEFAULT_FEE_RATES = [0.0, 0.005, 0.01, 0.015]
DEFAULT_STARTING_CASH = 1000.0
DEFAULT_SIGNAL_PROVIDER = "confluence"

# Existing callers/tests patch AlphaSense.analysis.backtest.compute_confluence;
# resolve it at provider-build time so that patch point keeps working.
register_provider("confluence", lambda **kwargs: ConfluenceProvider(compute_fn=compute_confluence, **kwargs))


@dataclass
class Trade:
    action: str  # "BUY", "SELL", "SELL (short)", "BUY (cover)", or a *liquidated at end variant
    time: object
    price: float
    fee_paid: float


@dataclass
class BacktestResult:
    symbol: str
    threshold: int
    fee_rate: float
    starting_cash: float
    final_value: float
    direction: str = "long"
    trades: list = field(default_factory=list)
    benchmark_final_value: float | None = None
    equity_curve: pd.Series | None = None
    bars_per_year: float | None = None
    benchmark_sharpe: float | None = None
    benchmark_sortino: float | None = None
    benchmark_max_drawdown_pct: float | None = None
    benchmark_calmar: float | None = None

    @property
    def return_pct(self) -> float:
        return (self.final_value / self.starting_cash - 1) * 100

    @property
    def sharpe(self) -> float | None:
        if self.equity_curve is None or self.bars_per_year is None:
            return None
        return metrics.sharpe_ratio(self.equity_curve.pct_change(), self.bars_per_year)

    @property
    def sortino(self) -> float | None:
        if self.equity_curve is None or self.bars_per_year is None:
            return None
        return metrics.sortino_ratio(self.equity_curve.pct_change(), self.bars_per_year)

    @property
    def max_drawdown_pct(self) -> float | None:
        if self.equity_curve is None:
            return None
        drawdown = metrics.max_drawdown(self.equity_curve)
        return None if drawdown is None else drawdown * 100

    @property
    def calmar(self) -> float | None:
        if self.equity_curve is None or self.bars_per_year is None:
            return None
        return metrics.calmar_ratio(self.equity_curve, self.bars_per_year)

    @property
    def benchmark_return_pct(self) -> float | None:
        if self.benchmark_final_value is None:
            return None
        return (self.benchmark_final_value / self.starting_cash - 1) * 100

    @property
    def beats_benchmark(self) -> bool | None:
        if self.benchmark_final_value is None:
            return None
        # Strict: a tie does NOT beat the benchmark.
        return self.final_value > self.benchmark_final_value

    @property
    def num_round_trips(self) -> int:
        # Count position exits: SELL (closing a long, incl. liquidation) and
        # cover (closing a short, incl. liquidation). Exclude "SELL (short)"
        # which opens a short position rather than closing one.
        return sum(
            1
            for t in self.trades
            if "cover" in t.action or (t.action.startswith("SELL") and "short" not in t.action)
        )

    @property
    def is_win(self) -> bool:
        return self.final_value > self.starting_cash


def _vol_target_fraction_series(
    price_df: pd.DataFrame,
    target_vol: float | None,
    vol_lookback: int,
    bars_per_year: float | None,
    fallback_fraction: float = 1.0,
) -> pd.Series:
    """
    Per-bar fraction of the available cash to deploy, for volatility-targeted
    position sizing.

    target_vol is an ANNUALIZED volatility target expressed as a fraction
    (0.10 = 10%). The realized volatility at bar i is the rolling standard
    deviation of the log returns over the previous vol_lookback returns,
    annualized with bars_per_year. The fraction is target_vol / realized_vol,
    clamped to [0, 1]: this only ever REDUCES exposure (no leverage).

    Bars where the window is not full yet, realized volatility is zero, or the
    ratio is not finite fall back to fallback_fraction (1.0 by default, i.e.
    the legacy fully-invested sizing).

    target_vol=None disables the feature entirely and returns a constant 1.0
    series, without requiring bars_per_year.

    No lookahead: fraction.iloc[i] only uses close prices up to bar i (the
    rolling window ends at i), so a signal computed from bar i cannot see any
    future bar. The rolling window has min_periods=vol_lookback, hence the
    first vol_lookback values are always the fallback.
    """
    if target_vol is None:
        return pd.Series(1.0, index=price_df.index, dtype=float)
    if bars_per_year is None:
        raise ValueError("bars_per_year is required for volatility targeting (unknown interval)")
    if vol_lookback < 2:
        raise ValueError(f"vol_lookback must be >= 2, got {vol_lookback}")

    closes = price_df["close"].astype(float)
    log_returns = np.log(closes / closes.shift(1))
    realized_vol = (
        log_returns.rolling(window=vol_lookback, min_periods=vol_lookback).std(ddof=1)
        * np.sqrt(bars_per_year)
    )
    fraction = target_vol / realized_vol
    invalid = (~np.isfinite(fraction)) | (realized_vol <= 0)
    fraction = fraction.where(~invalid, fallback_fraction)
    return fraction.clip(lower=0.0, upper=1.0)


def _simulate_wallet_full(
    price_df: pd.DataFrame,
    confluence: dict,
    threshold: int,
    fee_rate: float,
    starting_cash: float,
    direction: str = "long",
    target_vol: float | None = None,
    vol_lookback: int = 20,
    fallback_fraction: float = 1.0,
    bars_per_year: float | None = None,
    entry_threshold: int | None = None,
    exit_threshold: int | None = None,
    fee_fixed: float = 0.0,
    slippage_bps: float = 0.0,
) -> tuple[float, list[Trade], pd.Series]:
    """
    Full simulation core: same trading rules as simulate_wallet(), plus the
    mark-to-market equity curve.

    Execution timing is unchanged: the signal computed from candle i is
    executed at candle (i+1)'s open. equity[0] is the starting cash (no
    execution can happen on the first bar), equity[t] is the wallet value
    (cash + shares * close[t]) right after executing candle (t-1)'s signal at
    candle t's open, and the final point is overwritten with the post-
    liquidation cash so that equity[-1] == the returned final value exactly.

    No point in the curve looks beyond its own bar, which keeps the existing
    anti-lookahead guarantee intact.

    target_vol (annualized fraction, e.g. 0.10 for 10%) opts into
    volatility-targeted sizing: each entry deploys only
    fraction * cash, where fraction is derived from the realized volatility
    known at the signal bar. target_vol=None keeps the legacy fully-invested
    sizing exactly (fraction == 1.0 everywhere).

    entry_threshold/exit_threshold implement threshold hysteresis: a new
    position is opened only once the signal count reaches entry_threshold,
    while an open position is closed (or flipped) only once the opposite
    count reaches exit_threshold. Both default to `threshold`, which reproduces
    the legacy single-threshold behaviour exactly. The two thresholds act
    independently per direction (a long exit never depends on the long entry
    count, and vice versa).

    fee_fixed is an absolute fee (same currency as starting_cash) charged on
    every paid fill: long/short entry, long exit/cover, each leg of a flip and
    the end-of-run liquidation. It is deducted from cash in addition to the
    existing percentage fee_rate fee, which stays applied to the notional.
    Unlike that percentage fee, the absolute fee does not change the position
    sizing: every fill costs exactly fee_fixed, so a run with N fills always
    ends N * fee_fixed below the same run without the fixed fee.
    fee_fixed=0.0 reproduces the legacy behaviour exactly.

    slippage_bps models execution slippage in basis points (1 bp = 0.01%),
    applied ONLY to the execution price and always adversely: a buy (long
    entry, short cover) fills at price * (1 + slippage_bps / 10000), a sell
    (long exit, short entry) fills at price * (1 - slippage_bps / 10000). It
    applies to every paid fill - entries, exits, both legs of a flip and the
    end-of-run liquidation - and is independent of both fee_rate (percentage
    of notional) and fee_fixed (absolute per fill). slippage_bps=0.0 leaves
    execution prices untouched, reproducing the legacy behaviour exactly.
    """
    if direction not in ("long", "short", "reverse"):
        raise ValueError(f"direction must be 'long', 'short' or 'reverse', got {direction!r}")
    if slippage_bps < 0.0:
        raise ValueError(f"slippage_bps must be >= 0, got {slippage_bps}")

    slippage_fraction = slippage_bps / 10000.0

    def _buy_price(price: float) -> float:
        return price if slippage_bps == 0.0 else price * (1.0 + slippage_fraction)

    def _sell_price(price: float) -> float:
        return price if slippage_bps == 0.0 else price * (1.0 - slippage_fraction)

    # Hysteresis thresholds are opt-in: None means "same as the legacy
    # threshold". Validate only explicitly supplied values so that legacy
    # calls (including any unusual historical `threshold` values) are
    # unaffected.
    if entry_threshold is not None and entry_threshold < 1:
        raise ValueError(f"entry_threshold must be >= 1, got {entry_threshold}")
    if exit_threshold is not None and exit_threshold < 1:
        raise ValueError(f"exit_threshold must be >= 1, got {exit_threshold}")
    if entry_threshold is None:
        entry_threshold = threshold
    if exit_threshold is None:
        exit_threshold = threshold

    # Computed once for the whole run; fraction.iloc[i] is the sizing to use
    # for a signal computed from bar i (executed at bar i+1's open).
    fractions = _vol_target_fraction_series(
        price_df, target_vol, vol_lookback, bars_per_year, fallback_fraction
    )

    times = list(price_df.index)
    cash = starting_cash
    # Cumulative absolute fees already charged. Adding them back when sizing a
    # new position keeps fee_fixed purely additive (it never changes the
    # notional, unlike the percentage fee which is part of it).
    fixed_fees_paid = 0.0
    shares = 0.0  # positive = long, negative = short
    trades: list[Trade] = []
    equity_values: list[float] = []

    if times:
        # Bar 0: no signal has been actionable yet, so the wallet is untouched.
        equity_values.append(float(starting_cash))

        # For each bar t >= 1, act on the signal from bar t-1 at bar t's open,
        # then mark the resulting wallet to bar t's close.
        for t in range(1, len(times)):
            i = t - 1
            conf = confluence.get(pd.Timestamp(times[i]).isoformat())
            if conf is not None:
                execution_price = float(price_df["open"].iloc[i + 1])
                execution_time = times[i + 1]
                fraction = float(fractions.iloc[i])

                if direction == "long":
                    if shares == 0.0 and conf["bullish_count"] >= entry_threshold and fraction > 0.0:
                        exec_price = _buy_price(execution_price)
                        notional = fraction * (cash + fixed_fees_paid)
                        fee_paid = notional * fee_rate
                        shares = (notional - fee_paid) / exec_price
                        cash = cash - notional - fee_fixed
                        fixed_fees_paid += fee_fixed
                        trades.append(Trade("BUY", execution_time, exec_price, fee_paid))
                    elif shares > 0.0 and conf["bearish_count"] >= exit_threshold:
                        exec_price = _sell_price(execution_price)
                        proceeds = shares * exec_price
                        fee_paid = proceeds * fee_rate
                        # Any cash left uninvested by volatility targeting stays
                        # in the wallet. When fraction == 1.0, cash is already
                        # 0 here, so this is exactly the legacy behaviour.
                        cash = cash + proceeds - fee_paid - fee_fixed
                        fixed_fees_paid += fee_fixed
                        shares = 0.0
                        trades.append(Trade("SELL", execution_time, exec_price, fee_paid))
                elif direction == "short":
                    if shares == 0.0 and conf["bearish_count"] >= entry_threshold and fraction > 0.0:
                        # Sell short the fraction * cash worth of shares.
                        # We receive the sale proceeds in cash and hold a negative share
                        # position; fees are deducted from the proceeds.
                        exec_price = _sell_price(execution_price)
                        notional = fraction * (cash + fixed_fees_paid)
                        fee_paid = notional * fee_rate
                        shares = -notional / exec_price
                        cash = cash + notional - fee_paid - fee_fixed
                        fixed_fees_paid += fee_fixed
                        trades.append(Trade("SELL (short)", execution_time, exec_price, fee_paid))
                    elif shares < 0.0 and conf["bullish_count"] >= exit_threshold:
                        # Buy to cover: repurchase shares to close the short position.
                        exec_price = _buy_price(execution_price)
                        cover_cost = (-shares) * exec_price
                        fee_paid = cover_cost * fee_rate
                        cash = cash - cover_cost - fee_paid - fee_fixed
                        fixed_fees_paid += fee_fixed
                        shares = 0.0
                        trades.append(Trade("BUY (cover)", execution_time, exec_price, fee_paid))
                else:  # direction == "reverse"
                    if shares == 0.0 and conf["bullish_count"] >= entry_threshold and fraction > 0.0:
                        # Flat + bullish: go long, same sizing as the "long" direction.
                        exec_price = _buy_price(execution_price)
                        notional = fraction * (cash + fixed_fees_paid)
                        fee_paid = notional * fee_rate
                        shares = (notional - fee_paid) / exec_price
                        cash = cash - notional - fee_fixed
                        fixed_fees_paid += fee_fixed
                        trades.append(Trade("BUY", execution_time, exec_price, fee_paid))
                    elif shares == 0.0 and conf["bearish_count"] >= entry_threshold and fraction > 0.0:
                        # Flat + bearish: go short, same sizing as the "short" direction.
                        exec_price = _sell_price(execution_price)
                        notional = fraction * (cash + fixed_fees_paid)
                        fee_paid = notional * fee_rate
                        shares = -notional / exec_price
                        cash = cash + notional - fee_paid - fee_fixed
                        fixed_fees_paid += fee_fixed
                        trades.append(Trade("SELL (short)", execution_time, exec_price, fee_paid))
                    elif shares > 0.0 and conf["bearish_count"] >= exit_threshold:
                        # Long + bearish: sell the long position, then - only if the
                        # opposite signal also clears entry_threshold - immediately
                        # sell short with fraction * the resulting cash. Both legs
                        # are sells, so both fill at the (adverse) sell price. A zero
                        # fraction, or a bearish count between exit_threshold and
                        # entry_threshold, closes the long and stays flat without a
                        # zero-size entry trade.
                        exec_price = _sell_price(execution_price)
                        proceeds = shares * exec_price
                        sell_fee_paid = proceeds * fee_rate
                        cash = cash + proceeds - sell_fee_paid - fee_fixed
                        fixed_fees_paid += fee_fixed
                        shares = 0.0
                        trades.append(Trade("SELL", execution_time, exec_price, sell_fee_paid))

                        if fraction > 0.0 and conf["bearish_count"] >= entry_threshold:
                            notional = fraction * (cash + fixed_fees_paid)
                            short_fee_paid = notional * fee_rate
                            shares = -notional / exec_price
                            cash = cash + notional - short_fee_paid - fee_fixed
                            fixed_fees_paid += fee_fixed
                            trades.append(Trade("SELL (short)", execution_time, exec_price, short_fee_paid))
                    elif shares < 0.0 and conf["bullish_count"] >= exit_threshold:
                        # Short + bullish: cover the short position, then - only if
                        # the opposite signal also clears entry_threshold -
                        # immediately buy long with fraction * the resulting cash.
                        # Both legs are buys, so both fill at the (adverse) buy
                        # price. A zero fraction, or a bullish count between
                        # exit_threshold and entry_threshold, covers the short and
                        # stays flat without a zero-size entry trade.
                        exec_price = _buy_price(execution_price)
                        cover_cost = (-shares) * exec_price
                        cover_fee_paid = cover_cost * fee_rate
                        cash = cash - cover_cost - cover_fee_paid - fee_fixed
                        fixed_fees_paid += fee_fixed
                        shares = 0.0
                        trades.append(Trade("BUY (cover)", execution_time, exec_price, cover_fee_paid))

                        if fraction > 0.0 and conf["bullish_count"] >= entry_threshold:
                            notional = fraction * (cash + fixed_fees_paid)
                            buy_fee_paid = notional * fee_rate
                            shares = (notional - buy_fee_paid) / exec_price
                            cash = cash - notional - fee_fixed
                            fixed_fees_paid += fee_fixed
                            trades.append(Trade("BUY", execution_time, exec_price, buy_fee_paid))

            equity_values.append(cash + shares * float(price_df["close"].iloc[t]))

        if shares > 0.0:
            last_price = _sell_price(float(price_df["close"].iloc[-1]))
            proceeds = shares * last_price
            fee_paid = proceeds * fee_rate
            cash = cash + proceeds - fee_paid - fee_fixed
            fixed_fees_paid += fee_fixed
            trades.append(Trade("SELL (liquidated at end)", times[-1], last_price, fee_paid))
        elif shares < 0.0:
            last_price = _buy_price(float(price_df["close"].iloc[-1]))
            cover_cost = (-shares) * last_price
            fee_paid = cover_cost * fee_rate
            cash = cash - cover_cost - fee_paid - fee_fixed
            fixed_fees_paid += fee_fixed
            trades.append(Trade("BUY (cover liquidated at end)", times[-1], last_price, fee_paid))

        # The run ends fully in cash: make the last curve point exact.
        equity_values[-1] = cash

    equity = pd.Series(equity_values, index=price_df.index, dtype=float)
    return cash, trades, equity


def simulate_wallet(
    price_df: pd.DataFrame,
    confluence: dict,
    threshold: int,
    fee_rate: float,
    starting_cash: float = DEFAULT_STARTING_CASH,
    direction: str = "long",
    target_vol: float | None = None,
    vol_lookback: int = 20,
    fallback_fraction: float = 1.0,
    bars_per_year: float | None = None,
    entry_threshold: int | None = None,
    exit_threshold: int | None = None,
    fee_fixed: float = 0.0,
    slippage_bps: float = 0.0,
) -> tuple[float, list[Trade]]:
    """
    Pure simulation core (no I/O, fully deterministic given its inputs - the
    thing to unit-test directly). Walks the candles in order:

    - direction="long": while flat, buys at the NEXT candle's open the first
      time bullish_count clears `threshold`; while holding, sells at the NEXT
      candle's open the first time bearish_count clears `threshold`.
    - direction="short": while flat, sells short at the NEXT candle's open the
      first time bearish_count clears `threshold`; while short, buys to cover
      at the NEXT candle's open the first time bullish_count clears `threshold`.
    - direction="reverse": while flat, behaves like "long" on a bullish signal
      and like "short" on a bearish signal. When already positioned, an
      opposite signal flips the wallet directly: long + bearish sells the long
      position and immediately sells short, short + bullish covers the short
      and immediately buys long. Both flip trades execute at the same
      NEXT candle's open.

    Any position still open after the last candle is force-liquidated at the
    final candle's close (fee included), so every run ends fully in cash and
    results are directly comparable across thresholds/fees/symbols.

    Fractional shares are allowed (this simulates dollar-value wallet
    performance, not literal whole-share broker execution).

    target_vol/vol_lookback/fallback_fraction/bars_per_year opt into
    volatility-targeted sizing (see _vol_target_fraction_series); leaving
    target_vol=None keeps the exact legacy behaviour.

    entry_threshold/exit_threshold opt into threshold hysteresis (see
    _simulate_wallet_full); leaving both None reproduces the legacy
    single-threshold rules exactly.

    fee_fixed is an absolute fee charged on every paid fill (see
    _simulate_wallet_full); leaving it at 0.0 reproduces the legacy behaviour
    exactly. It combines with fee_rate, which stays a percentage of notional.

    slippage_bps applies adverse execution slippage to every paid fill (see
    _simulate_wallet_full); leaving it at 0.0 reproduces the legacy behaviour
    exactly. It is independent of both fee mechanisms.
    """
    cash, trades, _ = _simulate_wallet_full(
        price_df, confluence, threshold, fee_rate, starting_cash, direction,
        target_vol=target_vol, vol_lookback=vol_lookback,
        fallback_fraction=fallback_fraction, bars_per_year=bars_per_year,
        entry_threshold=entry_threshold, exit_threshold=exit_threshold,
        fee_fixed=fee_fixed, slippage_bps=slippage_bps,
    )
    return cash, trades


def simulate_wallet_with_equity(
    price_df: pd.DataFrame,
    confluence: dict,
    threshold: int,
    fee_rate: float,
    starting_cash: float = DEFAULT_STARTING_CASH,
    direction: str = "long",
    target_vol: float | None = None,
    vol_lookback: int = 20,
    fallback_fraction: float = 1.0,
    bars_per_year: float | None = None,
    entry_threshold: int | None = None,
    exit_threshold: int | None = None,
    fee_fixed: float = 0.0,
    slippage_bps: float = 0.0,
) -> tuple[float, list[Trade], pd.Series]:
    """
    Same signature and trading rules as simulate_wallet(), but also returns
    the mark-to-market equity curve (one point per candle, indexed by the
    price DataFrame's index). See _simulate_wallet_full() for the exact
    contract; equity.iloc[-1] equals the returned final cash value.
    """
    return _simulate_wallet_full(
        price_df, confluence, threshold, fee_rate, starting_cash, direction,
        target_vol=target_vol, vol_lookback=vol_lookback,
        fallback_fraction=fallback_fraction, bars_per_year=bars_per_year,
        entry_threshold=entry_threshold, exit_threshold=exit_threshold,
        fee_fixed=fee_fixed, slippage_bps=slippage_bps,
    )


def buy_and_hold_value(
    price_df: pd.DataFrame,
    fee_rate: float,
    starting_cash: float = DEFAULT_STARTING_CASH,
) -> float:
    """
    Passive long buy-and-hold: buy at the first bar's OPEN, value/sell at the
    last bar's CLOSE, with the SAME fee assumptions as simulate_wallet
    (entry fee on the notional, exit fee on the proceeds).

    This is the benchmark a strategy must beat: it only depends on the bars
    and the fee rate, never on a threshold or direction.
    """
    entry_price = float(price_df["open"].iloc[0])
    exit_price = float(price_df["close"].iloc[-1])
    entry_fee = starting_cash * fee_rate
    shares = (starting_cash - entry_fee) / entry_price
    gross = shares * exit_price
    exit_fee = gross * fee_rate
    return gross - exit_fee


def buy_and_hold_equity_curve(
    price_df: pd.DataFrame,
    fee_rate: float,
    starting_cash: float = DEFAULT_STARTING_CASH,
) -> pd.Series:
    """
    Passive long buy-and-hold equity curve: one mark-to-market point per bar,
    valued at that bar's CLOSE, with the same fee assumptions as
    buy_and_hold_value().

    The entry fee is folded into the initial position: only
    (starting_cash - entry_fee) is invested at the first bar's OPEN, so the
    shares bought are already reduced by it and the entry fee does not show up
    again in the returns of the returned series. The exit fee is applied only
    to the final point, exactly where a passive holder would sell.

    Invariants: len(curve) == len(price_df) (same index), and the final point
    equals buy_and_hold_value(price_df, fee_rate, starting_cash). An empty
    price frame yields an empty float Series rather than an error.
    """
    if price_df.empty:
        return pd.Series(dtype=float)

    entry_price = float(price_df["open"].iloc[0])
    entry_fee = starting_cash * fee_rate
    shares = (starting_cash - entry_fee) / entry_price
    equity = shares * price_df["close"].astype(float)
    equity.iloc[-1] = equity.iloc[-1] * (1 - fee_rate)
    return equity


def backtest_symbol(
    symbol: str,
    start: datetime,
    end: datetime,
    interval: str,
    thresholds: list[int] = None,
    fee_rates: list[float] = None,
    starting_cash: float = DEFAULT_STARTING_CASH,
    direction: str = "long",
    signal_provider: str = DEFAULT_SIGNAL_PROVIDER,
    provider_kwargs: dict | None = None,
    target_vol: float | None = None,
    vol_lookback: int = 20,
    entry_threshold: int | None = None,
    exit_threshold: int | None = None,
    fee_fixed: float = 0.0,
    slippage_bps: float = 0.0,
) -> list[BacktestResult]:
    """
    Fetches price data and computes the selected signal ONCE for this symbol,
    then cheaply re-runs simulate_wallet() for every threshold x fee
    combination on that same in-memory result - one DB/API round trip per
    symbol regardless of how many threshold/fee combinations are swept.

    signal_provider names a registered SignalProvider; the default
    "confluence" preserves the original behaviour of using
    compute_confluence() directly.

    provider_kwargs (optional) is forwarded to that provider's factory. It
    must contain only options the selected factory accepts - e.g.
    {"lookbacks": [...]} for "tsmom" - because the engine intentionally does
    not know provider-specific parameter names.

    When direction="both", runs both a long and a short simulation for each
    threshold x fee combination and concatenates the results.
    When direction="reverse", runs a single simulation that flips directly
    between long and short positions.

    entry_threshold/exit_threshold are forwarded to the simulation core
    unchanged (None keeps the legacy single-threshold rules).

    fee_fixed is forwarded to every simulation unchanged (0.0 keeps the legacy
    percentage-only fee behaviour).

    slippage_bps is forwarded to every simulation unchanged (0.0 keeps the
    legacy no-slippage behaviour).
    """
    if direction not in ("long", "short", "both", "reverse"):
        raise ValueError(f"direction must be 'long', 'short', 'both' or 'reverse', got {direction!r}")

    thresholds = thresholds or DEFAULT_THRESHOLDS
    fee_rates = DEFAULT_FEE_RATES if fee_rates is None else fee_rates

    df = get_price_candles_dataframe(symbol, start, end, interval)
    if df.empty or len(df) < 3:
        return []
    df = df.sort_index()

    provider = get_provider(
        signal_provider, symbol, df.index.min(), df.index.max(), interval, **(provider_kwargs or {})
    )
    confluence = provider.build(df)

    # Annualization only matters for the risk metrics; an unusual interval
    # must not abort an otherwise valid backtest.
    try:
        bars_per_year = metrics.annualization_factor(interval)
    except ValueError:
        bars_per_year = None

    # The passive benchmark depends only on the (sorted) bars and the fee
    # rate - not on threshold or direction - so compute its value and its risk
    # metrics once per fee, then reuse them for every result row.
    benchmark_by_fee = {
        fee_rate: buy_and_hold_value(df, fee_rate, starting_cash) for fee_rate in fee_rates
    }
    benchmark_curve_by_fee = {
        fee_rate: buy_and_hold_equity_curve(df, fee_rate, starting_cash) for fee_rate in fee_rates
    }
    benchmark_metrics_by_fee = {}
    for fee_rate, curve in benchmark_curve_by_fee.items():
        bench_returns = curve.pct_change()
        bench_max_drawdown = metrics.max_drawdown(curve)
        benchmark_metrics_by_fee[fee_rate] = {
            "benchmark_sharpe": (
                None if bars_per_year is None else metrics.sharpe_ratio(bench_returns, bars_per_year)
            ),
            "benchmark_sortino": (
                None if bars_per_year is None else metrics.sortino_ratio(bench_returns, bars_per_year)
            ),
            "benchmark_max_drawdown_pct": (
                None if bench_max_drawdown is None else bench_max_drawdown * 100
            ),
            "benchmark_calmar": (
                None if bars_per_year is None else metrics.calmar_ratio(curve, bars_per_year)
            ),
        }

    results = []
    directions = ("long", "short") if direction == "both" else (direction,)
    for dir_ in directions:
        for threshold in thresholds:
            for fee_rate in fee_rates:
                final_value, trades, equity = simulate_wallet_with_equity(
                    df, confluence, threshold, fee_rate, starting_cash, dir_,
                    target_vol=target_vol, vol_lookback=vol_lookback,
                    bars_per_year=bars_per_year,
                    entry_threshold=entry_threshold, exit_threshold=exit_threshold,
                    fee_fixed=fee_fixed, slippage_bps=slippage_bps,
                )
                results.append(BacktestResult(
                    symbol, threshold, fee_rate, starting_cash, final_value, dir_, trades,
                    benchmark_final_value=benchmark_by_fee[fee_rate],
                    equity_curve=equity,
                    bars_per_year=bars_per_year,
                    **benchmark_metrics_by_fee[fee_rate],
                ))
    return results


def run_backtest(
    symbols: list[str] = None,
    start: datetime = None,
    end: datetime = None,
    interval: str = "1h",
    thresholds: list[int] = None,
    fee_rates: list[float] = None,
    starting_cash: float = DEFAULT_STARTING_CASH,
    direction: str = "long",
    signal_provider: str = DEFAULT_SIGNAL_PROVIDER,
    provider_kwargs: dict | None = None,
    equity_out: dict | None = None,
    target_vol: float | None = None,
    vol_lookback: int = 20,
    entry_threshold: int | None = None,
    exit_threshold: int | None = None,
    fee_fixed: float = 0.0,
    slippage_bps: float = 0.0,
    broker_preset: str | None = None,
    universe: str = "all",
) -> pd.DataFrame:
    """
    Runs backtest_symbol() across a list of symbols (default: the curated
    quick-list) and flattens the results into one long-format DataFrame.

    provider_kwargs is passed through unchanged to every backtest_symbol()
    call, so all symbols in a run use the same provider configuration.

    equity_out (optional) is filled in-place with the per-run equity curves,
    keyed by (symbol, direction, threshold, fee_pct). The curves are kept out
    of the returned DataFrame, which stays flat and cheap to aggregate.

    entry_threshold/exit_threshold are passed through unchanged to every
    backtest_symbol() call (None keeps the legacy single-threshold rules).

    fee_fixed is passed through unchanged to every backtest_symbol() call
    (0.0 keeps the legacy percentage-only fee behaviour).

    slippage_bps is passed through unchanged to every backtest_symbol() call
    (0.0 keeps the legacy no-slippage behaviour).

    broker_preset applies indicative broker costs with precedence:
    explicit fee_rates/fee_fixed/slippage_bps override preset values.

    universe ('all'|'us'|'eu') selects the default symbol list when symbols
    is not provided; an explicit symbols list always wins.
    """
    # Resolve broker preset with precedence: explicit args override preset
    fee_rates_resolved = fee_rates
    fee_fixed_resolved = fee_fixed
    slippage_bps_resolved = slippage_bps
    if broker_preset is not None:
        # Only non-default explicit values override the preset.
        if isinstance(fee_rates, (list, tuple)):
            explicit_fee_pct = fee_rates[0] if fee_rates else None
        elif isinstance(fee_rates, (int, float)) and not isinstance(fee_rates, bool):
            explicit_fee_pct = float(fee_rates)
        else:
            explicit_fee_pct = None
        resolved_fee_pct_single, resolved_fee_fixed, resolved_slippage_bps, _label = resolve_broker_preset(
            broker_preset,
            explicit_fee_pct,
            fee_fixed if fee_fixed != 0.0 else None,
            slippage_bps if slippage_bps != 0.0 else None,
        )
        # A preset supplies a single fee rate; an explicit list stays authoritative.
        if fee_rates is None:
            fee_rates_resolved = [resolved_fee_pct_single]
        fee_fixed_resolved = resolved_fee_fixed
        slippage_bps_resolved = resolved_slippage_bps

    symbols = symbols or get_universe_symbols(universe)
    end = end or datetime.now()
    start = start or (end - timedelta(days=60))

    rows = []
    for symbol in symbols:
        try:
            results = backtest_symbol(
                symbol, start, end, interval, thresholds, fee_rates_resolved, starting_cash, direction,
                signal_provider, provider_kwargs,
                target_vol=target_vol, vol_lookback=vol_lookback,
                entry_threshold=entry_threshold, exit_threshold=exit_threshold,
                fee_fixed=fee_fixed_resolved, slippage_bps=slippage_bps_resolved,
            )
        except Exception as e:
            rows.append({"symbol": symbol, "threshold": None, "fee_pct": None, "direction": None, "error": str(e)})
            continue
        if not results:
            rows.append({"symbol": symbol, "threshold": None, "fee_pct": None, "direction": None, "error": "no data / too few candles"})
            continue
        for r in results:
            if equity_out is not None:
                equity_out[(r.symbol, r.direction, r.threshold, round(r.fee_rate * 100, 2))] = r.equity_curve
            rows.append({
                "symbol": r.symbol,
                "threshold": r.threshold,
                "fee_pct": round(r.fee_rate * 100, 2),
                "direction": r.direction,
                "final_value": round(r.final_value, 2),
                "return_pct": round(r.return_pct, 2),
                "benchmark_return_pct": None if r.benchmark_return_pct is None else round(r.benchmark_return_pct, 2),
                "beats_benchmark": r.beats_benchmark,
                "sharpe": None if r.sharpe is None else round(r.sharpe, 2),
                "sortino": None if r.sortino is None else round(r.sortino, 2),
                "max_drawdown_pct": None if r.max_drawdown_pct is None else round(r.max_drawdown_pct, 2),
                "calmar": None if r.calmar is None else round(r.calmar, 2),
                "benchmark_sharpe": None if r.benchmark_sharpe is None else round(r.benchmark_sharpe, 2),
                "benchmark_sortino": None if r.benchmark_sortino is None else round(r.benchmark_sortino, 2),
                "benchmark_max_drawdown_pct": (
                    None if r.benchmark_max_drawdown_pct is None else round(r.benchmark_max_drawdown_pct, 2)
                ),
                "benchmark_calmar": None if r.benchmark_calmar is None else round(r.benchmark_calmar, 2),
                "num_trades": r.num_round_trips,
                "win": r.is_win,
                "error": None,
            })
    return pd.DataFrame(rows)
