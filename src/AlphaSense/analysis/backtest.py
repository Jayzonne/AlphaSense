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

import pandas as pd

from AlphaSense.frontend.data_access import get_price_candles_dataframe
from AlphaSense.analysis.confluence import compute_confluence
from AlphaSense.analysis.signals import ConfluenceProvider, get_provider, register_provider
from AlphaSense.requests.quotes import US_MEGACAPS, EU_MEGACAPS

DEFAULT_SYMBOLS = [symbol for symbol, _ in US_MEGACAPS + EU_MEGACAPS]
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

    @property
    def return_pct(self) -> float:
        return (self.final_value / self.starting_cash - 1) * 100

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


def simulate_wallet(
    price_df: pd.DataFrame,
    confluence: dict,
    threshold: int,
    fee_rate: float,
    starting_cash: float = DEFAULT_STARTING_CASH,
    direction: str = "long",
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
    """
    if direction not in ("long", "short", "reverse"):
        raise ValueError(f"direction must be 'long', 'short' or 'reverse', got {direction!r}")

    times = list(price_df.index)
    cash = starting_cash
    shares = 0.0  # positive = long, negative = short
    trades: list[Trade] = []

    # range(len(times) - 1): the final candle's own signal can never be acted
    # on, since there's no "next candle" left to execute at - an unavoidable
    # edge of any window-bounded backtest, not a bug.
    for i in range(len(times) - 1):
        conf = confluence.get(pd.Timestamp(times[i]).isoformat())
        if conf is None:
            continue
        execution_price = float(price_df["open"].iloc[i + 1])
        execution_time = times[i + 1]

        if direction == "long":
            if shares == 0.0 and conf["bullish_count"] >= threshold:
                fee_paid = cash * fee_rate
                shares = (cash - fee_paid) / execution_price
                cash = 0.0
                trades.append(Trade("BUY", execution_time, execution_price, fee_paid))
            elif shares > 0.0 and conf["bearish_count"] >= threshold:
                proceeds = shares * execution_price
                fee_paid = proceeds * fee_rate
                cash = proceeds - fee_paid
                shares = 0.0
                trades.append(Trade("SELL", execution_time, execution_price, fee_paid))
        elif direction == "short":
            if shares == 0.0 and conf["bearish_count"] >= threshold:
                # Sell short the equivalent of the available cash worth of shares.
                # We receive the sale proceeds in cash and hold a negative share
                # position; fees are deducted from the proceeds.
                fee_paid = cash * fee_rate
                short_value = cash  # notional value sold short
                shares = -short_value / execution_price
                cash = cash + short_value - fee_paid
                trades.append(Trade("SELL (short)", execution_time, execution_price, fee_paid))
            elif shares < 0.0 and conf["bullish_count"] >= threshold:
                # Buy to cover: repurchase shares to close the short position.
                cover_cost = (-shares) * execution_price
                fee_paid = cover_cost * fee_rate
                cash = cash - cover_cost - fee_paid
                shares = 0.0
                trades.append(Trade("BUY (cover)", execution_time, execution_price, fee_paid))
        else:  # direction == "reverse"
            if shares == 0.0 and conf["bullish_count"] >= threshold:
                # Flat + bullish: go long, same sizing as the "long" direction.
                fee_paid = cash * fee_rate
                shares = (cash - fee_paid) / execution_price
                cash = 0.0
                trades.append(Trade("BUY", execution_time, execution_price, fee_paid))
            elif shares == 0.0 and conf["bearish_count"] >= threshold:
                # Flat + bearish: go short, same sizing as the "short" direction.
                fee_paid = cash * fee_rate
                short_value = cash  # notional value sold short
                shares = -short_value / execution_price
                cash = cash + short_value - fee_paid
                trades.append(Trade("SELL (short)", execution_time, execution_price, fee_paid))
            elif shares > 0.0 and conf["bearish_count"] >= threshold:
                # Long + bearish: sell the long position, then immediately sell
                # short with the resulting cash - both at the same execution price.
                proceeds = shares * execution_price
                sell_fee_paid = proceeds * fee_rate
                cash = proceeds - sell_fee_paid
                shares = 0.0
                trades.append(Trade("SELL", execution_time, execution_price, sell_fee_paid))

                short_fee_paid = cash * fee_rate
                short_value = cash  # notional value sold short
                shares = -short_value / execution_price
                cash = cash + short_value - short_fee_paid
                trades.append(Trade("SELL (short)", execution_time, execution_price, short_fee_paid))
            elif shares < 0.0 and conf["bullish_count"] >= threshold:
                # Short + bullish: cover the short position, then immediately
                # buy long with the resulting cash - both at the same execution price.
                cover_cost = (-shares) * execution_price
                cover_fee_paid = cover_cost * fee_rate
                cash = cash - cover_cost - cover_fee_paid
                shares = 0.0
                trades.append(Trade("BUY (cover)", execution_time, execution_price, cover_fee_paid))

                buy_fee_paid = cash * fee_rate
                shares = (cash - buy_fee_paid) / execution_price
                cash = 0.0
                trades.append(Trade("BUY", execution_time, execution_price, buy_fee_paid))

    if shares > 0.0:
        last_price = float(price_df["close"].iloc[-1])
        proceeds = shares * last_price
        fee_paid = proceeds * fee_rate
        cash = proceeds - fee_paid
        trades.append(Trade("SELL (liquidated at end)", times[-1], last_price, fee_paid))
    elif shares < 0.0:
        last_price = float(price_df["close"].iloc[-1])
        cover_cost = (-shares) * last_price
        fee_paid = cover_cost * fee_rate
        cash = cash - cover_cost - fee_paid
        trades.append(Trade("BUY (cover liquidated at end)", times[-1], last_price, fee_paid))

    return cash, trades


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

    results = []
    directions = ("long", "short") if direction == "both" else (direction,)
    for dir_ in directions:
        for threshold in thresholds:
            for fee_rate in fee_rates:
                final_value, trades = simulate_wallet(df, confluence, threshold, fee_rate, starting_cash, dir_)
                results.append(BacktestResult(symbol, threshold, fee_rate, starting_cash, final_value, dir_, trades))
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
) -> pd.DataFrame:
    """
    Runs backtest_symbol() across a list of symbols (default: the curated
    quick-list) and flattens the results into one long-format DataFrame.

    provider_kwargs is passed through unchanged to every backtest_symbol()
    call, so all symbols in a run use the same provider configuration.
    """
    symbols = symbols or DEFAULT_SYMBOLS
    end = end or datetime.now()
    start = start or (end - timedelta(days=60))

    rows = []
    for symbol in symbols:
        try:
            results = backtest_symbol(
                symbol, start, end, interval, thresholds, fee_rates, starting_cash, direction,
                signal_provider, provider_kwargs,
            )
        except Exception as e:
            rows.append({"symbol": symbol, "threshold": None, "fee_pct": None, "direction": None, "error": str(e)})
            continue
        if not results:
            rows.append({"symbol": symbol, "threshold": None, "fee_pct": None, "direction": None, "error": "no data / too few candles"})
            continue
        for r in results:
            rows.append({
                "symbol": r.symbol,
                "threshold": r.threshold,
                "fee_pct": round(r.fee_rate * 100, 2),
                "direction": r.direction,
                "final_value": round(r.final_value, 2),
                "return_pct": round(r.return_pct, 2),
                "num_trades": r.num_round_trips,
                "win": r.is_win,
                "error": None,
            })
    return pd.DataFrame(rows)
