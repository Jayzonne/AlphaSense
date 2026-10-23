"""
CLI for AlphaSense.analysis.backtest.

Examples:
    python backtest.py
    python backtest.py --preset YTD
    python backtest.py --symbols AAPL MSFT --interval 1h --start 2026-06-01 --end 2026-09-01
    python backtest.py --thresholds 1 2 3 --fees 0 1
"""
import argparse
from datetime import datetime, timedelta

import pandas as pd

from AlphaSense.analysis.backtest import run_backtest, DEFAULT_SYMBOLS, DEFAULT_THRESHOLDS, DEFAULT_FEE_RATES
from AlphaSense.frontend.data_access import RANGE_PRESETS, resolve_range_preset


"""
CLI for AlphaSense.analysis.backtest.

Examples:
    python backtest.py
    python backtest.py --preset YTD
    python backtest.py --symbols AAPL MSFT --interval 1h --start 2026-06-01 --end 2026-09-01
    python backtest.py --thresholds 1 2 3 --fees 0 1
    python backtest.py --direction short
    python backtest.py --direction both
"""
import argparse
from datetime import datetime, timedelta

import pandas as pd

from AlphaSense.analysis.backtest import (
    run_backtest,
    DEFAULT_SYMBOLS,
    DEFAULT_THRESHOLDS,
    DEFAULT_FEE_RATES,
    get_universe_symbols,
)
from AlphaSense.analysis.signals import available_providers
from AlphaSense.frontend.data_access import RANGE_PRESETS, resolve_range_preset


def _print_symbol_table(symbol: str, symbol_df: pd.DataFrame) -> None:
    print(f"\n=== {symbol} ===")
    error = symbol_df["error"].iloc[0]
    if pd.notna(error):
        print(f"  (skipped: {error})")
        return

    directions = symbol_df["direction"].dropna().unique()
    for direction in directions:
        dir_df = symbol_df[symbol_df["direction"] == direction]
        if dir_df.empty:
            continue
        print(f"\n--- {direction.upper()} ---")
        # The passive buy-and-hold benchmark is direction-independent, so it is
        # safe to print it in every direction block. Older/error-only DataFrames
        # may not carry the column (or carry only NaNs): in that case stay
        # silent rather than crashing or printing "nan".
        if "benchmark_return_pct" in dir_df.columns and dir_df["benchmark_return_pct"].notna().any():
            pivot_bench = dir_df.pivot(index="threshold", columns="fee_pct", values="benchmark_return_pct")
            bench_parts = []
            for fee in pivot_bench.columns:
                values = pivot_bench[fee].dropna()
                if values.empty:
                    continue
                bench_parts.append(f"Fee {fee:g}% = {values.iloc[0]:+.2f}%")
            if bench_parts:
                print("buy & hold: " + " | ".join(bench_parts))
        pivot_return = dir_df.pivot(index="threshold", columns="fee_pct", values="return_pct")
        pivot_trades = dir_df.pivot(index="threshold", columns="fee_pct", values="num_trades")
        pivot_win = dir_df.pivot(index="threshold", columns="fee_pct", values="win")

        # Only widen the cells when the risk metrics actually carry data;
        # older/partial frames keep the exact original 18-char layout.
        has_metrics = (
            "max_drawdown_pct" in dir_df.columns
            and "sharpe" in dir_df.columns
            and dir_df["max_drawdown_pct"].notna().any()
            and dir_df["sharpe"].notna().any()
        )
        if has_metrics:
            pivot_mdd = dir_df.pivot(index="threshold", columns="fee_pct", values="max_drawdown_pct")
            pivot_sharpe = dir_df.pivot(index="threshold", columns="fee_pct", values="sharpe")

        col_width = 30 if has_metrics else 18
        # Cast to int explicitly for display: when other symbols in the same run
        # produced an error row (threshold=None), pandas upcasts the WHOLE
        # threshold/fee_pct columns to float64 across the entire combined
        # DataFrame - not just that symbol's rows - so a perfectly good integer
        # threshold like 1 can otherwise print as "1.0" here.
        header = f"{'Threshold':>9} | " + " | ".join(f"Fee {fee:g}%".rjust(col_width) for fee in pivot_return.columns)
        print(header)
        print("-" * len(header))
        for threshold in pivot_return.index:
            cells = []
            for fee in pivot_return.columns:
                ret = pivot_return.loc[threshold, fee]
                trades = int(pivot_trades.loc[threshold, fee])
                tag = "WIN " if pivot_win.loc[threshold, fee] else "LOSS"
                if has_metrics:
                    mdd = pivot_mdd.loc[threshold, fee]
                    sharpe = pivot_sharpe.loc[threshold, fee]
                    mdd_text = "n/a" if pd.isna(mdd) else f"{mdd:.1f}%"
                    sharpe_text = "n/a" if pd.isna(sharpe) else f"{sharpe:.2f}"
                    cell = f"{tag} {ret:+.2f}% ({trades}t) DD {mdd_text} SR {sharpe_text}"
                else:
                    cell = f"{tag} {ret:+.2f}% ({trades}t)"
                cells.append(cell.rjust(col_width))
            print(f"{int(threshold):>9} | " + " | ".join(cells))


def _print_aggregate_summary(results: pd.DataFrame) -> None:
    valid = results[results["error"].isna()]
    if valid.empty:
        print("\nNo symbols produced usable data - nothing to summarize.")
        return
    print("\n=== Aggregate across all symbols (win rate = share of symbols profitable net of fees) ===")
    group_cols = ["direction", "threshold", "fee_pct"] if "direction" in valid.columns and valid["direction"].notna().any() else ["threshold", "fee_pct"]
    aggregations = {
        "win_rate_pct": ("win", lambda s: round(s.mean() * 100, 1)),
        "avg_return_pct": ("return_pct", lambda s: round(s.mean(), 2)),
    }
    if "beats_benchmark" in valid.columns and valid["beats_benchmark"].notna().any():
        aggregations["beat_rate_pct"] = (
            "beats_benchmark",
            lambda s: round(s.dropna().astype(bool).mean() * 100, 1),
        )
    aggregations["symbols_tested"] = ("win", "size")
    # Risk metrics are optional too: only surface the averages when at least
    # one numeric value exists, leaving frames without the columns untouched.
    for column, name in (
        ("sharpe", "avg_sharpe"),
        ("sortino", "avg_sortino"),
        ("max_drawdown_pct", "avg_maxdd_pct"),
        ("calmar", "avg_calmar"),
    ):
        if column in valid.columns and valid[column].notna().any():
            aggregations[name] = (
                column,
                lambda s: round(pd.to_numeric(s, errors="coerce").dropna().mean(), 2),
            )
    summary = valid.groupby(group_cols).agg(**aggregations).reset_index()
    if "threshold" in summary.columns:
        summary["threshold"] = summary["threshold"].astype(int)  # same float-upcast reason as _print_symbol_table
    print(summary.to_string(index=False))


def _print_benchmark_summary(results: pd.DataFrame) -> None:
    """
    Print the passive buy-and-hold benchmark aggregated BY FEE RATE only, with
    both its return and its risk metrics.

    The benchmark is independent of direction and threshold, so duplicated
    rows for the same (symbol, fee_pct) are collapsed before aggregating -
    otherwise a run with several directions/thresholds would weight some
    symbols more heavily than others. Older/partial result frames (missing the
    benchmark columns, or carrying only benchmark_return_pct) print nothing or
    a reduced table rather than crashing.
    """
    if "benchmark_return_pct" not in results.columns:
        return
    valid = results[results["error"].isna()]
    if valid.empty:
        return
    if valid["benchmark_return_pct"].notna().sum() == 0:
        return

    # One observation per symbol and fee: the benchmark does not depend on
    # direction or threshold, so those duplicates must not skew the averages.
    bench = valid.drop_duplicates(subset=["symbol", "fee_pct"])

    aggregations = {}
    for column, name in (
        ("benchmark_return_pct", "avg_benchmark_pct"),
        ("benchmark_sharpe", "avg_benchmark_sharpe"),
        ("benchmark_sortino", "avg_benchmark_sortino"),
        ("benchmark_max_drawdown_pct", "avg_benchmark_maxdd_pct"),
        ("benchmark_calmar", "avg_benchmark_calmar"),
    ):
        if column in bench.columns and bench[column].notna().any():
            aggregations[name] = (
                column,
                lambda s: round(pd.to_numeric(s, errors="coerce").dropna().mean(), 2),
            )
    aggregations["symbols_tested"] = ("symbol", "nunique")

    summary = bench.groupby("fee_pct").agg(**aggregations).reset_index()
    print("\n=== Benchmark (buy & hold) across all symbols ===")
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Backtest a confluence-signal buy/sell strategy: buy on bullish-flagged candles, "
                    "sell on bearish-flagged ones (long), sell short on bearish-flagged candles "
                    "and cover on bullish-flagged ones (short), or flip directly between the two "
                    "on opposite signals (reverse), starting from a $1000 wallet."
    )
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Symbols to test (default: the curated quick-list, US + EU megacaps)")
    parser.add_argument("--preset", type=str, default=None, choices=list(RANGE_PRESETS.keys()),
                        help="Named date-range + interval shortcut (e.g. YTD = start of this year at 1d candles, "
                             "same definition the app's preset buttons use). Overrides --start/--end/--interval.")
    parser.add_argument("--start", type=str, default=None, help="YYYY-MM-DD (default: 60 days before --end)")
    parser.add_argument("--end", type=str, default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--interval", type=str, default="1h", help="Candle interval, e.g. 5m, 1h, 1d (default: 1h)")
    parser.add_argument("--thresholds", nargs="+", type=int, default=None,
                        help=f"Min. agreeing indicators/patterns to trigger a trade (default: {DEFAULT_THRESHOLDS})")
    parser.add_argument("--fees", nargs="+", type=float, default=None,
                        help="Fee percentages per operation, e.g. 0 0.5 1 1.5 (default: 0, 0.5, 1, 1.5)")
    parser.add_argument("--starting-cash", type=float, default=1000.0)
    parser.add_argument("--direction", type=str, default="long", choices=["long", "short", "both", "reverse"],
                        help="Trade direction: long (buy on bullish, sell on bearish), short (sell short on bearish, "
                             "cover on bullish), both (run long and short simulations), or reverse (flip directly "
                             "between long and short on opposite signals) (default: long)")
    parser.add_argument("--signal-provider", type=str, default="confluence", choices=available_providers(),
                        help="Signal source to backtest (default: confluence)")
    parser.add_argument("--lookbacks", nargs="+", type=int, default=[21, 63, 252],
                        help="TSMOM lookback windows in bars, e.g. 21 63 252 "
                             "(default: 21 63 252 = 1/3/12 months at 1d); only used with "
                             "--signal-provider tsmom")
    parser.add_argument("--target-vol", type=float, default=None,
                        help="Annualized volatility target in percent, e.g. 10 (disabled by default)")
    parser.add_argument("--vol-lookback", type=int, default=20,
                        help="Rolling window in bars for realized volatility (default: 20)")
    parser.add_argument("--entry-threshold", type=int, default=None,
                        help="Entry threshold for signal count (hysteresis). Defaults to --threshold if not set.")
    parser.add_argument("--exit-threshold", type=int, default=None,
                        help="Exit/flip threshold for opposite signal count (hysteresis). Defaults to --entry-threshold/--threshold if not set.")
    parser.add_argument("--fee-fixed", type=float, default=0.0,
                        help="Fixed fee per fill in same currency as starting cash (default: 0.0)")
    parser.add_argument("--slippage-bps", type=float, default=0.0,
                        help="Slippage in basis points per fill (default: 0.0)")
    parser.add_argument("--broker-preset", type=str, default=None,
                        choices=["alpaca", "xtb", "ibkr", "ibkr-us-pro", "ibkr-eu-fixed", "degiro"],
                        help="Apply indicative broker fee/slippage preset. Explicit flags override preset values.")
    parser.add_argument("--universe", type=str, default="all", choices=["all", "us", "eu"],
                        help="Default symbol universe when --symbols is not given (default: all).")
    args = parser.parse_args()

    if args.preset:
        start, end, interval = resolve_range_preset(args.preset)
    else:
        end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now()
        start = datetime.strptime(args.start, "%Y-%m-%d") if args.start else end - timedelta(days=60)
        interval = args.interval

    # Resolve broker preset first to determine effective costs
    from AlphaSense.costs.brokers import resolve_broker_preset
    broker_label = None
    if args.broker_preset is not None:
        resolved_fee_pct_single, resolved_fee_fixed, resolved_slippage_bps, broker_label = resolve_broker_preset(
            args.broker_preset,
            args.fees[0] / 100 if args.fees and len(args.fees) > 0 else None,
            args.fee_fixed if args.fee_fixed != 0.0 else None,
            args.slippage_bps if args.slippage_bps != 0.0 else None,
        )
        fee_rates = [resolved_fee_pct_single]
        fee_fixed_eff = resolved_fee_fixed
        slippage_bps_eff = resolved_slippage_bps
    else:
        fee_rates = [f / 100 for f in args.fees] if args.fees else DEFAULT_FEE_RATES
        fee_fixed_eff = args.fee_fixed
        slippage_bps_eff = args.slippage_bps

    # An explicit --symbols list always wins over the universe selection.
    symbols = args.symbols or get_universe_symbols(args.universe)
    target_vol = None if args.target_vol is None else args.target_vol / 100.0

    header = (
        f"Backtesting {len(symbols)} symbol(s), {start.date()} to {end.date()}, interval={interval}, "
        f"starting cash=${args.starting_cash:.2f}, direction={args.direction}"
    )
    if not args.symbols and args.universe != "all":
        header += f", universe={args.universe}"
    if target_vol is not None:
        header += f", target_vol={args.target_vol:.1f}% vol_lookback={args.vol_lookback}"
    if args.entry_threshold is not None or args.exit_threshold is not None:
        header += f", entry_threshold={args.entry_threshold}, exit_threshold={args.exit_threshold}"
    if args.broker_preset is not None:
        header += f", broker={args.broker_preset}"
        print(header)
        print(f"Costs: fee_pct={fee_rates[0]*100:.3f}%, fee_fixed=${fee_fixed_eff:.2f}, slippage_bps={slippage_bps_eff:.1f}")
    else:
        if fee_fixed_eff != 0.0:
            header += f", fee_fixed=${fee_fixed_eff:.2f}"
        if slippage_bps_eff != 0.0:
            header += f", slippage_bps={slippage_bps_eff:.1f}"
        print(header)

    # Only tsmom understands --lookbacks. Gating here (rather than making every
    # provider factory swallow unknown kwargs) keeps each factory's signature
    # honest while still passing the value through the generic
    # provider_kwargs channel in run_backtest/backtest_symbol.
    provider_kwargs = {"lookbacks": args.lookbacks} if args.signal_provider == "tsmom" else None

    results = run_backtest(
        symbols=symbols, start=start, end=end, interval=interval,
        thresholds=args.thresholds, fee_rates=fee_rates, starting_cash=args.starting_cash,
        direction=args.direction, signal_provider=args.signal_provider,
        provider_kwargs=provider_kwargs,
        target_vol=target_vol, vol_lookback=args.vol_lookback,
        entry_threshold=args.entry_threshold, exit_threshold=args.exit_threshold,
        fee_fixed=fee_fixed_eff,
        slippage_bps=slippage_bps_eff,
        broker_preset=None,
    )

    for symbol in results["symbol"].unique():
        _print_symbol_table(symbol, results[results["symbol"] == symbol])

    _print_aggregate_summary(results)
    _print_benchmark_summary(results)


if __name__ == "__main__":
    main()

