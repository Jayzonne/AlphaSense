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

from AlphaSense.analysis.backtest import run_backtest, DEFAULT_SYMBOLS, DEFAULT_THRESHOLDS, DEFAULT_FEE_RATES
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
        pivot_return = dir_df.pivot(index="threshold", columns="fee_pct", values="return_pct")
        pivot_trades = dir_df.pivot(index="threshold", columns="fee_pct", values="num_trades")
        pivot_win = dir_df.pivot(index="threshold", columns="fee_pct", values="win")

        col_width = 18
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
                cells.append(f"{tag} {ret:+.2f}% ({trades}t)".rjust(col_width))
            print(f"{int(threshold):>9} | " + " | ".join(cells))


def _print_aggregate_summary(results: pd.DataFrame) -> None:
    valid = results[results["error"].isna()]
    if valid.empty:
        print("\nNo symbols produced usable data - nothing to summarize.")
        return
    print("\n=== Aggregate across all symbols (win rate = share of symbols profitable net of fees) ===")
    group_cols = ["direction", "threshold", "fee_pct"] if "direction" in valid.columns and valid["direction"].notna().any() else ["threshold", "fee_pct"]
    summary = valid.groupby(group_cols).agg(
        win_rate_pct=("win", lambda s: round(s.mean() * 100, 1)),
        avg_return_pct=("return_pct", lambda s: round(s.mean(), 2)),
        symbols_tested=("win", "size"),
    ).reset_index()
    if "threshold" in summary.columns:
        summary["threshold"] = summary["threshold"].astype(int)  # same float-upcast reason as _print_symbol_table
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
    args = parser.parse_args()

    if args.preset:
        start, end, interval = resolve_range_preset(args.preset)
    else:
        end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now()
        start = datetime.strptime(args.start, "%Y-%m-%d") if args.start else end - timedelta(days=60)
        interval = args.interval

    fee_rates = [f / 100 for f in args.fees] if args.fees else DEFAULT_FEE_RATES
    symbols = args.symbols or DEFAULT_SYMBOLS

    print(f"Backtesting {len(symbols)} symbol(s), {start.date()} to {end.date()}, interval={interval}, "
          f"starting cash=${args.starting_cash:.2f}, direction={args.direction}")

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
    )

    for symbol in results["symbol"].unique():
        _print_symbol_table(symbol, results[results["symbol"] == symbol])

    _print_aggregate_summary(results)


if __name__ == "__main__":
    main()

