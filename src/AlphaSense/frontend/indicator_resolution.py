import pandas as pd


def resolve_indicator(key: str, entry: dict, df: pd.DataFrame, symbol: str, interval: str,
                       precomputed: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Reuses the indicator values already computed by compute_confluence() for
    this exact df when available, instead of paying for _compute() a second
    time. compute_confluence() always evaluates every indicator regardless of
    what's toggled visible, so a precomputed frame is normally available for
    every key here.

    Falls back to computing directly only when there's no usable match - e.g.
    Dash re-firing this callback a beat before confluence-store has caught up
    to a just-changed symbol/date-range, where reusing the stale entry would
    silently show indicator values for the wrong data.
    """
    computed = precomputed.get(key)
    if computed is not None and not computed.empty and computed.index.equals(df.index):
        return computed
    return entry["class"](symbol, df.index.min(), df.index.max(), interval, price_data=df)._compute(df)
