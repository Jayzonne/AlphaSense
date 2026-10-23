"""
Pure, deterministic risk/return metrics for backtest equity curves.

Nothing in this module performs I/O or keeps state: every function is a plain
calculation over the inputs it receives, so it can be unit-tested directly and
reused by any caller holding a returns series or an equity curve.
"""
import numpy as np
import pandas as pd

# Bars per year used to annualize periodic returns. Intraday intervals assume
# the project's 6.5-hour trading session convention; see
# annualization_factor() for the exact derivation.
_ANNUALIZATION_FACTORS: dict[str, float] = {
    "5m": 19656,
    "15m": 6552,
    "30m": 3276,
    "1h": 1638,
    "4h": 409.5,
    "1d": 252,
    "1wk": 52,
    "1mo": 12,
    "3mo": 4,
}


def annualization_factor(interval: str) -> float:
    """
    Number of bars in a year for a given candle interval.

    The intraday figures follow the project's 6.5-hour trading session
    convention: a session holds 6.5 hours of bars, and there are 252 sessions
    per year. For example, 1h -> 6.5 * 252 = 1638 bars/year, and 5m ->
    (6.5 * 60 / 5) * 252 = 19656 bars/year. Daily/weekly/monthly intervals use
    252, 52, 12 and 4 bars per year respectively.

    Raises:
        ValueError: if the interval is not in the supported table.
    """
    try:
        return float(_ANNUALIZATION_FACTORS[interval])
    except KeyError:
        raise ValueError(
            f"Unknown interval {interval!r}; expected one of "
            f"{sorted(_ANNUALIZATION_FACTORS)}"
        ) from None


def _clean_returns(returns) -> np.ndarray:
    """Drop NaN/non-finite observations and return a float ndarray."""
    series = returns if isinstance(returns, pd.Series) else pd.Series(returns, dtype=float)
    clean = pd.to_numeric(series, errors="coerce").dropna()
    values = clean.to_numpy(dtype=float)
    return values[np.isfinite(values)]


def sharpe_ratio(
    returns,
    periods_per_year: float,
    risk_free: float = 0.0,
) -> float | None:
    """
    Annualized Sharpe ratio of periodic `returns`.

    `risk_free` is an ANNUALIZED rate; it is converted to a per-period rate
    before being subtracted from each return. Returns None when there are
    fewer than two usable observations or when their sample standard
    deviation (ddof=1) is non-finite or zero.
    """
    values = _clean_returns(returns)
    if values.size < 2:
        return None

    excess = values - risk_free / periods_per_year
    std = float(np.std(excess, ddof=1))
    if not np.isfinite(std) or std == 0.0:
        return None

    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns,
    periods_per_year: float,
    risk_free: float = 0.0,
) -> float | None:
    """
    Annualized Sortino ratio of periodic `returns`.

    Like sharpe_ratio, `risk_free` is annualized. The downside deviation is
    the root-mean-square of the negative excess returns, divided by the TOTAL
    number of observations (not only the negative ones). Returns None when
    that deviation is zero or non-finite.
    """
    values = _clean_returns(returns)
    if values.size == 0:
        return None

    excess = values - risk_free / periods_per_year
    downside = np.minimum(excess, 0.0)
    downside_dev = float(np.sqrt(np.mean(downside**2)))
    if not np.isfinite(downside_dev) or downside_dev == 0.0:
        return None

    return float(np.mean(excess) / downside_dev * np.sqrt(periods_per_year))


def max_drawdown(equity) -> float | None:
    """
    Maximum peak-to-trough drawdown as a POSITIVE magnitude (0.25 = 25%).

    NaN observations are dropped first. Returns None when the curve is empty
    or when any running peak is non-positive (which would make the drawdown
    ratio meaningless or divide by zero).
    """
    series = equity if isinstance(equity, pd.Series) else pd.Series(equity, dtype=float)
    eq = pd.to_numeric(series, errors="coerce").dropna()
    if eq.empty:
        return None

    running_max = eq.cummax()
    if (running_max <= 0).any():
        return None

    drawdown = eq / running_max - 1.0
    magnitude = float(-drawdown.min())
    # Canonicalize -0.0 to 0.0 so callers never render "-0.0%".
    return 0.0 if magnitude == 0.0 else magnitude


def calmar_ratio(equity, periods_per_year: float) -> float | None:
    """
    Calmar ratio: CAGR divided by maximum drawdown magnitude.

    Returns None when the curve has fewer than two observations, when there is
    no drawdown to divide by, or when the CAGR cannot be computed (non-positive
    endpoints or non-finite result).
    """
    series = equity if isinstance(equity, pd.Series) else pd.Series(equity, dtype=float)
    eq = pd.to_numeric(series, errors="coerce").dropna()
    if len(eq) < 2:
        return None

    mdd = max_drawdown(eq)
    if mdd is None or mdd <= 0.0:
        return None

    years = (len(eq) - 1) / periods_per_year
    if years <= 0:
        return None

    growth = float(eq.iloc[-1]) / float(eq.iloc[0])
    if growth <= 0.0:
        return None

    cagr = growth ** (1.0 / years) - 1.0
    if not np.isfinite(cagr):
        return None

    return float(cagr / mdd)
