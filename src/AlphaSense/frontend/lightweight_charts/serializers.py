import pandas as pd

from AlphaSense.requests.indicators import INDICATOR_REGISTRY

UP_COLOR = "#26a69a"
DOWN_COLOR = "#ef5350"
BULLISH_CONFLUENCE_BORDER = "#fbbf24"  # amber - distinct from the up/down body colors
BEARISH_CONFLUENCE_BORDER = "#a78bfa"  # violet
_LINE_PALETTE = ["#2962FF", "#FF6D00", "#AA00FF", "#FFD600", "#00C853"]


def _to_unix_seconds(ts) -> int:
    """ Lightweight Charts' UTCTimestamp is seconds, not milliseconds. """
    return int(pd.Timestamp(ts).timestamp())


def candles_to_lwc(df: pd.DataFrame, confluence: dict, threshold: int) -> list[dict]:
    """
    Converts the shared OHLCV dataframe into Lightweight Charts candle points.

    Confluence-threshold highlighting is baked directly into each qualifying
    point's borderColor - the library's documented way to recolor individual
    bars - rather than the separate background-rectangle shapes the Plotly
    version drew. A qualifying candle also gets a confluenceText field, which
    the component surfaces as a hover tooltip (the equivalent of the Plotly
    version's invisible-hover-marker trick).
    """
    threshold = threshold or 1
    confluence = confluence or {}
    points = []
    for ts, row in df.iterrows():
        point = {
            "time": _to_unix_seconds(ts),
            "open": float(row["open"]), "high": float(row["high"]),
            "low": float(row["low"]), "close": float(row["close"]),
        }
        conf = confluence.get(pd.Timestamp(ts).isoformat())
        if conf:
            if conf["bullish_count"] >= threshold:
                point["borderColor"] = BULLISH_CONFLUENCE_BORDER
                point["confluenceText"] = f"{conf['bullish_count']} bullish: " + ", ".join(conf["bullish_sources"])
            elif conf["bearish_count"] >= threshold:
                point["borderColor"] = BEARISH_CONFLUENCE_BORDER
                point["confluenceText"] = f"{conf['bearish_count']} bearish: " + ", ".join(conf["bearish_sources"])
        points.append(point)
    return points


def _line_points(values: pd.Series) -> list[dict]:
    return [{"time": _to_unix_seconds(ts), "value": float(v)} for ts, v in values.items() if pd.notna(v)]


def _histogram_points(values: pd.Series) -> list[dict]:
    # Per-point color (standard MACD-histogram styling: green above zero, red below),
    # set directly on each point rather than via a separate recoloring API.
    return [
        {"time": _to_unix_seconds(ts), "value": float(v), "color": UP_COLOR if v >= 0 else DOWN_COLOR}
        for ts, v in values.items() if pd.notna(v)
    ]


def _series_for_column(col: str, index: int, computed: pd.DataFrame) -> dict:
    is_histogram = col == "histogram"
    return {
        "id": col,
        "name": col.replace("_", " ").title(),
        "color": UP_COLOR if is_histogram else _LINE_PALETTE[index % len(_LINE_PALETTE)],
        "type": "histogram" if is_histogram else "line",
        "data": _histogram_points(computed[col]) if is_histogram else _line_points(computed[col]),
    }


def overlays_to_lwc(selected_indicators: list[str], resolve_indicator) -> list[dict]:
    """
    resolve_indicator(key, entry) -> the pd.DataFrame already _compute()'d for
    that indicator (reused from confluence-store when possible - see
    AlphaSense.frontend.plotly.chart_utils._resolve_indicator, shared as-is
    so both frontends apply the exact same reuse-vs-recompute rule).
    """
    overlays = []
    for key in selected_indicators:
        entry = INDICATOR_REGISTRY[key]
        if entry["display"] != "overlay":
            continue
        computed = resolve_indicator(key, entry)
        for i, col in enumerate(entry["columns"]):
            series = _series_for_column(col, i, computed)
            overlays.append({"id": f"{key}-{col}", "name": f"{entry['label']} ({col})",
                              "color": series["color"], "data": series["data"]})
    return overlays


def subplots_to_lwc(selected_indicators: list[str], resolve_indicator) -> list[dict]:
    subplots = []
    for key in selected_indicators:
        entry = INDICATOR_REGISTRY[key]
        if entry["display"] != "subplot":
            continue
        computed = resolve_indicator(key, entry)
        series = [_series_for_column(col, i, computed) for i, col in enumerate(entry["columns"])]
        subplots.append({"id": key, "title": entry["label"], "height": 130, "priceFormat": "price", "series": series})
    return subplots


def markers_to_lwc(pattern_rows: list[dict]) -> list[dict]:
    """ pattern_rows: the same {time, pattern, direction, interval_minutes} rows the pattern table renders. """
    markers = []
    for row in pattern_rows:
        bullish = row["direction"] == "Bullish"
        markers.append({
            "time": _to_unix_seconds(row["time"]),
            "position": "belowBar" if bullish else "aboveBar",
            "shape": "arrowUp" if bullish else "arrowDown",
            "color": UP_COLOR if bullish else DOWN_COLOR,
            "text": row["pattern"],
        })
    return markers
