import pandas as pd

from AlphaSense.requests.indicators import INDICATOR_REGISTRY
from AlphaSense.requests.candle_patterns import PATTERN_REGISTRY

UP_COLOR = "#26a69a"
DOWN_COLOR = "#ef5350"
# Deliberately far more saturated than the muted body up/down colors above,
# so a confluence-flagged candle reads instantly as "something else is going
# on here" rather than blending into ordinary bullish/bearish candles - and
# green-for-bullish/red-for-bearish keeps the color itself intuitive, which
# the previous amber/violet scheme (distinct, but not directionally obvious)
# didn't give you.
BULLISH_CONFLUENCE_COLOR = "#00e676"
BEARISH_CONFLUENCE_COLOR = "#ff1744"
_LINE_PALETTE = ["#2962FF", "#FF6D00", "#AA00FF", "#FFD600", "#00C853"]
# One color per registered pattern (cycles if more patterns are added than
# colors listed) - keyed by iteration order over PATTERN_REGISTRY, so newly
# registered patterns automatically get a color with no mapping to maintain.
_PATTERN_COLOR_PALETTE = [
    "#42a5f5", "#ffa726", "#ab47bc", "#26c6da", "#d4e157", "#ec407a", "#8d6e63", "#78909c",
]


def _to_unix_seconds(ts) -> int:
    """ Lightweight Charts' UTCTimestamp is seconds, not milliseconds. """
    return int(pd.Timestamp(ts).timestamp())


def _pattern_color(pattern_key: str) -> str:
    keys = list(PATTERN_REGISTRY.keys())
    if pattern_key not in keys:
        return _PATTERN_COLOR_PALETTE[0]
    return _PATTERN_COLOR_PALETTE[keys.index(pattern_key) % len(_PATTERN_COLOR_PALETTE)]


def pattern_legend() -> list[dict]:
    """ For the UI legend: one {key, label, direction, color} entry per registered pattern. """
    return [
        {"key": key, "label": entry["label"], "direction": entry["direction"], "color": _pattern_color(key)}
        for key, entry in PATTERN_REGISTRY.items()
    ]


def _patterns_by_time(pattern_rows: list[dict]) -> dict[str, list[str]]:
    by_time: dict[str, list[str]] = {}
    for row in pattern_rows or []:
        by_time.setdefault(pd.Timestamp(row["time"]).isoformat(), []).append(row["pattern"])
    return by_time


def candles_to_lwc(df: pd.DataFrame, confluence: dict, threshold: int, pattern_rows: list[dict] = None) -> list[dict]:
    """
    Converts the shared OHLCV dataframe into Lightweight Charts candle points.

    Confluence-threshold highlighting recolors the ENTIRE candle (body, not
    just border/wick) in vivid green/red - a color-only border tweak was too
    easy to miss while scanning a full chart. The tradeoff is that a
    flagged candle's own up/down read is briefly overridden by the
    confluence color; confluence_markers_to_lwc()'s arrow (see below) and
    the hover tooltip both restore that context.

    Every candle with either a confluence flag or a detected pattern also
    gets an infoText field, which the component surfaces as a hover tooltip
    (the equivalent of the Plotly version's invisible-hover-marker trick) -
    this is what actually answers "which pattern(s)/indicator(s) fired here",
    rather than requiring a click over to the separate pattern table.
    """
    threshold = threshold or 1
    confluence = confluence or {}
    patterns_by_time = _patterns_by_time(pattern_rows or [])
    points = []
    for ts, row in df.iterrows():
        point = {
            "time": _to_unix_seconds(ts),
            "open": float(row["open"]), "high": float(row["high"]),
            "low": float(row["low"]), "close": float(row["close"]),
        }
        ts_iso = pd.Timestamp(ts).isoformat()
        info_lines = []

        conf = confluence.get(ts_iso)
        if conf:
            if conf["bullish_count"] >= threshold:
                point["color"] = point["borderColor"] = point["wickColor"] = BULLISH_CONFLUENCE_COLOR
                info_lines.append(f"Confluence: {conf['bullish_count']} bullish ({', '.join(conf['bullish_sources'])})")
            elif conf["bearish_count"] >= threshold:
                point["color"] = point["borderColor"] = point["wickColor"] = BEARISH_CONFLUENCE_COLOR
                info_lines.append(f"Confluence: {conf['bearish_count']} bearish ({', '.join(conf['bearish_sources'])})")

        pattern_names = patterns_by_time.get(ts_iso)
        if pattern_names:
            info_lines.append("Pattern: " + ", ".join(pattern_names))

        if info_lines:
            point["infoText"] = "\n".join(info_lines)
        points.append(point)
    return points


def confluence_markers_to_lwc(confluence: dict, threshold: int, pattern_rows: list[dict] = None) -> list[dict]:
    """
    A large (size 2, vs. the default 1 pattern markers use), text-badged
    arrow for every candle that clears the confluence threshold - on top of
    candles_to_lwc()'s full-body recolor, this is the "impossible to miss
    while scanning the chart" layer a color change alone didn't give you.
    The badge text is the signal count (e.g. "3"), so strength is visible
    without hovering.

    Skipped whenever a pattern marker already occupies the same
    time+position, so the two marker layers can never visually collide -
    the candle's full-body recolor already makes that candle the most
    visually distinct one on the chart either way, collision or not.
    """
    threshold = threshold or 1
    confluence = confluence or {}
    occupied = {
        (row["time"] if isinstance(row["time"], str) else pd.Timestamp(row["time"]).isoformat(),
         "belowBar" if row["direction"] == "Bullish" else "aboveBar")
        for row in (pattern_rows or [])
    }
    markers = []
    for ts_iso, conf in confluence.items():
        if conf["bullish_count"] >= threshold:
            position, shape, color, count = "belowBar", "arrowUp", BULLISH_CONFLUENCE_COLOR, conf["bullish_count"]
        elif conf["bearish_count"] >= threshold:
            position, shape, color, count = "aboveBar", "arrowDown", BEARISH_CONFLUENCE_COLOR, conf["bearish_count"]
        else:
            continue
        if (ts_iso, position) in occupied:
            continue
        markers.append({
            "time": _to_unix_seconds(pd.Timestamp(ts_iso)),
            "position": position, "shape": shape, "color": color,
            "text": str(count), "size": 2,
        })
    return markers


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
    AlphaSense.frontend.indicator_resolution.resolve_indicator).
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
    """
    pattern_rows: the same {time, pattern, pattern_key, direction,
    interval_minutes} rows the pattern table renders. Each pattern TYPE gets
    its own marker color (see _pattern_color / the UI legend) instead of a
    flat bullish/bearish color for every pattern - previously every bullish
    pattern looked identical on the chart, so telling a Hammer from a
    Bullish Engulfing meant leaving the chart to check the table. Marker
    text is intentionally left blank to avoid cluttering the chart; the
    hover tooltip (via candles_to_lwc's infoText) carries the full names.
    """
    markers = []
    for row in pattern_rows:
        bullish = row["direction"] == "Bullish"
        markers.append({
            "time": _to_unix_seconds(row["time"]),
            "position": "belowBar" if bullish else "aboveBar",
            "shape": "arrowUp" if bullish else "arrowDown",
            "color": _pattern_color(row.get("pattern_key", "")),
        })
    return markers
