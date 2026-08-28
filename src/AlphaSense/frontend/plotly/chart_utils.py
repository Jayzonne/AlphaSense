import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Patch

from AlphaSense.frontend.plotly.data_access import to_minutes
from AlphaSense.requests.indicators import INDICATOR_REGISTRY


def get_range_breaks(df: pd.DataFrame, interval: str) -> list[dict]:
    """
    Detect big gaps in the data to build rangebreaks for Plotly.
    ...
    """
    normal_interval = pd.Timedelta(minutes=to_minutes(interval))
    if not isinstance(normal_interval, pd.Timedelta):
        raise ValueError(f"Invalid interval: {interval}")

    timestamps = df.index.sort_values()
    diffs = timestamps.to_series().diff().dropna()
    gaps = diffs.loc[diffs > normal_interval * 2]

    # Plotly's rangebreak bounds are left-inclusive, right-exclusive: bounds=[x0, x1]
    # hides [x0, x1). Without this nudge, gap_start = timestamp - gap works out to be
    # EXACTLY the previous real candle's own timestamp (simple algebra: timestamp -
    # (timestamp - previous_timestamp) = previous_timestamp) - so that candle sits
    # precisely on the inclusive edge and gets hidden along with the actual gap.
    range_breaks = []
    for timestamp, gap in gaps.items():
        gap_start = timestamp - gap + normal_interval
        range_breaks.append(dict(bounds=[gap_start.isoformat(), timestamp.isoformat()]))
    return range_breaks

def build_candlestick_figure(df: pd.DataFrame, symbol: str, interval: str, y_min: float, y_max: float) -> go.Figure:
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"]
    )])
    fig.update_layout(
        title=f"{symbol} - {interval}",
        yaxis=dict(range=[y_min, y_max]),
        xaxis_rangeslider_visible=True,
        xaxis=dict(rangebreaks=get_range_breaks(df, interval)),
    )
    return fig


def build_highlight_patch(row: dict) -> Patch:
    center = pd.to_datetime(row["time"])
    if not isinstance(center, pd.Timestamp):
        raise ValueError(f"Invalid time value: {row['time']}")

    half_width = pd.Timedelta(minutes=row["interval_minutes"])
    if not isinstance(half_width, pd.Timedelta):
        raise ValueError(f"Invalid interval_minutes value: {row['interval_minutes']}")
    half_width /= 2

    patched_fig = Patch()
    patched_fig["layout"]["shapes"] = [dict(
        type="rect", xref="x", yref="paper",
        x0=(center - half_width).isoformat(), x1=(center + half_width).isoformat(),
        y0=0, y1=1, fillcolor="rgba(255, 125, 0, 0.35)", line_width=0
    )]
    return patched_fig

def build_figure(df: pd.DataFrame, symbol: str, interval: str, y_min: float, y_max: float,
                  selected_indicators: list[str], confluence: dict, threshold: int) -> go.Figure:
    interval_minutes = to_minutes(interval)
    overlay_keys = [k for k in selected_indicators if INDICATOR_REGISTRY[k]["display"] == "overlay"]
    subplot_keys = [k for k in selected_indicators if INDICATOR_REGISTRY[k]["display"] == "subplot"]
    n_rows = 1 + len(subplot_keys)
    row_heights = [0.6] + [0.4 / len(subplot_keys)] * len(subplot_keys) if subplot_keys else [1.0]

    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, row_heights=row_heights, vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"],
                                  close=df["close"], name=symbol), row=1, col=1)

    for key in overlay_keys:
        entry = INDICATOR_REGISTRY[key]
        computed = entry["class"](symbol, df.index.min(), df.index.max(), interval, price_data=df)._compute(df)
        for col in entry["columns"]:
            fig.add_trace(go.Scatter(x=df.index, y=computed[col], name=f"{entry['label']} ({col})",
                                      line=dict(width=1)), row=1, col=1)

    for i, key in enumerate(subplot_keys, start=2):
        entry = INDICATOR_REGISTRY[key]
        computed = entry["class"](symbol, df.index.min(), df.index.max(), interval, price_data=df)._compute(df)
        for col in entry["columns"]:
            fig.add_trace(go.Scatter(x=df.index, y=computed[col], name=f"{entry['label']} ({col})"), row=i, col=1)
        if "y_range" in entry:
            fig.update_yaxes(range=entry["y_range"], row=i, col=1)

    threshold = threshold or 1
    half_width = pd.Timedelta(minutes=interval_minutes) / 2
    shapes, hover_x, hover_y, hover_text = [], [], [], []
    for ts_str, conf in (confluence or {}).items():
        ts = pd.to_datetime(ts_str)
        if ts not in df.index:
            continue
        if conf["bullish_count"] >= threshold:
            color, sources = "rgba(0, 200, 0, 0.25)", conf["bullish_sources"]
        elif conf["bearish_count"] >= threshold:
            color, sources = "rgba(200, 0, 0, 0.25)", conf["bearish_sources"]
        else:
            continue
        shapes.append(dict(type="rect", xref="x", yref="y domain",
                            x0=(ts - half_width).isoformat(), x1=(ts + half_width).isoformat(),
                            y0=0, y1=1, fillcolor=color, line_width=0))
        hover_x.append(ts)
        hover_y.append(df.loc[ts, "high"] * 1.002)
        hover_text.append(f"{conf['bullish_count']} bullish / {conf['bearish_count']} bearish<br>" + ", ".join(sources))

    fig.add_trace(go.Scatter(x=hover_x, y=hover_y, mode="markers",
                              marker=dict(size=6, color="rgba(0,0,0,0)"),
                              hovertext=hover_text, hoverinfo="text", showlegend=False), row=1, col=1)

    fig.update_yaxes(range=[y_min, y_max], row=1, col=1)
    fig.update_xaxes(rangebreaks=get_range_breaks(df, interval))
    fig.update_xaxes(rangeslider_visible=True, row=n_rows, col=1)
    fig.update_layout(title=f"{symbol} - {interval}", shapes=shapes)
    return fig
