import pandas as pd
import plotly.graph_objects as go
from dash import callback, Input, Output, State, no_update

from AlphaSense.requests.candle_patterns import PATTERN_REGISTRY
from AlphaSense.frontend.plotly.data_access import get_price_candles_dataframe, to_minutes
from AlphaSense.frontend.plotly.chart_utils import build_candlestick_figure, build_highlight_patch, build_figure
from AlphaSense.analysis.confluence import compute_confluence


@callback(
    Output("candle-chart", "figure"),
    Output("symbol-error", "children"),
    Output("y-axis-slider", "min"),
    Output("y-axis-slider", "max"),
    Output("y-axis-slider", "value"),
    Input("symbol-input", "value"),
    Input("interval-select", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("y-axis-slider", "value"),
    Input("indicator-checklist", "value"),
    Input("confluence-threshold", "value"),
    Input("confluence-store", "data"),
)
def update_chart(symbol, interval, start_date, end_date, y_range,
                  selected_indicators, threshold, confluence):
    """
    Builds the candlestick figure AND the y-axis slider bounds together.
    These were two separate callbacks sharing the same trigger inputs at one
    point, which let them fire independently and race - the slider could
    reset to stale bounds a beat after the chart already redrew for a new
    date range. One callback guarantees the figure and slider always agree
    on the same data_min/data_max for a given render.
    """
    if not symbol or not end_date:
        return go.Figure(), "", 0, 100, [0, 100]

    try:
        df_candles = get_price_candles_dataframe(
            symbol.upper(), pd.to_datetime(start_date), pd.to_datetime(end_date), interval
        )
        if df_candles.empty:
            return go.Figure(), f"No data found for symbol '{symbol.upper()}'", 0, 100, [0, 100]

        data_min = df_candles["low"].min()
        data_max = df_candles["high"].max()
        y_min = y_range[0] if y_range and data_min <= y_range[0] <= data_max else data_min
        y_max = y_range[1] if y_range and data_min <= y_range[1] <= data_max else data_max

        fig = build_figure(df_candles, symbol.upper(), interval, y_min, y_max, selected_indicators or [], confluence or {}, threshold)
        return fig, "", data_min, data_max, [y_min, y_max]

    except Exception as e:
        return go.Figure(), f"Error fetching data for @{end_date}@ '{symbol.upper()}': {str(e)}", 0, 100, [0, 100]


@callback(
    Output("pattern-table", "data"),
    Input("symbol-input", "value"),
    Input("interval-select", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("pattern-checklist", "value")
)
def update_pattern_table(symbol, interval, start_date, end_date, selected_patterns):
    if not symbol or not selected_patterns:
        return []

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    interval_minutes = to_minutes(interval)

    rows = []
    for key in selected_patterns:
        entry = PATTERN_REGISTRY[key]
        try:
            instances = entry["class"](symbol.upper(), start, end, interval).get_pattern()
        except Exception:
            continue  # one bad pattern shouldn't take down the whole table

        for candle in instances:
            if "time" not in candle:
                continue  # defensive guard against the historical [{}] placeholder bug
            rows.append({
                "time": pd.Timestamp(candle["time"]).isoformat(),
                "pattern": entry["label"],
                "direction": entry["direction"],
                "interval_minutes": interval_minutes,
            })

    rows.sort(key=lambda r: r["time"])
    return rows


@callback(
    Output("candle-chart", "figure", allow_duplicate=True),
    Input("pattern-table", "active_cell"),
    State("pattern-table", "data"),
    prevent_initial_call=True
)
def highlight_on_click(active_cell, table_data):
    if not active_cell or not table_data:
        return no_update
    return build_highlight_patch(table_data[active_cell["row"]])

@callback(
    Output("confluence-store", "data"),
    Input("symbol-input", "value"),
    Input("interval-select", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
)
def update_confluence(symbol, interval, start_date, end_date):
    if not symbol:
        return {}
    start, end = pd.to_datetime(start_date), pd.to_datetime(end_date)
    df = get_price_candles_dataframe(symbol.upper(), start, end, interval)
    if df.empty:
        return {}
    return compute_confluence(symbol.upper(), start, end, interval, df)
