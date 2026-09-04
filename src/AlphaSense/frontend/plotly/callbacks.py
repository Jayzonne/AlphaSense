import pandas as pd
import plotly.graph_objects as go
from dash import callback, Input, Output, State, no_update

from AlphaSense.requests.candle_patterns import PATTERN_REGISTRY
from AlphaSense.frontend.plotly.data_access import (
    get_price_candles_dataframe, to_minutes, dataframe_to_records, dataframe_from_records,
)
from AlphaSense.frontend.plotly.chart_utils import build_candlestick_figure, build_highlight_patch, build_figure
from AlphaSense.analysis.confluence import compute_confluence


@callback(
    Output("price-data-store", "data"),
    Input("symbol-input", "value"),
    Input("interval-select", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
)
def update_price_data(symbol, interval, start_date, end_date):
    """
    Single source of truth for OHLCV data over the current symbol/interval/
    date range. The chart, confluence, and pattern-table callbacks below all
    read from this store instead of each independently instantiating a
    RequestData and hitting the DB for the exact same rows - that used to add
    up to as many as ~8 DB round trips (1 chart + 1 confluence + up to 6
    per-pattern fetches) for a single symbol change, now down to 1.
    """
    if not symbol or not end_date:
        return {"error": None, "records": []}
    try:
        df = get_price_candles_dataframe(
            symbol.upper(), pd.to_datetime(start_date), pd.to_datetime(end_date), interval
        )
    except Exception as e:
        return {"error": str(e), "records": []}
    return {"error": None, "records": dataframe_to_records(df)}


@callback(
    Output("candle-chart", "figure"),
    Output("symbol-error", "children"),
    Output("y-axis-slider", "min"),
    Output("y-axis-slider", "max"),
    Output("y-axis-slider", "value"),
    Input("price-data-store", "data"),
    Input("y-axis-slider", "value"),
    Input("indicator-checklist", "value"),
    Input("confluence-threshold", "value"),
    Input("confluence-store", "data"),
    State("symbol-input", "value"),
    State("interval-select", "value"),
)
def update_chart(store_data, y_range, selected_indicators, threshold, confluence_data,
                  symbol, interval):
    """
    Builds the candlestick figure AND the y-axis slider bounds together.
    These were two separate callbacks sharing the same trigger inputs at one
    point, which let them fire independently and race - the slider could
    reset to stale bounds a beat after the chart already redrew for a new
    date range. One callback guarantees the figure and slider always agree
    on the same data_min/data_max for a given render.

    Reads OHLCV data from price-data-store rather than fetching it directly,
    and passes confluence-store's precomputed indicator values through to
    build_figure so indicators toggled visible aren't computed a second time.
    """
    if not symbol:
        return go.Figure(), "", 0, 100, [0, 100]

    store_data = store_data or {}
    if store_data.get("error"):
        return go.Figure(), f"Error fetching data for '{symbol.upper()}': {store_data['error']}", 0, 100, [0, 100]

    df_candles = dataframe_from_records(store_data.get("records"))
    if df_candles.empty:
        return go.Figure(), f"No data found for symbol '{symbol.upper()}'", 0, 100, [0, 100]

    data_min = df_candles["low"].min()
    data_max = df_candles["high"].max()
    y_min = y_range[0] if y_range and data_min <= y_range[0] <= data_max else data_min
    y_max = y_range[1] if y_range and data_min <= y_range[1] <= data_max else data_max

    confluence_data = confluence_data or {}
    confluence = confluence_data.get("confluence", {})
    precomputed_indicators = {
        key: dataframe_from_records(records)
        for key, records in confluence_data.get("indicator_values", {}).items()
    }

    fig = build_figure(
        df_candles, symbol.upper(), interval, y_min, y_max,
        selected_indicators or [], confluence, threshold, precomputed_indicators,
    )
    return fig, "", data_min, data_max, [y_min, y_max]


@callback(
    Output("pattern-table", "data"),
    Input("price-data-store", "data"),
    Input("pattern-checklist", "value"),
    State("symbol-input", "value"),
    State("interval-select", "value"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
)
def update_pattern_table(store_data, selected_patterns, symbol, interval, start_date, end_date):
    """
    Reads OHLCV data from price-data-store and passes it through to each
    pattern class via price_data=. Previously each selected pattern built its
    own RequestData with no price_data, so ticking N patterns meant N
    redundant fetches of the identical candles - now it's zero, since the
    fetch already happened once in update_price_data.
    """
    if not symbol or not selected_patterns:
        return []

    df = dataframe_from_records((store_data or {}).get("records"))
    if df.empty:
        return []

    price_data = df.reset_index().to_dict("records")
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    interval_minutes = to_minutes(interval)

    rows = []
    for key in selected_patterns:
        entry = PATTERN_REGISTRY[key]
        try:
            instances = entry["class"](
                symbol.upper(), start, end, interval, price_data=price_data
            ).get_pattern()
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
    Input("price-data-store", "data"),
    State("symbol-input", "value"),
    State("interval-select", "value"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
)
def update_confluence(store_data, symbol, interval, start_date, end_date):
    """
    Reads OHLCV data from price-data-store instead of fetching it again -
    this was the second of up to 8 redundant DB round trips per symbol
    change. Also serializes the raw computed indicator values into the store
    alongside the bullish/bearish counts, so update_chart can reuse them
    instead of recomputing.
    """
    if not symbol:
        return {}
    df = dataframe_from_records((store_data or {}).get("records"))
    if df.empty:
        return {}
    start, end = pd.to_datetime(start_date), pd.to_datetime(end_date)
    result = compute_confluence(symbol.upper(), start, end, interval, df)
    return {
        "confluence": result["confluence"],
        "indicator_values": {
            key: dataframe_to_records(computed)
            for key, computed in result["indicator_values"].items()
        },
    }
