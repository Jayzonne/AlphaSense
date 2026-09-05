import pandas as pd
from dash import callback, Input, Output, State

from AlphaSense.requests.candle_patterns import PATTERN_REGISTRY
from AlphaSense.frontend.plotly.data_access import (
    get_price_candles_dataframe, to_minutes, dataframe_to_records, dataframe_from_records,
)
from AlphaSense.frontend.plotly.chart_utils import _resolve_indicator
from AlphaSense.analysis.confluence import compute_confluence
from AlphaSense.frontend.lightweight_charts.serializers import (
    candles_to_lwc, overlays_to_lwc, subplots_to_lwc, markers_to_lwc,
)

# update_price_data / update_confluence / update_pattern_table below are the
# same glue over the same shared data layer (get_price_candles_dataframe,
# compute_confluence, PATTERN_REGISTRY) as AlphaSense.frontend.plotly.callbacks
# - duplicated here rather than shared because they're wired to a separate
# Dash app instance. If the Plotly frontend is retired, these three plus
# _resolve_indicator are the natural candidates to hoist into one shared module.


@callback(
    Output("price-data-store", "data"),
    Input("symbol-input", "value"),
    Input("interval-select", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
)
def update_price_data(symbol, interval, start_date, end_date):
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
    Output("confluence-store", "data"),
    Input("price-data-store", "data"),
    State("symbol-input", "value"),
    State("interval-select", "value"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
)
def update_confluence(store_data, symbol, interval, start_date, end_date):
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
            key: dataframe_to_records(computed) for key, computed in result["indicator_values"].items()
        },
    }


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
    if not symbol or not selected_patterns:
        return []
    df = dataframe_from_records((store_data or {}).get("records"))
    if df.empty:
        return []
    price_data = df.reset_index().to_dict("records")
    start, end = pd.to_datetime(start_date), pd.to_datetime(end_date)
    interval_minutes = to_minutes(interval)

    rows = []
    for key in selected_patterns:
        entry = PATTERN_REGISTRY[key]
        try:
            instances = entry["class"](symbol.upper(), start, end, interval, price_data=price_data).get_pattern()
        except Exception:
            continue
        for candle in instances:
            if "time" not in candle:
                continue
            rows.append({
                "time": pd.Timestamp(candle["time"]).isoformat(),
                "pattern": entry["label"],
                "direction": entry["direction"],
                "interval_minutes": interval_minutes,
            })
    rows.sort(key=lambda r: r["time"])
    return rows


@callback(
    Output("candle-chart", "candles"),
    Output("candle-chart", "overlays"),
    Output("candle-chart", "subplots"),
    Output("symbol-error", "children"),
    Input("price-data-store", "data"),
    Input("indicator-checklist", "value"),
    Input("confluence-threshold", "value"),
    Input("confluence-store", "data"),
    State("symbol-input", "value"),
    State("interval-select", "value"),
)
def update_chart(store_data, selected_indicators, threshold, confluence_data, symbol, interval):
    """
    Fills in the LightweightChart component's candles/overlays/subplots
    props. Reuses confluence-store's already-computed indicator values via
    the same _resolve_indicator() reuse-or-recompute rule the Plotly frontend
    uses (see chart_utils.py), so toggling the indicator checklist doesn't
    trigger a fresh _compute() call for anything confluence already computed.
    """
    if not symbol:
        return [], [], [], ""

    store_data = store_data or {}
    if store_data.get("error"):
        return [], [], [], f"Error fetching data for '{symbol.upper()}': {store_data['error']}"

    df = dataframe_from_records(store_data.get("records"))
    if df.empty:
        return [], [], [], f"No data found for symbol '{symbol.upper()}'"

    confluence_data = confluence_data or {}
    confluence = confluence_data.get("confluence", {})
    precomputed = {
        key: dataframe_from_records(records)
        for key, records in confluence_data.get("indicator_values", {}).items()
    }

    def resolve(key, entry):
        return _resolve_indicator(key, entry, df, symbol.upper(), interval, precomputed)

    selected_indicators = selected_indicators or []
    candles = candles_to_lwc(df, confluence, threshold)
    overlays = overlays_to_lwc(selected_indicators, resolve)
    subplots = subplots_to_lwc(selected_indicators, resolve)
    return candles, overlays, subplots, ""


@callback(
    Output("candle-chart", "markers"),
    Input("pattern-table", "data"),
)
def update_markers(pattern_rows):
    return markers_to_lwc(pattern_rows or [])


@callback(
    Output("candle-chart", "highlightTime"),
    Input("pattern-table", "active_cell"),
    State("pattern-table", "data"),
    prevent_initial_call=True,
)
def highlight_on_click(active_cell, table_data):
    if not active_cell or not table_data:
        return None
    row = table_data[active_cell["row"]]
    return int(pd.Timestamp(row["time"]).timestamp())
