import pandas as pd
from dash import callback, Input, Output, State, ALL, ctx, no_update

from AlphaSense.requests.candle_patterns import PATTERN_REGISTRY
from AlphaSense.requests.quotes import get_single_quote
from AlphaSense.frontend.data_access import (
    get_price_candles_dataframe, to_minutes, dataframe_to_records, dataframe_from_records,
    resolve_range_preset,
)
from AlphaSense.frontend.indicator_resolution import resolve_indicator
from AlphaSense.analysis.confluence import compute_confluence
from AlphaSense.frontend.lightweight_charts.serializers import (
    candles_to_lwc, overlays_to_lwc, subplots_to_lwc, markers_to_lwc, confluence_markers_to_lwc,
)

# update_price_data / update_confluence / update_pattern_table below are the
# glue over the shared data layer (get_price_candles_dataframe,
# compute_confluence, PATTERN_REGISTRY) that this is the only remaining
# frontend for - the Plotly version was retired once this one covered
# everything it did and more.


@callback(
    Output("date-range", "start_date"),
    Output("date-range", "end_date"),
    Output("interval-select", "value"),
    Input({"type": "range-preset", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def apply_range_preset(_):
    """
    One callback for every preset button (1D/1W/.../MAX), identified via
    Dash's pattern-matching IDs. Date math itself lives in
    AlphaSense.frontend.data_access.resolve_range_preset(), shared with the
    backtest CLI's --preset flag so "YTD" means the same thing in both places.
    """
    triggered = ctx.triggered_id
    resolved = resolve_range_preset(triggered["index"]) if triggered else None
    if resolved is None:
        return no_update, no_update, no_update
    return resolved


@callback(
    Output("symbol-input", "value"),
    Input("quick-pick-dropdown", "value"),
    prevent_initial_call=True,
)
def apply_quick_pick(selected_symbol):
    """ Copies a curated quick-pick selection into the free-text symbol field, which already drives everything else. """
    if not selected_symbol:
        return no_update
    return selected_symbol


@callback(
    Output("quick-pick-dropdown", "options"),
    Input("quick-pick-store", "data"),
)
def sync_quick_pick_options(quotes):
    """ Keeps the dropdown's options in sync with quick-pick-store, so add_to_quick_list() below can grow the list live. """
    quotes = quotes or []
    return [
        {"label": f"{q['name']} ({q['symbol']}) \u2014 {q['price']} {q['currency']}".strip(), "value": q["symbol"]}
        for q in quotes
    ]


@callback(
    Output("quick-pick-store", "data"),
    Output("quicklist-feedback", "children"),
    Input("add-to-quicklist-btn", "n_clicks"),
    State("symbol-input", "value"),
    State("quick-pick-store", "data"),
    prevent_initial_call=True,
)
def add_to_quick_list(n_clicks, symbol, current_list):
    """
    Adds whatever's currently in the symbol field to the quick-pick list,
    fetching its live price the same way the curated starting list does.
    Session-only, same caveat as quick-pick-store itself.
    """
    if not symbol or not symbol.strip():
        return no_update, "Enter a symbol first."
    symbol = symbol.strip().upper()
    current_list = current_list or []
    if any(q["symbol"] == symbol for q in current_list):
        return no_update, f"{symbol} is already in the quick list."
    quote = get_single_quote(symbol)
    if quote is None:
        return no_update, f"Couldn't fetch a quote for '{symbol}' - check the symbol and try again."
    return current_list + [quote], f"Added {quote['name']} ({symbol}) to the quick list."



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
                "pattern_key": key,
                "direction": entry["direction"],
                "interval_minutes": interval_minutes,
            })
    rows.sort(key=lambda r: r["time"])
    return rows


@callback(
    Output("candle-chart", "candles"),
    Output("candle-chart", "overlays"),
    Output("candle-chart", "subplots"),
    Output("candle-chart", "markers"),
    Output("symbol-error", "children"),
    Input("price-data-store", "data"),
    Input("indicator-checklist", "value"),
    Input("confluence-threshold", "value"),
    Input("confluence-store", "data"),
    Input("pattern-table", "data"),
    State("symbol-input", "value"),
    State("interval-select", "value"),
)
def update_chart(store_data, selected_indicators, threshold, confluence_data, pattern_rows, symbol, interval):
    """
    Fills in the LightweightChart component's candles/overlays/subplots/
    markers props. Reuses confluence-store's already-computed indicator
    values via resolve_indicator()'s reuse-or-recompute rule, so toggling the
    indicator checklist doesn't trigger a fresh _compute() call for anything
    confluence already computed. Also folds in pattern-table's rows so each
    candle's hover tooltip can show which pattern(s), not just which
    indicators, fired there, and so pattern/confluence markers don't collide
    (see confluence_markers_to_lwc).
    """
    if not symbol:
        return [], [], [], [], ""

    store_data = store_data or {}
    if store_data.get("error"):
        return [], [], [], [], f"Error fetching data for '{symbol.upper()}': {store_data['error']}"

    df = dataframe_from_records(store_data.get("records"))
    if df.empty:
        return [], [], [], [], f"No data found for symbol '{symbol.upper()}'"

    confluence_data = confluence_data or {}
    confluence = confluence_data.get("confluence", {})
    precomputed = {
        key: dataframe_from_records(records)
        for key, records in confluence_data.get("indicator_values", {}).items()
    }

    def resolve(key, entry):
        return resolve_indicator(key, entry, df, symbol.upper(), interval, precomputed)

    pattern_rows = pattern_rows or []
    selected_indicators = selected_indicators or []
    candles = candles_to_lwc(df, confluence, threshold, pattern_rows)
    overlays = overlays_to_lwc(selected_indicators, resolve)
    subplots = subplots_to_lwc(selected_indicators, resolve)
    markers = markers_to_lwc(pattern_rows) + confluence_markers_to_lwc(confluence, threshold, pattern_rows)
    return candles, overlays, subplots, markers, ""


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
