from dash import Dash, html, dcc, callback, Input, Output, State, dash_table, no_update, Patch
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta

from AlphaSense.requests.RequestData import RequestData
from AlphaSense.requests.candle_patterns import *

DEFAULT_SYMBOL = "AAPL"
DEFAULT_INTERVAL = "5m"
DEFAULT_START_DATE = datetime.now()-timedelta(days=7)
DEFAULT_END_DATE = datetime.now()
# TODO: Implement data source selection
app = Dash()
request_data = RequestData(DEFAULT_SYMBOL,
                           DEFAULT_START_DATE,
                           DEFAULT_END_DATE,
                           DEFAULT_INTERVAL
                           )
app.layout = html.Div([
        html.H1("AlphaSense Dashboard"),

        html.Div([
            dcc.Input(
                id="symbol-input",
                type="text",
                placeholder="Enter symbol (e.g AAPL)",
                value=DEFAULT_SYMBOL,
                debounce=True
            ),
            html.Div(id="symbol-error", style={"color": "red"}),
            dcc.Dropdown(
                id="interval-select",
                options=[{"label": i, "value": i} for i in request_data.get_authorized_intervals()],
                value=DEFAULT_INTERVAL
            ),
            dcc.DatePickerRange(
                id="date-range",
                start_date=DEFAULT_START_DATE,
                end_date=DEFAULT_END_DATE
            ),
        ]),
        html.Div([
            html.Div([
                dcc.Graph(
                    id="candle-chart",
                    style={"width": "100%", 'height':"600px"},
                    config={"responsive": True}
                ),
                dcc.RangeSlider(
                        id="y-axis-slider",
                        vertical=True,
                        verticalHeight=600,
                        step=0.1,
                        min=0,
                        max=100,
                        value=[0, 100],
                        allowCross=False,
                        tooltip={"placement": "left", "always_visible": True}
                    )
                ], style={"display": "grid", "gridTemplateColumns": "1fr 60px", "alignItems": "center", "width": "100%", "gap":"10px"}),


            dash_table.DataTable(
                id="pattern-table",
                columns=[
                    {"name": "Time", "id": "time"},
                    {"name": "Pattern", "id": "pattern"},
                    {"name": "Direction", "id": "direction"},
                ],
                data=[],
                sort_action="none",
                filter_action="none",
                page_action="none",
                style_cell={"textAlign": "left"},
                style_data_conditional=[
                    {"if": {"filter_query": "{direction} = 'Bullish'"}, "backgroundColor": "#d4f7d4"},
                    {"if": {"filter_query": "{direction} = 'Bearish'"}, "backgroundColor": "#f7d4d4"},
                ],
                style_table={"width": "100%", "overflowY": "auto", "maxHeight": "400px"}
            )
            ], style={"display": "flex", "flexDirection": "column", "width": "100%", "gap": "20px"}),

        html.Div([
            html.Label("Patterns to detect:"),
            dcc.Checklist(
                id="pattern-checklist",
                options=[{"label": v["label"], "value":k} for k, v in PATTERN_REGISTRY.items()],
                value=[],
                inline=True
            )
        ])
])

# dcc.Dropdown(id="interval-select"),
# @callback(
#       Output("interval-select","options"),
#       Output("interval-selct","value"),
#       Input("datasouece-input","value"),
# )
# def update_intervals(symbol):
#   intervals = request_data.get_authorized_intervals()
#   return [{"label": i, "value": i} for i in intervals], intervals[0] if intervals else None
@callback(
    Output("candle-chart", "figure"),
    Output("symbol-error", "children"),
    Input("symbol-input", "value"),
    Input("interval-select", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("y-axis-slider", "value")
)
def update_chart(symbol, interval, start_date, end_date, y_range):
    if not symbol:
        return go.Figure(), ""
    try:
        request_data.set_action_symbol(symbol.upper())
        request_data.set_interval(interval)
        print(interval)
        request_data.set_start_date(pd.to_datetime(start_date))
        request_data.set_end_date(pd.to_datetime(end_date))
        df_candles = request_data.get_price_candles_dataframe()
        if df_candles.empty:
            return go.Figure(), f"No data found for symbol '{symbol.upper()}'"

        fig = go.Figure(data=[go.Candlestick(
            x=df_candles.index,
            open=df_candles["open"],
            high=df_candles["high"],
            low=df_candles["low"],
            close=df_candles["close"]
        )])
        y_min = y_range[0] if y_range else df_candles["low"].min
        y_max = y_range[1] if y_range else df_candles["high"].max
        fig.update_layout(
                title=f"{symbol.upper()} - {interval}",
                yaxis=dict(range=[y_min, y_max]),
                xaxis_rangeslider_visible=True,
                xaxis=dict(rangebreaks=get_range_breaks(df_candles, interval))
        )
        return fig, ""
    except Exception as e:
        return go.Figure(), f"Error fetching data for '{symbol.upper()}': {str(e)}"


@callback(
        Output("y-axis-slider", "min"),
        Output("y-axis-slider", "max"),
        Output("y-axis-slider", "value"),
        Input("symbol-input", "value"),
        Input("interval-select", "value"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date")
)
def update_slider_bounds(symbol, interval, start_date, end_date):
    if not symbol:
        return 0, 100, [0, 100]
    request_data.set_action_symbol(symbol.upper())
    request_data.set_interval(interval)
    request_data.set_start_date(pd.to_datetime(start_date))
    request_data.set_end_date(pd.to_datetime(end_date))
    df_candles = request_data.get_price_candles_dataframe()

    if df_candles.empty:
        return 0, 100, [0, 100]

    y_min = df_candles["low"].min()
    y_max = df_candles["high"].max()
    return y_min, y_max, [y_min, y_max]

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
    interval_minutes = pd.Timedelta(interval).total_seconds() / 60

    rows = []
    for key in selected_patterns:
        entry = PATTERN_REGISTRY[key]
        try:
            instances = entry["class"](symbol.upper(), start, end, interval).get_pattern()
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

def _highlight_patch(row: dict) -> Patch:
    center = pd.to_datetime(row["time"])
    half_width = pd.Timedelta(minutes=row["interval_minutes"]) / 2
    patched_fig = Patch()
    patched_fig["layout"]["shapes"] = [dict(
        type="rect", xref="x", yref="paper",
        x0=(center - half_width).isoformat(), x1=(center + half_width).isoformat(),
        y0=0, y1=1, fillcolor="rgba(255,125,0,0.35)", line_width=0
    )]
    return patched_fig

@callback(
        Output("candle-chart","figure", allow_duplicate=True),
        Input("pattern-table", "active_cell"),
        State("pattern-table","data"),
        prevent_initial_call=True
)
def highlight_on_click(active_cell, table_data):
    if not active_cell or not table_data:
        return no_update
    return _highlight_patch(table_data[active_cell["row"]])

def get_range_breaks(df: pd.DataFrame, interval: str) -> list[dict]:
    """
    Detect big gaps in API data to have rangebreaks for Plotly
    It avoid to have big hole for market close hour in charts, by avoiding type category in plotly
    that would hide some legitimates holes between candles
    The treshold has been set to 2x the candle size this is pretty arbitrary and debatable
    But that will do the trick for now
    """
    interval_in_min = pd.Timedelta(interval).total_seconds() / 60

    timestamps = df.index.sort_values()
    diffs = timestamps.to_series().diff().dropna()
    normal_interval = pd.Timedelta(minutes=interval_in_min)

    if not isinstance(normal_interval, pd.Timedelta):
        raise ValueError(f"Invalid interval: {interval_in_min}")
    gaps = diffs.loc[diffs > normal_interval * 2]
    range_breaks = []
    for timestamp, gap in gaps.items():
        gap_start = timestamp - gap
        gap_end = timestamp
        range_breaks.append(dict(
            bounds=[gap_start.isoformat(), gap_end.isoformat()]
        ))
    return range_breaks


if __name__ == "__main__":
    app.run(debug=True)
