from dash import Dash, html, dcc, callback, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta

from AlphaSense.requests.RequestData import RequestData

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
app.layout = [
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
            dcc.RadioItems(
                id="scale-select",
                options=[{"label": "Linear", "value": "linear"},
                         {"label": "Log", "value": "log"}],
                value="linear"
            )
        ]),

        dcc.Graph(id="candle-chart")
]

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
    Input("scale-select", "value")
)
def update_chart(symbol, interval, start_date, end_date, scale):
    if not symbol:
        return go.Figure(), ""
    try:
        request_data.set_action_symbol(symbol.upper())
        request_data.set_interval(interval)
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

        fig.update_layout(
                title=f"{symbol.upper()} - {interval}",
                yaxis_type=scale,
                xaxis_rangeslider_visible=True,
                xaxis=dict(rangebreaks=get_range_breaks(df_candles, interval))
        )
        return fig, ""
    except Exception as e:
        return go.Figure(), f"Error fetching data for '{symbol.upper()}': {str(e)}"


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
    print(gaps)
    range_breaks = []
    for timestamp, gap in gaps.items():
        gap_start = timestamp - gap
        gap_end = timestamp
        range_breaks.append(dict(
            bounds=[gap_start.isoformat(), gap_end.isoformat()]
        ))
    print(range_breaks)
    return range_breaks


if __name__ == "__main__":
    app.run(debug=True)
