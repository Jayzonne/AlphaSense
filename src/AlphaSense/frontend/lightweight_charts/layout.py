from dash import html, dcc, dash_table
from datetime import datetime

from dash_lightweight_charts import LightweightChart

from AlphaSense.requests.candle_patterns import PATTERN_REGISTRY
from AlphaSense.requests.indicators import INDICATOR_REGISTRY


def build_layout(
    default_symbol: str,
    default_interval: str,
    default_start_date: datetime,
    default_end_date: datetime,
    authorized_intervals: list[str],
) -> html.Div:
    return html.Div([
        html.H1("AlphaSense Dashboard"),

        html.Div([
            dcc.Input(
                id="symbol-input",
                type="text",
                placeholder="Enter symbol (e.g AAPL)",
                value=default_symbol,
                debounce=True
            ),
            html.Div(id="symbol-error", style={"color": "red"}),
            dcc.Dropdown(
                id="interval-select",
                options=[{"label": i, "value": i} for i in authorized_intervals],
                value=default_interval
            ),
            dcc.DatePickerRange(
                id="date-range",
                start_date=default_start_date,
                end_date=default_end_date
            ),
        ]),

        html.Div([
            # No range slider / y-axis slider here: Lightweight Charts' native
            # drag-to-pan, scroll-to-zoom, and drag-to-rescale on the price
            # axis cover the same ground without extra UI chrome.
            LightweightChart(id="candle-chart", height=500),

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
                options=[{"label": v["label"], "value": k} for k, v in PATTERN_REGISTRY.items()],
                value=[],
                inline=True
            )
        ]),
        # Single source of truth for OHLCV data: fetched once per symbol/interval/
        # date-range change and fanned out to the chart, confluence, and pattern
        # table callbacks below, instead of each of them hitting the DB on its own.
        dcc.Store(id="price-data-store"),
        dcc.Store(id="confluence-store"),

        html.Div([
            html.Label("Indicators to display:"),
            dcc.Checklist(
                id="indicator-checklist",
                options=[{"label": v["label"], "value": k} for k, v in INDICATOR_REGISTRY.items()],
                value=[],
                inline=True
            )
        ]),
        html.Div([
            html.Label("Highlight candles with at least this many agreeing signals:"),
            dcc.Input(id="confluence-threshold", type="number", min=1, step=1, value=3)
        ]),
    ])
