from dash import html, dcc, dash_table
from datetime import datetime

from dash_lightweight_charts import LightweightChart

from AlphaSense.requests.candle_patterns import PATTERN_REGISTRY
from AlphaSense.requests.indicators import INDICATOR_REGISTRY
from AlphaSense.frontend.lightweight_charts.serializers import (
    BULLISH_CONFLUENCE_COLOR, BEARISH_CONFLUENCE_COLOR, pattern_legend,
)

_RANGE_PRESET_LABELS = ["1D", "1W", "1M", "YTD", "1Y", "5Y", "MAX"]


def _legend() -> html.Div:
    """
    Directly answers "which patterns/signals affect the detection": the
    color coding used for confluence highlighting and per-pattern chart
    markers, generated from the same registries/palette the chart itself
    uses (serializers.pattern_legend()) so it can't drift out of sync.
    """
    swatch = {"marginRight": "16px", "whiteSpace": "nowrap"}
    return html.Div([
        html.Span("Confluence: ", style={"fontWeight": "bold"}),
        html.Span("\u25CF Bullish", style={**swatch, "color": BULLISH_CONFLUENCE_COLOR}),
        html.Span("\u25CF Bearish", style={**swatch, "color": BEARISH_CONFLUENCE_COLOR, "marginRight": "28px"}),
        html.Span("Patterns: ", style={"fontWeight": "bold"}),
        *[
            html.Span(f"\u25CF {entry['label']}", style={**swatch, "color": entry["color"]})
            for entry in pattern_legend()
        ],
    ], style={"fontSize": "13px", "margin": "6px 0 14px 0", "display": "flex", "flexWrap": "wrap"})


def build_layout(
    default_symbol: str,
    default_interval: str,
    default_start_date: datetime,
    default_end_date: datetime,
    authorized_intervals: list[str],
    curated_quotes: list[dict] = None,
) -> html.Div:
    curated_quotes = curated_quotes or []

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
            html.Button("+ Add to quick list", id="add-to-quicklist-btn", n_clicks=0),
            dcc.Dropdown(
                id="quick-pick-dropdown",
                options=[],  # populated by sync_quick_pick_options() from quick-pick-store
                placeholder="Search big US/EU stocks...",
                searchable=True,
                value=None,
                style={"minWidth": "340px"},
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
        ], style={"display": "flex", "gap": "10px", "alignItems": "center", "flexWrap": "wrap"}),

        html.Div(id="quicklist-feedback", style={"fontSize": "12px", "color": "#666", "marginTop": "4px"}),

        html.Div([
            html.Button(label, id={"type": "range-preset", "index": label}, n_clicks=0)
            for label in _RANGE_PRESET_LABELS
        ], style={"display": "flex", "gap": "6px", "margin": "10px 0"}),

        _legend(),

        # Holds the current quick-pick list (starts as the curated US/EU
        # megacaps, grows as symbols are added via the button above). Session
        # only - a page refresh resets it back to the curated starting list,
        # since there's no persistence layer behind this menu.
        dcc.Store(id="quick-pick-store", data=curated_quotes),

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

