import json
import os as _os

from dash.development.base_component import Component, _explicitize_args

_basepath = _os.path.dirname(__file__)
with open(_os.path.join(_basepath, "package-info.json")) as _f:
    _package = json.load(_f)

__version__ = _package["version"]

_js_dist = [
    {
        "relative_package_path": "dash_lightweight_charts.min.js",
        "namespace": "dash_lightweight_charts",
    }
]


class LightweightChart(Component):
    """A LightweightChart component.

    Wraps TradingView's Lightweight Charts v5 (a native candlestick chart
    with per-indicator panes) as a Dash component, in place of a
    dcc.Graph/Plotly figure.

    Keyword arguments:

    - id (string; optional):
        The ID of this component, used to identify it in Dash callbacks.

    - candles (list of dicts; optional):
        OHLC candles: [{time, open, high, low, close, color?, borderColor?,
        wickColor?}]. `time` is a UTCTimestamp (unix seconds). The optional
        per-point color fields are how confluence-threshold highlighting is
        applied to individual candles.

    - overlays (list of dicts; optional):
        Line series sharing the price pane, e.g. Bollinger Bands:
        [{id, name, color, data: [{time, value}]}].

    - subplots (list of dicts; optional):
        Indicator subplots, each rendered in its own pane below price:
        [{id, title, height, priceFormat, series: [{id, name, color, type,
        data: [{time, value}]}]}]. `type` is "line" or "histogram".

    - markers (list of dicts; optional):
        Candle pattern markers: [{time, position, shape, color, text?}].

    - highlightTime (number; optional):
        Unix-seconds time of a single candle to flash-highlight, e.g. when a
        pattern-table row is clicked.

    - height (number; default 400):
        Height in px of the main price pane. Subplot panes add to this.

    - clickedTime (number; optional):
        Read-only output: unix-seconds time of the last candle the user
        clicked on the chart.
    """

    _children_props = []
    _base_nodes = ["children"]
    _namespace = "dash_lightweight_charts"
    _type = "LightweightChart"

    @_explicitize_args
    def __init__(
        self,
        id=Component.UNDEFINED,
        candles=Component.UNDEFINED,
        overlays=Component.UNDEFINED,
        subplots=Component.UNDEFINED,
        markers=Component.UNDEFINED,
        highlightTime=Component.UNDEFINED,
        height=Component.UNDEFINED,
        clickedTime=Component.UNDEFINED,
        **kwargs,
    ):
        self._prop_names = [
            "id", "candles", "overlays", "subplots", "markers",
            "highlightTime", "height", "clickedTime",
        ]
        self._valid_wildcard_attributes = []
        self.available_properties = self._prop_names
        self.available_wildcard_properties = []
        _explicit_args = kwargs.pop("_explicit_args")
        _locals = locals()
        _locals.update(kwargs)
        args = {k: _locals[k] for k in _explicit_args if k != "children"}

        super(LightweightChart, self).__init__(**args)


LightweightChart._js_dist = _js_dist
