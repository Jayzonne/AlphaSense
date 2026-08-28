# AlphaSense/requests/indicators/__init__.py
from .RequestIndicator import RequestIndicator
from .RequestRSI import RequestRSI
from .RequestMACD import RequestMACD
from .RequestBollingerBands import RequestBollingerBands
from .RequestStochastic import RequestStochastic

INDICATOR_REGISTRY = {
    "rsi":             {"label": "RSI (14)",            "class": RequestRSI,             "display": "subplot", "columns": ["rsi"], "y_range": [0, 100]},
    "macd":            {"label": "MACD",                 "class": RequestMACD,            "display": "subplot", "columns": ["macd", "signal", "histogram"]},
    "bollinger_bands": {"label": "Bollinger Bands (20)",  "class": RequestBollingerBands,  "display": "overlay", "columns": ["upper_band", "middle_band", "lower_band"]},
    "stochastic":      {"label": "Stochastic (14,3)",     "class": RequestStochastic,      "display": "subplot", "columns": ["percent_k", "percent_d"], "y_range": [0, 100]},
}
__all__ = ['RequestIndicator',
           'RequestRSI',
           'RequestMACD',
           'RequestBollingerBands',
           'RequestStochastic',
           'INDICATOR_REGISTRY'
           ]
