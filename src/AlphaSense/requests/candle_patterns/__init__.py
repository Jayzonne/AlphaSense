from .RequestBearishEngulfing import RequestBearishEngulfing
from .RequestBullishEngulfing import RequestBullishEngulfing
from .RequestCandlePatterns import RequestCandlePatterns
from .RequestEveningStar import RequestEveningStar
from .RequestHammers import RequestHammers
from .RequestMorningStar import RequestMorningStar
from .RequestShootingStar import RequestShootingStar

PATTERN_REGISTRY = {
    "hammer":
              {
                "label": "Hammer",
                "class": RequestHammers,
                "direction": "Bullish"
              },
    "shooting_star":
              {
                 "label": "Shooting Star",
                 "class": RequestShootingStar,
                 "direction": "Bearish"
              },
    "morning_star":
              {
                  "label": "Morning Star",
                  "class": RequestMorningStar,
                  "direction": "Bullish"
               },
    "evening_star":
               {
                  "label": "Evening Star",
                  "class": RequestEveningStar,
                  "direction": "Bearish"
               },
    "bullish_engulfing":
                {
                   "label": "Bullish Engulfing",
                   "class": RequestBullishEngulfing,
                   "direction": "Bullish"
                },
    "bearish_engulfing":
                {
                    "label": "Bearish Engulfing",
                    "class": RequestBearishEngulfing,
                    "direction": "Bearish"
                },
}
__all__ = ['RequestBearishEngulfing',
           'RequestBullishEngulfing',
           'RequestCandlePatterns',
           'RequestEveningStar',
           'RequestHammers',
           'RequestMorningStar',
           'RequestShootingStar',
           'PATTERN_REGISTRY'
           ]
