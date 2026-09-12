from .RequestBearishEngulfing import RequestBearishEngulfing
from .RequestBullishEngulfing import RequestBullishEngulfing
from .RequestCandlePatterns import RequestCandlePatterns
from .RequestEveningStar import RequestEveningStar
from .RequestHammers import RequestHammers
from .RequestHeadAndShoulders import RequestHeadAndShoulders
from .RequestInverseHeadAndShoulders import RequestInverseHeadAndShoulders
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
    "head_and_shoulders":
                {
                    "label": "Head & Shoulders",
                    "class": RequestHeadAndShoulders,
                    "direction": "Bearish"
                },
    "inverse_head_and_shoulders":
                {
                    "label": "Inverse Head & Shoulders",
                    "class": RequestInverseHeadAndShoulders,
                    "direction": "Bullish"
                },
}
__all__ = ['RequestBearishEngulfing',
           'RequestBullishEngulfing',
           'RequestCandlePatterns',
           'RequestEveningStar',
           'RequestHammers',
           'RequestHeadAndShoulders',
           'RequestInverseHeadAndShoulders',
           'RequestMorningStar',
           'RequestShootingStar',
           'PATTERN_REGISTRY'
           ]
