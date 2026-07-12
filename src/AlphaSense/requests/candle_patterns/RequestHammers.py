from AlphaSense.requests.candle_patterns.RequestCandlePatterns import RequestCandlePatterns


class RequestHammers(RequestCandlePatterns):

    def _is_pattern(self, candle: dict) -> bool:
        high = candle["high"]
        low = candle["low"]
        close = candle["close"]
        open = candle["open"]
        body = abs(close - open)
        lower_wick = min(open, close) - low
        upper_wick = high - max(open, close)
        return (
                lower_wick >= 2 * body and
                upper_wick <= 0.1 * body and
                body > 0
        )
