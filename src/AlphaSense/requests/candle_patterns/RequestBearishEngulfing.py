from AlphaSense.requests.candle_patterns.RequestMultipleCandlesPattern import RequestMultipleCandlePatterns


# Could need improvement on pattern detection
# The detection here could be a little be too permissive
class RequestBearishEngulfing(RequestMultipleCandlePatterns):

    def _is_complex_pattern(self, candles: list[dict]) -> bool:
        previous_candle, current_candle, _ = candles
        return (
                current_candle["close"] < current_candle["open"] and
                previous_candle["close"] > previous_candle["open"] and
                current_candle["open"] >= previous_candle["close"] and
                current_candle["close"] <= previous_candle["open"]
                )
