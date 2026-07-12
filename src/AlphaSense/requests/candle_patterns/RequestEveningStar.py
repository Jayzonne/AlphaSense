from AlphaSense.requests.candle_patterns.RequestMultipleCandlesPattern import RequestMultipleCandlePatterns


# Could need improvement on pattern detection
# The detection here could be a little be too permissive
class RequestEveningStar(RequestMultipleCandlePatterns):

    def _is_complex_pattern(self, candles: list[dict]) -> bool:
        previous_candle, current_candle, next_candle = candles
        body_previous_candle = previous_candle["close"] - previous_candle["open"]
        body_current_candle = abs(current_candle["close"] - current_candle["open"])
        body_next_candle = next_candle["open"] - next_candle["close"]
        midpoint_previous_candle = previous_candle["open"] + body_previous_candle / 2
        return (
                body_previous_candle > 0 and
                body_current_candle <= 0.25 * body_previous_candle and
                body_next_candle > 0 and
                next_candle["close"] < midpoint_previous_candle
        )
