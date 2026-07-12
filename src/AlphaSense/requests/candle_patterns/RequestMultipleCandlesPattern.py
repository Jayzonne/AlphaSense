from AlphaSense.requests.candle_patterns.RequestCandlePatterns import RequestCandlePatterns
from abc import abstractmethod


class RequestMultipleCandlePatterns(RequestCandlePatterns):
    """
    RequestMultipleCandlePatterns
    This class can be used to request candles patterns with three candles
    It is allowing to have some pattern like the morning star
    """

    def get_pattern(self) -> list[dict]:
        """ Get list of candle that match a certain pattern """
        data_requests_pattern = []
        for i, candle in enumerate(self._data_request[1:-1]):
            previous_candle = self._data_request[i]
            next_candle = self._data_request[i+2]
            if self._is_complex_pattern([previous_candle, candle, next_candle]):
                data_requests_pattern.append(candle)
        return data_requests_pattern

    # If subclass need this method to determinated if a complex pattern
    # Based on the fact a candle is another pattern, it will override it
    # Otherwise, this method is useless because it does not allow
    # us to calculate multiple candles patterns
    def _is_pattern(self, candle: dict) -> bool:
        return False

    @abstractmethod
    def _is_complex_pattern(self, candles: list[dict]) -> bool:
        pass
