from AlphaSense.requests.candle_patterns.RequestCandlePatterns import RequestCandlePatterns
from AlphaSense.requests.candle_patterns.swing_utils import find_pivots, is_head_and_shoulders


class RequestHeadAndShoulders(RequestCandlePatterns):
    """
    Head & Shoulders (bearish reversal): three peaks - left shoulder, head,
    right shoulder - with the head higher than both roughly-symmetric
    shoulders, separated by a roughly level neckline. Spans however many
    candles separate those five points, which neither RequestCandlePatterns'
    single-candle window nor RequestMultipleCandlePatterns' fixed 3-candle
    window can express - so get_pattern() is fully overridden here, scanning
    over detected local-high/low pivots (see swing_utils.py) instead of a
    fixed window. The right shoulder's candle is reported as the pattern's
    location.

    pivot_order: how many candles on each side a point must beat to count as
    a local high/low (bigger = coarser, less noisy swings).
    """

    def __init__(self, *args, pivot_order: int = 3, **kwargs):
        self._pivot_order = pivot_order
        super().__init__(*args, **kwargs)

    def _is_pattern(self, candle: dict) -> bool:
        return False  # unused - get_pattern() is fully overridden below

    def get_pattern(self) -> list[dict]:
        peaks, troughs = find_pivots(self._data_request, self._pivot_order)
        matches = []
        for p1, p2, p3 in zip(peaks, peaks[1:], peaks[2:]):
            t1 = next((t for t in reversed(troughs) if p1 < t < p2), None)
            t2 = next((t for t in troughs if p2 < t < p3), None)
            if t1 is None or t2 is None:
                continue
            left_shoulder, head, right_shoulder, neckline_1, neckline_2 = (
                self._data_request[i] for i in (p1, p2, p3, t1, t2)
            )
            if is_head_and_shoulders(left_shoulder, head, right_shoulder, neckline_1, neckline_2):
                matches.append(self._data_request[p3])
        return matches
