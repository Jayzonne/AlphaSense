from AlphaSense.requests.candle_patterns.RequestCandlePatterns import RequestCandlePatterns
from AlphaSense.requests.candle_patterns.swing_utils import find_pivots, is_inverse_head_and_shoulders


class RequestInverseHeadAndShoulders(RequestCandlePatterns):
    """
    Inverse Head & Shoulders (bullish reversal): the mirror of
    RequestHeadAndShoulders - three troughs (left shoulder, head, right
    shoulder) with the head lower than both roughly-symmetric shoulders,
    separated by a roughly level neckline (two peaks). See
    RequestHeadAndShoulders for why get_pattern() is fully overridden rather
    than using either existing base class's fixed windowing. The right
    shoulder's candle is reported as the pattern's location.
    """

    def __init__(self, *args, pivot_order: int = 3, **kwargs):
        self._pivot_order = pivot_order
        super().__init__(*args, **kwargs)

    def _is_pattern(self, candle: dict) -> bool:
        return False  # unused - get_pattern() is fully overridden below

    def get_pattern(self) -> list[dict]:
        peaks, troughs = find_pivots(self._data_request, self._pivot_order)
        matches = []
        for t1, t2, t3 in zip(troughs, troughs[1:], troughs[2:]):
            p1 = next((p for p in reversed(peaks) if t1 < p < t2), None)
            p2 = next((p for p in peaks if t2 < p < t3), None)
            if p1 is None or p2 is None:
                continue
            left_shoulder, head, right_shoulder, neckline_1, neckline_2 = (
                self._data_request[i] for i in (t1, t2, t3, p1, p2)
            )
            if is_inverse_head_and_shoulders(left_shoulder, head, right_shoulder, neckline_1, neckline_2):
                matches.append(self._data_request[t3])
        return matches
