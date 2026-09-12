def find_pivots(candles: list[dict], order: int = 3) -> tuple[list[int], list[int]]:
    """
    Returns (peak_indices, trough_indices): positions of local highs/lows in
    the candle list, where a candle counts as a peak if its high is the
    maximum within +/- `order` candles on either side (and the mirror for a
    trough on low). Bigger `order` means fewer, coarser swings.

    This is how Head & Shoulders is located from raw OHLC data: unlike the
    single-candle and fixed-3-candle patterns, the number of candles between
    the shoulders/head/neckline varies, so detection works over these
    detected swing points rather than a fixed-size window.
    """
    peaks, troughs = [], []
    n = len(candles)
    for i in range(order, n - order):
        window = candles[i - order: i + order + 1]
        if candles[i]["high"] == max(c["high"] for c in window):
            peaks.append(i)
        if candles[i]["low"] == min(c["low"] for c in window):
            troughs.append(i)
    return peaks, troughs


def is_head_and_shoulders(
    left_shoulder: dict, head: dict, right_shoulder: dict, neckline_1: dict, neckline_2: dict,
    shoulder_tolerance: float = 0.15, neckline_tolerance: float = 0.10,
) -> bool:
    """
    Bearish reversal shape: a head higher than two roughly-symmetric
    shoulders, separated by a roughly level neckline (two troughs).

    - shoulder_tolerance: how different the two shoulder highs may be,
      relative to how far the head rises above the lower of the two.
    - neckline_tolerance: how different the two neckline troughs may be,
      relative to their average.
    """
    head_high, left_high, right_high = head["high"], left_shoulder["high"], right_shoulder["high"]
    if not (head_high > left_high and head_high > right_high):
        return False

    head_prominence = head_high - min(left_high, right_high)
    if head_prominence <= 0:
        return False
    if abs(left_high - right_high) / head_prominence > shoulder_tolerance:
        return False

    neck_1, neck_2 = neckline_1["low"], neckline_2["low"]
    neck_avg = (neck_1 + neck_2) / 2
    if neck_avg <= 0 or abs(neck_1 - neck_2) / neck_avg > neckline_tolerance:
        return False

    # Sanity guard: both troughs should sit below both shoulders, or this
    # isn't really a "neck" separating them.
    if neck_1 >= min(left_high, right_high) or neck_2 >= min(left_high, right_high):
        return False

    return True


def is_inverse_head_and_shoulders(
    left_shoulder: dict, head: dict, right_shoulder: dict, neckline_1: dict, neckline_2: dict,
    shoulder_tolerance: float = 0.15, neckline_tolerance: float = 0.10,
) -> bool:
    """ Bullish reversal mirror of is_head_and_shoulders(): troughs/peaks swapped. """
    head_low, left_low, right_low = head["low"], left_shoulder["low"], right_shoulder["low"]
    if not (head_low < left_low and head_low < right_low):
        return False

    head_prominence = min(left_low, right_low) - head_low
    if head_prominence <= 0:
        return False
    if abs(left_low - right_low) / head_prominence > shoulder_tolerance:
        return False

    neck_1, neck_2 = neckline_1["high"], neckline_2["high"]
    neck_avg = (neck_1 + neck_2) / 2
    if neck_avg <= 0 or abs(neck_1 - neck_2) / neck_avg > neckline_tolerance:
        return False

    if neck_1 <= max(left_low, right_low) or neck_2 <= max(left_low, right_low):
        return False

    return True
