"""
Lightweight, best-effort current-quote lookups for the symbol-picker menu.

Deliberately separate from RequestData/TimescaleDB: RequestData exists to
build and cache a queryable history of OHLCV candles for charting/pattern/
indicator analysis. This is a fire-and-forget "what's it trading at right
now" lookup for a handful of curated tickers, with no DB persistence and no
caching layer - a different concern, kept out of the core data pipeline
entirely so that pipeline stays untouched.
"""
import yfinance as yf

# (ticker, display name). Yahoo Finance ticker conventions verified directly
# against finance.yahoo.com for the European suffixes, which are the easy
# ones to get subtly wrong (e.g. Nestle is NESN.SW, not NESN.VX).
US_MEGACAPS = [
    ("AAPL", "Apple"),
    ("MSFT", "Microsoft"),
    ("GOOGL", "Alphabet"),
    ("AMZN", "Amazon"),
    ("NVDA", "NVIDIA"),
    ("META", "Meta Platforms"),
    ("TSLA", "Tesla"),
    ("BRK-B", "Berkshire Hathaway"),
    ("JPM", "JPMorgan Chase"),
    ("LLY", "Eli Lilly"),
]

EU_MEGACAPS = [
    ("ASML.AS", "ASML"),
    ("MC.PA", "LVMH"),
    ("NOVO-B.CO", "Novo Nordisk"),
    ("SAP.DE", "SAP"),
    ("NESN.SW", "Nestle"),
    ("AZN.L", "AstraZeneca"),
    ("OR.PA", "L'Oreal"),
    ("SIE.DE", "Siemens"),
    ("TTE.PA", "TotalEnergies"),
]


def get_curated_quotes() -> list[dict]:
    """
    Best-effort current price for the curated US/EU megacap list, fetched in
    one batched yfinance call rather than one request per symbol. Any symbol
    that fails to resolve a price is silently dropped rather than breaking
    the whole menu - this is a "nice to have" quick-pick list, not part of
    the core charting pipeline, so it degrades gracefully (e.g. offline) down
    to an empty list rather than raising.
    """
    all_tickers = US_MEGACAPS + EU_MEGACAPS
    names = dict(all_tickers)
    symbols = [t for t, _ in all_tickers]
    quotes = []
    try:
        batch = yf.Tickers(" ".join(symbols))
    except Exception:
        return quotes

    for symbol in symbols:
        try:
            fast_info = batch.tickers[symbol].fast_info
            price = fast_info["lastPrice"]
            currency = fast_info["currency"]
            if price is None:
                continue
            quotes.append({
                "symbol": symbol,
                "name": names[symbol],
                "price": round(float(price), 2),
                "currency": currency or "",
            })
        except Exception:
            continue
    return quotes


def get_single_quote(symbol: str) -> dict | None:
    """
    Best-effort current price + display name for one arbitrary symbol - used
    by the "add to quick list" button, so it works for any ticker the person
    searches, not just the curated list above. Returns None (never raises) if
    the symbol can't be resolved, so the caller can show a plain "not found"
    message rather than a stack trace.
    """
    symbol = symbol.strip().upper()
    if not symbol:
        return None
    try:
        ticker = yf.Ticker(symbol)
        fast_info = ticker.fast_info
        price = fast_info["lastPrice"]
        currency = fast_info["currency"]
        if price is None:
            return None
    except Exception:
        return None

    name = symbol
    try:
        short_name = ticker.info.get("shortName")  # slower / more failure-prone than fast_info, so best-effort only
        if short_name:
            name = short_name
    except Exception:
        pass

    return {"symbol": symbol, "name": name, "price": round(float(price), 2), "currency": currency or ""}
