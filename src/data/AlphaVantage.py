import requests
import polars as pl
from typing import Optional


class AlphaVantageConnector:
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _make_request(self, params: dict) -> Optional[dict]:
        params['apikey'] = self.api_key
        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            if "Error Message" in data or "Note" in data:
                raise ValueError(data.get("Error Message", data.get("Note", "Unknown error")))
            return data
        except Exception as e:
            print(f"Erreur lors de la requête : {e}")
            return None

    def get_intraday_1min(self, symbol: str, output_size: str = "compact") -> Optional[pl.DataFrame]:
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": "1min",
            "outputsize": output_size,
            "datatype": "json"
        }
        data = self._make_request(params)
        if not data:
            return None

        time_series_key = next((k for k in data if "Time Series" in k), None)
        if not time_series_key:
            return None

        time_series = data[time_series_key]

        rows = [
            {
                "timestamp": ts,
                "open": float(values["1. open"]),
                "high": float(values["2. high"]),
                "low": float(values["3. low"]),
                "close": float(values["4. close"])
            }
            for ts, values in time_series.items()
        ]

        df = pl.DataFrame(rows)
        df = df.sort("timestamp")
        df = df.with_columns([
            pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias("timestamp")
        ])
        return df

    def save_to_csv(self, df: pl.DataFrame, filename: str):
        df.write_csv(filename, separator=";")


# --- Code exécuté directement ici (plus de if __name__ == "__main__") ---

API_KEY = "VOTRE_CLE_API_ICI"
SYMBOL = "AAPL"
FILENAME = f"{SYMBOL.lower()}_intraday_1min.csv"

connector = AlphaVantageConnector(API_KEY)
df = connector.get_intraday_1min(SYMBOL)

if df is not None:
    connector.save_to_csv(df, FILENAME)
    print(f"✅ Données sauvegardées dans : {FILENAME}")
else:
    print("❌ Échec de récupération des données.")
