from datetime import datetime, timedelta
from dash import Dash

from AlphaSense.frontend.data_access import get_authorized_intervals
from AlphaSense.frontend.lightweight_charts.layout import build_layout
from AlphaSense.requests.quotes import get_curated_quotes
import AlphaSense.frontend.lightweight_charts.callbacks  # noqa: F401 - imported for its @callback registration side-effect

DEFAULT_SYMBOL = "AAPL"
DEFAULT_INTERVAL = "5m"
DEFAULT_START_DATE = datetime.now() - timedelta(days=7)
DEFAULT_END_DATE = datetime.now()

app = Dash(__name__)
app.layout = build_layout(
    default_symbol=DEFAULT_SYMBOL,
    default_interval=DEFAULT_INTERVAL,
    default_start_date=DEFAULT_START_DATE,
    default_end_date=DEFAULT_END_DATE,
    authorized_intervals=get_authorized_intervals(),
    curated_quotes=get_curated_quotes(),  # best-effort; renders an empty (but usable) quick-pick if this comes back empty
)

if __name__ == "__main__":
    app.run(debug=True)
