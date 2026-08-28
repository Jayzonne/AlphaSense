from datetime import datetime
from AlphaSense.requests.RequestData import RequestData
from abc import ABC, abstractmethod


class RequestCandlePatterns(ABC):
    """
    RequestCandlePatterns
    Class to easily request candle patterns given an action, a start_date an end_date and interval
    Different Candle pattens can be requested 
    And this class can be extended to return more Candle patterns
    action_symbol -> symbol of the stock action you want the data from
    start_date -> date you want the requested data to start from, datetime format
    end_date -> date you want the requested data to end, datetime format
    interval -> interval between two stock point (string), avaiable:
    Depends on the data_source
    data_source -> One of the data source avaiable, current data sources are:
        YAHOO, ALPHAVANTAGE
    """

    _data_request: list[dict]

    def __init__(
            self,
            action_symbol: str,
            start_date: datetime,
            end_date: datetime,
            interval: str,
            data_source="YAHOO",
            price_data: list[dict] | None = None,
            ):
        if price_data is not None:
            self._data_request = price_data
        else:
            self._data_request = RequestData(
               action_symbol,
               start_date,
               end_date,
               interval,
               data_source
               ).get_price_candles()

    def get_pattern(self) -> list[dict]:
        """ Get list of candle that match a certain pattern """
        data_requests_pattern = []
        for candle in self._data_request:
            if self._is_pattern(candle):
                data_requests_pattern.append(candle)
        return data_requests_pattern

    @abstractmethod
    def _is_pattern(self, candle: dict) -> bool:
        pass
