from abc import ABC, abstractmethod
from typing import Dict
from typing import List


class GenericAPI(ABC):
    """
    GenericClass

    Generic class for api requesting differents stock markets APIs
    Class contain the following fields:
    url -> Contains the main url of the api
    start_date -> datetime format, date from which the data start.
    Depending on api in can only retain month or it can retain exact date
    end_date -> datetime format, date from which the data end.
    Same commentary as start_date
    interval -> interval between two stock points (string),
    avaiable value can differs depending on API
    action_symbol -> symbol of the action you want the data from
    api_token -> If needed for the API, your API token
    """
    _url = ""
    _start_date = None
    _end_date = None
    _interval = ""
    _action_symbol = None
    _api_token = None
    _interval_authorized_values: List

    @property
    def interval(self) -> str:
        return self._interval

    @interval.setter
    def interval(self, interval: str) -> None:
        if interval not in self._interval_authorized_values:
            raise ValueError(
                f"Interval must have one of the following value:\
{self._interval_authorized_values}"
            )
        self._interval = interval

    @abstractmethod
    def get_json_api(self) -> Dict:
        """
        This function will request the stock market API
        It will return a json containing informations
        depending of API return format
        Return is under python Dict format
        """

        pass

    @abstractmethod
    def get_standard_json(self) -> Dict:
        """
        This function will return the get_json_api under a
        standardize format for all API for easier database
        feed. This format as been choosen as the yahoo api
        format
        """

        pass
