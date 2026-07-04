#!/usr/bin/env python
from abc import ABC, abstractmethod


class databaseGenericInput(ABC):
    @abstractmethod
    def feed_api_data_to_database():
        pass
