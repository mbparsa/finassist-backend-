
import pandas as pd


class merchant():
    _instance = None
    _df = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(merchant, cls).__new__(cls)
        return cls._instance

    def set_merchant(self, data):
        self._df = pd.DataFrame(data)

    def get_merchant(self):
        return self._df