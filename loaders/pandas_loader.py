import pandas as pd
from core.base_processor import BaseProcessor


class PandasLoader(BaseProcessor):

    def load_data(self):
        self.df = pd.read_csv(self.path)
        return self

    def drop_missing(self):
        self.df = self.df.dropna()
        return self

    def get_dataframe(self):
        return self.df
