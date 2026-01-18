import dask.dataframe as dd
from core.base_processor import BaseProcessor


class DaskLoader(BaseProcessor):

    def load_data(self):
        self.df = dd.read_csv(self.path)
        return self

    def drop_missing(self):
        self.df = self.df.dropna()
        return self

    def get_dataframe(self):
        return self.df.compute()
