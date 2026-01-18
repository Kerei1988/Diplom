from abc import ABC, abstractmethod


class BaseProcessor(ABC):
    """
    Абстрактный базовый класс для загрузки и подготовки данных.
    """

    def __init__(self, path: str):
        self.path = path
        self.df = None

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def drop_missing(self):
        pass

    @abstractmethod
    def get_dataframe(self):
        pass
