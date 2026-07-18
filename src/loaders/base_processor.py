from abc import ABC, abstractmethod

import pandas as pd


class BaseProcessor(ABC):
    """
    Абстрактный базовый класс для загрузки и подготовки данных.
    
    Определяет интерфейс для всех загрузчиков данных.
    Наследники должны реализовать методы load_data(), drop_missing(), get_dataframe().
    
    Attributes:
        path (str): Путь к файлу с данными.
        df (pd.DataFrame | None): Загруженный датафрейм (None до вызова load_data).
    """

    def __init__(self, path: str) -> None:
        """
        Инициализация загрузчика.
        
        Args:
            path (str): Путь к CSV-файлу с данными.
        """
        self.path: str = path
        self.df: pd.DataFrame | None = None
  
    @abstractmethod
    def load_data(self):
        """Загружает данные из файла в self.df. Возвращает self для цепочки вызовов."""
        pass

    @abstractmethod
    def drop_missing(self):
        """Удаляет строки с пропущенными значениями. Возвращает self для цепочки вызовов."""
        pass

    @abstractmethod
    def get_dataframe(self) -> pd.DataFrame:
        """
        Возвращает готовый DataFrame с данными.
        
        Returns:
            pd.DataFrame: Очищенный датафрейм.
        """
        pass
