import logging

import pandas as pd
from loaders.base_processor import BaseProcessor

logger = logging.getLogger(__name__)


class PandasLoader(BaseProcessor):
    """
    Загрузчик данных через Pandas.
    
    Подходит для небольших и средних объёмов данных,
    которые помещаются в оперативную память.
    """

    def load_data(self) -> 'PandasLoader':
        """
        Загружает CSV-файл в pandas DataFrame.
        
        Returns:
            PandasLoader: self для method chaining.
        """
        logger.info(f'Pandas: чтение файла {self.path}.')
        self.df = pd.read_csv(self.path)
    
        return self

    def drop_missing(self) -> 'PandasLoader':
        """
        Удаляет строки с пропущенными значениями.
        
        Returns:
            PandasLoader: self для method chaining.
        """
        before = len(self.df)
        self.df = self.df.dropna()
        logger.info(f"Pandas: удалено {before - len(self.df)} строк с пропусками.")
        return self

    def get_dataframe(self) -> pd.DataFrame:
        """
        Возвращает готовый pandas DataFrame.
        
        Returns:
            pd.DataFrame: Очищенные данные.
        """
        logger.info(f"Pandas: итоговый размер датафрейма — {len(self.df)} строк")
        return self.df
