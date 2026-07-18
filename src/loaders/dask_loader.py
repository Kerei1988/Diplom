import logging

import dask.dataframe as dd
import pandas as pd
from loaders.base_processor import BaseProcessor


logger = logging.getLogger(__name__)

class DaskLoader(BaseProcessor):
    """
    Загрузчик данных через Dask.
    
    Использует ленивые вычисления и подходит для больших объёмов данных,
    не помещающихся в оперативную память.
    """

    def load_data(self) -> 'DaskLoader':
        """
        Загружает CSV-файл в Dask DataFrame (ленивый режим).
        Данные не загружаются в память до вызова .compute().
        
        Returns:
            DaskLoader: self для method chaining.
        """
        logger.info(f'Dask: чтение файла {self.path}')
        self.df: dd.DataFrame = dd.read_csv(self.path)
        return self

    def drop_missing(self) -> 'DaskLoader':
        """
        Удаляет строки с пропущенными значениями (лениво).
        
        Returns:
            DaskLoader: self для method chaining.
        """
        self.df = self.df.dropna()
        logger.info(f'Dask: пропуски будут удалены при compute()')
        return self

    def get_dataframe(self) -> pd.DataFrame:
        """
        Выполняет все отложенные операции и возвращает pandas DataFrame.
        
        Returns:
            pd.DataFrame: Очищенные данные в виде pandas DataFrame.
        """
        logger.info("Dask: выполнение compute()...")
        result = self.df.compute()
        logger.info(f"Dask: итоговый размер датафрейма — {len(self.df)} строк")
        return result
