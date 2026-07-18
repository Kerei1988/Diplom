import logging

import pandas as pd


logger = logging.getLogger(__name__)

class SalaryProcessor:
    """
    Класс для обработки столбца Salary.
    
    Преобразует различные форматы зарплат (месячные, почасовые, диапазоны)
    в единый формат: средняя зарплата в тысячах долларов ($K).
    
    Attributes:
        salary (pd.Series): Исходный столбец с зарплатами в строковом формате.
    """

    def __init__(self, salary_series: pd.Series) -> None:
        """
        Инициализация обработчика.
        
        Args:
            salary_series (pd.Series): Столбец с зарплатами из датафрейма.
        """
        self.salary = salary_series.astype(str)

    def process(self) -> pd.DataFrame:
        """
        Обрабатывает зарплаты, разделяя на почасовые и месячные.
        
        Returns:
            pd.Series: Обработанные зарплаты в $K, отсортированные по индексу.
        """
        if self.salary.empty:
            logger.warning(f'Столбец Salary пуст.')
            return pd.Series(name='Average salary, $K', dtype=float)
        
        salary = self.salary.str.replace('$', '', regex=False)

        hourly = salary[salary.str.contains('Per', na=False)]
        monthly = salary[~salary.str.contains('Per', na=False)]

        hourly_salary = self._process_hourly(hourly)
        monthly_salary = self._process_monthly(monthly)

        result = pd.concat([hourly_salary, monthly_salary])
        result.name = 'Average salary, $K'
        return result.sort_index()

    def _process_hourly(self, s:pd.Series) -> pd.Series:
        """
        Пересчитывает почасовые зарплаты в месячные.
        
        Формула: среднее_в_час * 8 часов * 22 дня / 1000 = $K в месяц.
        
        Args:
            s (pd.Series): Строки с почасовыми зарплатами.
            
        Returns:
            pd.Series: Месячные зарплаты в $K.
        """
        extracted = s.str.extract(r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)|(\d+\.?\d*)')
        avg = (
            (extracted[0].astype(float) + extracted[1].astype(float)) / 2
        ).fillna(extracted[2].astype(float))

        return (avg * 8 * 22 / 1000).round(1)

    def _process_monthly(self, s:pd.Series) -> pd.Series:
        """
        Парсит месячные зарплаты из формата '$50K - $70K' или '$60K'.
        
        Args:
            s (pd.Series): Строки с месячными зарплатами.
            
        Returns:
            pd.Series: Зарплаты в $K.
        """
        extracted = s.str.extract(r'(\d+)K\s*-\s*(\d+)K|(\d+)K')
        avg = (
            (extracted[0].astype(float) + extracted[1].astype(float)) / 2
        ).fillna(extracted[2].astype(float))

        return avg.round(1)
