import logging

import pandas as pd

logger = logging.getLogger(__name__)

class JobAnalytics:
    """
    Аналитика рынка вакансий Data Science.
    
    Предоставляет методы для расчёта ключевых метрик:
    - Топ локаций, должностей, работодателей
    - Средняя зарплата по рынку
    - Рейтинг компаний с фильтрацией
    
    Attributes:
        df (pd.DataFrame): Датафрейм с обработанными данными.
        COL_LOCATION (str): Название колонки с локацией.
        COL_JOB_TITLE (str): Название колонки с должностью.
        COL_COMPANY_NAME (str): Название колонки с компанией.
        COL_AVG_SALARY (str): Название колонки со средней зарплатой.
        COL_RATING (str): Название колонки с рейтингом компании.
    """
    
    COL_LOCATION = 'Location'
    COL_JOB_TITLE = 'Job Title'
    COL_COMPANY_NAME = 'Company Name'
    COL_AVG_SALARY = 'Average salary, $K'
    COL_RATING = 'Company Rating'

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Инициализация аналитики.
        
        Args:
            df (pd.DataFrame): Датафрейм с загруженными и обработанными данными.
        """
        required_columns = [    
            self.COL_LOCATION,
            self.COL_JOB_TITLE,
            self.COL_COMPANY_NAME,
            self.COL_AVG_SALARY,
            self.COL_RATING
            ]
        
        missing = [col for col in df.columns if col not in required_columns]
        
        if missing:
            raise ValueError(f'Отсутствуют колонки: {missing}')
        self.df = df

    def top_locations(self, n: int=10) -> pd.Series:
        """
        Возвращает топ-N локаций по количеству вакансий.
        
        Args:
            n (int): Количество возвращаемых локаций (по умолчанию 10).
            
        Returns:
            pd.Series: Локации и количество вакансий, отсортированные по убыванию.
        """
        return self.df[self.COL_LOCATION].value_counts().head(n)

    def top_jobs(self, n: int=10) -> pd.Series:
        """
        Возвращает топ-N должностей по количеству вакансий.
        
        Args:
            n (int): Количество возвращаемых должностей (по умолчанию 10).
            
        Returns:
            pd.Series: Должности и количество вакансий, отсортированные по убыванию.
        """
        return self.df[self.COL_JOB_TITLE].value_counts().head(n)

    def top_employers(self, n: int=10) -> pd.Series:
        """
        Возвращает топ-N работодателей по количеству вакансий.
        
        Args:
            n (int): Количество возвращаемых работодателей (по умолчанию 10).
            
        Returns:
            pd.Series: Компании и количество вакансий, отсортированные по убыванию.
        """
        return self.df[self.COL_COMPANY_NAME].value_counts().head(n)

    def average_salary(self) -> float:
        """
        Рассчитывает среднюю зарплату по всем вакансиям.
        
        Returns:
            float: Средняя зарплата в тысячах долларов ($K).
        """
        return self.df[self.COL_AVG_SALARY].astype(float).mean()

    def best_rated_companies(self, min_rating: int=4.0) -> pd.DataFrame:
        """
        Возвращает компании с рейтингом не ниже заданного.
        
        Убирает дубликаты и сортирует по убыванию рейтинга.
        Очищает названия компаний от спецсимволов.
        
        Args:
            min_rating (float): Минимальный рейтинг для фильтрации (по умолчанию 4.0).
            
        Returns:
            pd.DataFrame: Колонки 'Company Name' и 'Company Rating'.
        """
        result = (
            self.df[[self.COL_COMPANY_NAME, self.COL_RATING]]
            .drop_duplicates()
            .sort_values(self.COL_RATING, ascending=False)
            .query(f'`{self.COL_RATING}` >= @min_rating')
        )
        result[self.COL_COMPANY_NAME] = result[self.COL_COMPANY_NAME].str.replace(r'[\r\n]+', ' ', regex=True) 
        
        return result