from decimal import Decimal
import pandas as pd
from pandas import DataFrame


class SalaryProcessor:
    """
    Класс для обработки столбца Salary:
    - почасовая
    - месячная
    - диапазоны
    """

    def __init__(self, salary_series: pd.Series):
        self.salary = salary_series.astype(str)

    def process(self) -> DataFrame:
        salary = self.salary.str.replace('$', '', regex=False)

        hourly = salary[salary.str.contains('Per', na=False)]
        monthly = salary[~salary.str.contains('Per', na=False)]

        hourly_salary = self._process_hourly(hourly)
        monthly_salary = self._process_monthly(monthly)

        result = pd.concat([hourly_salary, monthly_salary])
        result.name = 'Average salary, $K'
        return result.sort_index()

    def _process_hourly(self, s):
        extracted = s.str.extract(r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)|(\d+\.?\d*)')
        avg = (
            (extracted[0].astype(float) + extracted[1].astype(float)) / 2
        ).fillna(extracted[2].astype(float))

        return (avg * 8 * 22 / 1000).apply(
            lambda x: Decimal(x).quantize(Decimal('1.0'))
        )

    def _process_monthly(self, s):
        extracted = s.str.extract(r'(\d+)K\s*-\s*(\d+)K|(\d+)K')
        avg = (
            (extracted[0].astype(float) + extracted[1].astype(float)) / 2
        ).fillna(extracted[2].astype(float))

        return avg.apply(lambda x: Decimal(x).quantize(Decimal('1.0')))
