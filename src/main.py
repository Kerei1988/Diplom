"""
Главный модуль для запуска пайплайна анализа рынка вакансий.

Выполняет:
1. Загрузку данных из CSV
2. Обработку зарплат
3. Аналитику и визуализацию
4. Сохранение результатов в output/
"""

import logging

from loaders.pandas_loader import PandasLoader
# from loaders.dask_loader import DaskLoader
from processors.salary_processor import SalaryProcessor
from analytics.job_analytics import JobAnalytics
from visualization.plot import Plotter


DATA_PATH = 'data/Data-Science-Jobs.csv'
DATA_OUTPUT = 'src/output'


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)


def main():
    """Точка входа. Запускает полный пайплайн обработки и анализа данных."""
    try:
        import os

        os.makedirs(DATA_OUTPUT, exist_ok=True)

        loader = PandasLoader(DATA_PATH)
        # loader = DaskLoader(DATA_PATH)

        logger.info("Загрузка данных.")
        df = (
            loader
            .load_data()
            .drop_missing()
            .get_dataframe()
        )

        if df.empty:
            raise ValueError("Датафрейм пусть после загрузки и очистки.")
        
        logger.info(f"Загружено {len(df)} записей.")

        logger.info('Обработка данных столбца зарплат...')
        salary_processor = SalaryProcessor(df['Salary'])
        df['Average salary, $K'] = salary_processor.process()

        logger.info('Формирование аналитики.')
        analytics = JobAnalytics(df)
        avg_salary = analytics.average_salary()
        
        logger.info("Сохранение результатов...")
        top_locations = analytics.top_locations()
        top_locations.to_csv(f'{DATA_OUTPUT}/top_locations.csv', header=['Count'])
        Plotter.pie(data=top_locations,
                    title='TOP 10 Data Science Hiring Locations',
                    filename=f"{DATA_OUTPUT}/top_locations.png")

        top_companies = analytics.best_rated_companies().head(10)
        Plotter.bar(
            x=top_companies['Company Name'],
            y=top_companies['Company Rating'],
            title='Best Rated Companies',
            xlabel='Company',
            ylabel='Rating',
            filename=f'{DATA_OUTPUT}/best_rated_company.png'
        )

        top_jobs = analytics.top_jobs()
        top_jobs.to_csv(f"{DATA_OUTPUT}/top_jobs.csv", header=['Count'])

        logger.info("Формирование отчёта...")
        with open(f'{DATA_OUTPUT}/report.txt', mode='w', encoding='utf-8') as file:
            file.write(f"Средняя заработная плата - {avg_salary}\n")
            file.write(f'Всего вакансий - {len(df)}\n')
            file.write(f"Всего работодателей - {df['Company Name'].nunique()}\n")
            
            file.write(f'Топ 5 локаций:\n')
            for city, country in top_locations[:5].items():
                file.write(f'   {city} --- {country}\n')
            
            file.write(f'Топ 5 должностей:\n')
            for jobs, country in top_jobs[:5].items():
                file.write(f'   {jobs} --- {country}\n')    
        
        logger.info(f"Результаты сохранены в папку '{DATA_OUTPUT}/'")
    except FileNotFoundError:
        logger.error(f'Файл не найден в {DATA_PATH}')
        logger.error("Поместите Data-Science-Jobs.csv в папку data/")
    except ValueError as e:
        logger.error(f"Ошибка в данных - {e}")
    except Exception as e:
        logger.error(f"Непредвиденная ошибка - {e}")

if __name__ == '__main__':
    main()
