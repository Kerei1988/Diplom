from loaders.pandas_loader import PandasLoader
# from loaders.dask_loader import DaskLoader

from processors.salary_processor import SalaryProcessor
from analytics.job_analytics import JobAnalytics
from visualization.plots import Plotter


DATA_PATH = 'data/Data-Science-Jobs.csv'


def main():
    loader = PandasLoader(DATA_PATH)
    # loader = DaskLoader(DATA_PATH)

    df = (
        loader
        .load_data()
        .drop_missing()
        .get_dataframe()
    )

    salary_processor = SalaryProcessor(df['Salary'])
    df['Average salary, $K'] = salary_processor.process()

    analytics = JobAnalytics(df)

    print("Средняя зарплата:", analytics.average_salary())

    top_locations = analytics.top_locations()
    Plotter.pie(top_locations, 'TOP 10 Data Science Hiring Locations')

    top_companies = analytics.best_rated_companies()
    Plotter.bar(
        top_companies['Company Name'],
        top_companies['Company Rating'],
        'Best Rated Companies',
        'Company',
        'Rating'
    )


if __name__ == '__main__':
    main()
