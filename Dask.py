# Тема дипломной:
# № 5. Анализ и сравнение различных способов обработки и хранения больших данных:
# Pandas, Dask и Apache Spark

import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from decimal import Decimal

from mpl_toolkits.mplot3d.art3d import rotate_axes

# считываем базу данных при помощи библиотеки Dask
ddf = dd.read_csv('Data-Science-Jobs.csv')

# Выводим первые пять строк базы данных, для визуального просмотра
pd.set_option('display.max_columns', 9)
# print(ddf.compute().head())

# запросим информацию по базе данных: количество строк, тип данных, которые они содержат
# print(ddf.compute().info())

number_employers = ddf['Company Name'].unique().compute() # количество работодателей
# колонки: 'Salary', 'Logo', 'Company Rating' имеют заполненные строки 438, 436, 439 соответственно.
# Что составляет 12.4%, 12.8%, 12.2% соответственно

complete_ddf = ddf.dropna()
# print(complete_ddf.compute().info())

# Определим TOP 10 популярных мест для найма специалистов в области обработки данных США, визуализируем для удобного восприятия.
# # график показывает что наибольшее количество вакансий Riverwoods, IL(31), Remote (22), New York, NY (21)
locations = complete_ddf['Location'].compute().value_counts()
top_20_loc = locations.head(10)
top_20_loc.plot.pie(title='TOP 10 Data Science Hiring Location')
# plt.show()


# Найдем города с наибольшим и наименьшим спросом на специалистов.
# Количество вакансии меньше 6 - 117 регион
# Количество вакансий больше 15 - 3 региона
location_min = locations[locations < 6]
location_max = locations[locations > 15]
# print(location_min)
# print(location_min.count())
# print(location_max)
# print(location_max.count())

# Самые востребованные специалисты:
# 1 - Data Scientist (22)
# 2 - Senior Data Scientist (10)
# 3 - Senior Manager Data Scientist(8), Principal Data Science (8)
most_special = complete_ddf[['Job Title']].compute(ascending=False).value_counts()
# print(most_special.head(10))

# количество специальностей - 223
number_spec = complete_ddf[['Job Title']].nunique().compute() # количество специальностей - 223
# print(number_spec)

# количество наименее востребованных специалистов, с одной вакансией на рынке труда - 154 вакансий
least_specialists = most_special[most_special < 2].count()
print(least_specialists)


# # найдем среднее количество дней актуальности вакансий - 19+
# минимальный срок актуальности - 1
# максимальное количество дней - 30
date = complete_ddf['Date'].str.replace('d', '')
date = date.str.replace('+', '')
date = date.str.replace('24h', '1')
date = date.astype(int)
mmm_date = date.describe().compute()
# print(mmm_date)


# найдем самого активного работодателя на рынке труда из базы данных - 'Discover Financial Services'
# Общее количество работодателей 259
# отобразим на графике 10 самых активных работодателей

name_company = ddf['Company Name']
employers_vacancies = name_company.value_counts('count').compute()  # список количества вакансий на каждого работодателя
total_employers = employers_vacancies.values.sum()
employer_1 = employers_vacancies.index[0]  # работодатель с наибольшим количеством вакансий
name_company = name_company.drop_duplicates().compute()
number_companies = len(name_company)  # количество работодателей 259
part = (employers_vacancies / total_employers) * 100
part.name = 'fraction, %'
part_top_10 = pd.DataFrame(part.head(10))
# sns.catplot(part_top_10, x='Company Name', y='fraction, %', hue='Company Name', height=2, aspect=3)
plt.xticks(rotation=30)
# plt.show()


# Анализ з\п. Столбец "Salary" является типом 'object', для анализа оплаты труда требуется
# извлечь цифровые значения, преобразовать в тип 'int'. Т.к. есть данные по часовой оплате, и с диапазоном (min, max),
# вычислим среднею почасовую, и умножим на полный часовой рабочий день в США(8 часов, с понедельника по пятницу)
# Сведем полученные данные в таблицу для дальнейшего анализа


#  данные столбца 'Salary' переведем в тип данных float

# данные почасовой оплаты
salary = complete_ddf['Salary']
salary = salary.str.replace('$', '')
salary_per = salary[salary.str.contains('Per')]
hourly_payment_1 = salary_per.str.extract(r'(\d{,3}\.\d{,2}) - (\d{,3}\.\d{,2}).|(\d{,3}\.\d{,2})').astype(float)
average_hour_1 = (((hourly_payment_1[0].dropna() + hourly_payment_1[1].dropna()) / 2) * 8 * 22)
average_hour_2 = hourly_payment_1[2].dropna() * 8 * 22
wages_hour = dd.concat([average_hour_1, average_hour_2], axis=0)
wages_hour = wages_hour.apply(lambda x: float(x / 1000), meta=(None, float)).persist()

# Данные ежемесячной оплаты
monthly_payment = salary.str.extract(r'(\d{,3})K - (\d{,3})K|(\d{,3})K').astype(float)  # полный список оплаты в месяц только цифры
average_payment_1 = (monthly_payment[0].dropna() + monthly_payment[1].dropna()) / 2  # средняя оплата труда из столбца с диапазоном
average_payment_2 = monthly_payment[2].dropna()  # оплата труда фиксированная в месяц
wages_month = dd.concat([average_payment_1, average_payment_2], axis=0).persist()

# сводим полученные данные из столбца 'Salary' в таблицу
wages = dd.concat([wages_hour, wages_month]).apply(lambda x: Decimal(x).quantize(Decimal('1.0')), meta=(None, float))
wages = wages.repartition(npartitions=1)
wages.name = 'Average salary, $K'
title_job = complete_ddf[['Job Title', 'Company Name']]
wages = dd.concat([wages, title_job], axis=1, interleave_partitions=True).persist()

# # Проведем анализ обработанных данных
# Максимальная оплата труда вакансия 'Reliability, Availability and Serviceability Expert,
# Datacenter AI Products Development'($272.5 K) от компании NVIDIA
# Минимальная оплата труда предлагает компания 'Tennis Express'($2.3 K) на вакансию Junior 'Data Science'
salary_top_20 = wages.sort_values('Average salary, $K', ascending=False).head(30).set_index('Job Title')
salary_min_30 = wages.sort_values('Average salary, $K', ascending=True).head(30).set_index('Job Title')
# print(salary_top_20)
# print(salary_min_30)

#Средняя заработная плата по рынку составляет 117.413

average_salary = wages['Average salary, $K'].mean()
# print(average_salary.compute())


# Лучшие работодатели по рейтингу
company_rating = complete_ddf[['Company Rating', 'Company Name']].set_index('Company Name').sort_values('Company Rating', ascending=False)
company_rating = company_rating.drop_duplicates()
company_rating_top = company_rating[company_rating['Company Rating'] > 4]
plt.figure(figsize=(10, 6))
x = company_rating_top.index.compute()
y = company_rating_top['Company Rating'].compute()
plt.xticks(rotation=25)
plt.bar(x, y, align='edge')
plt.xlabel('Company name')
plt.ylabel('Company rating')
plt.title('Rating of companies')
plt.grid(True)
plt.show()
