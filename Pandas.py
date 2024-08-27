from decimal import Decimal
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# считываем базу данных при помощи библиотеки Pandas

df = pd.read_csv('Data-Science-Jobs.csv')
# Выводим первые пять строк базы данных, для визуального просмотра
# print(df.head())
# запросим информацию по базе данных: Какие столбцы и строки у нас есть, тип данных
# print((df.info()))
# print(df.describe())
# колонки: 'Salary', 'Logo', 'Company Rating' имеют заполненные строки 438, 436, 439 соответственно.
# Что составляет 12.4%, 12.8%, 12.2% соответственно
#
complete_df = df.dropna()
# print(complete_df.describe())

# При удалении всех строк, которые содержат пустые значения, описательная статистика полученнная методом describe(),
# отличается на незначительное значения. Анализируем базу данных без строк, содержащих пустые значения.
# Определим TOP 10 популярныx мест для найма специалистов в области обработки данных США, визуализируем для удобного восприятия.
# график показывает что наибольшее количество вакансий Riverwoods, IL(31), Remote (22), New York, NY (21)
locations = complete_df[['Location']].value_counts()
top_20_loc = locations.head(10)
# top_20_loc.plot.pie(title='TOP 10 Data Science Hiring Location')
# plt.show()


# определим города с количеством вакансий меньше 3, получим 84 города с наименьшим спросом на специалистов
small_number = locations[locations < 3]
# print(small_number)   # список городов
# print(small_number.count())   #количество городов

most_special = complete_df.value_counts(
    'Job Title')  #Определим какие специалисты по обработке данных наиболе востребованы сейчас
# print(most_special)
number_spec = complete_df['Job Title'].nunique()  # количество специальностей - 223
# print(number_spec)

# найдем среднее количество дней актуальности вакансий - 19+
date = complete_df['Date'].str.replace('d', '')
date = date.str.replace('+', '')
date = date.str.replace('24h', '1')
date = date.astype(int)
mean_date = date.mean()
# print(mean_date)


# найдем самого активного работодателя на рынке труда из базы данных - 'Discover Financial Services'
# Общее количество работодателей 259
# отобразим на графике 10 самых активных работодателей

name_company = df['Company Name']
employers_vacancies = name_company.value_counts()  # список количества вакансий на каждого работодателя
total_employers = employers_vacancies.values.sum() #количество вакансий
employer_1 = employers_vacancies.index[0]  # работодатель с наибольшим количеством вакансий
name_company = name_company.unique()
number_companies = len(name_company)  # количество работодателей 259
part = (employers_vacancies / total_employers)*100
part.name = 'fraction, %'
part = pd.DataFrame(part)
part_top_10 = part.head(10)  # десятка активных работодателей базы данных
# sns.catplot(part_top_10, x='Company Name', y='fraction, %', hue='Company Name')
# plt.show()

# Для анализа оплаты труда. Столбец "Salary" является типом 'object', для анализа оплаты труда требуется
# извлечь цифровые значения, преобразовать в тип 'int'. Т.к. есть данные по часовой оплате, и с диапазоном (min, max),
# вычислим среднею почасовую, и умножим на полный часовой рабочий день в США(8 часов, с понедельника по пятницу)
# Сведем полученные данные в таблицу для дальнейшего анализа

salary = complete_df['Salary']

salary = salary.str.replace('$', '')
salary_per = salary[salary.str.contains('Per')]
hourly_payment_1 = salary_per.str.extract(r'(\d{,3}\.\d{,2}) - (\d{,3}\.\d{,2}).|(\d{,3}\.\d{,2})')
average_hour_1 = ((hourly_payment_1[0].dropna().astype(float) + hourly_payment_1[1].dropna().astype(float)) / 2) * 8 * 22
average_hour_2 = hourly_payment_1[2].dropna().astype(float) * 8 * 22
wages_hour = pd.concat((average_hour_1, average_hour_2), axis=0).apply(
    lambda x: Decimal(x / 1000).quantize(Decimal('1.0')))

monthly_payment = salary.str.extract(r'(\d{,3})K - (\d{,3})K|(\d{,3})K')  # полный список оплаты в месяц только цифры
average_payment_1 = (monthly_payment[0].dropna().astype(float) + monthly_payment[1].dropna().astype(float)) / 2  # средняя оплата труда из столбца с диапазоном
average_payment_2 = monthly_payment[2].dropna().astype(float)  # оплата труда фиксированная в месяц
wages_month = pd.concat((average_payment_1, average_payment_2), axis=0)

salary = pd.concat((wages_month, wages_hour), axis=0).sort_index(axis=0)
salary.name = 'Average salary, $K'

# # Проведем анализ обработанных данных
# Максимальная оплата труда вакансия 'Reliability, Availability and Serviceability Expert, Datacenter AI Products Development'($272.5 K) от компании NVIDIA
# Минимальная оплата труда предлагает компания 'Tennis Express'($2.3 K) на вакансию Junior 'Data Science'

salary = pd.concat((salary, complete_df[['Job Title', 'Company Name']]), axis=1)
salary_top_20 = salary.sort_values('Average salary, $K', ascending=False).head(20).set_index('Job Title')
salary_min_30 = salary.sort_values('Average salary, $K', ascending=True).head(30).set_index('Job Title')
# print(salary_min_30)
# print(salary_top_20)

#Средняя заработная плата по рынку состовляет 117.413

average_salary = salary['Average salary, $K'].astype(float).mean()
# print(average_salary)


# Лучшие работодатели по рейтингу
company_rating = complete_df[['Company Rating', 'Company Name']].set_index('Company Name').sort_values('Company Rating', ascending=False)
company_rating = company_rating.drop_duplicates()
company_rating.plot.bar()
plt.show()

