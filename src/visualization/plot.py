import logging

import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


logger = logging.getLogger(__name__)

class Plotter:
    """
    Класс для визуализации результатов анализа.
    
    Предоставляет статические методы для построения графиков:
    - Круговые диаграммы (pie chart)
    - Столбчатые диаграммы (bar chart)
    
    Графики отображаются на экране и опционально сохраняются в файл.
    """

    @staticmethod
    def pie(data: pd.Series, title: str, filename: str | None = None) -> None:
        """
        Строит круговую диаграмму.
        
        Args:
            data (pd.Series): Данные для диаграммы (значения — размеры долей).
            title (str): Заголовок диаграммы.
            filename (str | None): Путь для сохранения графика. Если None — не сохраняется.
        """
        if len(data) == 0:
            logger.warning("Нет данных для построения круговой диаграммы.")
            return

        data.plot.pie(title=title, autopct='%1.1f%%')
        plt.ylabel('')
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()

    @staticmethod
    def bar(x: pd.Series, y: pd.Series, title: str, 
            xlabel: str, ylabel: str,
            filename: str | None = None) -> None:
        """
        Строит столбчатую диаграмму.
        
        Args:
            x (pd.Series): Значения для оси X (подписи столбцов).
            y (pd.Series): Значения для оси Y (высота столбцов).
            title (str): Заголовок диаграммы.
            xlabel (str): Подпись оси X.
            ylabel (str): Подпись оси Y.
            filename (str | None): Путь для сохранения графика. Если None — не сохраняется.
        """

        if len(x)==0 or len(y)==0:
            logger.warning("Нет данных для построения столбчaтой диаграммы.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.bar(x, y)
        plt.xticks(rotation=45)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
