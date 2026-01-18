import matplotlib.pyplot as plt


class Plotter:

    @staticmethod
    def pie(data, title):
        data.plot.pie(title=title, autopct='%1.1f%%')
        plt.ylabel('')
        plt.show()

    @staticmethod
    def bar(x, y, title, xlabel, ylabel):
        plt.figure(figsize=(10, 6))
        plt.bar(x, y)
        plt.xticks(rotation=30)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.show()
