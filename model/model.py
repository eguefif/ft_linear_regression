import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Model:
    def __init__(self, filename: str, sep: str = ","):
        self.data = pd.read_csv(filename, sep=sep)
        if len(self.data.columns) == 2:
            self.data_x = self.data.columns[0]
            self.data_y = self.data.columns[1]
        else:
            print("Invalid data")

    def plotData(self):
        try:
            plot = sns.scatterplot(data=self.data, x=self.data_x, y=self.data_y)
            plot.set(xlabel="km", ylabel = "price")
            plt.show()
            figure = plot.get_figure()
            figure.savefig("plot.png")
        except (Exception) as e:
            print(f"Error: {e}")

