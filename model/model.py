import pandas as pd
import seaborn as sns
from sklearn.preprocessing import normalize
import numpy as np
import matplotlib.pyplot as plt


class Model:
    def __init__(self, filename: str, sep: str = ",", eta: float = 0.1, n_iterations: int = 50):
        self.data = np.genfromtxt(filename, delimiter=sep, skip_header=1)
        self.x = self.data[:,0]
        self.y = self.data[:,1]
        self.m = len(self.x)
        self.x = self.x[:, np.newaxis]
        self.y = self.y[:, np.newaxis]
        self.theta = np.random.randn(2,1)
        self.x_legend, self.y_legend = self.getlegend(filename) 
        self.eta = eta
        self.n_iterations = n_iterations

    def getlegend(self, filename: str):
        with open(filename, "r") as f:
            first_row = f.readlines()[0]
            splits = first_row.split(",")
        return splits[0].strip(), splits[1].strip()


    def plotData(self):
        self.plotLine()
        self.plotScatter()
        plt.show()

    def plotScatter(self):
        data = pd.DataFrame({"km": self.data[:,0], "price": self.data[:,1]})
        try:
            sns.scatterplot(data=data, x="km", y="price")
        except Exception as e:
            print(f"Error: {e}")

    def plotLine(self):
        y = self.data[:,0] * self.theta[1][0] + self.theta[0][0]
        predictions = pd.DataFrame({"km": self.data[:,0], "price": y})
        sns.relplot(data=predictions, x="km", y="price", kind="line", color="red")

    def train(self):
        x = np.c_[np.ones((self.m, 1)), self.x]
        for _ in range(self.n_iterations):
            import pdb; pdb.set_trace()
            gradients = 2 / self.m * x.T.dot(x.dot(self.theta) - self.y)
            self.theta = self.theta - self.eta * gradients

    def displayResultWithLinalg(self):
        x = np.c_[np.ones((self.m, 1)), self.x]
        theta_best = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(self.y)
        print(theta_best)
