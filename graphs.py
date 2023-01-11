import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.size'] = '15'
file = r"C:\Users\t8852921\Documents\python\numeri5\statistics_numerical_ex5\hysteresis eta = 1.csv"


def excel_to_np(f, cols):
    a = []
    for i in cols:
        df = pd.read_csv(f)
        data = df.iloc[:, i].to_numpy()[1:]
        a.append(np.array(data, dtype=float))
    return a


def graph():
    x, y = excel_to_np(file, cols=[1, 4])
    plt.scatter(x, y)
    plt.xlabel(r"$\eta$")
    plt.ylabel("U")
    plt.show()


graph()
