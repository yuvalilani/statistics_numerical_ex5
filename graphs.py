import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.size'] = '16'
# csv path:
file_0 = r"C:\Users\t8852921\Documents\python\numeri5\statistics_numerical_ex5\final data\h = 0.csv"
file_01 = r"C:\Users\t8852921\Documents\python\numeri5\statistics_numerical_ex5\final data\h = 0.1.csv"
file_05 = r"C:\Users\t8852921\Documents\python\numeri5\statistics_numerical_ex5\final data\h = 0.5.csv"
file_1 = r"C:\Users\t8852921\Documents\python\numeri5\statistics_numerical_ex5\final data\h = 1.csv"

N = 32 * 32


def excel_to_np(f, cols):
    a = []
    for i in cols:
        df = pd.read_csv(f)
        data = df.iloc[:, i].to_numpy()[1:]
        a.append(np.array(data, dtype=float))
    return a


def graph(file, cols, xlable, ylabel, func=lambda x: x, show=1):
    x, y = excel_to_np(file, cols=cols)  # cols to take from the csv
    plt.scatter(x, func(y))
    plt.xlabel(xlable)
    plt.ylabel(ylabel)
    if show:
        plt.show()


def onsager(eta):
    z = np.exp(-2 * eta)
    return ((1 + z ** 2) ** (1 / 4) + (1 - 6 * z ** 2 + z ** 4) ** (1 / 8)) / (1 - z ** 2) ** (1 / 2)


def A():
    x, y = excel_to_np(file_0, cols=[1, 2])  # cols to take from the csv
    plt.scatter(x, np.abs(y) / N, color="red", label="simulation")
    plt.plot(x, onsager(x), label="Onsager")
    plt.xlabel(r"$\eta$")
    plt.ylabel(r"$|<M>|/N$")
    plt.legend()
    plt.show()

def plot_U(file, label, show=1):
    x, y = excel_to_np(file, cols=[1, 4])  # cols to take from the csv
    plt.plot(x, np.abs(y) / N, label=label)
    plt.xlabel(r"$\eta$")
    plt.ylabel("U")
    plt.legend()
    if show:
        plt.show()


def plot_cv(file, label, show=1):
    x, y, usq = excel_to_np(file, cols=[1, 4, 5])
    cv = (usq - y ** 2) / N
    plt.plot(x, cv / N, label=label)
    plt.xlabel(r"$\eta$")
    plt.ylabel(r"$c_v$")
    plt.legend()
    if show:
        plt.show()

def B():
    plot_U(file_0, "B=0")
    plot_cv(file_0, "B=0")

def C():
    plot_U(file_0, "B=0", 0)
    plot_U(file_01, "B=0.1", 0)
    plot_U(file_05, "B=0.5", 0)
    plot_U(file_1, "B=1")


    plot_cv(file_0, "B=0", 0)
    plot_cv(file_01, "B=0.1", 0)
    plot_cv(file_05, "B=0.5", 0)
    plot_cv(file_1, "B=1")


A()
B()
C()