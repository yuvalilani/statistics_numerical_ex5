import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.size'] = '16'
# csv path:
file_0 = r"final data\h = 0.csv"
file_01 = r"final data\h = 0.1.csv"
file_05 = r"final data\h = 0.5.csv"
file_1 = r"final data\h = 1.csv"
file_2 = r"final data\h = 2.csv"

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
    return ((1 + z ** 2) ** (1 / 4) * (1 - 6 * z ** 2 + z ** 4) ** (1 / 8)) / (1 - z ** 2) ** (1 / 2)


def A():
    x, y = excel_to_np(file_0, cols=[1, 2])  # cols to take from the csv
    plt.scatter(x, np.abs(y) / N, color="red", label="simulation")
    plt.plot(x, onsager(x), label="Onsager")
    plt.xlabel(r"$\eta$")
    plt.ylabel(r"$|<M>|/N$")
    plt.legend()
    plt.show()


def plot_U(file, label, scatter=False, show=1):
    x, y = excel_to_np(file, cols=[1, 4])  # cols to take from the csv
    if scatter:
        plt.scatter(x, y / N, label=label)
    else:
        plt.plot(x, y / N, label=label)
    plt.xlabel(r"$\eta$")
    plt.ylabel("U")
    plt.legend()
    if show:
        plt.show()


def plot_cv(file, label, scatter=False, show=1):
    x, y, usq = excel_to_np(file, cols=[1, 4, 5])
    cv = (usq - y ** 2) / N
    if scatter:
        plt.scatter(x, cv / N, label=label)
    else:
        plt.plot(x, cv / N, label=label)
    plt.xlabel(r"$\eta$")
    plt.ylabel(r"$c_v$")
    plt.legend()
    if show:
        plt.show()


def plot_m(file, label, scatter=False, show=1):
    x, y = excel_to_np(file, cols=[1, 2])
    if scatter:
        plt.scatter(x, np.abs(y), label=label)
    else:
        plt.plot(x, np.abs(y), label=label)
    plt.xlabel(r"$\eta$")
    plt.ylabel(r"$|<M>|/N$")
    plt.legend()
    if show:
        plt.show()


def flip_percent(file, label, scatter=False, show=True):
    x, flip_count, k = excel_to_np(file, cols=[1, 6, 7])
    y = flip_count / (20 * k * N)
    if scatter:
        plt.scatter(x, y, label=label)
    else:
        plt.plot(x, y, label=label)
    plt.xlabel(r"$\eta$")
    plt.ylabel(r"$|<M>|/N$")
    plt.legend()
    if show:
        plt.show()


def B():
    plot_U(file_0, "B=0", True)
    plot_cv(file_0, "B=0", True)


def C():
    plot_U(file_0, "B=0", False, False)
    plot_U(file_01, "B=0.1", False, False)
    plot_U(file_05, "B=0.5", False, False)
    plot_U(file_1, "B=1")

    plot_cv(file_0, "B=0", False, False)
    plot_cv(file_01, "B=0.1", False, False)
    plot_cv(file_05, "B=0.5", False, False)
    plot_cv(file_1, "B=1")

    plot_m(file_0, "B=0", False, False)
    plot_m(file_01, "B=0.1", False, False)
    plot_m(file_05, "B=0.5", False, False)
    plot_m(file_1, "B=1")

    flip_percent(file_0, "B=0", False, False)
    flip_percent(file_01, "B=0.1", False, False)
    flip_percent(file_05, "B=0.5", False, False)
    flip_percent(file_1, "B=1", False, False)
    flip_percent(file_2, "B=2")


def plot_hysteresis(eta):
    file_name = r"final data\hysteresis eta = " + str(eta) + ".csv"
    x, y = excel_to_np(file_name, cols=[1, 2])  # cols to take from the csv
    plt.plot(x, y / N)
    plt.xlabel(r"$\eta$")
    plt.ylabel(r"$|<M>|/N$")


def plot_critical_eta():
    x, y = excel_to_np(r"final data\h = 0 with initial direction.csv", cols=[1, 2])
    plt.scatter(x, np.abs(y) / N)
    x, y = excel_to_np(file_0, cols=[1, 2])
    plt.scatter(x, np.abs(y) / N)
    plt.xlabel(r"$\eta$")
    plt.ylabel(r"$|<M>|/N$")
    plt.show()


if __name__ == '__main__':
    # eta_arr = [0.3, 0.5, 0.59, 0.62, 0.65, 1]
    # for eta in eta_arr:
    #     plot_hysteresis(eta)
    # plt.legend(["eta = " + str(eta) for eta in eta_arr])
    # plt.xlim(-0.5, 0.5)
    # plt.show()
    # plot_critical_eta()
    A()
    B()
    C()
