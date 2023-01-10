import numpy as np
from numba import njit
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

p_flip_cache = {}


@njit
def calc_p_flip(energy):
    return 1 / (np.exp(-2 * energy) + 1)


@njit
def create_p_flip(h, eta):
    p_flip = np.zeros((9, 2))
    for n in range(9):
        for s in range(2):
            energy = -((n - 4) * eta + h) * (2 * s - 1)
            p_flip[n, s] = calc_p_flip(energy)
    return p_flip


@njit
def flip_spin(spin_lattice, neighbor_sum, lattice_size, x, y, p, p_flip):
    if p < p_flip[int(4 + neighbor_sum[x, y]), int((spin_lattice[x, y] + 1) / 2)]:
        spin_lattice[x, y] *= -1
        ds = 2 * spin_lattice[x, y]
        if x < lattice_size - 1:
            neighbor_sum[x + 1, y] += ds
        if y < lattice_size - 1:
            neighbor_sum[x, y + 1] += ds
        if x > 0:
            neighbor_sum[x - 1, y] += ds
        if y > 0:
            neighbor_sum[x, y - 1] += ds


@njit
def flip_all(spin_lattice, neighbor_sum, lattice_size, p, p_flip):
    for i in range(lattice_size):
        for j in range(lattice_size):
            flip_spin(spin_lattice, neighbor_sum, lattice_size, i, j, p[i, j], p_flip)


@njit
def dont_stop(first, second, k):
    m1 = first[0]
    m2 = second[0]
    delta = 10 ** -3
    return not (abs(m1 - m2) / abs(m2) < delta or k > 10 ** 5)


def spin_setup(lattice_size):
    return 2.0 * np.random.randint(2, size=(lattice_size, lattice_size)) - 1.0


@njit
def neighbor_setup(spin_lattice, neighbor_sum, lattice_size):
    """
    CHANGES THE neighbor_sum
    """
    for x in range(lattice_size):
        for y in range(lattice_size):
            if x < lattice_size - 1:
                neighbor_sum[x, y] += spin_lattice[x + 1, y]
            if y < lattice_size - 1:
                neighbor_sum[x, y] += spin_lattice[x, y + 1]
            if x > 0:
                neighbor_sum[x, y] += spin_lattice[x - 1, y]
            if y > 0:
                neighbor_sum[x, y] += spin_lattice[x, y - 1]


def single_run(spin_lattice, neighbor_sum, lattice_size, k, n_sweep, h, eta, p_flip):
    p = np.random.random((k * n_sweep, lattice_size, lattice_size))
    M = np.zeros(k)
    M_sqr = np.zeros(k)
    U = np.zeros(k)
    U_sqr = np.zeros(k)
    for i in range(k):
        for j in range(n_sweep):
            flip_all(spin_lattice, neighbor_sum, lattice_size, p[i * n_sweep + j], p_flip)
        m = np.sum(spin_lattice)
        u = 0.5 * eta * np.dot(neighbor_sum.reshape(lattice_size ** 2), spin_lattice.reshape(lattice_size ** 2)) - m * h
        M[i] = m
        M_sqr[i] = m ** 2
        U[i] = u
        U_sqr[i] = u ** 2
    return np.average(M), np.average(M_sqr), np.average(U), np.average(U_sqr)


def full_run(spin_lattice, neighbor_sum, k, h, eta, lattice_size):
    n_sweep = 5
    p_flip = create_p_flip(h, eta)
    single_run(spin_lattice, neighbor_sum, lattice_size, k, n_sweep, h, eta, p_flip)
    first = single_run(spin_lattice, neighbor_sum, lattice_size, k, n_sweep, h, eta, p_flip)
    second = single_run(spin_lattice, neighbor_sum, lattice_size, 2 * k, n_sweep, h, eta, p_flip)
    while dont_stop(first, second, k):
        k = 2 * k
        first = second
        second = single_run(spin_lattice, neighbor_sum, lattice_size, 2 * k, n_sweep, h, eta, p_flip)
    return second


def simulate(k, h, eta, lattice_size):
    spin_lattice = spin_setup(lattice_size)
    neighbor_sum = np.zeros((lattice_size, lattice_size), float)
    neighbor_setup(spin_lattice, neighbor_sum, lattice_size)
    return full_run(spin_lattice, neighbor_sum, k, h, eta, lattice_size)


def full_simulation(h, eta, log_data=False):
    k = 50
    lattice_size = 32
    data = np.zeros((5, len(eta)))
    data[0] = eta
    for n in tqdm(range(len(eta))):
        data[1:, n] = simulate(k, h, eta[n], lattice_size)
    plt.plot(eta, abs(data[1]), '.')
    plt.show()
    plt.plot(eta, data[3], '.')
    plt.show()
    #  log data to csv
    if log_data:
        pd.DataFrame(data.transpose(), columns=["eta", "M", "M_sqr", "U", "U_sqr"]).to_csv("h = " + str(h) + ".csv")


def simulate_hysteresis(h, eta, log_data=False):
    k = 50
    lattice_size = 32
    spin_lattice = spin_setup(lattice_size)
    neighbor_sum = np.zeros((lattice_size, lattice_size), float)
    neighbor_setup(spin_lattice, neighbor_sum, lattice_size)
    data = np.zeros((5, len(h)))
    data[0] = h
    for n in tqdm(range(len(h))):
        data[1:, n] = full_run(spin_lattice, neighbor_sum, k, h[n], eta, lattice_size)
    plt.plot(h, data[1], '.')
    plt.show()
    #  log data to csv
    if log_data:
        pd.DataFrame(data.transpose(), columns=["eta", "M", "M_sqr", "U", "U_sqr"]).to_csv(
            "hysteresis eta = " + str(eta) + ".csv")


if __name__ == '__main__':
    # h = 0
    # eta = np.concatenate((np.arange(0.1, 0.42, 0.05), np.arange(0.42, 0.5, 0.005), np.arange(0.55, 0.85, 0.05)))
    # full_simulation(h, eta, True)
    h = np.concatenate((np.linspace(1, -1, 400), np.linspace(-1, 1, 400)))
    eta = 0.5
    simulate_hysteresis(h, eta, True)
