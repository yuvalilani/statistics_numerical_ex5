import numpy as np
from numba import njit

p_flip_cache = {}


def calc_p_flip(energy):
    return 1 / (np.exp(-2 * energy) + 1)


def create_p_flip(h, eta):
    p_flip = np.zeros((9, 2))
    for n in range(9):
        for s in range(2):
            energy = ((n - 4) * eta - h) * (2 * s - 1)
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


def dont_stop(first, second, k):
    m1 = first[0]
    m2 = second[0]
    delta = 10 ** -3
    return not (abs(m1 - m2) / abs(m2) < delta or k > 10 ** 5)


def neighbor_setup(spin_lattice, neighbor_sum, lattice_size, h, eta):
    """
    CHANGES THE neighbor_sum
    """
    for x in range(lattice_size):
        for y in range(lattice_size):
            if x < lattice_size - 1:
                neighbor_sum[x, y] -= spin_lattice[x + 1, y]
            if y < lattice_size - 1:
                neighbor_sum[x, y] -= spin_lattice[x, y + 1]
            if x > 0:
                neighbor_sum[x, y] -= spin_lattice[x - 1, y]
            if y > 0:
                neighbor_sum[x, y] -= spin_lattice[x, y - 1]


@njit
def single_run(spin_lattice, neighbor_sum, lattice_size, k, n_sweep, h, eta, p_flip):
    p = np.random.random((k * n_sweep, lattice_size, lattice_size))
    M = 0
    M_sqr = 0
    U = 0
    U_sqr = 0
    for i in range(k):
        for j in range(n_sweep):
            flip_all(spin_lattice, neighbor_sum, lattice_size, p[i * n_sweep + j], p_flip)
        m = np.sum(spin_lattice)
        u = 0.5 * eta * np.dot(neighbor_sum.reshape(lattice_size ** 2), spin_lattice.reshape(lattice_size ** 2)) - m * h
        M += m
        M_sqr += m ** 2
        U += u
        U_sqr += u ** 2
    M /= k
    U /= k
    M_sqr /= k
    U_sqr /= k
    return M, M_sqr, U, U_sqr


def simulate(k, h, eta):
    n_sweep = 5
    lattice_size = 32
    spin_lattice = np.ones((lattice_size, lattice_size))
    neighbor_sum = np.zeros((lattice_size, lattice_size))
    neighbor_setup(spin_lattice, neighbor_sum, lattice_size, h, eta)
    p_flip = create_p_flip(h, eta)
    first = single_run(spin_lattice, neighbor_sum, lattice_size, k, n_sweep, h, eta, p_flip)
    second = single_run(spin_lattice, neighbor_sum, lattice_size, 2 * k, n_sweep, h, eta, p_flip)
    while dont_stop(first, second, k):
        k = 2 * k
        first = second
        second = single_run(spin_lattice, neighbor_sum, lattice_size, 2 * k, n_sweep, h, eta, p_flip)
        print(k, second)
    return second


if __name__ == '__main__':
    # parameters
    k = 50
    h = 0
    eta = 0.5
    output = simulate(k, h, eta)
    print(output)
