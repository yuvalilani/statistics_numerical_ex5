import numpy as np


def p_flip(energy):
    return 1 / (np.exp(-2 * energy) + 1)


def flip_spin(spin_lattice, energy_lattice, lattice_size, x, y, p, eta):
    energy = energy_lattice[x, y]
    if p < p_flip(energy):
        spin_lattice[x, y] *= -1
        energy_delta = 2 * eta * spin_lattice[x, y]
        if x < lattice_size - 1:
            energy_lattice[x + 1, y] += energy_delta * spin_lattice[x + 1, y]
        if y < lattice_size - 1:
            energy_lattice[x, y + 1] += energy_delta * spin_lattice[x, y + 1]
        if x > 0:
            energy_lattice[x - 1, y] += energy_delta * spin_lattice[x - 1, y]
        if y > 0:
            energy_lattice[x, y - 1] += energy_delta * spin_lattice[x, y - 1]


def flip_all(spin_lattice, energy_lattice, lattice_size, p, eta):
    for i in range(lattice_size):
        for j in range(lattice_size):
            flip_spin(spin_lattice, energy_lattice, lattice_size, i, j, p[i, j], eta)


def dont_stop(first, second, k):
    m1 = first[0]
    m2 = second[0]
    delta = 10**-3
    return not (abs(m1 - m2) / abs(m2) < delta or k > 10 ** 8)


def energy_setup(spin_lattice, energy_lattice, lattice_size, h, eta):
    """
    CHANGES THE energy_lattice
    :param energy_lattice:
    :return:
    """
    for x in range(lattice_size):
        for y in range(lattice_size):
            if x < lattice_size - 1:
                energy_lattice[x, y] -= eta * spin_lattice[x + 1, y]
            if y < lattice_size - 1:
                energy_lattice[x, y] -= eta * spin_lattice[x, y + 1]
            if x > 0:
                energy_lattice[x, y] -= eta * spin_lattice[x - 1, y]
            if y > 0:
                energy_lattice[x, y] -= eta * spin_lattice[x, y - 1]
            energy_lattice[x, y] -= h
            energy_lattice[x, y] *= spin_lattice[x, y]


def single_run(spin_lattice, energy_lattice, lattice_size, k, n_sweep, h, eta):
    p = np.random.random((k*n_sweep, lattice_size, lattice_size))
    M = 0
    M_sqr = 0
    U = 0
    U_sqr = 0
    for i in range(k):
        for j in range(n_sweep):
            flip_all(spin_lattice, energy_lattice, lattice_size, p[i*n_sweep+j], eta)
        m = np.sum(spin_lattice)
        u = 2*np.sum(energy_lattice) + m * h
        M += m
        M_sqr += m**2
        U += u
        U_sqr += u**2
    M /= k
    U /= k
    M_sqr /= k
    U_sqr /= k
    return M, M_sqr, U, U_sqr


def simulate(k, h, eta):
    n_sweep = 5
    lattice_size = 32
    spin_lattice = np.ones((lattice_size, lattice_size))
    energy_lattice = np.zeros((lattice_size, lattice_size))
    energy_setup(spin_lattice, energy_lattice, lattice_size, h, eta)
    first = single_run(spin_lattice, energy_lattice, lattice_size, k, n_sweep, h, eta)
    second = single_run(spin_lattice, energy_lattice, lattice_size, 2*k, n_sweep, h, eta)
    while dont_stop(first, second, k):
        k = 2 * k
        first = second
        second = single_run(spin_lattice, energy_lattice, lattice_size, 2*k, n_sweep, h, eta)
        print(k, second)
    return second


if __name__ == '__main__':
    # parameters
    k = 50
    h = 0
    eta = 0.5
    M, M_sqr, U, U_sqr = simulate(k, h, eta)
