import numpy as np


def flip_spin(spin_lattice, energy_lattice, x, y, p, eta):
    pass


def flip_all(spin_lattice, energy_lattice):
    rows = len(spin_lattice)
    cols = len(spin_lattice[0])
    p = np.random.random((rows, cols))
    for i in range(rows):
        for j in range(cols):
            flip_spin(spin_lattice, energy_lattice, i, j, p[i, j], eta)


def stop(f1, f2):
    pass


def energy_setup(spin_lattice, energy_lattice):
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
            energy_lattice -= h
            energy_lattice *= spin_lattice[x, y]


def single_run(spin_lattice, energy_lattice, k):
    for i in range(k):
        flip_all(spin_lattice, energy_lattice)

    return M, U


def simulate(k):
    spin_lattice = np.ones((lattice_size, lattice_size))
    energy_lattice = np.zeros((lattice_size, lattice_size))
    energy_setup(spin_lattice, energy_lattice)
    first = single_run(k)
    second = single_run(2 * k)
    while stop(first, second):
        k = 2 * k
        first = second
        second = single_run(2 * k)


if __name__ == '__main__':
    # parameters
    lattice_size = 32
    h = 1
    eta = 0.5
    n_sweep = 5
    simulate()
