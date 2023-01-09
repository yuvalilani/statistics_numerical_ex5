import numpy as np


def p_flip(energy):
    return 1 / (np.exp(-2 * energy) + 1)


def flip_spin(spin_lattice, energy_lattice, x, y, p, eta):
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
    pass


def flip_all():
    pass


def stop():
    pass


def energy_setup(spin_lattice, energy_lattice, h, eta):
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


def simulate():
    spin_lattice = np.ones((lattice_size, lattice_size))
    energy_lattice = np.zeros((lattice_size, lattice_size))


if __name__ == '__main__':
    # parameters
    lattice_size = 32
    h = 1
    eta = 0.5
    n_sweep = 5
    simulate()
