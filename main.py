import numpy as np


def flip_spin(i, j, p):
    pass


def flip_all():
    pass


def stop():
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
