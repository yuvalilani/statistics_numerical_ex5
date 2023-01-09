import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # parameters
    lattice_size = 32
    h = 1
    eta = 0.5
    n_sweep = 5
    spin_lattice = np.ones((lattice_size, lattice_size))
    energy_lattice = np.zeros((lattice_size, lattice_size))
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
