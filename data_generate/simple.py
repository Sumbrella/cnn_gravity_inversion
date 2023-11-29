import numpy as np
import os
from utils.gravity_model import GravityModel
import matplotlib.pyplot as plt


def main():
    nx = 128 // 2
    ny = 128 // 2
    nz = 64 // 2

    xmax = 400
    ymax = 400
    zmax = 400

    obs_z = -5

    gm = GravityModel(-xmax, xmax, -ymax, ymax, zmax, nx, ny, nz, obs_z)
    sample_size = 200
    gm.rho = np.ones(gm.rho.shape) * 0.1

    gm.rho[np.sqrt(gm.X ** 2 + gm.Y ** 2 + (gm.Z - 150) ** 2) <= 100] = 0.5
    gm.fast_forward()

    data = np.dstack([gm.obs, *np.gradient(gm.obs)])
    train_data = np.tile(data, (sample_size, 1, 1, 1))

    train_label = np.tile(gm.rho, (sample_size, 1, 1, 1))

    np.save(os.path.join("../data", 'simple_data_32'), train_data)
    np.save(os.path.join("../data", 'simple_label_32'), train_label)

main()
