import sys
sys.path.append("..")

import numpy as np
import random
import os
from utils.gravity_model import GravityModel


def main():
    nx = 128 // 2
    ny = 128 // 2
    nz = 64 // 2

    xmax = 400
    ymax = 400
    zmax = 400

    obs_z = -5

    gm = GravityModel(-xmax, xmax, -ymax, ymax, zmax, nx, ny, nz, obs_z)
    # sample_size = 5000

    train_data = np.zeros((5, nx, ny, 3))
    train_label = np.zeros((5, nx, ny, nz))

    # Rec
    gm.rho = np.zeros(gm.rho.shape)
    gm.rho[(gm.X > -75) & (gm.X < 75) & (gm.Y > -75) & (gm.Y < 75) & (gm.Z > 50) & (gm.Z < 150)] = 1
    gm.fast_forward()
    train_data[0] = np.dstack([gm.obs, *np.gradient(gm.obs)])
    train_label[0] = gm.rho.copy()

    # two Rec vertical
    gm.rho = np.zeros(gm.rho.shape)
    gm.rho[(gm.X > -75) & (gm.X < 75) & (gm.Y > -75) & (gm.Y < 75) & (gm.Z > 50) & (gm.Z < 100)] = 1
    gm.rho[(gm.X > -75) & (gm.X < 75) & (gm.Y > -75) & (gm.Y < 75) & (gm.Z > 150) & (gm.Z < 200)] = 1
    gm.fast_forward()
    train_data[1] = np.dstack([gm.obs, *np.gradient(gm.obs)])
    train_label[1] = gm.rho.copy()

    # two Rec hor
    gm.rho = np.zeros(gm.rho.shape)
    gm.rho[(gm.X > -150) & (gm.X < -50) & (gm.Y > -50) & (gm.Y < 50) & (gm.Z > 50) & (gm.Z < 100)] = 1
    gm.rho[(gm.X > 50) & (gm.X < 150) & (gm.Y > -50) & (gm.Y < 50) & (gm.Z > 50) & (gm.Z < 100)] = 1
    gm.fast_forward()
    train_data[2] = np.dstack([gm.obs, *np.gradient(gm.obs)])
    train_label[2] = gm.rho.copy()

    # dike
    gm.rho = np.zeros(gm.rho.shape)
    gm.rho[(gm.X > -50) & (gm.X < 50) & (gm.Y > -50) & (gm.Y < 50) & (gm.Z > 50) & (gm.Z <= 75)] = 1
    gm.rho[(gm.X > 0) & (gm.X < 100) & (gm.Y > -50) & (gm.Y < 50) & (gm.Z > 75) & (gm.Z <= 100)] = 1
    gm.rho[(gm.X > 50) & (gm.X < 150) & (gm.Y > -50) & (gm.Y < 50) & (gm.Z > 100) & (gm.Z <= 125)] = 1
    gm.rho[(gm.X > 100) & (gm.X < 200) & (gm.Y > -50) & (gm.Y < 50) & (gm.Z > 125) & (gm.Z <= 150)] = 1
    gm.fast_forward()
    train_data[3] = np.dstack([gm.obs, *np.gradient(gm.obs)])
    train_label[3] = gm.rho.copy()

    np.save(os.path.join("../data", 'test_model_data'), train_data)
    np.save(os.path.join("../data", 'test_model_label'), train_label)

main()
