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
    sample_size = 5000

    train_data = np.zeros((sample_size, nx, ny, 3))
    train_label = np.zeros((sample_size, nx, ny, nz))

    for i in range(sample_size):
        print("{} / {}".format(i + 1, sample_size))
        gm.rho = np.zeros(gm.rho.shape, dtype=int)

        for _ in range(2):
            sx = int(np.random.random() * gm.nx)
            sy = int(np.random.random() * gm.ny)
            sz = int(np.random.random() * gm.nz)
            # target_rho = np.random.random() * 0.8 + 0.1
            target_rho = 1

            for step in range(np.random.randint(80, 160)):
                # dx, dy, dz = np.random.random_sample()
                dx, dy, dz = random.choice([
                    [0, 0, -1],
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, -1, 0],
                    [1, 0, 0],
                    [-1, 0, 0]
                ])

                sx = sx + dx if 3 <= sx + dx < nx - 3 else sx
                sy = sy + dy if 3 <= sy + dy < ny - 3 else sy
                sz = sz + dz if 3 <= sz + dz < nz - 3 else sz

                gm.rho[sx:sx+2, sy:sy+2, sz:sz+2] = target_rho

        gm.fast_forward()
        data = np.dstack([gm.obs, *np.gradient(gm.obs)])
        label = gm.rho
        train_data[i] = data
        train_label[i] = label
        gm.obs = gm.obs * 0

    np.save(os.path.join("../data", '5000_01_travel_data_32'), train_data)
    np.save(os.path.join("../data", '5000_01_travel_label_32'), train_label)

main()
