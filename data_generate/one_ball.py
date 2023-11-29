import sys
sys.path.append("..")

import numpy as np
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
    sample_size = 500
    train_data = np.zeros((sample_size, nx, ny, 3))
    train_label = np.zeros((sample_size, nx, ny, nz))
    for i in range(sample_size):
        print("{} / {}".format(i + 1, sample_size))
        gm.rho = np.zeros(gm.rho.shape)

        x, y = (np.random.random(2) * 0.9 - 0.5) * 400
        z = (np.random.random() + 0.2) * 200
        max_value = np.abs([x, y, z]).max()
        radius = (np.random.random() + 0.1) * 150
        # radius = min((xmax - max_value) * 0.8, radius)

        gm.rho[np.sqrt((gm.X - x)**2 + (gm.Y - y)**2 + (gm.Z - z)**2) <= radius] = (np.random.random()) * 0.9
        # gm.rho[((gm.X - 100)**2 + gm.Y ** 2 + (gm.Z - 100) ** 2) <= 50] = 5
        # gm.fast_forward()
        # gm.save_npy('./data/', '{}'.format(i))
        gm.fast_forward()
        data = np.dstack([gm.obs, *np.gradient(gm.obs)])
        train_data[i] = data.copy()
        train_label[i] = gm.rho.copy()
    np.save(os.path.join("../data", '500_one_ball_data_32'), train_data)
    np.save(os.path.join("../data", '500_one_ball_label_32'), train_label)


if __name__ == '__main__':
    main()