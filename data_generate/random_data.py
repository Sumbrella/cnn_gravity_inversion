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
    sample_size = 50
    train_data = np.zeros((sample_size, nx, ny, 3))
    train_label = np.zeros((sample_size, nx, ny, nz))
    for i in range(sample_size):
        print("{} / {}".format(i + 1, sample_size))
        gm.rho = np.random.randn(*gm.rho.shape)
        # gm.rho[np.sqrt(gm.X**2 + gm.Y**2 + (gm.Z - 300)**2) <= 200] = 10
        # gm.rho[((gm.X - 100)**2 + gm.Y ** 2 + (gm.Z - 100) ** 2) <= 50] = 5
        # gm.fast_forward()
        # gm.save_npy('./data/', '{}'.format(i))
        gm.fast_forward()
        data = np.dstack([gm.obs, *np.gradient(gm.obs)])
        label = gm.rho
        train_data[i] = data
        train_label[i] = label
    np.save(os.path.join("../data", 'random_data_32'), train_data)
    np.save(os.path.join("../data", 'random_label_32'), train_label)

main()