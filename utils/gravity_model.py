import os
import sys
sys.path.append(".")

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.io as sio

# from constants import *
G = 6.67e-11



class GravityModel:
    def __init__(self, xmin, xmax, ymin, ymax, zmax, nx, ny, nz, obs_z):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmax = zmax
        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.x = np.linspace(xmin, xmax, nx)
        self.y = np.linspace(ymin, ymax, ny)
        self.z = np.linspace(0   , zmax, nz)

        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dz = self.z[1] - self.z[0]

        self.obs_z = obs_z
        self.rho = np.zeros((nx, ny, nz))

        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z)

        self.obs = np.zeros((nx, ny))

    def fast_forward(self):
        self.obs = np.zeros((self.nx, self.ny))
        with tqdm(total=self.nz) as tbar:
            for k in range(self.nz):
                t = np.zeros((2 * self.nx, 2 * self.ny))
                h = np.zeros((self.nx, self.ny))
                for i in range(self.nx):
                    for j in range(self.ny):
                        h[i, j] = self.cal_h(self.x[0], self.y[0], self.x[i], self.y[j], self.z[k])

                t[self.nx:, self.ny:] = h
                t[1:self.nx, self.ny:] = h[-1:0:-1, :]
                t[self.nx:, 1:self.ny] = h[:, -1:0:-1]
                t[1:self.nx, 1:self.ny] = h[-1:0:-1, -1:0:-1]

                c = np.zeros_like(t)
                c[:self.nx, :self.ny] = t[self.nx:, self.ny:]
                c[self.nx:, self.ny:] = t[:self.nx, :self.ny]
                c[:self.nx, self.ny:] = t[self.nx:, :self.ny]
                c[self.nx:, :self.ny] = t[:self.nx, self.ny:]

                g = np.zeros_like(c)
                g[:self.nx, :self.ny] = self.rho[:, :, k]

                cc = np.fft.fft2(c)
                gg = np.fft.fft2(g)
                ff = cc * gg
                f = np.fft.ifft2(ff)

                gz = np.abs(f[:self.nx, :self.ny])
                self.obs += gz
                tbar.update(1)

    def cal_h(self, x, y, xi, eta, zeta):
        X = [xi - 0.5 * self.dx - x, xi + 0.5 * self.dx - x]
        Y = [eta - 0.5 * self.dy - y, eta + 0.5 * self.dy - y]
        Z = [zeta - 0.5 * self.dz - self.obs_z, zeta + 0.5 * self.dz - self.obs_z]
        h = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    R = np.sqrt(X[i] ** 2 + Y[j] ** 2 + Z[k] ** 2)
                    mu = (-1) ** (i + j + k + 3)
                    tmp = G * mu * (
                            Z[k] * np.arctan(X[i] * Y[j] / Z[k] / R) -
                            X[i] * np.log(R + Y[j]) -
                            Y[j] * np.log(R + X[i])
                    )
                    h = h + tmp
        return h

    def save_npy(self, path, name):
        np.save(os.path.join(path, '{}_gz'.format(name)), self.obs)
        np.save(os.path.join(path, '{}_dg'.format(name)), np.gradient(self.obs))
        np.save(os.path.join(path, '{}_rho'.format(name)), self.rho)

