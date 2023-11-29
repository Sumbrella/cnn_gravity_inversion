import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

G = 6.67

def cal_h(x, y, z, xi, eta, zeta, dx, dy, dz):
    X = [xi - 0.5 * dx - x, xi + 0.5 * dx - x]
    Y = [eta - 0.5 * dy - y, eta + 0.5 * dy - y]
    Z = [zeta - 0.5 * dz - z, zeta + 0.5 * dz - z]

    h = 0

    for i in range(2):
        for j in range(2):
            for k in range(2):
                R = np.sqrt(X[i]**2 + Y[i]**2 + Z[k]**2)
                mu = (-1) ** (i + j + k + 3)
                tmp = G * mu * (
                    Z[k] * np.arctan(X[i] * Y[j] / Z[k] / R) -
                    X[i] * np.log(R + Y[j]) -
                    Y[j] * np.log(R + X[i])
                )
                h = h + tmp
    return h


def main():
    nx = 100
    ny = 100
    nz = 50

    xmax = 800
    ymax = 800
    zmax = 600

    X = np.linspace(-xmax, xmax, nx)
    Y = np.linspace(-ymax, ymax, ny)
    Z = np.linspace(0, zmax, nz)

    XO = np.linspace(-xmax, xmax, nx)
    YO = np.linspace(-ymax, ymax, ny)

    dx = X[1] - X[0]
    dy = Y[1] - Y[0]
    dz = Z[1] - Z[0]

    z = 0

    obs = np.zeros((nx, ny))
    rho = np.ones((nx, ny, nz))

    XX, YY, ZZ = np.meshgrid(X, Y, Z)
    rho[np.sqrt(XX**2 + YY**2 + (ZZ - 100)**2) <= 100] = 5

    rho[np.sqrt((XX - 400) ** 2 + (YY + 400) ** 2 + (ZZ - 100) ** 2) <= 100] = 5

    with tqdm(total=nz) as tbar:
        for k in range(nz):
            t = np.zeros((2 * nx, 2 * ny))
            h = np.zeros((nx, ny))
            for i in range(nx):
                for j in range(ny):
                    h[i, j] = cal_h(XO[0], YO[0], z, X[i], Y[j], Z[k], dx, dy, dz)

            t[nx:, ny:] = h
            t[1:nx, ny:] = h[-1:0:-1, :]
            t[nx:, 1:ny] = h[:, -1:0:-1]
            t[1:nx, 1:ny] = h[-1:0:-1, -1:0:-1]

            c = np.zeros_like(t)
            c[:nx, :ny] = t[nx:, ny:]
            c[nx:, ny:] = t[:nx, :ny]
            c[:nx, ny:] = t[nx:, :ny]
            c[nx:, :ny] = t[:nx, ny:]

            g = np.zeros_like(c)
            g[:nx, :ny] = rho[:, :, k]

            cc = np.fft.fft2(c)
            gg = np.fft.fft2(g)
            ff = cc * gg
            f = np.fft.ifft2(ff)

            gz = np.abs(f[:nx, :ny])
            obs += gz
            tbar.update(1)
    plt.imshow(obs)
    # plt.xticks(ticks=np.linspace(nx), labels=XO)
    # plt.yticks(ticks=np.linspace(ny), labels=YO)
    plt.show()

main()