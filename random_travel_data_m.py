import numpy as np
import matplotlib.pyplot as plt
import pygimli as pg
from pygimli.viewer import pv
from pygimli.physics.gravimetry import GravityModelling

nx = 64
ny = 64
nz = 32

x = np.linspace(-400, 400, nx)
y = np.linspace(-400, 400, ny)
z = np.linspace(0., 400, nz)
grid = pg.createGrid(x, y, z)

v = np.zeros((len(z)-1, len(y)-1, len(x)-1))
for i in range(7):
    v[1+i, 11-i:16-i, 7:13] = 1

grid["synth"] = v.ravel()

xx, yy = np.meshgrid(x, y)
points = np.column_stack((xx.ravel(), yy.ravel(), -np.ones(np.prod(xx.shape))))
fop = GravityModelling(grid, points)
data = fop.response(grid["synth"])
noise_level = 0.1
data += np.random.randn(len(data)) * noise_level
plt.contourf(yy, xx, np.reshape(data, xx.shape))
plt.colorbar()
plt.show()