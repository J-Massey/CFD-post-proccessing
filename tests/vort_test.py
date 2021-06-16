# -*- coding: utf-8 -*-
"""
@author: B. Font Garcia
@description: package tests
@contact: b.fontgarcia@soton.ac.uk
"""

# Imports
import postproc.io as io
import postproc.calc as averages
import postproc.plotter as plotter
import matplotlib.pyplot as plt
import numpy as np
from numpy import ma
from matplotlib import ticker, cm
import seaborn as sns

plt.style.use(['science', 'grid'])
D = 48
fname = '/home/masseyjmo/Workspace/Lotus/projects/cylinder_dns/sims/eps_test/eps-2/3D/datp/spTAv.3.pvtr'
data = io.read_vtr(fname)
u, v, w = data[0]
u, v = np.squeeze(u), np.squeeze(v)
vort = averages.vortZ(u, v)
p = np.squeeze(data[1])

x, y, z = data[2]
print(np.shape(x))
# x = x[-2 * length_scale: 2 * length_scale]
# Y = Y[-2 * length_scale: 2 * length_scale]
X, Y = np.meshgrid(x/D, y/D)

fig, ax = plt.subplots(figsize=(7, 5))
cs = ax.contourf(X, Y, np.transpose(u), 20, cmap=sns.color_palette("icefire", as_cmap=True))
cbar = fig.colorbar(cs)
plt.xlim(-1, 2)
plt.ylim(-1, 1)
ax.set_aspect(1)


plt.title(r'$ \overline{u} $')

plt.savefig('u.png', dpi=700)

# plotter.plot_2D(np.transpose(vort), cmap='bwr', lvls=100, lim=[-0.15, 0.15], fn='vort.pdf', x=[xmin,xmax], Y=[ymin,ymax])


# fname = '/home/masseyjmo/Workspace/Lotus/projects/cylinder_dns/sims/eps_test/eps-1/3D/datp/flu2d.3.pvtr'
# dat = io.read_vtr(fname)
# u, v, w = dat[0]

