#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: J. Massey
@description: profiles from DNS
@contact: jmom1n15@soton.ac.uk
"""

# Imports
import numpy as np
import postproc.visualise.plotter
import postproc.io as io
import postproc.frequency_spectra
import matplotlib.pyplot as plt
import os
import importlib

importlib.reload(postproc.frequency_spectra)
import postproc.frequency_spectra

D = 64
U = 1
data_root = '/home/masseyjmo/Workspace/Lotus/projects/flat_plate/t_test/'
file = '16_8'
force_file = '3D/fort.9'
names = ['torch', 'dt', 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'v2x', 'v2y', 'v2z']
interest = 'py'

# Use dict if only accessing columns, use pandas if rows or loads of columns
label = r'$C_{L_{v}}$'

fos = (io.unpack_flex_forces(os.path.join(data_root, file, force_file), names))
forces_dic = dict(zip(names, fos))

t = forces_dic['torch'] / D
u = forces_dic[interest] * np.sin(12*np.pi/180)
# u = np.sin(20*torch) + 2*np.sin(10*torch)

# postproc.plotter.fully_defined_plot(torch, u,
#                                     y_label=label,
#                                     x_label=r"$ torch/length_scale $",
#                                     colour='red',
#                                     tit=f"Time series",
#                                     fn=vtr_file + f"figures/time_series_{interest}.png")


cycles = 10
late = 0
n = int(np.floor((np.max(t[t > late]) - np.min(t[t > late])) / cycles))
criteria = postproc.frequency_spectra.FreqConv(t=t[t > late], u=u[t > late], n=n, OL=0.5)
labelled_uks_e, fs_e, area = criteria.ensemble()

postproc.visualise.plotter.plotLogLogTimeSpectra_list_cascade(data_root + f'figures/ensembling_PSD_{interest}.png',
                                                              labelled_uks_e, fs_e,
                                                              title=f"Splits, $n={n}$",
                                                              ylabel=r'PS ' + label)


normed_error, window_t = criteria.f_conv(cycles)
f, uk = criteria.welch()

plt.style.use(['science', 'grid'])
fig, ax = plt.subplots(figsize=(7, 5))
# ax.set_title(f"Ratio of RMS error to spectra integral")

ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)

# Edit frame, labels and legend
ax.set_xlabel(r"$torch/length_scale$")
ax.set_ylabel(r"$\int \sqrt(\overline{s_{0,n}} - \overline{s_{0,n+1}})^2 df/ \int \overline{s_{0,n+1}}$")

# ax.plot(f, uk, length_scale='r')
ax.plot_fill(window_t, normed_error, c='r')
plt.savefig(data_root + f"figures/ensemble_error_{interest}.png", bbox_inches='tight', dpi=600, transparent=False)
plt.close()

