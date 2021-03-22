#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: J. Massey
@description: profiles from DNS
@contact: jmom1n15@soton.ac.uk
"""

# Imports
# Imports
import numpy as np
import postproc.plotter
import postproc.io as io
import postproc.frequency_spectra
import matplotlib.pyplot as plt
import os
import torch
import importlib
import seaborn as sns

colours = sns.color_palette("husl", 14)

D = 64
U = 1
data_root = '/home/masseyjmo/Workspace/Lotus/projects/flat_plate/t_test/'
file = '32_32'
force_file = '3D/fort.9'
names = ['t', 'dt', 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'v2x', 'v2y', 'v2z']

# Use dict if only accessing columns, use pandas if rows or loads of columns
labels = [r'$ c(32,32, 0.25) $']


from collections import deque

fos = (io.unpack_flex_forces(os.path.join(data_root, file, force_file), names))
forces_dic = dict(zip(names, fos))
#%%
importlib.reload(postproc.plotter)
importlib.reload(postproc.frequency_spectra)
import postproc.plotter
t, u = forces_dic['t'] / D, forces_dic['py']
# u = u[t > 10.8]; t = t[t > 10.8]
postproc.plotter.fully_defined_plot(t, u,
                                    y_label=r"$ C_{L_{p}} $",
                                    x_label=r"$ t/D $",
                                    colour='red',
                                    title=f"Fourier bin variances",
                                    file=data_root + f"figures/t_test/time_series.svg")
n = 7
uks, fs, means, variances = postproc.frequency_spectra.freq_spectra_convergence(t, u, n=n, OL=0.5)

postproc.plotter.plotTimeSpectra_list(data_root + f'figures/t_test/time_spec_convergence.svg', uks, fs,
                                      title=f"Splits, $n={n}$",
                                      xlim=[0, 5])
plt.close()

you, fre = postproc.frequency_spectra.freq_spectra(t, u, windowing=False)


postproc.plotter.fully_defined_plot(you[1:], (fre)[1:],
                                    y_label=r"$ PSD $",
                                    x_label=r"$ f/D $",
                                    colour='red',
                                    title=f"Fourier bin variances",
                                    file=data_root + f"figures/t_test/spec_t_test.svg")

# postproc.plotter.fully_defined_plot(t, variances,
#                                     y_label=r"$ Var(s_{n}) $",
#                                     x_label=r"$ t/D $",
#                                     colour='red',
#                                     title=f"Fourier bin variances",
#                                     file=data_root + f"figures/t_test/fourier_var_t_test.svg", marker='+')
#
# postproc.plotter.fully_defined_plot(t, means,
#                                     y_label=r"$ \overline{(s_{n})} $",
#                                     x_label=r"$ t/D $",
#                                     colour='red',
#                                     title=f"Fourier bin means",
#                                     file=data_root + f"figures/t_test/fourier_mean_t_test.svg", marker='+')

plt.close()

