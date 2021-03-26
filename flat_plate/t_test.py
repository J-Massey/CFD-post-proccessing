#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: J. Massey
@description: profiles from DNS
@contact: jmom1n15@soton.ac.uk
"""

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
from collections import deque


colours = sns.color_palette("husl", 14)

D = 64
U = 1
data_root = '/home/masseyjmo/Workspace/Lotus/projects/flat_plate/t_test/'
file = '16_8'
force_file = '3D/fort.9'
names = ['t', 'dt', 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'v2x', 'v2y', 'v2z']
interest = 'vy'

# Use dict if only accessing columns, use pandas if rows or loads of columns
labels = [r'$ c(16,8, 0.25) $']

fos = (io.unpack_flex_forces(os.path.join(data_root, file, force_file), names))
forces_dic = dict(zip(names, fos))

t, u = forces_dic['t'] / D, forces_dic[interest] * np.sin(12*np.pi/180)
postproc.plotter.fully_defined_plot(t, u,
                                    y_label=r"$ C_{L_{p}} $",
                                    x_label=r"$ t/D $",
                                    colour='red',
                                    title=f"Time series",
                                    file=data_root + f"figures/time_series_{interest}.png")

late = 0
cycles = 20
n = int(np.floor((np.max(t[t > late]) - np.min(t[t > late])) / cycles))
uk, f = postproc.frequency_spectra.freq_spectra_Welch(t[t > late], u[t > late], n=n)

plt.style.use(['science', 'grid'])
fig, ax = plt.subplots(figsize=(7, 5))
ax.set_title(f"Splits, $n={n}$", )

ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)

# Edit frame, labels and legend
ax.set_xlabel(r"$fc/U$")
ax.set_ylabel(r"$ PS (C_{L_{p}}) $")

ax.loglog(uk, f, c='r')
plt.savefig(data_root + f"figures/welch_fspec.png", bbox_inches='tight', transparent=False)


labelled_uks_e, fs_e = postproc.frequency_spectra \
    .freq_spectra_ensembling(t[t > late], u[t > late], n=n, OL=0.5, lowpass=True)

postproc.plotter.plotLogLogTimeSpectra_list_cascade(data_root + f'figures/ensembling_PSD_{interest}.png',
                                                    labelled_uks_e, fs_e,
                                                    title=f"Splits, $n={n}$",
                                                    ylabel=r"$ PS (\sqrt{{r^{\prime}}^{2} + {\theta^{\prime}}^{2}}) $")

uks_e = deque()
for loop1 in labelled_uks_e:
    uks_e.append(loop1[1])

uks_e = np.array(uks_e)

off1 = uks_e[1:]
off2 = uks_e[:-1]

diff_rms = np.sqrt((off1 - off2)**2)

l2 = np.trapz(diff_rms, fs_e[1:])
area = np.trapz(uks_e, fs_e)
normed_diff = l2/area[1:]

window_t = (np.linspace(min(t[t > late]) + cycles, max(t[t > late]), len(l2)))

plt.style.use(['science', 'grid'])
fig, ax = plt.subplots(figsize=(7, 5))
ax.set_title(f"Ratio of RMS error to spectra integral")

ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)

# Edit frame, labels and legend
ax.set_xlabel(r"$t/D$")
ax.set_ylabel(r"$\int \sqrt(\overline{s_{0,n}} - \overline{s_{0,n+1}})^2 df/ \int \overline{s_{0,n+1}} df$")

ax.plot(window_t, normed_diff, c='r', marker='+')
plt.savefig(data_root + f"figures/ensemble_error_{interest}.png", bbox_inches='tight', dpi=600, transparent=False)
