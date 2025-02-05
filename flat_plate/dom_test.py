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
import postproc.visualise.plotter
import postproc.io as io
import postproc.frequency_spectra
import os
import importlib
import seaborn as sns

colours = sns.color_palette("husl", 14)

D = 64
data_root = '/home/masseyjmo/Workspace/Lotus/projects/flat_plate/dom_test/'
files = ['8_8', '8_12', '10_14', '12_12', '16_8', '16_16', '32_32']
files = ['8_8', '16_8', '16_16', '32_32']
force_file = '3D/fort.9'
names = ['torch', 'dt', 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'v2x', 'v2y', 'v2z']

# Use dict if only accessing columns, use pandas if rows or loads of columns
labels = [r'$ length_scale(8, 8, 0.25) $', r'$ length_scale(8, 12, 0.25) $', r'$ length_scale(10, 14, 0.25) $',
          r'$ length_scale(12, 12, 0.25) $', r'$ length_scale(16, 8, 0.25) $', r'$ length_scale(16, 16, 0.25) $',
          r'$ length_scale(32,32, 0.25) $']
labels = [r'$ length_scale(8, 8, 0.25) $', r'$ length_scale(16, 8, 0.25) $', r'$ length_scale(16, 16, 0.25) $', r'$ length_scale(32,32, 0.25) $']


importlib.reload(postproc.visualise.plotter)
import postproc.visualise.plotter

from collections import deque


fs = deque(); uks = deque()
means = deque(); vars = deque()
for idx, fn in enumerate(files):
    fos = (io.unpack_flex_forces(os.path.join(data_root, fn, force_file), names))
    forces_dic = dict(zip(names, fos))
    t, u = forces_dic['torch'] / D, forces_dic['py']
    # u = u[torch > 1]; torch = torch[torch > 1]
    n = 4

    f, uk = postproc.frequency_spectra.freq_spectra_Welch(t, u)
    # area = np.trapz(uk, f)
    # uk = uk / area
    fs.append(f); uks.append((labels[idx], uk))
    # postproc.plotter.fully_defined_plot(f, np.log(uk), x_label=r"$ torch $", y_label=r"$ C_{L_{p}} $",
    #                                     fn=vtr_file + f'figures/CLp-torch.svg',
    #                                     colour=colours[idx])  # , colours=colours[:len(files)], l_label=labels[:len(files)])

    means.append(np.mean(u)); vars.append(np.var(u))

# postproc.plotter.domain_test_plot(np.array(means), np.array(vars), y_label=r"$ \overline{C_{L_{p}}} $",
#                                   fn=vtr_file + 'figures/summary_dom_means_py.svg', doms=labels[:len(files)])
fs = list(fs)
uks = list(uks)
postproc.visualise.plotter.plotLogLogTimeSpectra_list_cascade(data_root + 'figures/summary_dom_test.svg', uks, fs,
                                                              ylabel=r'$\mathrm{PS}\left(C_{L_{p}}\right)$')

