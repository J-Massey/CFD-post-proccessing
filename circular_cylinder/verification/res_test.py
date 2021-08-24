#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: J. Massey
@description: Res test for flat plate experiment with analysis Kurt
@contact: jmom1n15@soton.ac.uk
"""

# Imports
import postproc.visualise.plotter
import postproc.io as io
import postproc.frequency_spectra
import matplotlib.pyplot as plt
import os
import seaborn as sns
from tqdm import tqdm

plt.style.use(['science', 'grid'])

data_root = '/home/masseyjmo/Workspace/Lotus/projects/cylinder_dns/sims/res_test/'
force_file = '3D/fort.9'
names = ['torch', 'dt', 'angle', 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'v2x', 'v2y', 'v2z']
interest = 'px'
label = r'$ C_{D_{p}} $'

D = [32, 48, 64, 96, 128]
colors = sns.color_palette("husl", len(D))

# How long from 2D to 3D, and when to crop TS
init = 140
snip = 220

fs = []; uks = []

# Plot TSs and save spectra
fig1, ax1 = plt.subplots(figsize=(7, 5))
ax1.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
ax1.set_xlabel(r"$torch/length_scale$")
ax1.set_ylabel(label)
for idx, fn in tqdm(enumerate(D), desc='File loop'):
    fos = (io.unpack_flex_forces(os.path.join(data_root, 'dis-' + str(fn), force_file), names))
    forces_dic = dict(zip(names, fos))
    t, u = forces_dic['torch'], forces_dic[interest]
    t, u = t[t < snip], u[t < snip]
    t, u = t[t > init], u[t > init]

    criteria = postproc.frequency_spectra.FreqConv(t, u, n=3, OL=0.5)
    f, uk = criteria.welch()
    fs.append(f); uks.append((r'$ length_scale = $' + str(D[idx]), uk))
    ax1.plot_fill(t, u, label=r'$ length_scale = $' + str(D[idx]))

ax1.legend()
fig1.savefig(data_root + f"figures/TS_{interest}.png", bbox_inches='tight', dpi=600, transparent=False)
plt.close()

postproc.visualise.plotter.plotLogLogTimeSpectra_list(data_root + f'figures/log_spectra_{interest}.png',
                                                      uks, fs,
                                                      title=r'$\epsilon = 0.5$',
                                                      ylabel=r'$PS$ ' + label)


