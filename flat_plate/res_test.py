#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: J. Massey
@description: Res test for flat plate experiment with Melike Kurt
@contact: jmom1n15@soton.ac.uk
"""

# Imports
import numpy as np
import postproc.plotter
import postproc.io as io
import postproc.frequency_spectra
import matplotlib.pyplot as plt
import os
import seaborn as sns
from tqdm import tqdm

plt.style.use(['science', 'grid'])

data_root = '/home/masseyjmo/Workspace/Lotus/projects/flat_plate/thin_res_test/'
force_file = '3D/fort.9'
names = ['t', 'dt', 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'v2x', 'v2y', 'v2z']
interest = 'v2'
label = r'$ C_{L_{p}} $'

D = [64, 96, 128, 192, 256]
D = [64, 96, 128]
colors = sns.color_palette("husl", len(D))

# How long from 2D to 3D, and when to crop TS
init = 20
snip = 200


# Write out labels so they're interpretable by latex
labels = [r'$ c=64 $', r'$ c=96 $', r'$ c=128 $', r'c=192', r'$c=256 $']
labels = [r'$ c=64 $', r'$ c=96 $', r'$ c=128 $']

fs = []; uks = []

# Plot TSs and save spectra
fig1, ax1 = plt.subplots(figsize=(7, 5))
ax1.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
ax1.set_xlabel(r"$t/D$")
ax1.set_ylabel(label)
for idx, fn in tqdm(enumerate(D), desc='File loop'):
    fos = (io.unpack_flex_forces(os.path.join(data_root, str(fn), force_file), names))
    forces_dic = dict(zip(names, fos))
    t, u = forces_dic['t']/D[idx], np.array((forces_dic[interest+'x'], forces_dic[interest+'y']))
    # Transform the forces into the correct plane
    theta = np.radians(12)
    rot = np.array((np.cos(theta), -np.sin(theta)),
                   (np.sin(theta), np.cos(theta)))
    rot = np.array((np.sin(theta), np.cos(theta)))
    # This needs to be changed depending if we want the force in x or y
    u = rot.dot(u)
    t, u = t[t < snip], u[t < snip]
    t, u = t[t > init], u[t > init]

    # Append the Welch spectra to a list in order to compare
    criteria = postproc.frequency_spectra.FreqConv(t, u, n=3, OL=0.5)
    f, uk = criteria.welch()
    fs.append(f); uks.append((labels[idx], uk))
    ax1.plot(t, u, label=labels[idx])

ax1.legend()
fig1.savefig(data_root + f"figures/TS_{interest}.png", bbox_inches='tight', dpi=600, transparent=False)
plt.close()

postproc.plotter.plotLogLogTimeSpectra_list(data_root + f'figures/log_spectra_{interest}.png',
                                                    uks, fs,
                                                    title=r'$\epsilon = 0.5$',
                                                    ylabel=r'$PS$ ' + label)


