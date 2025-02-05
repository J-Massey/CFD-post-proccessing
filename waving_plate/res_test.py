#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: J. Massey
@description: Res test for flat plate experiment with analysis Kurt
@contact: jmom1n15@soton.ac.uk
"""

# Imports
import numpy as np
import postproc.visualise.plotter
import postproc.io as io
import postproc.frequency_spectra
import matplotlib.pyplot as plt
import os
import seaborn as sns
from tqdm import tqdm

plt.style.use(['science', 'grid'])

data_root = '/home/masseyjmo/Workspace/Lotus/projects/waving_plate/smooth'
force_file = '3D/fort.9'
names = ['t', 'dt', 'px', 'py', 'pz']
interest = 'p'
label = r'$ C_{D} $'
tit = r'$\epsilon = 0.5$'

c = [192, 256]
theta = np.radians(2)
colors = sns.color_palette("husl", len(c))

# How long from 2D to 3D, and when to crop TS
init = 40
snip = 200

# Write out labels so they're interpretable by latex
labels = [r'$ c='+str(i)+' $' for i in c]

fs, uks_labelled, uks = [], [], []

# Plot TSs and save spectra
fig1, ax1 = plt.subplots(figsize=(7, 5))
ax1.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
ax1.set_xlabel(r'$ t/c $')
ax1.set_ylabel(label)
for idx, fn in tqdm(enumerate(c), desc='File loop'):
    fos = (io.unpack_flex_forces(os.path.join(data_root, str(fn), force_file), names))
    forces_dic = dict(zip(names, fos))
    t, u = forces_dic['t'] / c[idx], np.array(forces_dic[interest + 'x'])

    # # Transform the forces into the correct plane
    # rot = np.array((np.cos(theta), -np.sin(theta)),
    #                (np.sin(theta), np.cos(theta)))
    # rot = np.array((np.sin(theta), np.cos(theta)))
    # # For the res test normalise pressure forces by frontal area
    # old_norm = (c[idx]/45.71*np.sin(np.radians(theta))*c[idx]*0.25*2)
    # new_norm = ((c[idx]/45.71*np.cos(np.radians(theta))  # Thickness
    #              + c[idx] * np.sin(np.radians(theta)))   # Body
    #             * c[idx]*0.25*2)                         # z
    #
    # u = rot.dot(u)*old_norm/new_norm
    t, u = t[t < snip], u[t < snip]
    t, u = t[t > init], u[t > init]
    sv = np.hstack((t, u))

    # Append the Welch spectra to a list in order to compare
    criteria = postproc.frequency_spectra.FreqConv(t, u, n=3, OL=0.5)
    f, uk = criteria.welch()
    fs.append(f)
    uks.append(uk)
    uks_labelled.append((labels[idx], uk))
    ax1.plot_fill(t, u, label=labels[idx])

ax1.legend()
fig1.savefig(os.path.join(data_root, 'figures/TS_'+interest+'.pdf'), bbox_inches='tight', dpi=30, transparent=False)
plt.close()

postproc.visualise.plotter.plot_fft(os.path.join(data_root, 'figures/spectra_' + interest + '.pdf'),
                                    uks, fs, xlim=[0, 1],
                                    l_label=labels, colours=colors, title=tit)

postproc.visualise.plotter.plotLogLogTimeSpectra_list(os.path.join(data_root, 'figures/log_spectra_' + interest + '.pdf'),
                                                      uks_labelled, fs,
                                                      title=tit,
                                                      ylabel=r'$PS$ ' + label)
