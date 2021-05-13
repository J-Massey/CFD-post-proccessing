# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: J. Massey
@description: profiles from DNS
@contact: jmom1n15@soton.ac.uk
"""

# Imports
import numpy as np
import postproc.plotter
import postproc.io
import postproc.frequency_spectra
from postproc.boundary_layer import ProfileDataset
import os
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from tqdm import tqdm

plt.style.use(['science', 'grid'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_root = '/home/masseyjmo/Workspace/Lotus/projects/cylinder_dns/sims/res_test/'

# What are the folder names of the simulations
files = ['d-32', 'd-48', 'd-64', 'd-96', 'd-128']
# files = ['d-96', 'd-128']

# Length scales we are comparing
D = [32, 48, 64, 96, 128]
# length_scale = [96, 128]
colors = sns.color_palette("husl", len(D))

# Path to  where we printed the mean forces
force_file = '3D/fort.9'
names = ['t', 'dt', 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'v2x', 'v2y', 'v2z']

# What's the name and label of the thing we're interested in?
interest = 'rms'
label = r'\sqrt{u^{\prime}^{2}+v^{\prime}^{2}}'

# Hold Force spectra for plotting later, angle is the number of profiles around the circle we printed
n_angles = 5
angled_dic_cd = [[] for _ in range(n_angles)]
angled_dic_cl = [[] for _ in range(n_angles)]

fig1, ax1 = plt.subplots(figsize=(7, 5))
ax1.set_xlabel(r"$t/length_scale$")
ax1.set_ylabel(r"$C_{D_{f}}$")

for idx, fn in tqdm(enumerate(files), desc='File loop', ascii=True):
    # Unpack mean forces
    fos = (postproc.io.unpack_flex_forces(os.path.join(data_root, fn, force_file), names))
    forces_dic = dict(zip(names, fos))
    t_min = min(forces_dic['t'])
    t_max = max(forces_dic['t'])
    t = forces_dic['t']

    # Wasn't consistent with print res so correct for this
    if D[idx] <= 64:
        res = 128
    else:
        res = 256
    # Get the profiles
    data = ProfileDataset(os.path.join(data_root, fn, '3D'), True)
    angles = data.angles

    # Get the O(1) F-length_scale
    cd, cl = data.fd_1(length_scale=D[idx], print_res=res, print_len=3)
    ax1.plot(t, cd[2], color=colors[idx], label=f"length_scale = ${D[idx]}$")

    for idx1, (loop1, loop2) in tqdm(enumerate(zip(cd, cl)), desc='Loooop', ascii=True):
        # Restructure so that the angles can be directly compared at different resolutions
        angled_dic_cd[idx1].append(loop1)
        angled_dic_cl[idx1].append(loop2)
        # Could compare spectra, better convergence idea

plt.legend()
fig1.savefig(data_root + f"figures/time_series_cf.png", bbox_inches='tight', dpi=600, transparent=False)
plt.close(fig1)

# # Plot spectra
# for idx, (loop_f, loop_uk) in enumerate(zip(angled_dic_cd, angled_dic_cl)):
#     postproc.plotter.plotLogLogTimeSpectra_list(vtr_file + f'figures/log_spectra_{idx}.png',
#                                                 loop_uk, loop_f,
#                                                 tit=r'$ \theta$' + f'$ = {round(angles[idx], 2)}^r $',
#                                                 ylabel=r'$PS(\sqrt{u^{\prime^{2}}+v^{\prime^{2}}})$')
