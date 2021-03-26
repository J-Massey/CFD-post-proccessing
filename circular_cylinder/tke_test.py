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
from postproc.profile_convergence import ProfileDataset
import matplotlib
import os
import importlib
from collections import deque
import matplotlib.pyplot as plt
import torch
import seaborn as sns

matplotlib.use('TkAgg')

colors = sns.color_palette("husl", 14)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

t_comparisons = 'd-64/3D/'
data_root = '/home/masseyjmo/Workspace/Lotus/projects/cf_lotus/ml/real_profiles/DNS/sims/t_test/'

files = ['d-64']
force_file = '3D/fort.9'
names = ['t', 'dt', 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'v2x', 'v2y', 'v2z']

for idx, fn in enumerate(files):
    fos = (postproc.io.unpack_flex_forces(os.path.join(data_root, fn, force_file), names))
    forces_dic = dict(zip(names, fos))
    t_min = min(forces_dic['t'])
    t_max = max(forces_dic['t'])
    t = forces_dic['t']

importlib.reload(postproc.profile_convergence)

rs = ProfileDataset(data_root + t_comparisons).bl_poincare_limit(single_point=True,
                                                                 position=0.1, length_scale=64,
                                                                 print_res=128, print_len=3)[0]

azis = ProfileDataset(data_root + t_comparisons).bl_poincare_limit(single_point=True,
                                                                   position=0.1, length_scale=64,
                                                                   print_res=128, print_len=3)[1]
angles = ProfileDataset(data_root + t_comparisons).angles

rs = torch.tensor(rs, device=device)
azis = torch.tensor(azis, device=device)

r_dash = torch.mean(rs, dim=1).unsqueeze(1).repeat_interleave(rs.size()[1], dim=1)
azi_dash = azis - torch.mean(azis, dim=1).unsqueeze(1).repeat_interleave(azis.size()[1], dim=1)
instant_tke = (0.5 * (r_dash ** 2 + azi_dash ** 2)).cpu().numpy()


for idx, loop in enumerate(instant_tke):

    ti = t[0:len(loop)]
    late = 0
    cycles = 10

    n = int(np.floor((np.max(ti[ti > late]) - np.min(ti[ti > late])) / cycles))
    uk, f = postproc.frequency_spectra.freq_spectra_Welch(ti[ti > late], loop[ti > late], n=n)

    plt.style.use(['science', 'grid'])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(f"Splits, $n={n}$ at $ {angles[idx]:.2f}r$", )

    ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)

    # Edit frame, labels and legend
    ax.set_xlabel(r"$fD/U$")
    ax.set_ylabel(r"$ PS (\sqrt{{r^{\prime}}^{2} + {\theta^{\prime}}^{2}}) $")

    ax.loglog(uk, f, c='r')
    plt.savefig(data_root + f"figures/fspec_welch_{idx}.png", dpi=600, bbox_inches='tight', transparent=False)

    labelled_uks_e, fs_e = postproc.frequency_spectra \
        .freq_spectra_ensembling(ti[ti > late], loop[ti > late], n=n, OL=0.5)

    postproc.plotter.plotLogLogTimeSpectra_list_cascade(data_root + f'figures/ensembling_PSD_{idx}.png',
                                                        labelled_uks_e, fs_e,
                                                        title=f"Splits, $n={n}$ at $ {angles[idx]:.2f}r$",
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

    normed_error = (l2/area[:1])

    window_t = (np.linspace(min(t[t > late]) + cycles, max(t[t > late]), len(l2)))

    plt.style.use(['science', 'grid'])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(f"Ratio of RMS error to spectra integral")

    ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)

    # Edit frame, labels and legend
    ax.set_xlabel(r"$t/D$")
    ax.set_ylabel(r"$\int \sqrt(\overline{s_{0,n}} - \overline{s_{0,n+1}})^2 df/ \int \overline{s_{0,n+1}} df$")

    ax.plot(window_t, normed_error, c='r', marker='+')
    plt.savefig(data_root + f"figures/ensemble_error.png", bbox_inches='tight', dpi=600, transparent=False)
    plt.close()

