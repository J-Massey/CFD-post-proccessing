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
interest = 'rms'
label = r'\sqrt{u^{\prime}^{2}+v^{\prime}^{2}}'

for idx, fn in enumerate(files):
    fos = (postproc.io.unpack_flex_forces(os.path.join(data_root, fn, force_file), names))
    forces_dic = dict(zip(names, fos))
    t_min = min(forces_dic['t'])
    t_max = max(forces_dic['t'])
    t = forces_dic['t']
#%%
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
# %%
importlib.reload(postproc.plotter)
importlib.reload(postproc.frequency_spectra)

ti = t[0:len(instant_tke[0])]

# for idx, loop in enumerate(instant_tke):
idx = 0
u = instant_tke[idx]
print(len(instant_tke[idx]), len(ti))
# %%
importlib.reload(postproc.plotter)
importlib.reload(postproc.frequency_spectra)
label = r"$\sqrt{u^{\prime}^{2}+v^{\prime}^{2}}$"
label = "FUCKOFF"

postproc.plotter.fully_defined_plot(ti, u,
                                    y_label=label,
                                    x_label=r"$ t/D $",
                                    colour='red',
                                    title=f"Time series",
                                    file=data_root + f"figures/time_series_{interest}.png")

#%%

importlib.reload(postproc.frequency_spectra)

cycles = 20
late = 0
n = int(np.floor((np.max(ti[ti > late]) - np.min(ti[ti > late])) / cycles))
criteria = postproc.frequency_spectra.FreqConv(t=ti[ti > late], u=u[ti > late], n=n, OL=0.5)
labelled_uks_e, fs_e, area = criteria.ensemble()

postproc.plotter.plotLogLogTimeSpectra_list_cascade(data_root + f'figures/ensembling_PSD_{interest}.png',
                                                    labelled_uks_e, fs_e,
                                                    title=f"Splits, $n={n}$",
                                                    ylabel=r'PS ' + label)

normed_error, window_t = criteria.f_conv(cycles)
f, uk = criteria.welch()

plt.style.use(['science', 'grid'])
fig, ax = plt.subplots(figsize=(7, 5))
ax.set_title(f"Ratio of RMS error to spectra integral")

ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)

# Edit frame, labels and legend
ax.set_xlabel(r"$t/D$")
ax.set_ylabel(r"$\int \sqrt(\overline{s_{0,n}} - \overline{s_{0,n+1}})^2 df/ \int \overline{s_{0,n+1}}$")

# ax.plot(f, uk, c='r')
ax.plot(window_t, normed_error, c='r')
plt.savefig(data_root + f"figures/ensemble_error_{interest}.png", bbox_inches='tight', dpi=600, transparent=False)
plt.close()


























