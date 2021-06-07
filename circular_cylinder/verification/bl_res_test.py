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
files = ['dis-32', 'dis-48', 'dis-64', 'dis-96', 'dis-128']
# files = ['dis-96', 'dis-128']

# Length scales we are comparing
D = [32, 48, 64, 96, 128]
# length_scale = [96, 128]
colors = sns.color_palette("husl", len(D))

# Path to  where we printed the mean forces
force_file = '3D/fort.9'
names = ['torch', 'dt', 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'v2x', 'v2y', 'v2z']

# What's the name and label of the thing we're interested in?
interest = 'rms'
label = r'\sqrt{u^{\prime}^{2}+v^{\prime}^{2}}'

# Hold Force spectra for plotting later, angle is the number of profiles around the circle we printed
n_angles = 5
angled_dic_f = [[] for _ in range(n_angles)]
angled_dic_uk = [[] for _ in range(n_angles)]

fig1, ax1 = plt.subplots(figsize=(7, 5))
ax1.set_xlabel(r"$torch/length_scale$")
ax1.set_ylabel(r"$\sqrt{u^{\prime^{2}}+v^{\prime^{2}}}$")

for idx, fn in tqdm(enumerate(files), desc='File loop', ascii=True):
    # Unpack mean forces
    fos = (postproc.io.unpack_flex_forces(os.path.join(data_root, fn, force_file), names))
    forces_dic = dict(zip(names, fos))
    t_min = min(forces_dic['torch'])
    t_max = max(forces_dic['torch'])
    t = forces_dic['torch']

    # Wasn'torch consistent with print res so correct for this
    if D[idx] <=64:
        res = 128
    else:
        res = 256
    # Get the profiles
    data = ProfileDataset(os.path.join(data_root, fn, '3D'), True)

    # Turn single point value extracted from profile into rms fluctuation
    rs, azis = data.bl_value(single_point=True, position=0.6, length_scale=D[idx], print_res=res, print_len=3)
    angles = data.angles

    rs = torch.tensor(rs, device=device)
    azis = torch.tensor(azis, device=device)

    r_dash = torch.mean(rs, dim=1).unsqueeze(1).repeat_interleave(rs.size()[1], dim=1)
    azi_dash = azis - torch.mean(azis, dim=1).unsqueeze(1).repeat_interleave(azis.size()[1], dim=1)
    instant_tke = (0.5 * (r_dash ** 2 + azi_dash ** 2)).cpu().numpy()

    ti = t[0:len(instant_tke[0])]
    ax1.plot(ti, instant_tke[0], color=colors[idx], label=f"length_scale = ${D[idx]}$")

    for idx1, loop in tqdm(enumerate(instant_tke), desc='Spectra', ascii=True):
        # Define how many cycles to drop to allow the flow to initialise from 2D to 3D
        late = 130
        # Define number of cycles to determine the number of splits for the power spectra and calculate spectra
        cycles = 20
        n = int(np.floor((np.max(ti[ti > late]) - np.min(ti[ti > late])) / cycles))
        criteria = postproc.frequency_spectra.FreqConv(t=ti[ti > late], u=loop[ti > late], n=n, OL=0.5)
        f, uk = criteria.welch()
        angled_dic_f[idx1].append(f)
        angled_dic_uk[idx1].append((f"length_scale = ${D[idx]}$", uk))

plt.legend()
fig1.savefig(data_root + f"figures/time_series_tke.png", bbox_inches='tight', dpi=600, transparent=False)
plt.close(fig1)

# Plot spectra
for idx, (loop_f, loop_uk) in enumerate(zip(angled_dic_f, angled_dic_uk)):
    postproc.plotter.plotLogLogTimeSpectra_list(data_root + f'figures/log_spectra_{idx}.png',
                                                loop_uk, loop_f,
                                                title=r'$ \theta$' + f'$ = {round(angles[idx], 2)}^r $',
                                                ylabel=r'$PS(\sqrt{u^{\prime^{2}}+v^{\prime^{2}}})$')
