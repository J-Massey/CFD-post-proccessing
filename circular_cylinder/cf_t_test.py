# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: J. Massey
@description: Check how long it took for a simulation to converge, or, whether it did...
@contact: jmom1n15@soton.ac.uk
"""

# Imports
import numpy as np
import postproc.io
import postproc.frequency_spectra
import postproc.boundary_layer_convergence
import os
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

plt.style.use(['science', 'grid'])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_root = '/home/masseyjmo/Workspace/Lotus/projects/cylinder_dns/sims/res_test/'

fn = 'd-32'
force_file = '3D/fort.9'
names = ['t', 'dt', 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'v2x', 'v2y', 'v2z']
interest = 'rms'

fos = (postproc.io.unpack_flex_forces(os.path.join(data_root, fn, force_file), names))
forces_dic = dict(zip(names, fos))
t_min = min(forces_dic['t'])
t_max = max(forces_dic['t'])
t = forces_dic['t']

data = postproc.boundary_layer_convergence.ProfileDataset(os.path.join(data_root, fn, '3D'), True)
cd, cl = data.fd_1(length_scale=32, print_res=128, print_len=3)
# rs = torch.tensor(rs, device=device)
# azis = torch.tensor(azis, device=device)
#
# r_dash = torch.mean(rs, dim=1).unsqueeze(1).repeat_interleave(rs.size()[1], dim=1)
# azi_dash = azis - torch.mean(azis, dim=1).unsqueeze(1).repeat_interleave(azis.size()[1], dim=1)
# instant_tke = (0.5 * (r_dash ** 2 + azi_dash ** 2)).cpu().numpy()
#
#
# t = forces_dic['t']
# for idx, loop in tqdm(enumerate(instant_tke), ascii=True, desc='Calculate spectra'):
#     u = loop
#     ti = t[0:len(loop)]
#
#     cycles = 7
#     late = 100
#
#     n = int(np.floor((np.max(ti[ti > late]) - np.min(ti[ti > late])) / cycles))
#     criteria = postproc.frequency_spectra.FreqConv(t=ti[ti > late], u=u[ti > late], n=n, OL=0.5)
#     f, uk = criteria.welch()
#     normed_error, window_t = criteria.f_conv(cycles)
#
#     fig, ax = plt.subplots(figsize=(7, 5))
#     ax.set_title(r'$\theta = $' + f'$ {round(angles[idx], 2)} $')
#     ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
#     ax.set_xlabel(r"$t/D$")
#     ax.set_ylabel(r'$\int \sqrt{(\overline{s_{0,n}} - \overline{s_{0,n+1}})^2} df/ \int \overline{s_{0,n+1}}$')
#
#     ax.plot(window_t, normed_error, c='r')
#     plt.savefig(data_root + f"figures/{interest}_converged_{idx}.png", bbox_inches='tight', dpi=600,
#                 transparent=False)
#     plt.close()


























