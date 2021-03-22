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
import postproc.gmm
import postproc.frequency_spectra
from postproc.profile_convergence import ProfileDataset, plot_poincare
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os, sys
import torch
import itertools

# %%

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# %%

dtprint = 0.1;
t_print = 30;
n = t_print / dtprint;
t_min = 100;
t_max = 500

t_comparisons = 't_long'
data_root = '/home/masseyjmo/Workspace/Lotus/projects/cf_lotus/ml/real_profiles/DNS/sims/t_test/'
# %%
rs = torch.tensor(ProfileDataset(data_root + t_comparisons)
                  .bl_poincare_limit(single_point=True, position=0.2, length_scale=32, print_res=256)[0]).to(device)
azis = torch.tensor(ProfileDataset(data_root + t_comparisons)
                    .bl_poincare_limit(single_point=True, position=0.2, length_scale=32, print_res=256)[1]).to(device)
angles = ProfileDataset(data_root + t_comparisons).angles
# %%
# torch.mean(data, 1).unsqueeze(1).repeat_interleave(data.size()[1], dim=1)
r_dash = torch.mean(rs, dim=1).unsqueeze(1).repeat_interleave(rs.size()[1], dim=1)
azi_dash = azis - torch.mean(azis, dim=1).unsqueeze(1).repeat_interleave(azis.size()[1], dim=1)
instant_tke = (0.5 * (r_dash ** 2 + azi_dash ** 2)).cpu().numpy()
# %%
import importlib
importlib.reload(postproc.plotter)
importlib.reload(postproc.frequency_spectra)

for idx, loop in enumerate(instant_tke):
    t = np.arange(0, len(loop) * dtprint, dtprint)
    n = 4
    uks, fs, means, variances = postproc.frequency_spectra.freq_spectra_convergence(t, loop, n=n, OL=0.5)

    postproc.plotter.fully_defined_plot(t, loop,
                                        y_label=r"$ \sqrt{{r^{\prime}}^{2} + {\theta^{\prime}}^{2}} $",
                                        x_label=r"$ t/D $",
                                        colour=colours[idx],
                                        title=f"TKE at ${angles[idx]:.2f}r$",
                                        file=data_root + f"figures/tke_{idx}.svg")

    postproc.plotter.plotTimeSpectra_list(data_root + f'figures/time_spec_convergence_{idx}.svg', uks, fs,
                                          title=f"Splits, $n={n}$ at $ {angles[idx]:.2f}r$")
    plt.close()

    n = len(loop)*dtprint/10  # bin every 10 convection cycles
    uks, fs, means, variances = postproc.frequency_spectra.freq_spectra_convergence(t, loop, n=n)
    t = np.linspace(0, len(loop) / 10., len(variances))
    postproc.plotter.fully_defined_plot(t, variances,
                                        y_label=r"$ Var(s_{n}) $",
                                        x_label=r"$ t/D $",
                                        colour='red',
                                        title=f"Fourier bin variance at ${angles[idx]:.2f}r$",
                                        file=data_root + f"figures/fourier_var_tke_{idx}.svg", marker='+')
    postproc.plotter.fully_defined_plot(t, means,
                                        y_label=r"$ \overline{(s_{n})} $",
                                        x_label=r"$ t/D $",
                                        colour='red',
                                        title=f"Fourier bin variance at ${angles[idx]:.2f}r$",
                                        file=data_root + f"figures/fourier_mean_tke_{idx}.svg", marker='+')


    plt.close()
