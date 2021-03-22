
#!/usr/bin/env python3
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
import matplotlib.pyplot as plt
import os, sys
import torch
import itertools

#%%

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#%%

dtprint = 0.1; t_print = 30; n = t_print/dtprint; t_min=100; t_max=500
colours = ['red', 'green', 'blue', 'purple', 'orange', 'magenta', 'black']

t_comparisons=['t_long']

# t_comparisons=['t_500/']
data_root_t = '/home/masseyjmo/Workspace/Lotus/projects/cf_lotus/ml/real_profiles/DNS/sims/t_test/'
#%%

xs = []; ys = [];
for idx,case in enumerate(t_comparisons):
    xs.append((ProfileDataset(data_root_t+case).bl_poincare_limit(single_point=True,length_scale=32))[0])
    ys.append((ProfileDataset(data_root_t+case).bl_poincare_limit(single_point=True,length_scale=32))[1])
    angles = ProfileDataset(data_root_t+case).angles
    print(f"Test angles: {[round(a, 2) for a in angles]}")
# %%
import importlib
importlib.reload(postproc.plotter)
import postproc.plotter
#%%
importlib.reload(postproc.frequency_spectra)
import postproc.frequency_spectra


# %%

for idx in range(len(xs[0])):
    for case, (x, y) in enumerate(zip(xs, ys)):
        # Plot part of the t_100 case
        data = torch.tensor([x[idx],y[idx]]).to(device)
        # Flatten the data
        data = torch.transpose(data,0,1)
        print("Made data a Tensor"); print("Running GMM")
        # Next, the Gaussian mixture is instantiated and ..
        n_components = 2
        model = postproc.gmm.GaussianMixture(n_components, np.shape(data)[1]).cuda()
        model.fit(data, delta=1e-5, n_iter=1e6)
        print("Gmm fit")
        # .. used to predict the data points as they where shifted
        y = model.predict(data)
        likelihood = model.score_samples(data).cpu().numpy()

        # postproc.gmm.plot_gmm(data.cpu(), y.cpu(),
        #             y_label=r"$ \theta $", x_label=r"$ r $", label=['3D','2D'],
        #             file=data_root_t+f"figures/group_{idx}.svg",
        #             title=f"$ {round(angles[idx], 2)}^r $ from the front",
        #             colours=colours)

        # plt.close()

        t=np.linspace(t_min,t_max,len(data))
        n=8

        label = lambda t: f"$ {t_min} \leq t \leq {t} $"
        labels = [label(int(t)) for t in np.arange(t_min, t_max, t_max / n)]

        fs, ts = postproc.frequency_spectra.freq_spectra_Welch(t, likelihood, n=n,
                                                               windowing=False,
                                                               expanding_windowing=True)
        for w_idx, (f, t) in enumerate(zip(fs, ts)):
            postproc.plotter.simple_plot(f, t,
                                         y_label=r"$ \ln[\mathcal{L}(\mu_k|x_k)] $", x_label=r"$ f/D $",
                                         colour=colours[w_idx],
                                         colours=colours[:len(fs)],
                                         title=f"$ {round(angles[idx], 2)}^r $ from the front",
                                         l_label=labels,
                                         file=data_root_t+f"figures/pow_spec_welch_{idx}.svg")

        # for loop in range(1,8):
        #     window = postproc.frequency_spectra._window(loop * t / n)
        #
        #     # pow_spec = postproc.frequency_spectra.freq_spectra_Welch(t[:int(loop*len(t)/5)],likelihood[:int(loop*len(t)/5)])
        #
        #     postproc.plotter.simple_plot(*pow_spec,
        #                 y_label=r"$ \ln[\mathcal{L}(\mu_k|x_k)] $", x_label=r"$ f/D $", colour=colours[loop-1],
        #                 colours=colours[:len(xs)], title=f"$ {round(angles[idx], 2)}^r $ from the front",
        #                 file=data_root_t+f"figures/pow_spec_welch_{idx}.svg")
        #
        # plt.close()

#%%



