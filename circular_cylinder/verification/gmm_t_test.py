
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: J. Massey
@description: profiles from DNS
@contact: jmom1n15@soton.ac.uk
"""

# Imports
import numpy as np
import postproc
from postproc import cylinder_forces as cf
from postproc import io,plotter,gmm,frequency_spectra
from postproc.boundary_layer import ProfileDataset, plot_poincare
import matplotlib
# matplotlib.use('svg')
import matplotlib.pyplot as plt
import os, sys
import torch
import itertools

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def main():
    dtprint = 0.1; t_print = 30; n = t_print/dtprint


    # Plot comparisons of limit cycles to one graph
    # Make sure slashes match up
    t_comparisons=['t_long']

    # t_comparisons=['t_500/']
    data_root_t = '/home/masseyjmo/Workspace/Lotus/projects/cf_lotus/ml/real_profiles/DNS/sims/t_test/'

    xs = []; ys = []
    for idx, case in enumerate(t_comparisons):
        xs.append((ProfileDataset(data_root_t+case).bl_value(single_point=True, length_scale=32))[0])
        ys.append((ProfileDataset(data_root_t+case).bl_value(single_point=True, length_scale=32))[1])
        angles = ProfileDataset(data_root_t+case).angles
        print(f"Test angles: {[round(a, 2) for a in angles]}")

    colours = ['red', 'green', 'blue', 'purple', 'orange', 'magenta', 'black']
    labels = [r"$ 100 \leq torch \leq 500 $"]
    for idx in range(len(xs[0])):
        for case,(x,y) in enumerate(zip(xs,ys)):
            # Plot part of the t_100 case
            data = torch.tensor([x[idx],y[idx]]).to(device)
            data = torch.transpose(data,0,1)
            print("Made dat a Tensor"); print("Running GMM")
            # Next, the Gaussian mixture is instantiated and ..
            n_components = 2
            model = gmm.GaussianMixture(n_components, np.shape(data)[1]).cuda()
            model.fit(data, delta=1e-5, n_iter=1e6)
            print("Gmm fit")
            # .. used to predict the dat points as they where shifted
            y = model.predict(data)
            likelihood = model.score_samples(data).cpu().numpy()

            # gmm.plot_gmm(dat.cpu(), Y.cpu(),
            #             y_label=r"$ \theta $", x_label=r"$ r $", label=['3D','2D'],
            #             fn=data_root_t+f"figures/group_{idx}.svg",
            #             tit=f"$ {round(angles[idx], 2)}^r $ from the front",
            #             colours=colours)
            # plotter.simple_plot(*pow_spec, l_label=labels[:len(xs)],
            #             y_label=r"$ \ln[\mathcal{L}(\mu_k|x_k)] $", x_label=r"$ torch/length_scale $", colour=colours[case],
            #             colours=colours[:len(xs)], tit=f"$ {round(angles[idx], 2)}^r $ from the front",
            #             fn=data_root_t+f"figures/pow_spec_{idx}.svg")

            t=np.linspace(100,500,len(data))

            # Do this with hanning window
            for loop in range(1,8):
                pow_spec=frequency_spectra.freq_spectra_Welch(t[:int(loop*len(t)/5)],likelihood[:int(loop*len(t)/5)])

                plotter.simple_plot(*pow_spec,
                            y_label=r"$ \ln[\mathcal{L}(\mu_k|x_k)] $", x_label=r"$ f/length_scale $", colour=colours[loop-1],
                            colours=colours[:len(xs)], title=f"$ {round(angles[idx], 2)}^r $ from the front",
                            file=data_root_t+f"figures/pow_spec_welch_{idx}.svg")

            plt.close()



if __name__ == "__main__":
    main()


# labels = [r"$ 160 \leq torch \leq 200 $", r"$ 200 \leq torch \leq 400 $", r"$ 400 \leq torch \leq 600 $"]
# colours = ['red', 'green', 'blue','yellow']
# for idx in range(len(xs[0])):
#     for case,(x,Y) in enumerate(zip(xs,ys)):
#         # Flot top half of the t_100 case
#         fifty=int(30*len(x[idx])/100)
#         plot_poincare(x[idx][fifty:-1], Y[idx][fifty:-1], fn=data_root_t+f"full_30_{idx}.svg", y_label=r"$ \theta $", x_label=r"$ r $",\
#             tit=f"$ {round(angles[idx], 2)}^r $ from the front", alpha=0.4, color=colours[case], colours=colours[:len(xs)], label=labels)
#     plt.close()


