# -*- coding: utf-8 -*-
"""
@author: J. Massey
@description: Linear regression of DNS dat
@contact: jmom1n15@soton.ac.uk
"""

import numpy as np
import pandas as pd
import postproc.io
import postproc.boundary_layer
import os
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

plt.style.use(['science', 'grid'])

data_root = '/home/masseyjmo/Workspace/Lotus/projects/cylinder_dns/sims/res_test/'
fn = 'd-96'
pressure = 'fort.10'
force_file = 'fort.9'
u_0 = 'profiles'

names = ['t', 'dt', 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'v2x', 'v2y', 'v2z']
fos = (postproc.io.unpack_flex_forces(os.path.join(data_root, fn, '3D', force_file), names))
forces_dic = dict(zip(names, fos))
t_min = min(forces_dic['t'])
t_max = max(forces_dic['t'])
t = forces_dic['t']

profiles = postproc.boundary_layer.ProfileDataset(os.path.join(data_root, fn, '3D'), True)
rs, u0 = profiles.bl_poincare_limit(single_point=True, position=0.5, length_scale=96, print_res=256, print_len=3)


if __name__ == "__main__":
    print("Importing dat")

    print("Imported dat")
