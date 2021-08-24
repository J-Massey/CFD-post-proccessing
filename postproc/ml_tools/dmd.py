# -*- coding: utf-8 -*-
"""
@author: Jonathan Massey
@description: Unpack flow field and plot the contours
@contact: jmom1n15@soton.ac.uk
"""

# Imports
import matplotlib.pyplot as plt
import numpy as np
from pydmd import DMD, FbDMD
from tqdm import tqdm

plt.style.use(['science', 'grid'])


def dmd_mag(snap: np.array):
    u, v, _ = snap[:, 2:-1].T

    snapshots = []
    for loop1, loop2 in tqdm(zip(u, v)):
        U, V = np.mean(loop1, axis=2).T, np.mean(loop2, axis=2).T
        # Filter snapshots so that we only use region of interest around the foil
        mag = np.sqrt(U ** 2 + V ** 2)
        snapshots.append(mag)
    snapshots = snapshots - np.mean(snapshots, axis=0)
    # dmd = DMD(svd_rank=-1, tlsq_rank=2, exact=True, opt=True).fit(snapshots)
    opt_dmd = FbDMD(svd_rank=-1, exact=True).fit(snapshots)
    print('\n Done DMD fit, freeing up cores')
    return opt_dmd


def dmd_pressure(snap: np.array):
    p = snap[:, -1]
    snapshots = []
    for loop1 in tqdm(p):
        snapshots.append(np.mean(loop1, axis=0).T)
    # snapshots = snapshots - np.mean(snapshots, axis=0)
    opt_dmd = FbDMD(svd_rank=-1, exact=True).fit(snapshots)
    return opt_dmd

