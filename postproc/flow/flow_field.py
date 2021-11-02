# -*- coding: utf-8 -*-
"""
@author: Jonathan Massey
@description: Hold object for flow field.
            Load from npy if available otherwise go through and save batches to .npy
@contact: jmom1n15@soton.ac.uk

To do: - Add 'snapshot()' to different class
"""

import numpy as np
from tqdm import tqdm


class FlowBase:
    """
    Simple class to hold the main flow quantities of a 2D flow field so you can call it on
    a data_root and it will look for a quick binary file to read ".npy". Or read all the files
    from the vtr logs.

    This class has been built for the analysis of flow snapshots, and hasn't been tested on
    single 2D matrices.
    """
    def __init__(self, snapshots: np.array, **dimension: bool):
        """
        We take file path inputs so it's important to get these correct.
        Args:
            snapshots: loaded from load_snaps.py: LoadSnap.snapshots
            **dimension: slice, spanwise_avg, 3D
        """
        self.snaps = snapshots
        self.dimensions = dimension

    @property
    def x(self):
        return self.snaps[0, 0]

    @property
    def y(self):
        return self.snaps[0, 1]

    @property
    def u(self):
        if self.dimensions.get('slice'):
            _u = np.array([loop[:, :, np.shape(loop)[2]//2].T for loop in self.snaps[:, 2].T])
        elif self.dimensions.get('3D'):
            _u = np.array([loop.T for loop in self.snaps[:, 2].T])
        else:
            if self.dimensions.get('spanwise_avg') is None: print('\nNo averaging so assuming spanwise avg')
            _u = np.array([np.mean(loop, axis=2).T for loop in self.snaps[:, 2].T])
        return _u

    @property
    def v(self):
        if self.dimensions.get('slice'):
            _v = np.array([loop[:, :, np.shape(loop)[2] // 2].T for loop in self.snaps[:, 3].T])
        elif self.dimensions.get('3D'):
            _v = np.array([loop.T for loop in self.snaps[:, 3].T])
        else:
            if self.dimensions.get('spanwise_avg') is None: print('\nNo averaging so assuming spanwise avg')
            _v = np.array([np.mean(loop, axis=2).T for loop in self.snaps[:, 3].T])
        return _v

    @property
    def w(self):
        _w = np.array([loop.T for loop in self.snaps[:, 4].T])
        return _w

    @property
    def p(self):
        if self.dimensions.get('slice'):
            _p = np.array([loop[np.shape(loop)[2] // 2, :, :] for loop in self.snaps[:, -1].T])
        elif self.dimensions.get('3D'):
            _p = np.array([loop for loop in self.snaps[:, -1].T])
        else:
            if self.dimensions.get('spanwise_avg') is None: print('\nNo averaging so assuming spanwise avg')
            _p = np.array([np.mean(loop, axis=0) for loop in self.snaps[:, -1]])
        return _p

    @property
    def magnitude(self):
        return np.array([np.sqrt(loop1 ** 2 + loop2 ** 2)
                         for loop1, loop2 in tqdm(zip(self.u, self.v))])

    @property
    def vorticity(self):
        return np.array([np.gradient(loop2, axis=0, edge_order=2) - np.gradient(loop1, axis=1, edge_order=2)
                         for loop1, loop2 in tqdm(zip(self.u, self.v))])


