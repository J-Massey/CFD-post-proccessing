# -*- coding: utf-8 -*-
"""
@author: Jonathan Massey
@description: Object for flow
@contact: jmom1n15@soton.ac.uk
"""

import os
from tkinter import Tcl
import numpy as np
import postproc.io as io


class LoadSnaps:
    """
    This class acts as a file handler for the flow_field class
    """
    def __init__(self, data_root: str, length_scale, case: str):
        """
        We take file path inputs so it's important to get these correct.
        Args:
            data_root: The whole path to the simulation folder (Where did Lotus write)
            length_scale: What length scale is your time etc based on
            case: The 5 letter vtr filename
        """
        self.length_scale = length_scale
        self.data_root = data_root
        self.case = case

    @property
    def snapshots(self):
        """
        Load snaps from npy file

        Returns: np.array of (*, n, m) snapshots
        """
        try:
            file = self.npy_exist()
            snaps = np.load(os.path.join(self.data_root, self.npy_exist()), allow_pickle=True)
            print('\nFound '+file+' file, going from there')
            return snaps
        except FileNotFoundError:
            print('\nNo saved .npy file, the analysis will be quicker if you save it first')

    @property
    def average_flow(self):
        if self.case.endswith('vti'):
            snap = io.read_vti(os.path.join(self.data_root, 'datp', self.case))
        elif self.case.endswith('vtr'):
            snap = io.read_vtr(os.path.join(self.data_root, 'datp', self.case))
        else:
            print('\nYou need to add a file extension')
            raise FileNotFoundError

        return self._unpack_read_vt_2D(snap)

    def _unpack_read_vt_2D(self, snap):
        """
        Unpacks the values from
        Args:
            snap: the call from io.read_vtr/i

        Returns: X, Y - coordinates (useful for indexing)
                 U, V - rotated velocity components
                 w    - z velocity component
                 p    - pressure field

        """
        # Get the grid
        x, y, z = snap[2]
        # X, Y = np.meshgrid(x / self.length_scale, y / self.length_scale)

        u, v, w = snap[0]
        p = snap[1]
        p = np.reshape(p, [np.shape(p)[0], np.shape(p)[2], np.shape(p)[3]])
        return np.squeeze(x), np.squeeze(y), np.squeeze(u), np.squeeze(v), np.squeeze(w),\
            np.squeeze(p)

    def npy_exist(self):
        fns = os.listdir(self.data_root)
        numpy_saves = [fn for fn in fns if fn.startswith(self.case) and fn.endswith('.npy')]
        fns = Tcl().call('lsort', '-dict', numpy_saves)
        if len(fns) > 1:
            print("\nMultiple batches for POD? Using first batch")
            file = fns[0]
            return file
        elif len(fns) == 1:
            file = fns[0]
            return file
        elif len(fns) == 0:
            raise FileNotFoundError


if __name__ == '__main__':
    sim_dir = '/home/masseyjmo/Workspace/Lotus/projects/flat_plate/AoA_12/smooth/dom_test/5'
    snaps = LoadSnaps(sim_dir, 192, 'spTAv.1.pvti').average_flow

