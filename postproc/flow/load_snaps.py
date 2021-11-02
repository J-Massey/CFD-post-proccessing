# -*- coding: utf-8 -*-
"""
@author: Jonathan Massey
@description: Object for flow
@contact: jmom1n15@soton.ac.uk
"""

import os
from tkinter import Tcl
import numpy as np


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
    sim_dir = '/home/masseyjmo/Workspace/Lotus/projects/waving_plate'
    case = ['full_bumps', 'smooth']
    extension = 'quick_access_data'
    for loop in case:
        d_root = os.path.join(sim_dir, extension, loop)
        snaps = LoadSnaps(d_root, 256, 'flu2d').snapshots

