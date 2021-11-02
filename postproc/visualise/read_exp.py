# -*- coding: utf-8 -*-
"""
@author: Jonathan Massey
@description: Class to read and convert experimental data into usable arrays
@contact: jmom1n15@soton.ac.uk
"""

import h5py
import numpy as np
from postproc import calc


class PIVFramework:
    """
    Class that holds all the functions to extract dat from a .mat fn to a plottable form.
    """

    def __init__(self, exp, fn, **kwargs):
        rms = kwargs.get('rms', False)
        mag = kwargs.get('mag', True)
        vort = kwargs.get('vort', False)
        data = {}
        f = h5py.File(exp)[fn]
        for k, v in f.items():
            data[k] = np.array(v)
        # Normalise with the chord length
        l, U_inf = data['chord_length'], data['U_inf']
        print(l)
        self.X, self.Y = data['X'] / l, data['Y'] / l
        self.u, self.v = data['VY'] / 0.08, data['VX'] / 0.08
        if mag:
            self.mag_snap = np.sqrt((np.einsum('...jk,...jk->...jk', self.u, self.u) +
                                     np.einsum('...jk,...jk->...jk', self.v, self.v)))

            mean = np.mean(self.mag_snap, axis=0)
            self.U = mean
        if rms:
            mag = np.sqrt((np.einsum('...jk,...jk->...jk', self.u, self.u) +
                           np.einsum('...jk,...jk->...jk', self.v, self.v)))
            mean = np.array([np.mean(mag, axis=0)])
            fluc = np.sqrt((mag - mean) ** 2)
            self.U = np.mean(fluc, axis=0)
        if vort:  # ddx-ddy
            omega = []
            for idx, (snap_u, snap_v) in enumerate(zip(self.u, self.v)):
                omega.append(np.array(calc.vortZ(snap_u, snap_v, x=self.X[:, 0], y=self.Y[0], acc=2)))
            self.omega = np.sum(omega, axis=0) / len(self.U)
            self.omega = self.omega.T
            self.omega = data['vort'] / 0.08
            self.omega = np.squeeze(np.mean(self.omega, axis=0)).T
