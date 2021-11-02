# -*- coding: utf-8 -*-
"""
@author: Jonathan Massey
@description: Save flow snapshots to .npy file
@contact: jmom1n15@soton.ac.uk
"""
import vtk
from matplotlib import pyplot as plt
import os
from tkinter import Tcl
import numpy as np
from tqdm import tqdm
import warnings

plt.style.use(['science', 'grid'])


class SaveSnaps:
    def __init__(self, data_root, save_dir, length_scale, file_start):
        """
        Class that takes the 2D flow and saves snapshots
        Args:
            data_root: The path to the Lotus simulation directory.
            file_start: The quantity saved, 'flu2d', 'avRms', etc...
        """
        self.save_dir = save_dir
        self.length_scale = length_scale
        self.data_root = data_root
        self.case = file_start

    @staticmethod
    def read_vtr(fn):
        warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
        reader = vtk.vtkXMLPRectilinearGridReader()
        reader.SetFileName(fn)
        reader.Update()
        data = reader.GetOutput()
        pointData = data.GetPointData()
        sh = data.GetDimensions()[::-1]
        ndims = len(sh)

        # get vector field
        try:
            v = np.array(pointData.GetVectors("Velocity")).reshape(sh + (ndims,))
            vec = []
            for d in range(ndims):
                a = np.array(v[..., d])
                vec.append(a)
            vec = np.array(vec)
            # get scalar field
            sca = np.array(pointData.GetScalars('Pressure')).reshape(sh + (1,))

            # get grid
            x = np.array(data.GetXCoordinates())
            y = np.array(data.GetYCoordinates())
            z = np.array(data.GetZCoordinates())

            return np.transpose(vec, (0, 3, 2, 1)), np.transpose(sca, (0, 3, 2, 1)), np.array((x, y, z))
        except ValueError:
            print('\n' + fn + ' corrupt, skipping for now')

    def save_snaps(self, n=14, theta=12):
        os.system('mkdir -p ' + self.save_dir)
        datp_dir = os.path.join(self.data_root, 'datp')
        fns = [fn for fn in os.listdir(datp_dir) if fn.startswith(self.case) and fn.endswith('.pvtr')]
        fns = Tcl().call('lsort', '-dict', fns)
        batch = len(fns) // n
        if len(fns) == 1: n = 1
        for idx in range(n):
            snaps = []
            for fn in tqdm(fns[batch * idx:batch * (idx + 1)], desc='Saving some snaps'):
                snap = self.rotate(os.path.join(datp_dir, fn), self.length_scale, rotation=theta)
                snaps.append(snap)
            snaps = np.array(snaps)
            np.save(os.path.join(self.save_dir, self.case + '_' + str(idx) + '.npy'), snaps)
            del snaps
        if len(fns) % n != 0:
            snaps = []
            for fn in tqdm(fns[batch * (idx + 1):len(fns)], desc='Collecting leftovers'):
                snap = self.rotate(os.path.join(datp_dir, fn), self.length_scale, rotation=theta)
                snaps.append(snap)
            np.save(os.path.join(self.save_dir, self.case + '_' + str(n) + '.npy'), np.array(snaps))

    def rotate(self, fn, length_scale, rotation=0):
        """
        Rotates and scales vtr file
        Args:
            fn: The path to the 'datp' folder
            length_scale: length scale of the simulation
            rotation: Rotate the grid. If you're running a simulation with
                      an angle of attack, it's better to rotate the flow than
                      the foil because of the meshing.

        Returns: X, Y - coordinates (useful for indexing)
                 U, V - rotated velocity components
                 w    - un-rotated z velocity component
                 p    - pressure field

        """
        rot = rotation / 180 * np.pi
        data = self.read_vtr(fn)
        # Get the grid
        x, y, z = data[2]
        X, Y = np.meshgrid(x / length_scale, y / length_scale)
        X = np.cos(rot) * X + np.sin(rot) * Y
        Y = -np.sin(rot) * X + np.cos(rot) * Y

        u, v, w = data[0]
        U = np.cos(rot) * u + np.sin(rot) * v
        V = -np.sin(rot) * u + np.cos(rot) * v
        p = data[1]
        p = np.reshape(p, [np.shape(p)[0], np.shape(p)[2], np.shape(p)[3]])
        return X, Y, U, V, w, p


if __name__ == '__main__':
    sim_dir = '/home/masseyjmo/Workspace/Lotus/projects/waving_plate'
    case = ['full_bumps', 'half_bumps', 'riblet', 'smooth']
    case = ['smooth']
    extension = '256/save'
    for loop in case:
        d_root = os.path.join(sim_dir, loop, extension)
        save_folder = os.path.join(sim_dir, 'quick_access_data', loop)
        SaveSnaps(d_root, save_folder, 256, 'flu2d').save_snaps(n=1, theta=0)
