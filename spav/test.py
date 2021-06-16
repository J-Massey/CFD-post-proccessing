# -*- coding: utf-8 -*-
"""
@author: Jonathan Massey
@description: Unpack flow field and plot the contours
@contact: jmom1n15@soton.ac.uk
"""

# Imports
import os
import postproc.io as io
import postproc.calc as averages
import postproc.plotter as plotter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.colors as colors
from matplotlib import ticker, cm
import seaborn as sns
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tkinter import Tcl
import imageio
import tqdm
import h5py


def plot_2D_fp_isocontours(data, interest, fn_save, **kwargs):

    filled = kwargs.get('filled', True)
    title = kwargs.get('title', None)

    # plt.style.use(['science', 'grid'])
    # fig, ax = plt.subplots(figsize=(10, 7))
    # # cmap = sns.color_palette("icefire", as_cmap=True)
    # plt.title(title)
    # divider = make_axes_locatable(ax)
    # # Plot the window of interest
    # ax.set_xlim(-0.2, 1.4)
    # ax.set_ylim(-0.5, 0.5)

    X, Y = data.X, data.Y
    # Now plot what we are interested in
    if interest == 'p':
        vals = data.p
    elif interest == 'u':
        vals = data.U
    elif interest == 'v':
        vals = data.V
    elif interest == 'mag':
        U, V = data.U, data.V
        vals = np.sqrt(V ** 2 + U ** 2)
        # vals = vals * data.iter_correction(30)
    elif interest == 'rms':
        vals = data.rms()
    elif interest == 'rms_mag':
        vals = data.rms_mag()
    elif interest == 'vort':
        vals = data.vort

    lim = [np.min(vals), np.max(vals)]
    lim = kwargs.get('lims', lim)
    print(lim)
    # Put limits consistent with experimental data
    # norm = colors.Normalize(vmin=lim[0], vmax=lim[1])
    # lvls = kwargs.get('lvls', 11)
    # step = kwargs.get('step', None)
    # if step is not None:
    #     lvls = np.arange(lim[0], lim[1] + step, step)
    # else:
    #     lvls = np.linspace(lim[0], lim[1], lvls)
    #
    # if filled:
    #     cs = ax.contourf(X, Y, np.transpose(vals),
    #                      levels=lvls, vmin=lim[0], vmax=lim[1],
    #                      norm=norm, cmap=cmap)
    #     ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    #     fig.add_axes(ax_cb)
    #     plt.colorbar(cs, cax=ax_cb)
    #     ax_cb.yaxis.tick_right()
    #     ax_cb.yaxis.set_tick_params(labelright=True)
    # del X, Y, vals
    # ax.set_aspect(1)
    #
    # plt.savefig(fn_save)
    # plt.close()


def vti_to_mesh(fn, length_scale, rotation=0):
    rot = rotation / 180 * np.pi
    data = io.read_vti(fn)
    # Get the grid
    X, Y, z = data[2]
    X, Y = np.squeeze(X), np.squeeze(Y)
    # Move grid so front of the foil is at (0, 0) to match exp
    flo = np.nan_to_num(data[0], copy=True, nan=0.0, posinf=None, neginf=None)
    U, V, w = flo
    p = np.nan_to_num(data[1], copy=True, nan=0.0, posinf=None, neginf=None)
    p = np.reshape(p, [np.shape(p)[0], np.shape(p)[2], np.shape(p)[3]])
    # print(np.shape(p), np.shape(U), np.shape(X))
    return X, Y, U, V, w, p


class SimFramework:
    """
    Class that holds all the functions to extract dat from a paraview fn,
    average and plot the contours and an animation.
    """

    def __init__(self, sim_dir, fn_root, **kwargs):
        rms = kwargs.get('rms', False)
        down = kwargs.get('downsample', 1)
        self.sim_dir = sim_dir
        datp_dir = os.path.join(sim_dir, 'datp')
        rotation = kwargs.get('rotation', 0)
        self.rot = rotation / 180 * np.pi
        self.length_scale = kwargs.get('length_scale', 96)
        # Find what you're looking for
        fns = [fn for fn in os.listdir(datp_dir) if fn.startswith(fn_root) and fn.endswith('.pvti')]
        # Sort files
        fns = Tcl().call('lsort', '-dict', fns)

        if len(fns) > 1:
            print("More than one fn with this name. Taking time average.")
            # Store snapshots of field
            self.snaps = []
            for fn in fns[::down]:
                snap = vti_to_mesh(os.path.join(datp_dir, fn), self.length_scale)
                self.snaps.append(snap)
            del snap
            # Time average the flow field snaps
            mean_t = np.mean(np.array(self.snaps).T, axis=1)
            self.X, self.Y = mean_t[0:2]
            self.u, self.v, self.w = mean_t[2:-1]
            self.U, self.V = np.mean(self.u, axis=2), np.mean(self.v, axis=2)
            self.p = np.mean(mean_t[-1], axis=0)
            del mean_t
        else:
            # assert (len(fns) > 0), 'You dont have '+fn_root+'.pvtr in your datp folder'
            self.X, self.Y, self.U, self.V, self.W, self.p = vti_to_mesh(os.path.join(datp_dir, fns[0]),
                                                                         self.length_scale)
            self.U, self.V = np.squeeze(self.U), np.squeeze(self.V)
            self.p = np.squeeze(self.p)

    def rms(self):
        means = np.mean(np.array(self.snaps).T, axis=1)[2:-1]
        fluctuations = np.array(flow.snaps)[:, 2:-1] - means
        del means
        rms = np.mean((fluctuations[:, 0] ** 2 + fluctuations[:, 1] ** 2 + fluctuations[:, 2] ** 2) ** (1 / 2))
        del fluctuations
        return np.mean(rms, axis=2)

    def rms_mag(self):
        mean = np.mean(np.array(self.snaps).T, axis=1)[2:-1]
        mean = np.sqrt(np.sum(mean**2))
        mag = np.array(flow.snaps)[:, 2:-1]**2
        mag = np.sum(mag, axis=1)**0.5
        fluc = []
        for snap in mag:
            fluc.append(snap - mean)
        del mag, mean

        return np.mean(np.mean(fluc, axis=0), axis=2)

if __name__ == "__main__":
    plt.style.use(['science', 'grid'])
    c = 32
    tit = r'$ \overline{|U|} $'
    exp_data = '/home/masseyjmo/Workspace/Lotus/projects/flat_plate/flow_field/exp_data/smooth_Re10k_AoA_12.mat'
    data_root = '/home/masseyjmo/Workspace/Lotus/projects/rms_test/test/'
    tit = r'$ \overline{||U|^{\prime}|} $'
    flow = SimFramework(data_root, 'rms2D',
                        length_scale=c, rotation=0)
    field = 'mag'
    plot_2D_fp_isocontours(flow, field, os.path.join(data_root, 'figures/sim_rms_mag.pdf'),
                           title=tit)
    t = np.arange(0., 10.1, 0.1)
    print(np.sum(np.sqrt(np.sin(2*np.pi*t)**2))/len(t))

