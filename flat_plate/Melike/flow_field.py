# -*- coding: utf-8 -*-
"""
@author: Jonathan Massey
@description: Unpack flow field and plot the contours
@contact: jmom1n15@soton.ac.uk
"""

# Imports
import os
import postproc.io as io
import postproc.calc as calc
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

    plt.style.use(['science', 'grid'])
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = sns.color_palette("icefire", as_cmap=True)
    plt.title(title)
    divider = make_axes_locatable(ax)
    # Plot the window of interest
    ax.set_xlim(-0.2, 1.4)
    ax.set_ylim(-0.5, 0.5)

    X, Y = data.X, data.Y
    # Now plot what we are interested in
    if interest == 'p':
        vals = data.p
        # cmap = sns.color_palette("seismic", as_cmap=True)
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
        vals = calc.vortZ(data.U, data.V, x=X[0], y=Y[0], acc=2)
        # vals = -data.p * data.length_scale  # Need to scale by length scale
        cmap = sns.color_palette("seismic", as_cmap=True)
    elif interest == 'mat_file_vort':
        vals = data.omega
        cmap = sns.color_palette("seismic", as_cmap=True)
    elif interest == 'mat_file':
        vals = data.U.torch

    grey_color = '#dedede'
    rec = patches.Rectangle((0, -1 / 91.42), 1., 1/45.71, -12, linewidth=0.2, edgecolor='black', facecolor=grey_color)
    ax.add_patch(rec)

    lim = [np.min(vals), np.max(vals)]
    lim = kwargs.get('lims', lim)
    # Put limits consistent with experimental data
    norm = colors.Normalize(vmin=lim[0], vmax=lim[1])
    lvls = kwargs.get('lvls', 11)
    step = kwargs.get('step', None)
    if step is not None:
        lvls = np.arange(lim[0], lim[1]+step, step)
    else:
        lvls = np.linspace(lim[0], lim[1], lvls)

    if filled:
        cs = ax.contourf(X, Y, np.transpose(vals),
                         levels=lvls, vmin=lim[0], vmax=lim[1],
                         norm=norm, cmap=cmap)
        ax_cb = divider.new_horizontal(size="5%", pad=0.05)
        fig.add_axes(ax_cb)
        plt.colorbar(cs, cax=ax_cb)
        ax_cb.yaxis.tick_right()
        ax_cb.yaxis.set_tick_params(labelright=True)
        # plt.setp(ax_cb.get_yticklabels()[::2], visible=False)
        # ax.clabel(cs, cs.levels[::2], inline_spacing=-9, inline=1, fontsize=10, fmt='%1.2f')
    del X, Y, vals
    ax.set_aspect(1)

    plt.savefig(fn_save, dpi=300)
    plt.close()


def vtr_to_mesh(fn, length_scale, rotation=12):
    rot = rotation / 180 * np.pi
    data = io.read_vtr(fn)
    # Get the grid
    x, y, z = data[2]
    X, Y = np.meshgrid(x / length_scale, y / length_scale)
    # Move grid so front of the foil is at (0, 0) to match exp
    X += 0.5
    mask_bool = ((X >= 0.) & (X <= 1.) & (Y <= 1 / 91.42) & (Y >= -1 / 91.42)).T
    X = np.cos(rot) * X + np.sin(rot) * Y
    Y = -np.sin(rot) * X + np.cos(rot) * Y

    u, v, w = data[0]
    U = np.cos(rot) * u + np.sin(rot) * v
    # U = ma.masked_where(mask_bool, U)
    V = -np.sin(rot) * u + np.cos(rot) * v
    # V = ma.masked_where(mask_bool, V)
    p = data[1]
    p = np.reshape(p, [np.shape(p)[0], np.shape(p)[2], np.shape(p)[3]])
    return X, Y, U, V, w, p


class SimFramework:
    """
    Class that holds all the functions to extract dat from a paraview file,
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
        fns = [fn for fn in os.listdir(datp_dir) if fn.startswith(fn_root) and fn.endswith('.pvtr')]
        # Sort files
        fns = Tcl().call('lsort', '-dict', fns)

        if len(fns) > 1:
            print("More than one file with this name. Taking time average.")
            # Store snapshots of field
            self.snaps = []
            for fn in fns[::down]:
                snap = vtr_to_mesh(os.path.join(datp_dir, fn), self.length_scale)
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
            assert (len(fns) > 0), 'You dont have '+fn_root+'.pvtr in your datp folder'
            self.X, self.Y, self.U, self.V, self.W, self.p = vtr_to_mesh(os.path.join(datp_dir, fns[0]),
                                                                         self.length_scale)
            self.U, self.V = np.squeeze(self.U), np.squeeze(self.V)
            self.p = np.squeeze(self.p)
        self.z = np.ones(np.shape(self.X))
        # --- Unpack mean flow quantities ---
        names = kwargs.get('names', ['torch', 'dt', 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'v2x', 'v2y', 'v2z'])
        fos = (io.unpack_flex_forces(os.path.join(self.sim_dir, 'fort.9'), names))
        self.fos = dict(zip(names, fos))

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

    def vort_mag(self):
        return io.vort(self.U, self.V, self.W, x=self.X, y=self.Y, z=self.z)

    def downsample(self, skip=1):
        self.X, self.Y, self.U, self.V, self.p = np.mean(np.array(self.snaps[::(skip + 1)]).T, axis=1)

    def animate(self, folder, **kwargs):
        for idx, snap in enumerate(self.snaps):
            dat = np.array(snap).T
            plot_2D_fp_isocontours(dat, 'mag', folder+str(idx)+'.png', **kwargs)
        # Sort filenames to make sure they're in order
        fn_images = os.listdir(folder)
        fn_images = Tcl().call('lsort', '-dict', fn_images)
        images = []
        for filename in fn_images:
            images.append(imageio.imread(os.path.join(folder, filename)))
        imageio.mimsave(folder+'/flow.gif', images, duration=0.2, bitrate=1000)


class PIVFramework:
    """
    Class that holds all the functions to extract dat from a .mat file to a plottable form.
    """
    def __init__(self, exp, **kwargs):
        rms = kwargs.get('rms', False)
        mag = kwargs.get('mag', True)
        vort = kwargs.get('vort', False)
        data = {}
        f = h5py.File(exp)['smooth_Re10k_AoA_12']
        for k, v in f.items():
            data[k] = np.array(v)
        # Normalise with the chord length
        l, U_inf = data['chord_length'], data['U_inf']
        self.X, self.Y = data['X']/l, data['Y']/l
        self.u, self.v = data['VY']/U_inf, data['VX']/U_inf
        if mag:
            mag = np.sqrt((np.einsum('...jk,...jk->...jk', self.u, self.u) +
                           np.einsum('...jk,...jk->...jk', self.v, self.v)))
            mean = np.mean(mag, axis=0)
            self.U = mean
        if rms:
            mag = np.sqrt((np.einsum('...jk,...jk->...jk', self.u, self.u) +
                           np.einsum('...jk,...jk->...jk', self.v, self.v)))
            mean = np.array([np.mean(mag, axis=0)])
            fluc = np.sqrt((mag-mean)**2)
            self.U = np.mean(fluc, axis=0)
        if vort:  # ddx-ddy
            omega = []
            for idx, (snap_u, snap_v) in enumerate(zip(self.u, self.v)):
                omega.append(np.array(calc.vortZ(snap_u, snap_v, x=self.X[:, 0], y=self.Y[0], acc=2)))
            self.omega = np.sum(omega, axis=0)/len(self.U)
            self.omega = self.omega.torch
            self.omega = data['vort']/U_inf
            self.omega = np.squeeze(np.mean(self.omega, axis=0)).T
    #
    # def animate(dis, folder, **kwargs):
    #     for idx, snap in enumerate(dis.snaps):
    #         dat = np.array(snap).torch
    #         animate_isocontours(dat, 'mag', folder+str(idx)+'.png', **kwargs)
    #     # Sort filenames to make sure they're in order
    #     fn_images = os.listdir(folder)
    #     fn_images = Tcl().call('lsort', '-dict', fn_images)
    #     images = []
    #     for filename in fn_images:
    #         images.append(imageio.imread(os.path.join(folder, filename)))
    #     imageio.mimsave(folder+'/flow.gif', images, duration=0.2)


if __name__ == "__main__":
    plt.style.use(['science', 'grid'])
    c = 96
    exp_data = '/home/masseyjmo/Workspace/Lotus/projects/flat_plate/flow_field/exp_data/smooth_Re10k_AoA_12.mat'
    data_root = '/home/masseyjmo/Workspace/Lotus/projects/flat_plate/flow_field'
    data_root = '/home/masseyjmo/Workspace/Lotus/projects/flat_plate/circle_caps/eps-half'
    tit = r'$ \overline{|U|} $'
    flow = PIVFramework(exp_data)
    plot_2D_fp_isocontours(flow, 'mat_file', os.path.join(data_root, 'figures/exp_mag.pdf'),
                           title=tit, lims=[0, 1.4], step=0.1)
    tit = r'$ \overline{||U|^{\prime}|} $'
    flow = PIVFramework(exp_data, rms=True)
    plot_2D_fp_isocontours(flow, 'mat_file', os.path.join(data_root, 'figures/exp_rms_mag.pdf'),
                           title=tit, lims=[0, .4])
    tit = r'$ \overline{|\omega|} $'
    flow = PIVFramework(exp_data, vort=True)
    plot_2D_fp_isocontours(flow, 'mat_file_vort', os.path.join(data_root, 'figures/exp_vort.pdf'),
                           title=tit, lims=[-115, 85], lvls=11)
    tit = r'$ \overline{|U|} $'
    flow = SimFramework(os.path.join(data_root, str(c) + '/3D'), 'spTAv',
                        length_scale=c, rotation=12)
    field = 'mag'
    plot_2D_fp_isocontours(flow, field, os.path.join(data_root, 'figures/sim_mag.pdf'),
                           title=tit, lims=[0, 1.4], step=0.1)
    tit = r'$ \overline{||U|^{\prime}|} $'
    flow = SimFramework(os.path.join(data_root, str(c)+'/3D'), 'rms2D',
                        length_scale=c, rotation=12)
    field = 'p'
    plot_2D_fp_isocontours(flow, field, os.path.join(data_root, 'figures/sim_rms_mag.pdf'),
                           title=tit, step=0.04, lims=[0, .4])
    #
    flow = SimFramework(os.path.join(data_root, str(c) + '/3D'), 'slVor',
                        length_scale=c, rotation=12)
    tit = r'$ \overline{|\omega|} $'
    field = 'vort'
    plot_2D_fp_isocontours(flow, field, os.path.join(data_root, 'figures/sim_vort.pdf'),
                           title=tit, lims=[-115, 85], lvls=11)

