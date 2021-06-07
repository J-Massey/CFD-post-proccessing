# -*- coding: utf-8 -*-
"""
@author: Jonathan Massey
@description: Unpack flow field and plot the contours
@contact: jmom1n15@soton.ac.uk
"""

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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tkinter import Tcl
import imageio
from tqdm import tqdm
from matplotlib.ticker import FormatStrFormatter
from pygifsicle import optimize


def plot_2D_fp_isocontours(data, interest, fn_save):
    plt.style.use(['science'])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlabel(r'$x/D$')
    ax.set_ylabel(r'$y/D$')
    cmap = sns.color_palette("icefire", as_cmap=True)
    plt.title(title)
    divider = make_axes_locatable(ax)
    # Plot the window of interest
    ax.set_xlim(-2.5, 6)
    ax.set_ylim(-3, 3)

    X, Y = data[0:2]
    u, v, w = data[2:-1]
    p = data[-1][0]

    # Now plot what we are interested in
    if interest == 'p':
        vals = p*2
        cmap = sns.color_palette("PRGn_r", as_cmap=True)
    elif interest == 'u':
        vals = u
    elif interest == 'v':
        vals = np.mean(v, axis=2)
    elif interest == 'mag':
        U, V = np.mean(u, axis=2), np.mean(v, axis=2)
        vals = np.sqrt(V ** 2 + U ** 2)
        # vals = vals * data.iter_correction(30)
    elif interest == 'vort':
        U, V = np.mean(u, axis=2), np.mean(v, axis=2)
        vals = calc.vortZ(U, V)
        # vals = -data.p * data.length_scale  # Need to scale by length scale
        cmap = sns.color_palette("seismic", as_cmap=True)

    grey_color = '#dedede'
    circle = patches.Circle((0, 0), radius=0.5, linewidth=0.2, edgecolor='black', facecolor=grey_color)
    ax.add_patch(circle)

    lim = [np.min(vals), np.max(vals)]
    # lim = [0, 1.4]
    # lim = [-0.2, 0.2]
    lim = [-1.9, 1.]
    norm = colors.Normalize(vmin=lim[0], vmax=lim[1])
    # lvls = 121
    step = 0.01

    if step is not None:
        lvls = np.arange(lim[0], lim[1]+step, step)
    else:
        lvls = np.linspace(lim[0], lim[1], lvls)

    if filled:
        cs = ax.contourf(X, Y, np.transpose(vals),
                         levels=lvls, vmin=lim[0], vmax=lim[1],
                         norm=norm, cmap=cmap, extend='both')
        ax_cb = divider.new_horizontal(size="5%", pad=0.05)
        fig.add_axes(ax_cb)
        plt.colorbar(cs, cax=ax_cb)
        ax_cb.yaxis.tick_right()
        # ax_cb.yaxis.set_major_formatter(FormatStrFormatter('%1.1f'))
    else:
        cs = ax.contour(X, Y, np.transpose(vals),
                        levels=lvls, vmin=lim[0], vmax=lim[1],
                        colors=['k'], linewidths=0.4)

    ax.set_aspect(1)

    plt.savefig(fn_save, dpi=300)
    plt.close()


def save_frames(data, folder, interest):
    for idx, snap in tqdm(enumerate(data), desc='Plotting frames'):
        da = np.array(snap).T
        plot_2D_fp_isocontours(da, interest, os.path.join(folder, str(idx) + '.png'))


def animate(data, folder, interest):
    save_frames(data, folder, interest)
    # Sort filenames to make sure they're in order
    fn_images = os.listdir(folder)
    fn_images = Tcl().call('lsort', '-dict', fn_images)
    # Create gif
    gif_path = folder + '/flow'+interest+'.gif'
    with imageio.get_writer(gif_path, mode='I', duration=0.15) as writer:
        for filename in tqdm(fn_images[::4], desc='Loop images'):
            writer.append_data(imageio.imread(os.path.join(folder, filename)))
    optimize(gif_path)


class SnapShots:
    def __init__(self, snap):
        self.snaps = snap
        mean_t = np.mean(np.array(self.snaps).T, axis=1)
        self.X, self.Y = mean_t[0:2]
        self.u, self.v, self.w = mean_t[2:-1]
        self.U, self.V = np.mean(self.u, axis=2), np.mean(self.v, axis=2)
        self.p = np.mean(mean_t[-1], axis=0)


if __name__ == "__main__":
    snaps = np.load('snapshots/flow_snaps.npy', allow_pickle=True)

    data_root = '/home/masseyjmo/Workspace/Lotus/solver/postproc/circular_cylinder/figures/animations'

    interest = 'p'
    filled = True
    title = '$ p $'

    animate(snaps, os.path.join(data_root, 'frames_' + interest), interest)

    mean_ = np.mean(np.array(snaps).T, axis=1)
    fn_save = os.path.join(data_root + '/sim_' + interest + '.pdf')
    plot_2D_fp_isocontours(np.array(snaps).T[:, 102], interest, fn_save)
