# -*- coding: utf-8 -*-
"""
@author: Jonathan Massey
@description: Unpack flow field and plot the contours
@contact: jmom1n15@soton.ac.uk
"""

# Imports
import os.path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.colors as colors
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from postproc.flow.flow_field import FlowBase


def _rec(theta):
    grey_color = '#dedede'
    return patches.Rectangle((0, -1 / 91.42), 1., 1 / 45.71, -theta, linewidth=0.2, edgecolor='black',
                             facecolor=grey_color)


def plot_fill(x, y, vals, fn_save, **kwargs):
    plt.style.use(['science', 'grid'])
    fig, ax = plt.subplots(figsize=(7, 5))
    plt.title(kwargs.get('title', None))
    divider = make_axes_locatable(ax)
    # Plot the window of interest
    ax.set_xlim(kwargs.get('xlim', (-1, 2)))
    ax.set_ylim(kwargs.get('ylim', (-0.5, 0.5)))

    if kwargs.get('rec', False):
        rec = _rec(theta=12)
        ax.add_patch(rec)

    lim = [np.min(vals), np.max(vals)]
    lim = kwargs.get('lims', lim)
    if lim is None:
        lim = [np.min(vals), np.max(vals)]    # Hacky...

    # Put limits consistent with experimental data
    norm = colors.Normalize(vmin=lim[0], vmax=lim[1])
    levels = kwargs.get('levels', 11)
    step = kwargs.get('step', None)
    if step is not None:
        levels = np.arange(lim[0], lim[1] + step, step)
    else:
        levels = np.linspace(lim[0], lim[1], levels)

    cmap = kwargs.get('cmap', sns.color_palette("icefire", as_cmap=True))
    cs = ax.contourf(x, y, np.transpose(vals),
                     levels=levels, vmin=lim[0], vmax=lim[1],
                     norm=norm, cmap=cmap, extend='both')
    # ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    # fig.add_axes(ax_cb)
    # plt.colorbar(cs, cax=ax_cb)
    # ax_cb.yaxis.tick_right()
    # ax_cb.yaxis.set_tick_params(labelright=True)
    # # plt.setp(ax_cb.get_yticklabels()[::2], visible=False)
    #
    # del x, y, vals
    ax.set_aspect(1)

    plt.savefig(fn_save, dpi=300, transparent=True)
    plt.show()

def plot_line(x, y, vals, fn_save, **kwargs):
    plt.style.use(['science', 'grid'])
    fig, ax = plt.subplots(figsize=(7, 5))
    plt.title(kwargs.get('title', None))
    divider = make_axes_locatable(ax)
    # Plot the window of interest
    ax.set_xlim(kwargs.get('xlim', (-1, 2)))
    ax.set_ylim(kwargs.get('ylim', (-0.5, 0.5)))

    if kwargs.get('rec', False):
        rec = _rec(theta=12)
        ax.add_patch(rec)

    lim = [np.min(vals), np.max(vals)]
    lim = kwargs.get('lims', lim)
    if lim is None:
        lim = [np.min(vals), np.max(vals)]  # Hacky...

    # Put limits consistent with experimental data
    norm = colors.Normalize(vmin=lim[0], vmax=lim[1])
    levels = kwargs.get('levels', 11)

    cs = ax.contour(x, y, np.transpose(vals),
                    levels=levels, vmin=lim[0], vmax=lim[1],
                    norm=norm, colors=sns.color_palette("tab10"))
    ax.clabel(cs, cs.levels[::2], inline_spacing=1, inline=1, fontsize=12, fmt='%1.2f')
    del x, y, vals
    ax.set_aspect(1)

    plt.savefig(fn_save, dpi=300, transparent=True)
    plt.close()


def stack_contours(x, y, interesting_contours):
    """
    Stack contours from two different flow fields on top of each other for a good comparison.
    Returns: A contour plot

    """


if __name__ == '__main__':
    extension = '256/3D'
    sim_dir = '/home/masseyjmo/Workspace/Lotus/projects/flat_plate/AoA_12/full_bumps'
    flow = FlowBase(os.path.join(sim_dir, extension), 256, 'flu2d')
    save_figure_to = os.path.join(sim_dir, 'figures/test.png')
    flow.plot_fill(flow.x, flow.y, flow.magnitude[0].T, save_figure_to,
                   title='Test new plot method', lims=[0, 1.4], levels=101)
