# -*- coding: utf-8 -*-
"""
@author: Jonathan Massey
@description: Unpack flow field and plot the contours
@contact: jmom1n15@soton.ac.uk
"""

# Imports
import postproc.calc as calc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.colors as colors
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection


def _rec(theta):
    grey_color = '#dedede'
    return patches.Rectangle((0, -1 / 91.42), 1., 1 / 45.71, -theta, linewidth=0.2, edgecolor='red',
                             facecolor='red')


class Plot2DIsocontours:
    def __init__(self, data, interest, **kwargs):
        """
        Takes an input from the class postproc.flow_field and plots it with various options
        Args:
            data:
            interest:
            **kwargs: various options
        """
        self.title = kwargs.get('title', None)
        self.cmap = kwargs.get('cmap', sns.color_palette("seismic", as_cmap=True))
        try:
            self.X, self.Y = data.X, data.Y
        except AttributeError:
            pass
        # Now plot what we are interested in
        if interest == 'p':
            self.vals = data.p
        elif interest == 'u':
            self.vals = data.U
        elif interest == 'v':
            self.vals = data.V
        elif interest == 'mag':
            U, V = data.U, data.V
            self.vals = np.sqrt(V ** 2 + U ** 2)
            # vals = vals * data.iter_correction(30)
        elif interest == 'an_mag':
            self.X, self.Y = data[0:2]
            u, v, _ = data[2:-1]
            U, V = np.mean(u, axis=2), np.mean(v, axis=2)
            self.vals = np.sqrt(V ** 2 + U ** 2)
        elif interest == 'an_pres':
            self.X, self.Y = data[0:2]
            self.vals = np.mean(data[-1], axis=0)
            self.cmap = sns.color_palette("seismic", as_cmap=True)
        elif interest == 'an_vort':
            self.X, self.Y = data[0:2]
            u, v, _ = data[2:-1]
            U, V = np.mean(u, axis=2), np.mean(v, axis=2)
            dv_dx = np.gradient(V, axis=0, edge_order=2)
            du_dy = np.gradient(U, axis=1, edge_order=2)
            self.vals = dv_dx - du_dy
            self.cmap = sns.color_palette("seismic", as_cmap=True)
        elif interest == 'rms':
            self.vals = data.rms()
        elif interest == 'rms_mag':
            self.vals = data.rms_mag()
        elif interest == 'vort':
            dv_dx = np.gradient(data.V, axis=0, edge_order=2)
            du_dy = np.gradient(data.U, axis=1, edge_order=2)
            self.vals = dv_dx - du_dy
            self.cmap = sns.color_palette("seismic", as_cmap=True)
        elif interest == 'mat_file_vort':
            self.vals = data.omega
            self.cmap = sns.color_palette("seismic", as_cmap=True)
        elif interest == 'mat_file':
            self.X, self.Y = self.X-0.5, self.Y+0.5*np.sin(np.pi*12/180)
            self.vals = data.U.T
        elif interest == 'snap_mat_file':
            s = kwargs.get('snap', 0)
            self.vals = data.mag_snap[s].T

        self.kwargs = kwargs

    def plot_grid(self, fn_save):
        plt.style.use(['science'])
        fig, ax = plt.subplots(figsize=(7, 5))
        plt.title(self.title)
        ax.set_xlim((-4, 6))
        ax.set_ylim((-2.5, 2.5))

        ax.set_xlabel(r'$x/c^\star$')
        ax.set_ylabel(r'$y/c^\star$')

        x, y = self.X, self.Y
        ax.plot(x, y, c='k', linewidth=0.2)
        ax.plot(np.transpose(x), np.transpose(y), c='k', linewidth=0.2)

        rec = _rec(theta=0)
        ax.add_patch(rec)
        # ax.set_xticks()

        del self.X, self.Y, self.vals
        ax.set_aspect(1)
        ax.grid(False)

        plt.savefig(fn_save, dpi=800, transparent=True)
        plt.show()

    def plot_fill(self, fn_save):
        plt.style.use(['science', 'grid'])
        fig, ax = plt.subplots(figsize=(7, 5))
        plt.title(self.title)
        divider = make_axes_locatable(ax)
        # Plot the window of interest
        ax.set_xlim(self.kwargs.get('xlim', (-0.2, 2.3)))
        ax.set_ylim(self.kwargs.get('ylim', (-0.6, 0.6)))

        if self.kwargs.get('rec', False):
            rec = _rec(theta=0)
            ax.add_patch(rec)

        lim = [np.min(self.vals), np.max(self.vals)]
        lim = self.kwargs.get('lims', lim)
        if lim is None:
            lim = [np.min(self.vals), np.max(self.vals)]  # Hacky...

        # Put limits consistent with experimental data
        norm = colors.Normalize(vmin=lim[0], vmax=lim[1])
        levels = self.kwargs.get('levels', 101)
        step = self.kwargs.get('step', None)
        if step is not None:
            levels = np.arange(lim[0], lim[1] + step, step)
        else:
            levels = np.linspace(lim[0], lim[1], levels)

        cs = ax.contourf(self.X, self.Y, np.transpose(self.vals),
                         levels=levels, vmin=lim[0], vmax=lim[1],
                         norm=norm, cmap=self.cmap, extend='both')
        ax_cb = divider.new_horizontal(size="5%", pad=0.05)
        fig.add_axes(ax_cb)
        plt.colorbar(cs, cax=ax_cb)
        ax_cb.yaxis.tick_right()
        ax_cb.yaxis.set_tick_params(labelright=True)
        # plt.setp(ax_cb.get_yticklabels()[::2], visible=False)

        del self.X, self.Y, self.vals
        ax.set_aspect(1)

        plt.savefig(fn_save, dpi=300, transparent=True)
        plt.show()

    def plot_line(self, fn_save, **kwargs):
        plt.style.use(['science', 'grid'])
        fig, ax = plt.subplots(figsize=(8.5, 6))
        plt.title(self.title)
        divider = make_axes_locatable(ax)
        # Plot the window of interest
        ax.set_xlim(kwargs.get('xlim', (-0.2, 2.3)))
        ax.set_ylim(-0.5, 0.5)

        if kwargs.get('rec', False):
            rec = _rec(theta=12)
            ax.add_patch(rec)

        lim = [np.min(self.vals), np.max(self.vals)]
        lim = kwargs.get('lims', lim)
        # Put limits consistent with experimental data
        norm = colors.Normalize(vmin=lim[0], vmax=lim[1])
        levels = kwargs.get('levels', 6)
        step = kwargs.get('step', None)
        if step is not None:
            lvls = np.arange(lim[0], lim[1] + step, step)
        else:
            lvls = np.linspace(lim[0], lim[1], levels)

        cs = ax.contour(self.X, self.Y, np.transpose(self.vals),
                        levels=levels, vmin=lim[0], vmax=lim[1],
                        norm=norm, colors=kwargs.get('colors', sns.color_palette("tab10")))
        ax.clabel(cs, cs.levels[2::2], inline_spacing=1, inline=1, fontsize=12, fmt='%1.2f')
        del self.X, self.Y, self.vals
        ax.set_aspect(1)

        plt.savefig(fn_save, dpi=300, transparent=True)
        plt.show()

    def stack_contours(self, interesting_contour):
        """
        Stack contours from two different flow fields on top of each other for a good comparison.
        Returns: A contour plot

        """
