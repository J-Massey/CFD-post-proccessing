# -*- coding: utf-8 -*-
"""
@author: Jonathan Massey
@description: Loads in data for the SPOD analysis
@contact: jmom1n15@soton.ac.uk
"""

# We now show how to construct a data reader that can be passed
# to the constructor of pyspod to read data sequentially (thereby
# reducing RAM requirements)

# Reader for netCDF
import os
from tkinter import Tcl
from tqdm import tqdm
from postproc.flow.flow_field import FlowBase
from postproc.visualise.plot_flow_new import plot_fill
from pyspod.spod_low_ram import SPOD_low_ram
import numpy as np

import matplotlib.colors as colors
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class SPOD:
    def __init__(self, data_root, figure_root):
        """
        Class that takes the 2D flow and performs spectral POD.
        Args:
            data_root: The path to the Lotus simulation directory.
        """
        self.data_root = data_root
        self.figure_root = figure_root

    def params(self):
        # Let's define the required parameters into a dictionary
        params = dict()

        # -- required parameters
        params['dt'] = 1  # data time-sampling
        params['nt'] = 3999  # number of time snapshots (we consider all data)
        params['xdim'] = 2  # number of spatial dimensions (longitude and latitude)
        params['nv'] = 1  # number of variables
        params['n_FFT'] = 100  # length of FFT blocks (100 time-snapshots)

        # -- optional parameters
        params['conf_level'] = 0.95  # calculate confidence level
        params['n_freq'] = params['n_FFT'] / 2 + 1  # number of frequencies
        params['n_overlap'] = np.ceil(params['n_FFT'] * 0 / 100)  # dimension block overlap region
        params['mean'] = 'blockwise'  # type of mean to subtract to the data
        params['normalize'] = False  # normalization of weights by data variance
        params['savedir'] = os.path.join(self.figure_root, 'spod', 'simple_test')  # folder where to save results

        # params['weights'] = None  # if set to None, no weighting (if not specified, Default is None)
        # params['savefreqs'] = np.arange(0, params['n_freq'])  # frequencies to be saved
        # params['n_modes_save'] = 3  # modes to be saved
        # params['normvar'] = False  # normalize data by data variance
        # params['conf_level'] = 0.95  # calculate confidence level
        # params['savefft'] = True  # save FFT blocks to reuse them in the future (saves time)
        return params

    @staticmethod
    def read_data(data, t_0, t_end, variables, snapshots=3999):
        """
        Data handler for the SPOD package.

        Two considerations:
            - t_0-t_end sits in one file
            - t_0-t_end spans >= two files
        Args:
            data: path to data folder
            t_0: start time
            t_end: end time
            variables: blank call for compatibility
            snapshots: number of snapshots

        Returns:

        """
        # ----- Load and sort the data files ----- #
        fns = os.listdir(data)
        numpy_saves = [fn for fn in fns if fn.endswith('.npy')]
        fns = Tcl().call('lsort', '-dict', numpy_saves)
        # ----- Figure out which file we need depending on time stamp -----#
        last_file = np.shape(np.load(os.path.join(data, fns[-1]),
                             allow_pickle=True))[0]
        time_per_file = int((snapshots-last_file)/(len(fns)-1))

        # ----- How much time do we need from beginning and end file ----- #
        f_idx_0 = int(t_0 - time_per_file * (t_0 // time_per_file))
        f_idx_end = int(t_end - time_per_file * (t_end // time_per_file))

        # ----- If the time stamps fit within one file ----- #
        if t_0//time_per_file == t_end//time_per_file:
            X = FlowBase(np.load(os.path.join(data, fns[int(t_0//time_per_file)]),
                         allow_pickle=True), spanwise_avg=True).p
            X = X[f_idx_0:f_idx_end]
            # print('\nFinal shape', np.shape(X))
            return X

        # ----- If the time stamp spans multiple files ----- #
        elif t_end//time_per_file - t_0//time_per_file >= 1:
            t_0_file = FlowBase(np.load(os.path.join(data, fns[int(t_0 // time_per_file)]),
                                        allow_pickle=True), spanwise_avg=True).p
            t_end_file = FlowBase(np.load(os.path.join(data, fns[int(t_end // time_per_file)]),
                                          allow_pickle=True), spanwise_avg=True).p
            t_0_file, t_end_file = t_0_file[f_idx_0:], t_end_file[0:f_idx_end]

            # print('Before loop', np.shape(t_0_file))
            for mid_idx in range(1, int(t_end//time_per_file - t_0//time_per_file)-1):
                f_idx = int(t_0 // time_per_file) + mid_idx
                tmp = FlowBase(np.load(os.path.join(data, fns[f_idx]),
                                       allow_pickle=True), spanwise_avg=True).p
                t_0_file = np.concatenate((t_0_file, tmp), axis=0)
                # print('\nIn loop', np.shape(t_0_file))

            X = np.concatenate((t_0_file, t_end_file), axis=0)
            # print('\nFinal shape', np.shape(X))
            return X

    def fit_spod(self):
        # p = [loop.T for loop in self.p]
        spod_ls = SPOD_low_ram(
            data=self.data_root,
            params=self.params(),
            data_handler=self.read_data,
            variables=['p'])
        spod_ls.fit()
        return spod_ls


if __name__ == '__main__':
    parent_dir = '/home/masseyjmo/Workspace/Lotus/projects/flat_plate/AoA_12/quick_access_data'
    quantity = 'pres'
    case = ['full_bumps']
    case = ['smooth']
    extension = 'high_f'
    for idx, loop in enumerate(case):
        d_root = os.path.join(parent_dir, loop, extension)
        figure_path = os.path.join(parent_dir, loop, 'figures')
        spod = SPOD(d_root, figure_path).fit_spod()

#%%
flow = FlowBase(np.load(os.path.join(d_root, 'flu2d_0.npy'),
                        allow_pickle=True), spanwise_avg=True)
mode = np.load(spod.modes[0]).real
vals = np.squeeze(mode)[:, :, 0]
x, y, vals, fn_save = flow.x, flow.y, vals, os.path.join(figure_path, 'spod_mode_1.png')
#%%
plot_fill(x, y, vals, fn_save)
#%%
plt.style.use(['science', 'grid'])
fig, ax = plt.subplots(figsize=(7, 5))
# plt.title(kwargs.get('title', None))
divider = make_axes_locatable(ax)

# Plot the window of interest
# ax.set_xlim(kwargs.get('xlim', (-1, 2))
# ax.set_ylim(kwargs.get('ylim', (-0.5, 0.5)))

# if kwargs.get('rec', False):
#     rec = _rec(theta=12)
#     ax.add_patch(rec)

# lim = [np.min(vals), np.max(vals)]
# lim = kwargs.get('lims', lim)
# if lim is None:
#     lim = [np.min(vals), np.max(vals)]  # Hacky...

# Put limits consistent with experimental data
# norm = colors.Normalize(vmin=lim[0], vmax=lim[1])
# levels = kwargs.get('levels', 11)
# step = kwargs.get('step', None)
# if step is not None:
#     levels = np.arange(lim[0], lim[1] + step, step)
# else:
#     levels = np.linspace(lim[0], lim[1], levels)

# levels = np.linspace(lim[0], lim[1], 101)

# cmap = kwargs.get('cmap', sns.color_palette("icefire", as_cmap=True))

cmap = sns.color_palette("seismic", as_cmap=True)
ax.contourf(x, y, np.transpose(vals))
# cs = ax.contourf(x, y, np.transpose(vals),
#                  levels=levels, vmin=lim[0], vmax=lim[1],
#                  norm=norm, cmap=cmap, extend='both')
#
# ax_cb = divider.new_horizontal(size="5%", pad=0.05)
# fig.add_axes(ax_cb)
# plt.colorbar(cs, cax=ax_cb)
# ax_cb.yaxis.tick_right()
# ax_cb.yaxis.set_tick_params(labelright=True)
# # plt.setp(ax_cb.get_yticklabels()[::2], visible=False)

# del x, y, vals
# ax.set_aspect(1)

plt.savefig(fn_save, dpi=300, transparent=True)
plt.show()



