# -*- coding: utf-8 -*-
"""
@author: Jonathan Massey
@description: Load in the .npy data and cycle average
@contact: jmom1n15@soton.ac.uk
"""

from matplotlib import pyplot as plt
from postproc.flow.flow_field import FlowBase
from postproc.flow.load_snaps import LoadSnaps
import os

plt.style.use(['science', 'grid'])


class SPOD(FlowBase):
    def __init__(self, snapshots, figure_root):
        """
        Class that takes the 2D flow and performs spectral POD.
        Args:
            data_root: The path to the Lotus simulation directory.
            quantity: The quantity saved, 'flu2d', 'avRms', etc...
        """
        super().__init__(snapshots, spanwise_avg=True)
        self.figure_root = figure_root


if __name__ == '__main__':
    parent_dir = '/home/masseyjmo/Workspace/Lotus/projects/flat_plate/AoA_12/quick_access_data'
    quantity = 'pres'
    case = ['full_bumps']
    case = ['smooth']
    extension = 'high_f'
    for idx, loop in enumerate(case):
        data_root = os.path.join(parent_dir, loop, extension)
        snaps = LoadSnaps(data_root, 256, 'flu2d_0.npy').snapshots
# Imports
from postproc.visualise.plot_flow import *
from postproc.visualise.flow_field import *
from postproc.visualise.plot_gif import *

if __name__ == "__main__":
    plt.style.use(['science', 'grid'])
    cases = ['full_bumps']#, 'full_bumps']
    for case in cases:
        data_root = '/home/masseyjmo/Workspace/Lotus/projects/waving_plate/quick_access_data/' + case

        for c in [256]:
            # tit = r'$ \overline{|U|} $'
            #
            flow = SimFramework(os.path.join(data_root, str(c) + '/3D'), 'flu2d',
                                length_scale=c, rotation=0)
            # field = 'mag'
            # Plot2DIsocontours(flow, field)\
            #     .plot_grid(os.path.join(data_root,
            #                             '/home/masseyjmo/Workspace/Lotus/projects/'
            #                             'waving_plate/comparisons/sim_grid.png'))

            field = 'mag'
            Plot2DIsocontours(flow, field, title=None, lims=[0, 1.5], levels=101,
                              cmap=sns.color_palette("icefire", as_cmap=True), xlim=(-1, 2.5), ylim=(-0.6, 0.6)) \
                .plot_fill(os.path.join(data_root, 'figures/sim_pres.png'))
            #
            # tit = r'$ \overline{||U|^{\prime}|} $'
            # flow = SimFramework(os.path.join(data_root, str(c) + '/3D'), 'spRms',
            #                     length_scale=c, rotation=theta)
            # field = 'p'
            # Plot2DIsocontours(flow, field, title=tit, lims=[0, 1.4], step=0.1) \
            #     .plot(os.path.join(data_root, 'figures/' + str(c) + '_sim_rms_mag.pdf'))
            flow = SimFramework(os.path.join(data_root, str(c) + '/save'), 'flu2d',
                                length_scale=c, rotation=0)
            save_animate(flow, data_root, 'an_vort',
                         kwargs_plot={'title': r'$ \omega $', 'lims': [-0.5, 0.5], 'levels': 105, 'down': 14})
