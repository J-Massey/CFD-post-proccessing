# -*- coding: utf-8 -*-
"""
@author: Jonathan Massey
@description: Unpack flow field and plot the contours
@contact: jmom1n15@soton.ac.uk
"""

# Imports
from postproc.visualise.plot_flow import *
from postproc.visualise.flow_field import *
from postproc.visualise.plot_gif import *


if __name__ == "__main__":
    plt.style.use(['science', 'grid'])
    cases = ['half_bumps', 'smooth']
    # cases = ['full_bumps']
    for case in cases:
        data_root = '/home/masseyjmo/Workspace/Lotus/projects/waving_plate/' + case

        for c in [256]:
            # tit = r'$ \overline{|U|} $'
            # flow = SimFramework(os.path.join(data_root, str(c) + '/3D'), 'spTAv',
            #                     length_scale=c, rotation=0)
            # field = 'mag'
            # Plot2DIsocontours(flow, field, title=tit, lims=[0, 1.5], lvls=11)\
            #     .plot_line(os.path.join(data_root, 'figures/mag.png'))
            #
            # tit = r'$ \overline{||U|^{\prime}|} $'
            # flow = SimFramework(os.path.join(data_root, str(c) + '/3D'), 'spRms',
            #                     length_scale=c, rotation=theta)
            # field = 'p'
            # Plot2DIsocontours(flow, field, title=tit, lims=[0, 1.4], step=0.1) \
            #     .plot(os.path.join(data_root, 'figures/' + str(c) + '_sim_rms_mag.pdf'))
            flow = SimFramework(os.path.join(data_root, str(c) + '/3D'), 'flu2d',
                                length_scale=c, rotation=0)
            save_animate(flow, data_root, 'an_mag',
                         kwargs_plot={'title': r'$ \sqrt{u^2+v^2} $', 'lims': [0, 1.4], 'levels': 105})

