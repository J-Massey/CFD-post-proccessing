# -*- coding: utf-8 -*-
"""
@author: Jonathan Massey
@description: Import this to grab a few functions to plot a gif
@contact: jmom1n15@soton.ac.uk
"""
import os
from tkinter import Tcl

import imageio
from pygifsicle import optimize
from tqdm import tqdm

from postproc.visualise.plot_flow import Plot2DIsocontours


def animate(folder, interest, **kwargs):
    duration = kwargs.get('duration', 0.15)
    # Sort filenames to make sure they're in order
    fn_images = os.listdir(folder+'/animation')
    fn_images = Tcl().call('lsort', '-dict', fn_images)
    # Create gif
    gif_path = os.path.join(folder, interest + '.gif')
    with imageio.get_writer(gif_path, mode='I', duration=duration,  **kwargs) as writer:
        for filename in tqdm(fn_images[::1], desc='Loop images'):
            writer.append_data(imageio.imread(os.path.join(folder, 'animation', filename)))
    optimize(gif_path)


def save_piv_frames(data, folder, interest, tit=None):
    for snap in range(len(data.mag_snap)):
        Plot2DIsocontours(data, interest,
                          title=tit, lims=[0, 1.4], step=0.1, snap=snap).plot_fill(
                           os.path.join(folder, str(snap) + '.png'))


def save_sim_frames(data, folder, interest, **kwargs):
    os.system('mkdir -p '+os.path.join(folder, 'animation'))
    os.system('rm '+os.path.join(folder, 'animation')+'/*.png')
    for idx, snap in tqdm(enumerate(data.snaps), desc='Plotting snapshots'):
        Plot2DIsocontours(snap, interest, **kwargs) \
            .plot_fill(os.path.join(folder, 'animation', str(idx) + '.png'))


def save_animate(data, data_root, quantity, kwargs_plot={}, kwargs_animate={}):
    """
    Save the frames and animate the quantity of interest
    Args:
        data: the flow field defined by the class SimFramework
        data_root: file path to an empty folder to save frames to
        quantity: what we are animating
        **kwargs_plot: args for plotting the flow
        **kwargs_animate: extra args for imageio.get_writer

    Returns:

    """
    save_sim_frames(data, os.path.join(data_root, 'figures'), quantity, **kwargs_plot)
    animate(os.path.join(data_root, 'figures'), quantity, **kwargs_animate)

