#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: J. Massey
@description: Res test for flat plate experiment with analysis Kurt
@contact: jmom1n15@soton.ac.uk
"""

# Imports
import numpy as np
import postproc.visualise.plotter
import postproc.io as io
import postproc.frequency_spectra
import matplotlib.pyplot as plt
import os
import seaborn as sns
from tqdm import tqdm
from postproc.flow.flow_field import FlowBase
from scipy import signal

from flat_plate.forces import read_rot

plt.style.use(['science', 'grid'])
plt.switch_backend('TkAgg')  # png


def TS(data_root, case):
    data_root = data_root + case
    force_file = '3D/fort.9'
    names = ['t', 'dt', 'px', 'py', 'pz']
    interest = 'p'
    label = r'$ C_{L_{p}} $'
    tit = None

    c = [128, 192, 256]
    theta = np.radians(12)
    colors = sns.color_palette("husl", len(c))

    # How long from 2D to 3D, and when to crop TS
    init = 120
    snip = 200

    # Write out labels so they're interpretable by latex
    labels = [r'$ c^{\star}/\Delta x = ' + str(i) + ' $' for i in c]

    fs, uks_labelled, uks = [], [], []

    # Plot TSs and save spectra
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
    ax1.set_xlabel(r'$ t/c^{\star} $')
    ax1.set_ylabel(label)
    for idx, fn in tqdm(enumerate(c), desc='File loop'):
        fos = (io.unpack_flex_forces(os.path.join(data_root, str(fn), force_file), names))
        forces_dic = dict(zip(names, fos))
        t, u = forces_dic['t'] / c[idx], np.array(forces_dic[interest + 'y'])

        # # Transform the forces into the correct plane
        # rot = np.array((np.cos(theta), -np.sin(theta)),
        #                (np.sin(theta), np.cos(theta)))
        # rot = np.array((np.sin(theta), np.cos(theta)))
        # # For the res test normalise pressure forces by frontal area
        # old_norm = (c[idx]/45.71*np.sin(np.radians(theta))*c[idx]*0.25*2)
        # new_norm = ((c[idx]/45.71*np.cos(np.radians(theta))  # Thickness
        #              + c[idx] * np.sin(np.radians(theta)))   # Body
        #             * c[idx]*0.25*2)                         # z
        #
        # u = rot.dot(u)*old_norm/new_norm
        t, u = t[t < snip], u[t < snip]
        t, u = t[t > init], u[t > init]
        sv = np.hstack((t, u))

        # Append the Welch spectra to a list in order to compare
        criteria = postproc.frequency_spectra.FreqConv(t, u, n=3, OL=0.5)
        f, uk = criteria.welch()
        fs.append(f)
        uks.append(uk)
        uks_labelled.append((labels[idx], uk))
        ax1.plot(t, u, label=labels[idx])

    ax1.legend()
    fig1.savefig(os.path.join(data_root, 'figures/TS_' + interest + '.png'), bbox_inches='tight', dpi=300,
                 transparent=False)
    plt.show()

    # postproc.visualise.plotter.plot_fft(os.path.join(data_root, 'figures/spectra_' + interest + '.png'),
    #                                     uks, fs, xlim=[0, 1],
    #                                     l_label=labels, colours=colors, title=tit)

    postproc.visualise.plotter.plotLogLogTimeSpectra_list(
        os.path.join(data_root, 'figures/log_spectra_' + interest + '.png'),
        uks_labelled, fs,
        title=tit,
        ylabel=r'Power spectra ' + label)


def res_test(data_root, cs, case, **kwargs):
    mean = kwargs.get('mean', False)
    rms = kwargs.get('rms', False)

    fig, ax = plt.subplots(figsize=(3, 2))

    ax.set_xlabel(r'$c^{\star} / \Delta x$')
    if mean:
        ax.set_ylabel(r'$C_{L_p}$')
    else:
        ax.set_ylabel(r'$RMS(C_{L_p}^{\prime})$')

    init = 120
    snip = 200

    hs = np.array(cs)
    data_root = data_root + case
    for idx, res in enumerate(cs):
        t, ux, u = read_rot(data_root, res)

        t, u = t[t < snip], u[t < snip]
        t, u = t[t > init], u[t > init]
        # if res == np.max(np.array(cs[-1])):
        #     if mean:
        #         ax.fill_between(hs, np.mean(u)*1.04, np.mean(u)*.96, color='green', zorder=0, label=r'$\pm 5\%$')
        #     else:
        #         ax.fill_between(hs,
        #                         np.sqrt(np.sum((u-np.mean(u))**2)/len(u)) * 1.05,
        #                         np.sqrt(np.sum((u-np.mean(u))**2)/len(u)) * .95,
        #                         color='green', zorder=0, label=r'$\pm 5\%$')
        if idx > 2:
            co = 'purple'
        else:
            co = 'blue'

        if mean:
            ax.scatter(hs[idx], np.mean(u), color=co, marker='X', s=104)

        print(np.mean(u), np.sqrt(np.sum((u - np.mean(u)) ** 2) / len(u)))
        if rms:
            if idx > 2:
                co = 'purple'
            else:
                co = 'blue'
            ax.scatter(hs[idx], np.sqrt(np.sum((u - np.mean(u)) ** 2) / len(u)), color=co, marker='P', s=104)
            # print(np.sqrt(np.sum((u - np.mean(u)) ** 2) / len(u)))
    # ax.legend()
    if mean:
        ext = 'mean'
    if rms:
        ext = 'rms'

    fig.savefig(os.path.join(data_root, 'figures/cl_' + ext + '.png'),
                bbox_inches='tight', dpi=300, transparent=True)

    plt.close()


def flow_res_test():
    data_root = '/home/masseyjmo/Workspace/Lotus/projects/flat_plate/AoA_12/'
    labels = [r'Smooth', r'$ 70\% $']
    cs = [64, 96, 128, 192, 256]
    case = ['smooth', 'full_bumps']
    colours = ['blue', 'red']
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.set_xlabel(r'$\Delta x / c^{\star}$,  Downsampled $\Delta x / c^{\star}$')
    ax.set_ylabel(r'$L_{2}$ distance')
    for idx, loop in enumerate(case):
        flows = []
        for idx2, c in enumerate(cs):
            sim_dir = data_root + loop
            extension = str(c) + '/3D'
            flow = FlowBase(os.path.join(sim_dir, extension), os.path.join(sim_dir, extension), 256, 'spTAv')
            flows.append(np.squeeze(flow.p))

        n = 2
        for idx2 in range(len(cs) - n):
            if idx2 == 0:
                lab = labels[idx]
            else:
                lab = None
            arr = flows[idx2 + n]
            window = (1.0 / (2 * n)) * np.ones((n, n))
            res = signal.convolve2d(arr, window, mode='valid')[::n, ::n]
            ax.scatter(idx2, np.average(np.sqrt(abs(flows[idx2][0:-1, 0:-1] ** 2 - res ** 2))),
                       color=colours[idx], label=lab, marker='+', s=90)
    ax.set_xticks(range(len(cs) - n))
    ax.set_xticklabels([r'$L_{2}(64, 128)$', r'$L_{2}(96, 192)$', r'$L_{2}(128, 256)$'])
    plt.legend()
    fig.savefig(os.path.join(data_root, 'comparisons/l2_pressure.png'),
                bbox_inches='tight', dpi=700, transparent=True)


if __name__ == '__main__':
    c = [256]
    # flow_res_test()
    # plot_coeffs(drag=True)
    data_root = '/home/masseyjmo/Workspace/Lotus/projects/flat_plate/AoA_12/'
    res_test(data_root, c, 'full_bumps', mean=True)
    res_test(data_root, c, 'full_bumps', rms=True)
    # c = [128, 192, 256]
    # TS(data_root, 'smooth')
