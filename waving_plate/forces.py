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

plt.style.use(['science', 'grid'])
cases = ['full_bumps', 'smooth']
labels = [r'$ 75\% $', r'Smooth']


def read_rot(data_root, c):
    force_file = '3D/fort.9'
    names = ['t', 'dt', 'px', 'py', 'pz']
    interest = 'p'
    fos = (io.unpack_flex_forces(os.path.join(data_root, str(c), force_file), names))
    forces_dic = dict(zip(names, fos))
    t = forces_dic['t'] / c
    u = np.array((forces_dic[interest + 'x'], forces_dic[interest + 'y']))
    ux, uy = np.squeeze(np.array(u[0])), np.squeeze(np.array(u[1]))

    return t, ux, uy


def plot_coeffs(**kwargs):
    drag = kwargs.get('drag', False)
    lift = kwargs.get('lift', True)

    tit = r'Power spectra comparison'

    colors = sns.color_palette("husl", len(cases))

    # How long from 2D to 3D, and when to crop TS
    init = 40
    snip = 200

    fs, uks_labelled, uks = [], [], []
    # Plot TSs and save spectra
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.set_title(r'Resolution test')
    ax1.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
    ax1.set_xlabel(r'$ t/c $')

    if drag: label = r'$ C_{D} $'
    if lift: label = r'$ C_{L} $'

    ax1.set_ylabel(label)
    for idx2, case in enumerate(cases):
        data_root = '/home/masseyjmo/Workspace/Lotus/projects/waving_plate/' + case

        t, ux, uy = read_rot(data_root, c)

        if lift:
            u = uy
        if drag:
            u = ux

        t, u = t[t < snip], u[t < snip]
        t, u = t[t > init], u[t > init]

        # Append the Welch spectra to a list in order to compare
        criteria = postproc.frequency_spectra.FreqConv(t, u, n=5, OL=0.5)
        f, uk = criteria.welch()
        fs.append(f)
        uks.append(uk)
        uks_labelled.append((labels[idx2], uk))
        ax1.plot_fill(t, u, label=labels[idx2], color=colors[idx2])

    ax1.legend()
    if drag:
        fig1.savefig(os.path.join(data_root, '../comparisons/cd_comparison.pdf'),
                     bbox_inches='tight', dpi=30, transparent=False)
        postproc.visualise.plotter.plotLogLogTimeSpectra_list(
            os.path.join(data_root, '../comparisons/log_spectra_cd_comparison.pdf'),
            uks_labelled, fs,
            title=tit,
            ylabel=r'$PS$ ' + label,
            colors=colors)
    if lift:
        fig1.savefig(os.path.join(data_root, '../comparisons/cl_comparison.pdf'),
                     bbox_inches='tight', dpi=30, transparent=False)
        postproc.visualise.plotter.plotLogLogTimeSpectra_list(
            os.path.join(data_root, '../comparisons/log_spectra_cl_comparison.pdf'),
            uks_labelled, fs,
            title=tit,
            ylabel=r'$PS$ ' + label,
            colors=colors)
    plt.close()


def plot_cl_cd(aoas, theta):

    colors = sns.color_palette("husl", len(cases))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(r'$C_{L}/C_{D}$')
    ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
    ax.set_xlabel(r'Roughness case')
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels(labels)

    init = 100
    snip = 200
    for idx, alpha in enumerate(aoas):
        for idx2, case in enumerate(cases):
            data_root = '/home/masseyjmo/Workspace/Lotus/projects/waving_plate/' + str(alpha) + '/' + case
            t, ux, uy = read_rot(data_root, c, theta[idx])
            u = uy / ux
            t, u = t[t < snip], u[t < snip]
            t, u = t[t > init], u[t > init]

            if idx2 == 3:
                ax.axhline(np.mean(u), c=colors[idx], ls='--')

            if idx2 == 0:
                lab = r'AoA $' + str(theta[idx]) + r'$'
            else:
                lab = None
            ax.scatter(idx2, np.mean(u), color=colors[idx], label=lab)
    ax.legend()
    fig.savefig(os.path.join(data_root, '../../figures/cl_cd.pdf'),
                bbox_inches='tight', dpi=30, transparent=True)
    plt.close()


def res_test(cs, case, **kwargs):

    drag = kwargs.get('drag', False)
    lift = kwargs.get('lift', False)
    mean = kwargs.get('mean', False)
    rms = kwargs.get('rms', False)

    colors = sns.color_palette("husl", len(cases))

    fig, ax = plt.subplots(figsize=(5, 3))

    if drag:
        ax.set_title(r'$C_{D}$')
    if lift:
        ax.set_title(r'$C_{L}$')

    ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
    ax.set_xlabel(r'h')
    if mean:
        ax.set_ylabel(r'Mean')
    else:
        ax.set_ylabel(r'RMS')

    init = 30
    snip = 200

    hs = np.max(np.array(cs[-1]))/np.array(cs)
    for idx, res in enumerate(cs):
        data_root = '/home/masseyjmo/Workspace/Lotus/projects/waving_plate/' + case
        t, ux, uy = read_rot(data_root, res)

        if drag:
            u = ux
        if lift:
            u = uy

        t, u = t[t < snip], u[t < snip]
        t, u = t[t > init], u[t > init]
        if res == np.max(np.array(cs[-1])):
            if mean:
                ax.fill_between(hs, np.mean(u)*1.03, np.mean(u)*.97, color='green', zorder=0, label=r'$\pm 3\%$')
            else:
                ax.fill_between(hs,
                                np.sqrt(np.sum((u-np.mean(u))**2)/len(u)) * 1.03,
                                np.sqrt(np.sum((u-np.mean(u))**2)/len(u)) * .97,
                                color='green', zorder=0, label=r'$\pm 3\%$')

        if mean:
            if idx > 3:
                co = 'white'
            else:
                co = 'k'
            ax.scatter(hs[idx], np.mean(u), color=co, marker='X', s=104)
            print(np.mean(u))
        if rms:
            if idx > 3:
                co = 'white'
            else:
                co = 'k'
            ax.scatter(hs[idx], np.sqrt(np.sum((u-np.mean(u))**2)/len(u)), color=co, marker='P', s=104)
            print(np.sqrt(np.sum((u - np.mean(u)) ** 2) / len(u)))
    ax.legend()
    if mean:
        ext = 'mean'
    if rms:
        ext = 'rms'
        ax.set_title(r'RMS $C_{D_p}$')

    if drag:
        fig.savefig(os.path.join(data_root, 'figures/cd_'+ext+'.png'),
                    bbox_inches='tight', dpi=300, transparent=True)
    if lift:
        fig.savefig(os.path.join(data_root, 'figures/cl_'+ext+'.png'),
                    bbox_inches='tight', dpi=300, transparent=True)

    plt.close()


if __name__ == "__main__":
    c = [32, 64, 96, 128, 192, 256]
    # plot_coeffs(drag=True)
    res_test(c, 'smooth', drag=True, mean=True)
    res_test(c, 'smooth', drag=True, rms=True)
    # plot_cl_cd(AoAs, thetas)
    # plot_coeffs_rms(AoAs, thetas, lift=True, rms=True)
    # plot_coeffs_rms(AoAs, thetas, lift=True, mean=True)
