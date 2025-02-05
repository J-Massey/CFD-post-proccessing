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

plt.style.use(['science', 'grid'])
cases = ['full_bumps', 'smooth']
labels = [r'$ 75\% $', r'Smooth']
thetas = [12]


def read_rot(data_root, c, theta):
    force_file = '3D/fort.9'
    names = ['t', 'dt', 'px', 'py', 'pz']
    interest = 'p'
    fos = (io.unpack_flex_forces(os.path.join(data_root, str(c), force_file), names))
    forces_dic = dict(zip(names, fos))
    t = forces_dic['t'] / c
    u = np.array((forces_dic[interest + 'x'], forces_dic[interest + 'y']))

    # Transform the forces into the correct plane
    co, si = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    rot = np.matrix([[co, si], [-si, co]])
    m = np.dot(rot, [u[0], u[1]])

    old_norm = (c / 45.71 * np.sin(np.radians(theta)) * c * 0.25 * 2)
    new_norm = ((c / 45.71 * np.cos(np.radians(theta))  # Thickness
                 + c * np.sin(np.radians(theta)))  # Body
                * c * 0.25 * 2)  # z
    exp_norm = (c * (c * 0.25 * 2))
    ux, uy = np.squeeze(np.array(m[0])), np.squeeze(np.array(m[1]))

    return t, ux * old_norm / exp_norm, uy * old_norm / exp_norm


def plot_coeffs(theta, **kwargs):
    drag = kwargs.get('drag', False)
    lift = kwargs.get('lift', True)

    tit = r'Power spectra comparison'

    colors = sns.color_palette("husl", len(cases))

    # How long from 2D to 3D, and when to crop TS
    init = 100
    snip = 200

    fs, uks_labelled, uks = [], [], []
    # Plot TSs and save spectra
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.set_title(r'AoA = $' + str(theta) + r'$')
    ax1.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
    ax1.set_xlabel(r'$ t/c $')

    if drag: label = r'$ C_{D} $'
    if lift: label = r'$ C_{L} $'

    ax1.set_ylabel(label)
    for idx2, case in enumerate(cases):
        data_root = '/home/masseyjmo/Workspace/Lotus/projects/flat_plate/' + alpha + '/' + case

        t, ux, uy = read_rot(data_root, c, theta)

        old_norm = (c / 45.71 * np.sin(np.radians(theta)) * c * 0.25 * 2)
        new_norm = ((c / 45.71 * np.cos(np.radians(theta))  # Thickness
                     + c * np.sin(np.radians(theta)))  # Body
                    * c * 0.25 * 2)  # z
        exp_norm = (c * (c * 0.25 * 2))

        if lift:
            u = uy * old_norm / exp_norm  # Change normalisation to match experiments
        if drag:
            u = ux * old_norm / exp_norm

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
            data_root = '/home/masseyjmo/Workspace/Lotus/projects/flat_plate/' + str(alpha) + '/' + case
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


def plot_coeffs_rms(aoas, theta, **kwargs):

    drag = kwargs.get('drag', False)
    lift = kwargs.get('lift', False)
    mean = kwargs.get('mean', False)
    rms = kwargs.get('rms', False)

    colors = sns.color_palette("husl", len(cases))

    fig, ax = plt.subplots(figsize=(7, 5))

    if drag:
        ax.set_title(r'$C_{D}$')
    if lift:
        ax.set_title(r'$C_{L}$')

    ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
    ax.set_xlabel(r'Roughness case')
    if mean:
        ax.set_ylabel(r'Mean')
    else:
        ax.set_ylabel(r'RMS')
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels(labels)

    init = 100
    snip = 200
    for idx, alpha in enumerate(aoas):
        for idx2, case in enumerate(cases):
            data_root = '/home/masseyjmo/Workspace/Lotus/projects/flat_plate/' + str(alpha) + '/' + case
            t, ux, uy = read_rot(data_root, c, theta[idx])

            if drag:
                u = ux
            if lift:
                u = uy

            t, u = t[t < snip], u[t < snip]
            t, u = t[t > init], u[t > init]
            if idx2 == 3:
                if mean:
                    ax.axhline(np.mean(u), c=colors[idx], ls='--')
                else:
                    ax.axhline(np.sqrt(np.sum((u-np.mean(u))**2)/len(u)), c=colors[idx], ls='--')

            if idx2 == 0:
                lab = r'AoA $' + str(theta[idx]) + r'$'
            else:
                lab = None

            if mean:
                ax.scatter(idx2, np.mean(u), color=colors[idx], label=lab)
                print(np.mean(u))
            if rms:
                ax.scatter(idx2, np.sqrt(np.sum((u-np.mean(u))**2)/len(u)), color=colors[idx], label=lab)
                print(np.sqrt(np.sum((u - np.mean(u)) ** 2) / len(u)))
    ax.legend()
    if mean:
        ext = 'mean'
    if rms:
        ext = 'rms'
        ax.set_title(r'RMS $C_{L_p}$')

    if drag:
        fig.savefig(os.path.join(data_root, '../../meta_figures/cd_'+ext+'.pdf'),
                    bbox_inches='tight', dpi=30, transparent=True)
    if lift:
        fig.savefig(os.path.join(data_root, '../../meta_figures/cl_'+ext+'.pdf'),
                    bbox_inches='tight', dpi=30, transparent=True)

    plt.close()


if __name__ == "__main__":
    AoAs = ['AoA_12', '25k', 'AoA_2']
    c = 256
    for idx, alpha in tqdm(enumerate(AoAs)):
        plot_coeffs(thetas[idx], lift=True)
    #     plot_coeffs(thetas[idx], drag=True)
    # plot_cl_cd(AoAs, thetas)
    # plot_coeffs_rms(AoAs, thetas, lift=True, rms=True)
    # plot_coeffs_rms(AoAs, thetas, lift=True, mean=True)
