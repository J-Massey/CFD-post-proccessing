# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: J. Massey
@description: Load profiles from files and plot
@contact: jmom1n15@soton.ac.uk
"""
import numpy as np
from load_raw import LoadData
import matplotlib.pyplot as plt
import seaborn as sns

data_root = '/home/masseyjmo/Workspace/Lotus/projects/cylinder_dns/data'


def mu_0(dis):
    mu = np.empty(np.shape(dis))
    mu[abs(dis) < 2] = 0.5 * (1 + dis[abs(dis) < 2] / 2 + 1 / np.pi * np.sin(dis[abs(dis) < 2] / 2 * np.pi))
    mu[dis <= -2] = 0
    mu[dis >= 2] = 1
    return mu


def mu_1(dis):
    mu = np.empty(np.shape(dis))
    mu[abs(dis) < 2] = 2 * (1 / 4 - (dis[abs(dis) < 2] / (2 * 2)) ** 2 -
                            1 / (2 * np.pi)
                            * (dis[abs(dis) < 2] / 2 * np.sin(dis[abs(dis) < 2] * np.pi / 2)
                               + 1 / np.pi * (1 + np.cos(dis[abs(dis) < 2] * np.pi / 2))))
    mu[abs(dis) >= 2] = 0
    return mu


def sub_smear_prof(profile):
    dis = (np.linspace(0, 1, np.shape(profile)[-1]) - 1 / 2)*np.shape(profile)[-1]
    du = np.gradient(profile, axis=-1)
    for i in range(20):
        profile = profile * np.roll(mu_0(dis), -1) + np.roll(du, 0) * np.roll(mu_1(dis), -1)
        du = np.gradient(profile, axis=-1)
    return profile


def plot_profile():
    # Plot TSs and save spectra
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
    ax.set_xlabel(r'$n/D$')
    ax.set_ylabel(r'$u / \overline{U}$')

    # Pull a random profile from the DNS idx[theta, time]
    p = d.profiles.profiles_x[4, 100]
    y = (np.linspace(0, 1, len(p)) - 1 / 2)
    ax.plot_fill(p, y, label=r'Ground truth')

    for down in range(2, 8, 2):
        pro = sub_smear_prof(p[::down])
        len_ = len(pro)
        u0, cd = pro[int(len_/2)], pro[int(len_/2+2)] / 2
        ax.plot_fill(pro, y[::down], label=str(down) + r'$ \Delta x $ ')
        # ax.scatter(0, u0, c='k', marker='*')
    ax.legend()
    ax.set_ylim(-0.2, 0.4)
    plt.savefig('profile.pdf')
    plt.show()


def smooth_pol():
    # Plot TSs and save spectra
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
    ax.set_ylabel(r'$n/D$')
    ax.set_xlabel(r'$u / \overline{U}$')

    y = (np.linspace(-.5, 1.5, 128) - 1 / 2)
    ze = np.append(np.zeros(len(y[y < 0])), y[y >= 0])
    u = 2 * ze - 2 * ze ** 3 + ze ** 4
    ax.plot_fill(u[:-1], y[:-1], label=r'Polhausen profile', color='k', ls='--')
    pro = sub_smear_prof(u[::6])
    ax.plot_fill(pro[:-1], y[::6][:-1], label=r'Smoothed profile')
    ax.axvline(ls=':', label='Hard body boundary', c='k')
    ax.legend()
    # ax.set_xlim(-.2, .4)
    # ax.set_ylim(-.2, .2)
    plt.savefig('pol_profile.pdf')
    plt.close()


if __name__ == "__main__":
    colors = sns.color_palette("husl", 4)
    plt.style.use(['science', 'grid'])
    d = LoadData(data_root)
    plot_profile()

