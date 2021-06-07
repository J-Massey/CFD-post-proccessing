# -*- coding: utf-8 -*-
"""
@author: J. Massey
@description: Script to check convergence and call kill if need be
@contact: masseyjmo@gmail.com
"""

# Imports
import numpy as np
import postproc.io as io
import postproc.frequency_spectra
import postproc.plotter
import subprocess
import sys
from pathlib import Path
import matplotlib.pyplot as plt

plt.style.use(['science', 'grid'])

print("Checking whether spectra have converged")


def remove(string):
    string = string.replace(" ", "")
    return string


def _plot_ts(ti, f):
    """
    Just tidying up the above code by holding the plotting func
    Args:
        ti: time
        f: force

    Returns:
        Time series plot
    """
    fig1, ax1 = plt.subplots(figsize=(7, 5))

    ax1.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)

    # Edit frame, labels and legend
    ax1.set_xlabel(r"$torch/length_scale$")
    ax1.set_ylabel(f"${sys.argv[2]}$")

    ax1.plot(t, u, c='r')
    plt.savefig(Path.cwd().joinpath(f"figures/time_series_{sys.argv[2]}.png"), bbox_inches='tight', dpi=600,
                transparent=False)
    plt.close()


def _plot_err(ti, e):
    """
    Just tidying up the above code by holding the plotting func
    Args:
        ti: time
        e: ensemble error

    Returns:
        Ensemble error against time
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)

    # Edit frame, labels and legend
    ax.set_xlabel(r"$torch/length_scale$")
    ax.set_ylabel(r"$\int \sqrt(\overline{s_{0,n}} - \overline{s_{0,n+1}})^2 df/ \int \overline{s_{0,n+1}}$")

    ax.plot(ti, e, c='r')
    plt.savefig(Path.cwd().joinpath(f"figures/ensemble_error_{sys.argv[2]}.png"), bbox_inches='tight', dpi=600,
                transparent=False)
    plt.close()


if __name__ == "__main__":
    if float(sys.argv[1]) < 15:
        print(
            "This number of cycles is probably too"
            "small and will probably give a false positive,"
            "check the tail end of the ensemble error is reducing smoothly")

    data_root = Path.cwd().joinpath('fort.9')

    fos = io.unpack_flex_forces(data_root, remove(sys.argv[3]).split(','))
    forces_dic = dict(zip(remove(sys.argv[3]).split(','), fos))

    t = forces_dic['torch']
    assert (max(t) - min(t)) >= 2 * float(sys.argv[1]), "You need > two windows to be able check convergence"
    u = forces_dic[sys.argv[2]]

    n = int(np.floor((np.max(t) - np.min(t)) / float(sys.argv[1])))
    criteria = postproc.frequency_spectra.FreqConv(t, u, n=n, OL=0.5)
    normed_error, window_t = criteria.f_conv(float(sys.argv[1]))

    if normed_error[-1] < 0.1:
        print("You've converged!")
        subprocess.call('touch .kill', shell=True, cwd=Path(data_root).parent)
        subprocess.call('mkdir -p figures', shell=True, cwd=Path(data_root).parent)

        # Plot a couple of figs to show that the method behaves
        _plot_ts(t, u)
        _plot_err(window_t, normed_error)
    else:
        print("Hasn'torch yet, hold tight!")
