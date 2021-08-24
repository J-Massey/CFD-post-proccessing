import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import product
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import BoundaryNorm


colors = sns.color_palette("husl", 4)
plt.style.use(['science', 'grid'])


def plot_loss(epochs, cost, fn='cost.pdf'):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
    ax.set_xlabel(r"Epochs")
    ax.set_ylabel(r'$L_2$ loss')
    ax.plot_fill(np.linspace(0, epochs, len(cost)), cost, label=r'$L_{2}$')
    ax.legend()
    plt.savefig(fn)
    plt.show()


def plot_model(cd_hat, fos, Y, fn='model.pdf'):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
    ax.set_xlabel(r"$t/D$")
    ax.set_ylabel(r'$C_{D_f}$')
    ax.plot_fill(fos['t'], Y, label=r'Ground truth')
    ax.plot_fill(fos['t'], cd_hat, label=r'$\hat{C_{D_f}}$')
    ax.legend()
    plt.savefig(fn)
    plt.show()


def plot_BL_corruption():
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlabel(r'$x_n$')
    ax.set_ylabel(r'$y_n$', rotation=0)
    # Define grid
    D = 32
    eps = 2
    r = D / 2
    x, y = np.arange(-D, D + 1, 1), np.arange(-D, D + 1, 1)
    X, Y = np.meshgrid(x, y)

    # Body coordinates
    theta = np.linspace(0, 2 * np.pi, int(D * np.pi))

    Bx, By = r * np.cos(theta), r * np.sin(theta)
    ax.plot_fill(Bx, By, color='k', linewidth=2., label=r'Hard body boundary')

    Bepx, Bepy = (r + eps) * np.cos(theta), (r + eps) * np.sin(theta)
    ax.plot_fill(Bepx, Bepy, c='blue', linewidth=0.5, label=r'$D+\epsilon$')

    # Distance function from eps away from body edge
    dis = np.sqrt(X ** 2 + Y ** 2)

    # Cmap definition
    bs = iter((np.array([14, 15.8, 18.7, 22]) - 4.5) / D)
    colours = [(0, 'midnightblue'),
               (next(bs), 'midnightblue'),
               (next(bs), 'red'),
               (next(bs), 'green'),
               (next(bs), 'royalblue'),
               (1, 'royalblue')]
    cmap = LinearSegmentedColormap.from_list('corruption', colours, 256)

    cs = ax.imshow(dis, zorder=0, aspect="auto", extent=(-D, D, -D, D),
                   cmap=cmap, interpolation='bicubic')
    make_axes_locatable(ax)
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig.add_axes(ax_cb)
    cbar = plt.colorbar(cs, cax=ax_cb, ticks=[8, 16.4, 21, 32], extend='max')
    # ax_cb.yaxis.tick_right()
    cbar.ax.set_yticklabels([r'$\vec{b}$', r'$\vec{b}*\vec{f}$', r'$d|_{n \approx 0}$', r'$\vec{f}$'])
    cbar.ax.tick_params(which='both', size=0)
    ax.legend()
    plt.savefig('../figures/bl_corruption.pdf', dpi=300)
    plt.close()


def plot_pressure():
    data_root = '/home/masseyjmo/Workspace/Lotus/projects/cylinder_dns/validation'
    p = np.loadtxt(os.path.join(data_root, 'fort.10'), unpack=True)
    p = np.mean(p, axis=1)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r'$C_{p}$')
    ax.scatter(np.linspace(0, np.pi / 2, len(p)), p * 2, label=r'Pressure distribution', color='k', marker='+')
    ax.set_ylim(-2, 1)
    ax.legend()
    plt.savefig('pressure_theta.pdf')
    plt.show()


if __name__ == "__main__":
    plot_BL_corruption()
