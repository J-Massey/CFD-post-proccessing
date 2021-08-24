import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from postproc.boundary_layer import ProfileDataset

colors = sns.color_palette("husl", 4)
plt.style.use(['science', 'grid'])


def data_loader(d):
    data_root = '/home/masseyjmo/Workspace/Lotus/projects/cylinder_dns/data'
    names = ['t', 'dt', 'angle', 'px', 'py', 'pz']
    fos = np.loadtxt(os.path.join(data_root, str(d) + '/3D/fort.9'), unpack=True)
    fos = dict(zip(names, fos))
    dpdx = np.loadtxt(os.path.join(data_root, str(d) + '/3D/fort.10'), unpack=True)
    # self.ang = np.pi / np.shape(dpdx)[1]
    profiles = ProfileDataset(os.path.join(data_root, str(d)),
                              print_res=128, multi=8)
    # u0 = profiles.get_u(0, d, 1)[0]
    u2 = np.loadtxt(os.path.join(data_root, str(d) + '/3D/fort.12'), unpack=True)
    u_bl = np.loadtxt(os.path.join(data_root, str(d) + '/3D/fort.11'), unpack=True)
    # du_1 = ((profiles.get_u(2, d, 1)[0]).T / 2).T
    return dpdx, u2, u_bl


def plot_dpdx(D, dpdx):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)

    ax.set_xlabel(r"$\log D/\Delta x$")
    ax.set_ylabel(r'$ \log \big |\overline{\frac{dP}{dx}} \big |$')
    # ax.set_xlim(0, np.pi / 2)
    # ax.set_ylim(-2, 1)

    ax.scatter(D, -dpdx, color='red', marker='*')

    # ax.loglog()
    # x, y = loglogLine(p2=(96, 1e-2), p1x=16, m=-2)
    # ax.loglog(x, y, color='black', lw=1, ls='dotted', label=r'$ O(2) $')
    #
    # ax.legend()

    plt.savefig('figures/dpdx.pdf')
    plt.close()


def plot_u2(D, u2):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)

    ax.set_xlabel(r"$D/\Delta x$")
    ax.set_ylabel(r'$ \overline{U}_{\Delta x_n=2} $')
    # ax.set_xlim(0, np.pi / 2)
    # ax.set_ylim(-2, 1)

    ax.scatter(D, u2, color='red', marker='*')

    # ax.loglog()
    # x, y = loglogLine(p2=(96, 2e-1), p1x=16, m=-0.2)
    # ax.loglog(x, y, color='black', lw=1, ls='--', label=r'$ O(1) $')
    #
    # ax.legend()

    plt.savefig('figures/u2.pdf')
    plt.close()


def plot_u_bl(D, ubl):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)

    ax.set_xlabel(r"$D/\Delta x$")
    ax.set_ylabel(r'$ \overline{u}|_{(r=1.05, \theta=0.49^r)} $')
    # ax.set_xlim(0, np.pi / 2)
    # ax.set_ylim(-2, 1)

    ax.scatter(D, ubl, color='blue', marker='p')

    # ax.loglog()
    # x, y = loglogLine(p2=(96, 1e-1), p1x=16, m=-2)
    # ax.loglog(x, y, color='black', lw=1, ls='dotted', label=r'$ O(2) $')
    #
    # ax.legend()

    plt.savefig('figures/u_bl.pdf')
    plt.close()


if __name__ == "__main__":
    Ds = [16, 24, 32, 48, 64]
    dpdxs, u2s, u_bls = [], [], []
    for d in Ds:
        dpdx, u2, u_bl = data_loader(d)
        dpdxs.append(np.mean(dpdx))
        u2s.append(np.mean(u2))
        u_bls.append(np.mean(u_bl[0][3]))
    plot_dpdx(np.array(Ds), np.array(dpdxs))
    plot_u2(96/np.array(Ds), np.array(u2s)) #  /(96/np.array(Ds)))
    plot_u_bl(np.array(Ds), np.array(u_bls))

