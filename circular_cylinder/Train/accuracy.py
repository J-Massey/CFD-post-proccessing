# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: J. Massey
@description: Load data from files to numpy
@contact: jmom1n15@soton.ac.uk
"""
import numpy as np
import os
from load_raw import LoadData
from neural_net.nn_regression import LinearRegression as nn_1
from neural_net_2l.nn_regression import LinearRegression as nn_2
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from poly_feat_model.poly_model_sci import LrModel
import _pickle as cPickle
from sklearn.preprocessing import PolynomialFeatures

colors = sns.color_palette('husl', 8)
plt.style.use(['science', 'grid'])

# Define some global variables
# dp: np.ndarray
real_data_dir = '/home/masseyjmo/Workspace/Lotus/projects/cylinder_dns/data'


def get_gt(angles=32):
    # Get mean quantities
    with open('neural_net_2l/fos.pickle', "rb") as f:
        fos = cPickle.load(f)
    fos['t'] = fos['t'][fos['t'] > 190]
    chunk = angles * len(fos['t'])
    p_data = np.load('data.npy').astype(np.float32)
    gt = p_data[0:chunk, -1]
    gt = np.array([gt[i * len(fos['t']):(i + 1) * len(fos['t'])] for i in range(angles)])
    return np.mean(gt), np.var(gt)


def get_96():
    sim_dir = '/home/masseyjmo/Workspace/Lotus/projects/cylinder_dns/data'
    d = 96
    names = ['t', 'dt', 'angle', 'px', 'py', 'pz']
    fos = np.loadtxt(os.path.join(sim_dir, str(d) + '/3D/fort.9'), unpack=True)
    fos = dict(zip(names, fos))
    fos['t'] = fos['t'][fos['t'] > 190]
    chunk = 32 * len(fos['t'])
    p_data = np.load('data.npy').astype(np.float32)
    gt = p_data[len(p_data)-chunk:-1, -1]
    gt = np.array([gt[i * len(fos['t']):(i + 1) * len(fos['t'])] for i in range(32)])
    return np.mean(gt), np.var(gt)


def data_loader(d):
    return LoadData(real_data_dir, d)


def plot_dpdx():
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
    ax.set_xlabel(r'$\log $ $ D$')
    ax.set_ylabel(r'$ \frac{\partial p}{\partial x} \big |_{n=\epsilon}$')


def test_data(d):
    if d != 128 or d != 96:
        p = np.genfromtxt(os.path.join(real_data_dir, str(d), '3D', 'fort.10'))
        p = np.concatenate(p, axis=0)
        u0 = np.genfromtxt(os.path.join(real_data_dir, str(d), '3D', 'fort.11'))
        u0 = np.concatenate(u0, axis=0)
        du_1 = np.genfromtxt(os.path.join(real_data_dir, str(d), '3D', 'fort.12')) / (2 * d / 128)
        du_1 = np.concatenate(du_1, axis=0)
        return np.stack((p, u0, du_1), axis=1)


def get_real_error(d):
    # Get the real data from coarse simulations
    if d == 96:
        du_1 = data_loader(128).aug_96()[:, 2]
    else:
        du_1 = np.genfromtxt(os.path.join(real_data_dir, str(d), '3D', 'fort.12')) / (2 * d / 128)
        du_1 = np.concatenate(du_1, axis=0)
    gt_mean, gt_var = get_gt()
    mean_ = {'o1': abs(np.mean(du_1)-gt_mean)/gt_mean}
    rms_ = {'o1': abs(np.var(du_1)-gt_var)/gt_var}
    print(np.mean(du_1), gt_mean)
    return mean_, rms_


def get_nn_model_error(d, wd, device="cuda"):
    # Load model
    torch.manual_seed(1)
    np.random.seed(1)
    gt_mean, gt_var = get_gt()

    model = nn_1(3, 10)
    model.to(device)
    model.load_state_dict(torch.load('neural_net/models/'+str(wd)+'_nn_regression.pth'))

    # Get cd^hat
    if d == 128:
        data = data_loader(d)
        data = data.clean_data().astype(np.float32)
        data = data[:, 0:-1]
    elif d == 96:
        data = data_loader(128).aug_96().astype(np.float32)
        data = data[:, 0:-1]
    else:
        data = test_data(d).astype(np.float32)
    with torch.no_grad():
        cd_hat = (torch.squeeze(model(torch.tensor(data,
                                                   device=device)))
                  .cpu().detach().numpy())
    return abs(np.mean(cd_hat)-gt_mean)/gt_mean, abs(np.var(cd_hat)-gt_var)/gt_var


def get_nn_2l_model_error(d, wd, device="cuda", angles=32):
    # Load model
    torch.manual_seed(1)
    np.random.seed(1)
    gt_mean, gt_var = get_gt()

    model = nn_2(3)
    model.to(device)
    model.load_state_dict(torch.load('neural_net_2l/models/' + str(wd) + '_nn_regression.pth'))

    # Get cd^hat
    if d == 128:
        data = data_loader(d)
        data = data.clean_data().astype(np.float32)
        data = data[:, 0:-1]
    elif d == 96:
        data = data_loader(128).aug_96().astype(np.float32)
        data = data[:, 0:-1]
    else:
        data = test_data(d).astype(np.float32)
    with torch.no_grad():
        cd_hat = (torch.squeeze(model(torch.tensor(data,
                                                   device=device)))
                  .cpu().detach().numpy())
    return abs(np.mean(cd_hat) - gt_mean) / gt_mean, abs(np.var(cd_hat) - gt_var) / gt_var


def plot_nn_model_error(Ds):
    fig_m, ax_m = plt.subplots(figsize=(7, 4))
    fig_s, ax_s = plt.subplots(figsize=(7, 4))
    ax_m.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
    ax_s.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
    ax_m.set_xlabel(r"$  $ Grid refinement ratio")
    ax_s.set_xlabel(r"$  $ Grid refinement ratio")

    ax_m.set_ylabel(r'$ \log $ error')
    ax_s.set_ylabel(r'$ \log $ error')

    ax_m.set_title(r'Mean error of $\overline{C_{F}}$')
    ax_s.set_title(r'Variance error of $\overline{C_{F}}$')

    ax_m.set_yscale('log')
    ax_s.set_yscale('log')
    ax_m.set_ylim(1e-4, 1e2)
    ax_s.set_ylim(1e-2, 1e2)
    # ax_m.loglog()
    # ax_s.loglog()

    # ---------------- Plot simulation error -----------------------#
    for d in Ds:
        if d != 128:
            data = test_data(d)
            mean_err, var_err = get_real_error(d)
            ax_m.scatter(128/d, mean_err['o1'], color='k', marker='d', label=r'$O(1)$ F-D' if d == Ds[-1] else "")
            ax_s.scatter(128/d, var_err['o1'], color='k', marker='d', label=r'$O(1)$ F-D' if d == Ds[-1] else "")


        # ---------------- Plot nn model error -----------------------#
        wd = [0., 0.0001]
        mkrs = ['*', 'X', 'P', 'p']
        color1 = sns.color_palette('rocket', 5)
        color2 = sns.color_palette('mako', 5)
        for c_id, w in enumerate(wd):
            mean_err, var_err = get_nn_model_error(d, w)
            print('nn err = ', mean_err)
            ax_m.scatter(128/d, mean_err, color=color1[int(c_id+1)],
                         marker=mkrs[c_id], label=r'NN wd=$'+str(w)+'$' if d == Ds[-1] else "")
            ax_s.scatter(128/d, var_err, color=color1[int(c_id+1)],
                         marker=mkrs[c_id], label=r'NN wd=$' + str(w) + '$' if d == Ds[-1] else "")
        for c_id, w in enumerate(wd):
            mean_err, var_err = get_nn_2l_model_error(d, w)
            print('nn2 err = ', mean_err)
            ax_m.scatter(128 / d, mean_err, color=color2[int(c_id+1)],
                         marker=mkrs[c_id+2], label=r'NN2 wd=$' + str(w) + '$' if d == Ds[-1] else "")
            ax_s.scatter(128 / d, var_err, color=color2[int(c_id+1)],
                         marker=mkrs[c_id+2], label=r'NN2 wd=$' + str(w) + '$' if d == Ds[-1] else "")

    ax_m.legend(bbox_to_anchor=(1.0, 0.8))
    ax_s.legend(bbox_to_anchor=(1.0, 0.8))

    fig_m.savefig('./accuracy_figures/3_nn_mean_err.pdf')
    fig_s.savefig('./accuracy_figures/3_nn_var_err.pdf')
    plt.close()


def plot_best_model_error(Ds):
    fig_m, ax_m = plt.subplots(figsize=(7, 4))
    fig_s, ax_s = plt.subplots(figsize=(7, 4))
    ax_m.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
    ax_s.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
    ax_m.set_xlabel(r"$  $ Grid refinement ratio")
    ax_s.set_xlabel(r"$  $ Grid refinement ratio")

    ax_m.set_ylabel(r'$ \log $ error')
    ax_s.set_ylabel(r'$ \log $ error')

    ax_m.set_title(r'Mean error of $\overline{C_{F}}$')
    ax_s.set_title(r'Variance error of $\overline{C_{F}}$')

    ax_m.set_yscale('log')
    ax_s.set_yscale('log')
    ax_m.set_ylim(1e-3, 1e2)
    ax_s.set_ylim(1e-2, 1e4)

    # --------------- Plot poly model error --------------------- #
    def plot_lr(real_test_data):
        n = 1
        clean = LrModel(n).straight_linear()
        poly_feat = PolynomialFeatures(n)
        real_test_data = poly_feat.fit_transform(real_test_data)
        lr = clean.predict(real_test_data)


        mean_error, var_error = abs(np.mean(lr) - gt_mean) / gt_mean, abs(np.var(lr) - gt_var) / gt_var
        colours = sns.color_palette("BuGn_r", 7)
        ax_m.scatter(128 / d, mean_error, color=colours[3],
                     marker='h', label=r'poly $O('+str(n)+')$' if d == Ds[-1] else "")
        ax_s.scatter(128 / d, var_error, color=colours[3],
                     marker='h', label=r'poly $O('+str(n)+')$' if d == Ds[-1] else "")

    gt_mean, gt_var = get_gt()
    p_get = data_loader(128)
    mean_dpds_gt, var_dpds_gt = np.mean(p_get.clean_data()[:, 0]), np.var(p_get.clean_data()[:, 0])
    for d in Ds:
        if d == 128:
            data = data_loader(d).clean_data()[:, 0:-1]
            skip = True
        elif d == 96:
            data = data_loader(128).aug_96()[:, 0:-1]
            skip = False
        else:
            data = test_data(d)
            skip = False
    # -------------------- Plot simulation error --------------------- #
        if not skip:
            mean_err, var_err = get_real_error(d)
            ax_m.scatter(128 / d, mean_err['o1'], color='k', marker='d', label=r'$O(1)$ F-D' if d == Ds[0] else "")
            ax_s.scatter(128 / d, var_err['o1'], color='k', marker='d', label=r'$O(1)$ F-D' if d == Ds[0] else "")

        # ---------------- Plot pressure error ----------------------- #
            mean_dpds = abs((np.mean(data[:, 0]) - mean_dpds_gt)/mean_dpds_gt)
            var_dpds = abs((np.var(data[:, 0]) - var_dpds_gt) / var_dpds_gt)
            ax_m.scatter(128 / d, mean_dpds, color='k',
                         marker='P', label=r'Pressure error' if d == Ds[0] else "")
            ax_s.scatter(128 / d, var_dpds, color='k',
                         marker='P', label=r'Pressure error' if d == Ds[0] else "")
    # --------------- Plot poly model error --------------------- #
        plot_lr(data)

    # ---------------- Plot nn model error ----------------------- #
        mkrs = ['h', 'D', 'd']
        color1 = sns.color_palette('rocket', 5)
        mean_err, var_err = get_nn_2l_model_error(d, 0.)
        print('nn err = ', mean_err)
        ax_m.scatter(128 / d, mean_err, color=color1[2],
                     marker='X', label=r'NN2' if d == Ds[-1] else "")
        ax_s.scatter(128 / d, var_err, color=color1[2],
                     marker='X', label=r'NN2' if d == Ds[-1] else "")



        #
        # mean_err, var_err = abs(np.mean(rr) - gt_mean) / gt_mean, abs(np.var(rr) - gt_var) / gt_var
        #
        # colours = sns.light_palette("purple", 7)
        # ax_m.scatter(128 / d, mean_err, color=colours[3],
        #              marker='P', label=r'Ridge $\alpha = ' + str(1e-5) + '$' if d == Ds[-1] else "")
        # ax_s.scatter(128 / d, var_err, color=colours[int(1 + 2)],
        #              marker='P', label=r'Ridge $\alpha = ' + str(1e-5) + '$' if d == Ds[-1] else "")


    # x, y = loglogLine(p2=(64, 1e-1), p1x=16, m=-2)
    # ax_m.loglog(x, y, color='black', lw=1, ls='dotted', label=r'$O(2)$')
    # ax_s.loglog(x, y, color='black', lw=1, ls='dotted', label=r'$O(2)$')

    ax_m.legend()
    ax_s.legend()

    fig_m.savefig('./accuracy_figures/best_mean_err.png')
    fig_s.savefig('./accuracy_figures/best_var_err.png')
    plt.close()


if __name__ == "__main__":
    ds = [16, 24, 32, 48, 64, 96, 128]

    # plot_nn_model_error(ds)
    plot_best_model_error(ds)
    # np.shape(np.load('data.npy'))


