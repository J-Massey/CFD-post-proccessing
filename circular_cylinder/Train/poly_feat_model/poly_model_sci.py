# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: J. Massey
@description: Adapted from
@contact: jmom1n15@soton.ac.uk
"""
# we import necessary libraries and functions
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import seaborn as sns
from load_raw import LoadData
import _pickle as cPickle


real_data_dir = '/home/masseyjmo/Workspace/Lotus/projects/cylinder_dns/data'
plt.style.use(['science', 'grid'])


def get_gt(angles=32):
    # Get mean quantities
    with open('fos.pickle', "rb") as f:
        fos = cPickle.load(f)

    chunk = angles * len(fos['t'])
    p_data = np.load('data.npy').astype(np.float32)
    gt = p_data[0:chunk, -1]
    gt = np.array([gt[i * len(fos['t']):(i + 1) * len(fos['t'])] for i in range(angles)])
    return np.mean(gt), np.var(gt)


def data_loader(D):
    return LoadData(real_data_dir, D)


def get_real_error(data):
    # Get the real data from coarse simulations
    gt_mean, gt_var = get_gt()
    du_2 = data.ground_truth()
    du_1 = data.du_1
    mean_ = {'o2': abs(np.mean(du_2)-gt_mean)/gt_mean,
             'o1': abs(np.mean(du_1)-gt_mean)/gt_mean}
    rms_ = {'o2': abs(np.var(du_2)-gt_var)/gt_var,
            'o1': abs(np.var(du_1)-gt_var)/gt_var}
    print(np.mean(du_2), gt_mean)
    return mean_, rms_


class LrModel:
    def __init__(self, poly_n: int):
        self.poly_n = poly_n
        data = np.load('data.npy')
        poly = PolynomialFeatures(poly_n)
        poly = poly.fit_transform(data[:, 0:-1])
        self.data = np.concatenate((poly, data[:, -1][..., np.newaxis]), axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test =\
            train_test_split(self.data[:, 0:-1], self.data[:, -1], test_size=0.2, shuffle=True)

    def straight_linear(self):
        lr = LinearRegression(n_jobs=8)
        lr.fit(self.X_train, self.y_train)

        # evaluating the model on training and testing sets
        pred_train_lr = lr.predict(self.X_train)
        print('Lr train', r2_score(self.y_train, pred_train_lr))
        pred_test_lr = lr.predict(self.X_test)
        print('Lr test', r2_score(self.y_test, pred_test_lr))
        return lr

    def ridge(self, alpha):
        # here we define a Ridge regression model with lambda(alpha)=0.01
        rr = Ridge(alpha=alpha)
        rr.fit(self.X_train, self.y_train)
        pred_train_rr = rr.predict(self.X_train)
        print('Ridge train', r2_score(self.y_train, pred_train_rr))
        pred_test_rr = rr.predict(self.X_test)
        print('Ridge test', r2_score(self.y_test, pred_test_rr))
        return rr

    def lasso(self, alpha):
        # Define lasso
        model_lasso = Lasso(alpha=alpha)
        model_lasso.fit(self.X_train, self.y_train)
        pred_train_lasso = model_lasso.predict(self.X_train)
        print('Lasso train', r2_score(self.y_train, pred_train_lasso))

        pred_test_lasso = model_lasso.predict(self.X_test)
        print('Lasso test', r2_score(self.y_test, pred_test_lasso))

        return model_lasso

    def elastic(self, alpha):
        model_enet = ElasticNet(alpha=alpha)
        model_enet.fit(self.X_train, self.y_train)
        pred_train_enet = model_enet.predict(self.X_train)
        print('Elastic train', r2_score(self.y_train, pred_train_enet))

        pred_test_enet = model_enet.predict(self.X_test)
        print('Elastic test', r2_score(self.y_test, pred_test_enet))

        return model_enet


def plot_raw():
    fig_m, ax_m = plt.subplots(figsize=(7, 4))
    fig_s, ax_s = plt.subplots(figsize=(7, 4))
    ax_m.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
    ax_s.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
    ax_m.set_xlabel(r"Grid refinement ratio")
    ax_s.set_xlabel(r"Grid refinement ratio")

    ax_m.set_ylabel(r'$ \log $ error')
    ax_s.set_ylabel(r'$ \log $ error')

    ax_m.set_title(r'Mean error of $\overline{C_{F}}$')
    ax_s.set_title(r'Variance error of $\overline{C_{F}}$')

    ax_m.set_yscale('log')
    ax_s.set_yscale('log')
    ax_m.set_ylim(1e-2, 1e5)
    ax_s.set_ylim(1e-2, 1e9)

    Ds = [16, 24, 32, 48, 64, 96]

    # ax_m.loglog()
    # ax_s.loglog()

    gt_mean, gt_var = get_gt()

    poly_n = [7, 2, 1]
    lrs = [LrModel(n).straight_linear() for n in poly_n]
    # ---------------- Plot simulation error -----------------------#
    for d in Ds:
        data = data_loader(d)
        mean_err, var_err = get_real_error(data)
        if d != 96:
            ax_m.scatter(96 / d, mean_err['o2'], color='k', marker='h', label=r'$O(2)$ F-D' if d == Ds[-2] else "")
            ax_s.scatter(96 / d, var_err['o2'], color='k', marker='h', label=r'$O(2)$ F-D' if d == Ds[-2] else "")
        ax_m.scatter(96 / d, mean_err['o1'], color='k', marker='d', label=r'$O(1)$ F-D' if d == Ds[-1] else "")
        ax_s.scatter(96 / d, var_err['o1'], color='k', marker='d', label=r'$O(1)$ F-D' if d == Ds[-1] else "")

        # ---------------- Plot nn model error -----------------------#
        mkrs = ['*', 'X', 'P', 'p']
        colours = sns.color_palette("BuGn_r", 6)
        for c_id, n in enumerate(poly_n):
            data_test = data.clean_data()
            poly = PolynomialFeatures(n)
            poly = poly.fit_transform(data_test[:, 0:-1])
            data_test = np.concatenate((poly, data_test[:, -1][..., np.newaxis]), axis=1)
            cd_hat = lrs[c_id].predict(data_test[:, 0:-1])
            mean_err, var_err = abs(np.mean(cd_hat) - gt_mean) / gt_mean, abs(np.var(cd_hat) - gt_var) / gt_var
            ax_m.scatter(96 / d, mean_err, color=colours[int(1 + c_id)],
                         marker=mkrs[c_id], label=r'poly $O(' + str(n) + ')$' if d == Ds[-1] else "")
            ax_s.scatter(96 / d, var_err, color=colours[int(1 + c_id)],
                         marker=mkrs[c_id], label=r'poly $O(' + str(n) + ')$' if d == Ds[-1] else "")

    ax_m.legend(bbox_to_anchor=(1.22, 0.6))
    ax_s.legend(bbox_to_anchor=(1.22, 0.6))

    fig_m.savefig('../accuracy_figures/3_poly_mean_err.pdf')
    fig_s.savefig('../accuracy_figures/3_poly_var_err.pdf')
    plt.close()


def plot_ridge():
    fig_m, ax_m = plt.subplots(figsize=(7, 4))
    fig_s, ax_s = plt.subplots(figsize=(7, 4))
    ax_m.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
    ax_s.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
    ax_m.set_xlabel(r"Grid refinement ratio")
    ax_s.set_xlabel(r"Grid refinement ratio")

    ax_m.set_ylabel(r'$ \log $ error')
    ax_s.set_ylabel(r'$ \log $ error')

    ax_m.set_title(r'Mean error of $\overline{C_{F}}$')
    ax_s.set_title(r'Variance error of $\overline{C_{F}}$')

    ax_m.set_yscale('log')
    ax_s.set_yscale('log')
    ax_m.set_ylim(1e-3, 1e2)
    ax_s.set_ylim(1e-2, 1e4)

    Ds = [16, 24, 32, 48, 64, 96]

    gt_mean, gt_var = get_gt()

    n = 7
    alphas = [1e-4, 1e-5, 1e-6]
    ridges = [LrModel(n).ridge(a) for a in alphas]
    # lassos = [LrModel(n).lasso(a) for a in alphas]
    # elastics = [LrModel(n).elastic(a) for a in alphas]
    # ---------------- Plot O(1) poly -----------------------#
    for d in Ds:
        data = data_loader(d)
        clean = LrModel(1).straight_linear()
        data_t = data.clean_data()
        poly = PolynomialFeatures(1)
        poly = poly.fit_transform(data_t[:, 0:-1])
        data_t = np.concatenate((poly, data_t[:, -1][..., np.newaxis]), axis=1)
        lr = clean.predict(data_t[:, 0:-1])

        mean_err, var_err = abs(np.mean(lr) - gt_mean) / gt_mean, abs(np.var(lr) - gt_var) / gt_var
        colours = sns.color_palette("BuGn_r", 7)
        ax_m.scatter(96 / d, mean_err, color=colours[3],
                     marker='h', label=r'poly $O(1)$' if d == Ds[-1] else "")
        ax_s.scatter(96 / d, var_err, color=colours[3],
                     marker='h', label=r'poly $O(1)$' if d == Ds[-1] else "")

        # ---------------- Plot nn model error -----------------------#
        mkrs = ['*', 'X', 'P', 'p']
        colours = sns.light_palette("purple", 5)
        for c_id, a in enumerate(alphas):
            data_test = data.clean_data()
            poly = PolynomialFeatures(n)
            poly = poly.fit_transform(data_test[:, 0:-1])
            data_test = np.concatenate((poly, data_test[:, -1][..., np.newaxis]), axis=1)
            rr = ridges[c_id].predict(data_test[:, 0:-1])
            mean_err, var_err = abs(np.mean(rr) - gt_mean) / gt_mean, abs(np.var(rr) - gt_var) / gt_var
            ax_m.scatter(96 / d, mean_err, color=colours[int(1 + c_id)],
                         marker=mkrs[c_id], label=r'Ridge $\alpha = ' + str(a) + '$' if d == Ds[-1] else "")
            ax_s.scatter(96 / d, var_err, color=colours[int(1 + c_id)],
                         marker=mkrs[c_id], label=r'Ridge $\alpha = ' + str(a) + '$' if d == Ds[-1] else "")

    ax_m.legend(bbox_to_anchor=(1.01, 0.6))
    ax_s.legend(bbox_to_anchor=(1.01, 0.6))

    fig_m.savefig('../accuracy_figures/3_ridge_mean_err.pdf')
    fig_s.savefig('../accuracy_figures/3_ridge_var_err.pdf')
    plt.close()


if __name__ == "__main__":
    # plot_raw()
    plot_ridge()

