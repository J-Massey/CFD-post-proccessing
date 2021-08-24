import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from poly_model_sci import *
colours = sns.color_palette('mako', 4)
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
    ax.set_ylabel(r'$C_F$')
    ax.plot_fill(fos['t'], Y * 0.0010518, label=r'Ground truth', color='k')
    ax.plot_fill(fos['t'], cd_hat * 0.0010518, label=r'$\hat{Y}$', color=colours[2])
    ax.legend()
    plt.savefig(fn)
    plt.show()


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

    Ds = [16, 24, 32, 48, 64]

    # ax_m.loglog()
    # ax_s.loglog()

    gt_mean, gt_var = get_gt()

    poly_n = [7, 2, 1]
    lrs = [LrModel(n).straight_linear() for n in poly_n]
    # ---------------- Plot simulation error -----------------------'lp'
    for d in Ds:
        data = test_data(d)
        mean_err, var_err = get_real_error(d)
        if d != 96:
            ax_m.scatter(96 / d, mean_err['o1'], color='k', marker='d', label=r'$O(1)$ F-D' if d == Ds[-1] else "")
            ax_s.scatter(96 / d, var_err['o1'], color='k', marker='d', label=r'$O(1)$ F-D' if d == Ds[-1] else "")

        # ---------------- Plot nn model error -----------------------#
        mkrs = ['*', 'X', 'P', 'p']
        colours = sns.color_palette("BuGn_r", 6)
        for c_id, n in enumerate(poly_n):
            poly = PolynomialFeatures(n)
            data_t = poly.fit_transform(data)
            cd_hat = lrs[c_id].predict(data_t)
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

    Ds = [16, 24, 32, 48, 64]

    gt_mean, gt_var = get_gt()

    n = 7
    alphas = [1e-3, 1e-4, 1e-5]
    ridges = [LrModel(n).ridge(a) for a in alphas]
    # lassos = [LrModel(n).lasso(a) for a in alphas]
    # elastics = [LrModel(n).elastic(a) for a in alphas]
    # ---------------- Plot O(1) poly -----------------------#
    for d in Ds:
        data_t = test_data(d)
        clean = LrModel(1).straight_linear()
        poly = PolynomialFeatures(1)
        data_t = poly.fit_transform(data_t)
        lr = clean.predict(data_t)

        mean_err, var_err = abs(np.mean(lr) - gt_mean) / gt_mean, abs(np.var(lr) - gt_var) / gt_var
        colours = sns.color_palette("BuGn_r", 7)
        ax_m.scatter(96 / d, mean_err, color=colours[3],
                     marker='h', label=r'poly $O(1)$' if d == Ds[-1] else "")
        ax_s.scatter(96 / d, var_err, color=colours[3],
                     marker='h', label=r'poly $O(1)$' if d == Ds[-1] else "")

        # ---------------- Plot model error -----------------------#
        mkrs = ['*', 'X', 'P', 'p']
        colours = sns.light_palette("purple", 5)
        for c_id, a in tqdm(enumerate(alphas)):
            data_t = test_data(d)
            poly = PolynomialFeatures(n)
            data = poly.fit_transform(data_t)
            rr = ridges[c_id].predict(data)
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