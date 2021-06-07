import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
colors = sns.color_palette("husl", 4)
plt.style.use(['science', 'grid'])


def plot_loss(epochs, cost, fn='cost.pdf'):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
    ax.set_xlabel(r"Epochs")
    ax.set_ylabel(r'$L_2$ loss')
    ax.plot(np.linspace(0, epochs, len(cost)), cost, label=r'$L_{2}$')
    ax.legend()
    plt.savefig(fn)
    plt.show()


def plot_model(cd_hat, fos, Y, fn='model.pdf'):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
    ax.set_xlabel(r"$t/D$")
    ax.set_ylabel(r'$ \frac{ \partial \overline{u}}{\partial r}\big|_{r=0}$')
    ax.plot(fos['t'], Y, label=r'Ground truth')
    ax.plot(fos['t'], cd_hat, label=r'$\hat{Y}$')
    ax.legend()
    plt.savefig(fn)
    plt.show()
