import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import _pickle as cPickle
import torch

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


def plot_training(fos, Y, fn='model.pdf'):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
    ax.set_xlabel(r"$t/D$")
    ax.set_ylabel(r'$C_{D_f}$')
    for idx, truths in enumerate(Y):
        ax.plot_fill(fos['t'], truths, label=f'Sample {idx}')
    ax.legend()
    plt.savefig(fn)
    plt.show()


def vis_data():
    with open('fos.pickle', "rb") as f:
        fos = cPickle.load(f)
    p_data = np.load('data.npy').astype(np.float32)

    chunk = 32 * len(fos['t'])
    chunks = int(len(p_data)/chunk)
    train_list = []
    for s in range(chunks):
        tmp = (p_data[s*chunk:(s+1)*chunk, -1])
        print(np.mean(p_data[s*chunk:(s+1)*chunk], axis=0))
        train = np.array([tmp[i * len(fos['t']):(i + 1) * len(fos['t'])] for i in range(32)])
        train_list.append(np.mean(train, axis=0))
    return train_list


def compare_model(model, poly_n: int, device="cuda", angles=32):
    # Get mean quantities
    with open('fos.pickle', "rb") as f:
        fos = cPickle.load(f)

    chunk = angles * len(fos['t'])

    p_data = np.load('data.npy').astype(np.float32)
    gt = p_data[0:chunk, -1]

    with torch.no_grad():
        cd_hat = (torch.squeeze(model(torch.tensor(p_data[0:chunk, 0:-1],
                                                   device=device)))
                  .cpu().detach().numpy())
    cd_hat = np.array([cd_hat[i * len(fos['t']):(i + 1) * len(fos['t'])] for i in range(angles)])
    gt = np.array([gt[i * len(fos['t']):(i + 1) * len(fos['t'])] for i in range(angles)])

    plot_model(np.mean(cd_hat, axis=0), fos, np.mean(gt, axis=0), fn='model' + str(poly_n) + '.pdf')


if __name__ == "__main__":
    with open('fos.pickle', "rb") as f:
        fos = cPickle.load(f)
    lsd = vis_data()
    plot_training(fos, lsd, 'figures/vis.pdf')
