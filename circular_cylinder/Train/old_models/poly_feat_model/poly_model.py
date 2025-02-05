# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: J. Massey
@description: Train the model
@contact: jmom1n15@soton.ac.uk
"""

import torch
from plot import *
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from torch.autograd import Variable
import _pickle as cPickle
import signal
from sklearn.preprocessing import PolynomialFeatures
import time


def accuracy(model, X, Y):
    # Average L2 loss / mean(ground truth)
    with torch.no_grad():
        oupt = 100 * (1 - torch.abs(torch.squeeze(model(X)).mean() - Y.mean()) / Y.mean()).item()
    return oupt


def compare_model(model, poly_n: int, device="cuda", angles=32, fn='model.pdf'):
    # Get mean quantities
    with open('fos.pickle', "rb") as f:
        fos = cPickle.load(f)

    chunk = angles * len(fos['t'])

    p_data = np.load('data.npy').astype(np.float32)
    gt = p_data[0:chunk, -1]

    poly = PolynomialFeatures(poly_n)
    poly = poly.fit_transform(p_data[0:chunk, 0:-1])
    p_data = np.concatenate((poly, p_data[0:chunk, -1][..., np.newaxis]), axis=1)

    with torch.no_grad():
        cd_hat = (torch.squeeze(model(torch.tensor(p_data[0:chunk, 0:-1],
                                                   device=device)))
                  .cpu().detach().numpy())
    cd_hat = np.array([cd_hat[i * len(fos['t']):(i + 1) * len(fos['t'])] for i in range(angles)])
    gt = np.array([gt[i * len(fos['t']):(i + 1) * len(fos['t'])] for i in range(angles)])

    plot_model(np.mean(cd_hat, axis=0), fos, np.mean(gt, axis=0), fn=fn)


def handler(signum, frame):
    raise RuntimeError


def main(poly_n, eps):
    torch.manual_seed(1)
    np.random.seed(1)
    plt.style.use(['science', 'grid'])

    print("Let's get started")
    print('\nLoading data')
    start = time.time()
    # ---------------------Load data-------------------------#
    data = np.load('data.npy')
    poly = PolynomialFeatures(poly_n)
    poly = poly.fit_transform(data[:, 0:-1])
    data = np.concatenate((poly, data[:, -1][..., np.newaxis]), axis=1)

    x_train, x_test, y_train, y_test = train_test_split(data[:, 0:-1], data[:, -1],
                                                        test_size=0.2, shuffle=False)
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    local_batch = torch.tensor(x_train, device=device)
    local_truth = torch.tensor(y_train, device=device)
    # ---------------------Init model-----------------------#
    print(f'That took {(time.time() - start):.3f}s')
    start = time.time()
    print('\nInitialising the model')
    try:
        t_coeffs = torch.load('models/poly_coes_' + str(poly_n) + '.pt', map_location=torch.device(device))
        print('Found previous state')
    except FileNotFoundError:
        print('New model, initialising coefficients')
        t_coeffs = []
        for i in range(np.shape(x_train)[-1]):
            t_coeffs.append(torch.tensor(0., requires_grad=True, device=device))

    def model(x_input):
        iter_ = torch.clone(x_input)
        for deg in range(len(t_coeffs)):
            iter_[:, deg] = (x_input[:, deg] * t_coeffs[deg])
        return iter_.sum(dim=-1)

    # Loss function definition
    def loss(y_hat, y_target):
        return ((y_hat - y_target) ** 2).sum()

    max_epochs = eps
    lr = 1e-5
    optimiser = torch.optim.Adam
    # Setup the optimizer object, so it optimizes a and b.
    optimizer = optimiser(t_coeffs, lr=lr)

    print(f'That took {(time.time() - start):.3f}s')
    # ----------------------------Main optimization loop-------------------------------#
    start = time.time()
    print('\nStarting training')
    cost = []
    try:
        for t in tqdm(range(int(max_epochs)), desc='Optimisation', ascii=True):
            optimizer.zero_grad()  # Set the gradients to 0.
            y_predicted = model(Variable(local_batch))  # Compute the current predicted Y's from x_dataset
            current_loss = loss(y_predicted, local_truth)  # See how far off the prediction is
            if t % 10 == 0:
                cost.append(current_loss.item())
            current_loss.backward()  # Compute the gradient of the loss with respect to A and b.
            optimizer.step()  # Update A and b accordingly.
            signal.signal(signal.SIGINT, handler)
    except RuntimeError:
        print('Outta time bitch!')

    print("\nDone ")
    print(f'That took {(time.time() - start):.3f}s')
    # ----------------------------Save & get model accuracy-------------------------------#
    start = time.time()

    torch.save(t_coeffs, 'models/poly_coes_' + str(poly_n) + '.pt')

    print('\nGetting model accuracy')
    acc_train = accuracy(model, local_batch, local_truth)  # item-by-item
    print(f"Accuracy on training data = {acc_train:.4f} %")

    X_test = torch.from_numpy(x_test).to(device)
    Y_test = torch.from_numpy(y_test).to(device)

    acc_test = accuracy(model, X_test, Y_test)  # item-by-item
    print(f"Accuracy on test data = {acc_test:.4f} %")

    print(f"\nt = {t}, loss = {current_loss}, coeffs = {[round(co.item(), 8) for co in t_coeffs]}")

    # Plot Cost
    plot_loss(max_epochs, cost, fn='figures/cost_n_' + str(poly_n) + '.pdf')
    compare_model(model, poly_n, fn='figures/model_n_' + str(poly_n) + '.pdf')

    print('\n!!!!!!!!!!DONE!!!!!!!!!!')
    print(f'That took {(time.time() - start):.3f}s')


if __name__ == "__main__":
    main(2, 1e4)
    # for n, e in zip([1, 2], [3e6, 3e4]):
    #     main(n, e)
