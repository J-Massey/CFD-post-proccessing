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


def compare_model(model, device="cuda", angles=32, fn='model.pdf'):
    # Get mean quantities
    with open('fos.pickle', "rb") as f:
        fos = cPickle.load(f)

    chunk = angles * len(fos['t'])

    data = np.load('data.npy').astype(np.float32)
    gt = data[0:chunk, -1]

    with torch.no_grad():
        cd_hat = (torch.squeeze(model(torch.tensor(data[0:chunk, 0:-1],
                                                   device=device)))
                  .cpu().detach().numpy())
    cd_hat = np.array([cd_hat[i * len(fos['t']):(i + 1) * len(fos['t'])] for i in range(angles)])
    gt = np.array([gt[i * len(fos['t']):(i + 1) * len(fos['t'])] for i in range(angles)])

    plot_model(np.mean(cd_hat, axis=0), fos, np.mean(gt, axis=0), fn=fn)


def handler(signum, frame):
    raise RuntimeError


def main():
    plt.style.use(['science', 'grid'])

    data = np.load('data.npy')

    x_train, x_test, y_train, y_test = train_test_split(data[:, 0:-1], data[:, -1],
                                                        test_size=0.1, shuffle=False)
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    local_batch = torch.tensor(x_train, device=device)
    local_truth = torch.tensor(y_train, device=device)

    try:
        t_coeffs = torch.load('models/coes_scaling.pt', map_location=torch.device(device))
        print('Found previous state')
    except FileNotFoundError:
        print('New model, initialising coefficients')
        t_coeffs = []
        for i in range(3):
            t_coeffs.append(torch.tensor(1., requires_grad=True, device=device))

    # Define the prediction model
    def model(x_input):
        dpdx, u0, cf1 = x_input.t()
        return cf1 * (t_coeffs[0] * dpdx + t_coeffs[1] * u0 ** 2 + t_coeffs[2] * u0)

    # Loss function definition
    def loss(y_hat, y_target):
        return ((y_hat - y_target) ** 2).sum()

    max_epochs = 1e5
    lr = 1e-4
    optimiser = torch.optim.Adam
    # Setup the optimizer object, so it optimizes a and b.
    optimizer = optimiser(t_coeffs, lr=lr)

    # Main optimization loop
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
    torch.save(t_coeffs, 'models/coes_scaling.pt')
    print(f"t = {t}, loss = {current_loss}, coeffs = {[round(co.item(), 8) for co in t_coeffs]}")

    # Plot Cost
    plot_loss(max_epochs, cost, fn='figures/cost_scaling.pdf')
    compare_model(model, fn='figures/model_scaling.pdf')


if __name__ == "__main__":
    main()
