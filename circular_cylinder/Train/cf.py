# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: J. Massey
@description: profiles from DNS
@contact: jmom1n15@soton.ac.uk
"""
import seaborn as sns
import numpy as np
import os
import sys
import torch
import torch.optim as optim
import postproc.io as io
import postproc.boundary_layer
import importlib
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def plot1(x, y, label_x, label_y, title=None, legend=None, save=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    for i in enumerate(x):
        ax.plot(x[i[0]], y[i[0]], 'r')
    ax.set_xlabel(label_x, fontsize=16)
    ax.set_ylabel(label_y, fontsize=16)
    if title != None: ax.set_title(title)
    if legend != None: ax.legend(legend)
    plt.savefig('loss.png', dpi=200)
    plt.show()


data_root = '/home/masseyjmo/Workspace/Lotus/projects/cylinder_dns/validation'
names = ['t', 'dt', 'angle', 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'v2x', 'v2y', 'v2z', 'vforcex', 'vforcey', 'vforcez']

fos = np.loadtxt(os.path.join(data_root, 'd-96/3D/fort.9'), unpack=True)
fos = dict(zip(names, fos))
p = np.loadtxt(os.path.join(data_root, 'd-96/3D/fort.10'), unpack=True)
ang = np.pi / np.shape(p)[1]
profiles = postproc.boundary_layer.ProfileDataset(os.path.join(data_root, 'd-96/3D'))
u0, v0 = profiles.get_u(0, 96, 2, 256)
# Find 1st order fd at eps away from the boundary (is this good enough to be the answer?) show verification
u2, v2 = profiles.get_u(2, 96, 2, 256)
cd, cl = u2 / 2, v2 / 2
p = torch.tensor(p, device=device)
p = p.reshape([1, np.product(p.size())]).squeeze()
u0 = torch.tensor(u0, device=device)
u0 = u0.reshape([1, np.product(u0.size())]).squeeze()
cd = torch.tensor(cd, device=device)
cd = cd.reshape([1, np.product(cd.size())]).squeeze()

D = torch.stack((p, u0, cd), dim=1)
x_dataset = D[:, 0:2].t()
y_dataset = D[:, 2].t()

# %%
a, b = 0, 0, 0
# %%
lr = 1e-5
its = int(1e6)
optimiser = optim.Adam
# Provide some initial guesses from the visualisation
coeffs = [a, b]
t_coeffs = []
for i in coeffs:
    t_coeffs.append(torch.tensor(float(i), requires_grad=True, device=device))
a, b = t_coeffs


# Define the prediction model
def model(x_input):
    pressure, vel = x_input
    return a * pressure + b * vel


# Loss function definition
def loss(y_hat, y_target):
    return ((y_hat - y_target) ** 2).sum()


# Setup the optimizer object, so it optimizes a and b.
optimizer = optimiser([a, b], lr=lr)  # <----------

# Main optimization loop
cost = []
for t in tqdm(range(its), desc='Optimisation', ascii=True):
    optimizer.zero_grad()  # Set the gradients to 0.
    y_predicted = model(x_dataset)  # Compute the current predicted y's from x_dataset
    current_loss = loss(y_predicted, y_dataset)  # See how far off the prediction is
    cost.append(current_loss)
    current_loss.backward()  # Compute the gradient of the loss with respect to A and b.
    optimizer.step()  # Update A and b accordingly.
plot1([np.arange(its)], [cost], 'Iterations', 'Cost')
print(
    f"t = {t}, loss = {current_loss}, a = {a.item()}; b = {b.item()}"
)
a = a.item()
b = b.item()
#%%
cd_mod = model()

#%%
# Test on mean quantities

plt.style.use(['science', 'grid'])

D = 96
colors = sns.color_palette("husl", 4)


# Plot TSs and save spectra
fig, ax = plt.subplots(figsize=(5, 3))
ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
ax.set_xlabel(r"$t/c")
ax.set_ylabel(r'$C_{D_f}$')
# ax.plot(fos['t'], -fos['vforcex'], label=r'$U^{\prime}$')
ax.plot(fos['t'], fos['vx'], label=r'$O(1)$ FD')
ax.plot(fos['t'], fos['v2x'], label=r'$O(2)$ FD')
ax.legend()
plt.savefig('test.pdf')
plt.show()


