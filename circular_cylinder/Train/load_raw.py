# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: J. Massey
@description: Load data from files to numpy
@contact: jmom1n15@soton.ac.uk
"""
import numpy as np
import os
import postproc.boundary_layer
from tqdm import tqdm
import _pickle as cPickle
from sklearn.preprocessing import PolynomialFeatures


def mu_0(dis):
    mu = np.empty(np.shape(dis))
    mu[abs(dis) < 2] = 0.5 * (1 + dis[abs(dis) < 2] / 2 + 1 / np.pi * np.sin(dis[abs(dis) < 2] / 2 * np.pi))
    mu[dis <= -2] = 0
    mu[dis >= 2] = 1
    return mu


def mu_1(dis):
    mu = np.empty(np.shape(dis))
    mu[abs(dis) < 2] = 2 * (1 / 4 - (dis[abs(dis) < 2] / (2 * 2)) ** 2 -
                            1 / (2 * np.pi)
                            * (dis[abs(dis) < 2] / 2 * np.sin(dis[abs(dis) < 2] * np.pi / 2)
                               + 1 / np.pi * (1 + np.cos(dis[abs(dis) < 2] * np.pi / 2))))
    mu[abs(dis) >= 2] = 0
    return mu


class LoadData:
    """
    Load data for machine learning model
    """

    def __init__(self, sim_dir, d=128):
        # sim_dir = '/home/masseyjmo/Workspace/Lotus/projects/cylinder_dns/data'
        self.D = d
        self.sim_dir = sim_dir
        names = ['t', 'dt', 'angle', 'px', 'py', 'pz']
        fos = np.loadtxt(os.path.join(self.sim_dir, str(d) + '/3D/fort.9'), unpack=True)
        self.fos = dict(zip(names, fos))
        self.dpdx = np.loadtxt(os.path.join(self.sim_dir, str(d) + '/3D/fort.10'), unpack=True)
        self.ang = np.pi / np.shape(self.dpdx)[1]
        self.profiles = \
            postproc.boundary_layer.ProfileDataset(os.path.join(self.sim_dir, str(d), '3D'), print_res=128, multi=16)
        self.u0 = self.profiles.get_s(0, self.D, 1)
        self.du_1 = (np.array(self.profiles.get_s(2, self.D, 1)) / (2 * self.D / 128))

    def ground_truth(self, spacing=2):
        """
        What is the ground truth for this ML model
        Args:
            d: Length scale, added as a variable so we can augment d=96 to the model
            spacing: how many grid points are we using as spacing for the one side finite difference (epsilon?)
        Returns:
            O(2) one side finite difference

        """
        du_2 = -(3 * 0 - 4 * self.profiles.get_s(spacing, self.D, 1)
                 + self.profiles.get_s(2 * spacing, self.D, 1)) / (2 * spacing)
        return du_2

    def sub_smear_prof(self, down):
        profile = self.profiles.profiles[:, :, ::down]
        len_ = np.shape(profile)[-1]
        dis = (np.linspace(0, 1, len_) - 1 / 2) * len_
        du = np.gradient(profile, axis=-1)
        for i in range(20):
            profile = profile * np.roll(mu_0(dis), -1) + np.roll(du, 0) * np.roll(mu_1(dis), -1)
            du = np.gradient(profile, axis=-1)
        return profile[:, :, int(len_/2)], profile[:, :, int(len_/2+3)] / (2*self.D/128/down)

    def clean_data(self):
        du_2 = np.concatenate(self.ground_truth(), axis=0)
        p = np.concatenate(self.dpdx, axis=0)
        u0 = np.concatenate(self.u0, axis=0)
        du_1 = np.concatenate(self.du_1, axis=0)
        return np.stack((p, u0, du_1, du_2), axis=1)

    def aug_96(self):
        d = 96
        dpdx = np.loadtxt(os.path.join(self.sim_dir, str(d) + '/3D/fort.10'), unpack=True)
        profiles = \
            postproc.boundary_layer.ProfileDataset(os.path.join(self.sim_dir, str(d), '3D'), print_res=128, multi=16)
        du_2 = -(3 * 0 - 4 * profiles.get_s(2, d, 1)
                 + profiles.get_s(2 * 2, d, 1)) / (2 * 2)
        du_2 = np.concatenate(du_2, axis=0)
        u0 = profiles.get_s(0, d, 1)
        du_1 = (np.array(profiles.get_s(2, d, 1)) / (2 * d / 96))
        p = np.concatenate(dpdx, axis=0)
        u0 = np.concatenate(u0, axis=0)
        du_1 = np.concatenate(du_1, axis=0)
        return np.stack((p, u0, du_1, du_2), axis=1)

    def data(self):
        du_2 = np.concatenate(self.ground_truth(), axis=0)
        du_2_t = du_2
        p = np.concatenate(self.dpdx, axis=0)
        pt = p
        u0 = np.concatenate(self.u0, axis=0)
        du_1 = np.concatenate(self.du_1, axis=0)
        for down in tqdm(range(1, 2), desc='Sub sample and convolve', ascii=True):
            un, cn = self.sub_smear_prof(down)
            p = np.append(p, pt)
            u0 = np.append(u0, un)
            du_1 = np.append(du_1, cn)
            du_2 = np.append(du_2, du_2_t)
        labelled_data = np.stack((p, u0, du_1, du_2), axis=1)
        print('\n Labelled data =', np.shape(labelled_data))
        augmented_data = np.vstack((labelled_data, self.aug_96()))
        print('\n Augmented data =', np.shape(augmented_data))
        return labelled_data


def main():
    data_root = '/home/masseyjmo/Workspace/Lotus/projects/cylinder_dns/data'
    da = LoadData(data_root)
    fos = da.fos
    conv = da.data()
    with open(r"fos.pickle", "wb") as output_file:
        cPickle.dump(fos, output_file)
    np.save('data.npy', conv)


if __name__ == "__main__":
    main()

