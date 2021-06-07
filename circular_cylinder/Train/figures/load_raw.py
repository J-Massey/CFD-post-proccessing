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

    def __init__(self, sim_dir, D=96):
        # sim_dir = '/home/masseyjmo/Workspace/Lotus/projects/cylinder_dns/validation'
        self.D = D
        names = ['t', 'dt', 'angle', 'px', 'py', 'pz',
                 'vx', 'vy', 'vz', 'v2x', 'v2y', 'v2z', 'vforcex', 'vforcey', 'vforcez']
        fos = np.loadtxt(os.path.join(sim_dir, str(D) + '/fort.9'), unpack=True)
        self.fos = dict(zip(names, fos))
        self.dpdx = np.loadtxt(os.path.join(sim_dir, str(D) + '/fort.10'), unpack=True)
        self.ang = np.pi / np.shape(self.dpdx)[1]
        self.profiles = \
            postproc.boundary_layer.ProfileDataset(os.path.join(sim_dir, str(D)), print_res=128, multi=8)
        self.u0 = self.profiles.get_u(0, self.D, 1)[0]
        self.du_1 = ((self.profiles.get_u(2, self.D, 1)[0]).T / 2).T

    def ground_truth(self, spacing=2):
        """
        What is the ground truth for this ML model
        Args:
            spacing: how many grid points are we using as spacing for the one side finite difference
        Returns:
            O(2) one side finite difference

        """
        du_2 = -(3 * 0 - 4 * self.profiles.get_u(spacing, self.D, 1)[0]
                  + self.profiles.get_u(2 * spacing, self.D, 1)[0]) / (2 * spacing)
        return du_2

    def sub_smear_prof(self, down):
        profile = self.profiles.profiles_x[:, :, ::down]
        len_ = np.shape(profile)[-1]
        dis = (np.linspace(0, 1, len_) - 1 / 2) * len_
        du = np.gradient(profile, axis=-1)
        for i in range(20):
            profile = profile * np.roll(mu_0(dis), -1) + np.roll(du, 0) * np.roll(mu_1(dis), -1)
            du = np.gradient(profile, axis=-1)
        return profile[:, :, int(len_/2)], -profile[:, :, int(len_/2+2)] / 2

    def clean_data(self):
        du_2 = np.concatenate(self.ground_truth(), axis=0)
        p = np.concatenate(self.dpdx, axis=0)
        u0 = np.concatenate(self.u0, axis=0)
        du_1 = np.concatenate(self.du_1, axis=0)
        return np.stack((p, u0, du_1, du_2), axis=1)

    def data(self):
        du_2 = np.concatenate(self.ground_truth(), axis=0)
        du_2_t = du_2
        p = np.concatenate(self.dpdx, axis=0)
        pt = p
        u0 = np.concatenate(self.u0, axis=0)
        du_1 = np.concatenate(self.du_1, axis=0)
        for down in tqdm(range(1, 6), desc='Sub sample and convolve', ascii=True):
            un, cn = self.sub_smear_prof(down)
            p = np.append(p, pt)
            u0 = np.append(u0, un)
            du_1 = np.append(du_1, cn)
            du_2 = np.append(du_2, du_2_t)
        return np.stack((p, u0, du_1, du_2), axis=1)

    @staticmethod
    def add_poly_orders(data, poly_degree: int):
        """
        Adds data X^n to the front of the array
        Args:
            poly_degree: how many polynomial degrees you add
            data: the data to add polynomial orders to

        Returns: More data, yum

        """
        d = np.copy(data)
        p_t = d[:, 0:-1]
        p = d[:, 0:-1]
        for deg in tqdm(range(2, poly_degree + 1), desc='Create polynomial', ascii=True):
            p = np.concatenate((p_t ** deg, p), axis=-1)
        return np.hstack((p, d[:, -1][..., np.newaxis]))

    @staticmethod
    def add_poly_interactions(data, n):
        poly = PolynomialFeatures(n)
        poly = poly.fit_transform(data[:, 0:-1])
        return np.stack((poly, data[:, -1]), axis=1)


def main():
    data_root = '/home/masseyjmo/Workspace/Lotus/projects/cylinder_dns/data'
    da = LoadData(data_root)
    fos = da.fos
    conv = da.data()
    with open(r"scaling_correction/fos.pickle", "wb") as output_file:
        cPickle.dump(fos, output_file)
    np.save('scaling_correction/data.npy', conv)
    np.save('poly.npy', da.add_poly_orders(conv, 14))


if __name__ == "__main__":
    main()

