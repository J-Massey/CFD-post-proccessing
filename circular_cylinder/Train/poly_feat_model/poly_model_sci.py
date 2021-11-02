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


real_data_dir = '/home/masseyjmo/Workspace/Lotus/projects/cylinder_dns/data'
plt.style.use(['science', 'grid'])


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
        lr = LinearRegression(n_jobs=16, fit_intercept=False)
        reg = lr.fit(self.X_train, self.y_train)

        print('coeff', reg.coef_, 'inter', reg.intercept_)

        # evaluating the model on training and testing sets
        pred_train_lr = lr.predict(self.X_train)
        # print('Lr train', r2_score(self.y_train, pred_train_lr))
        pred_test_lr = lr.predict(self.X_test)
        # print('Lr test', r2_score(self.y_test, pred_test_lr))
        return reg

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


# if __name__ == "__main__":

