# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: J. Massey
@description: Batch load data to GPU
@contact: jmom1n15@soton.ac.uk
"""
import torch


class Dataset(torch.utils.data.Dataset):
    """
    Takes n x m dimensional numpy array
    """
    def __init__(self, X, Y):
        """Initialization"""
        self.X = torch.tensor(X)
        self.Y = torch.tensor(Y)

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.X)

    def __getitem__(self, index):
        """Generates one sample of data"""
        _X = self.X[index]
        _Y = self.Y[index]

        return _X, _Y


if __name__ == "__main__":
    from load_raw import LoadData
    data_root = '/home/masseyjmo/Workspace/Lotus/projects/cylinder_dns/validation'
    data = LoadData(data_root).data()



