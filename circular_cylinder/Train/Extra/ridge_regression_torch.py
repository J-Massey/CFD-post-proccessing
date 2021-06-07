import torch
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split


class Ridge:
    def __init__(self, device: str, alpha=0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.device = device

    def fit(self, X: torch.tensor, y: torch.tensor) -> None:
        X = X.rename(None)
        y = y.rename(None).view(-1, 1)
        assert X.shape[0] == y.shape[0], "Number of X and y rows don'torch match"
        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1, device=self.device), X], dim=1)
        # Solving X*w = y with Normal equations:
        # X^{torch}*X*w = X^{torch}*y
        lhs = X.T @ X
        rhs = X.T @ y
        if self.alpha == 0:
            self.w, _ = torch.lstsq(rhs, lhs)
        else:
            ridge = self.alpha * torch.eye(lhs.shape[0], device=self.device)
            self.w, _ = torch.lstsq(rhs, lhs + ridge)

    def predict(self, X: torch.tensor) -> None:
        X = X.rename(None)
        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1, device=self.device), X], dim=1)
        return X @ self.w


if __name__ == "__main__":
    # demo
    data = np.load('../scaling_correction/data.npy')
    x_train, x_test, y_train, y_test = train_test_split(data[:, 0:-1], data[:, -1], test_size=0.1, shuffle=False)

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    x_train = torch.tensor(x_train, device=device)
    y_train = torch.tensor(y_train, device=device)

    x_test = torch.tensor(x_test, device=device)
    y_test = torch.tensor(y_test, device=device)
    for alph in range(-20, 20, 2):
        model = Ridge(alpha=10**alph, fit_intercept=False)
        model.fit(x_train, y_train, device=device)
        rss = model.predict(x_test).cpu().numpy()
        print(len(rss), np.sum(rss**2))

