"""
@author: Lucas Deecke
@description: Simple implementation of Gaussian Mixture Model
@contact: jmom1n15@soton.ac.uk
"""

import torch
import numpy as np
from math import pi
import matplotlib.pyplot as plt


def check_size(x):
    if len(x.size()) == 2:
        # (n, d) --> (n, 1, d)
        x = x.unsqueeze(1)

    return x


def plot_gmm(data, y, file, x_label, y_label, title=None, alpha=1., colours=None, label=None):
    n = y.shape[0]

    ax = plt.gca()
    fig = plt.gcf()
    ax.set_title(title)

    ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)

    # Edit frame, labels and legend
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Make legend manually
    if label and colours is not None:
        from matplotlib.lines import Line2D
        legend_elements = []
        for idx, loop in enumerate(label):
            legend_elements.append(Line2D([0], [0], color=colours[idx], lw=4, label=loop))
        ax.legend(handles=legend_elements, loc='lower right')

    # plot the locations of all dat points ..
    for idx, point in enumerate(data.dat):
        ax.scatter(*point, alpha=alpha, color=colours[y[idx].cpu()])

    plt.savefig(file, transparent=False, bbox_inches='tight')


class GaussianMixture(torch.nn.Module):
    """
    Fits a mixture of k=1,..,K Gaussians to the input dat (K is supplied via n_components).
    Input tensors are expected to be flat with dimensions (n: number of samples, d: number of features).
    The model then extends them to (n, 1, d).
    The model parametrization (mu, sigma) is stored as (1, k, d), and probabilities are shaped (n, k, 1) if they
    relate to an individual sample, or (1, k, 1) if they assign membership probabilities to one of the mixture
    components.
    """

    def __init__(self, n_components, n_features, mu_init=None, var_init=None, eps=1.e-6):
        """
        Initializes the model and brings all tensors into their required shape.
        The class expects dat to be fed as a flat tensor in (n, d).
        The class owns:
            x:              torch.Tensor (n, 1, d)
            mu:             torch.Tensor (1, k, d)
            var:            torch.Tensor (1, k, d)
            pie:            torch.Tensor (1, k, 1)
            eps:            float
            n_components:   int
            n_features:     int
            log_likelihood: float
        args:
            n_components:   int
            n_features:     int
        options:
            mu_init:        torch.Tensor (1, k, d)
            var_init:       torch.Tensor (1, k, d)
            eps:            float
        """
        super(GaussianMixture, self).__init__()

        self.params_fitted = True
        self.n_components = n_components
        self.n_features = n_features

        self.mu_init = mu_init
        self.var_init = var_init
        self.eps = eps

        self.log_likelihood = -np.inf

        self._init_params()

    def _init_params(self):
        if self.mu_init is not None:
            assert self.mu_init.size() == (1, self.n_components, self.n_features), \
                "Input mu_init does not have required tensor dimensions (1, %i, %i)" \
                % (self.n_components, self.n_features)
            # (1, k, d)
            self.mu = torch.nn.Parameter(self.mu_init, requires_grad=False)
        else:
            self.mu = torch.nn.Parameter(torch.randn(1, self.n_components, self.n_features), requires_grad=False)

        if self.var_init is not None:
            assert self.var_init.size() == (1, self.n_components, self.n_features), \
                "Input var_init does not have required tensor dimensions (1, %i, %i)" \
                % (self.n_components, self.n_features)
            # (1, k, d)
            self.var = torch.nn.Parameter(self.var_init, requires_grad=False)
        else:
            self.var = torch.nn.Parameter(torch.ones(1, self.n_components, self.n_features), requires_grad=False)

        # (1, k, 1)
        self.pi = torch.nn.Parameter(torch.Tensor(1, self.n_components, 1), requires_grad=False) \
            .fill_(1. / self.n_components)

        self.params_fitted = False

    def bic(self, x):
        """
        Bayesian information criterion for a batch of samples.
        args:
            x:      torch.Tensor (n, d) or (n, 1, d)
        returns:
            bic:    float
        """
        x = check_size(x)
        n = x.shape[0]

        # Free parameters for covariance, means and mixture components
        free_params = self.n_features * self.n_components + self.n_features + self.n_components - 1

        bic = -2. * self.__score(x, sum_data=False).mean() * n + free_params * np.log(n)

        return bic

    def fit(self, x, delta=2e-3, n_iter=1000, warm_start=False):
        """
        Fits model to the dat.
        args:
            x:          torch.Tensor (n, d) or (n, k, d)
        options:
            delta:      float
            n_iter:     int
            warm_start: bool
        """
        if not warm_start and self.params_fitted:
            self._init_params()

        x = check_size(x)

        i = 0
        j = np.inf

        while (i <= n_iter) and (j >= delta):

            log_likelihood_old = self.log_likelihood
            mu_old = self.mu
            var_old = self.var

            self.__em(x)
            self.log_likelihood = self.__score(x)

            if (self.log_likelihood.abs() == float("Inf")) or (self.log_likelihood == float("nan")):
                # When the log-likelihood assumes inane values, reinitialize model
                self.__init__(self.n_components,
                              self.n_features,
                              mu_init=self.mu_init,
                              var_init=self.var_init,
                              eps=self.eps)

            i += 1
            j = self.log_likelihood - log_likelihood_old

            if j <= delta:
                # When score decreases, revert to old parameters
                self.__update_mu(mu_old)
                self.__update_var(var_old)

    def predict(self, x, probs=False):
        """
        Assigns input dat to one of the mixture components by evaluating the likelihood under each.
        If probs=True returns normalized probabilities of class membership.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
            probs:      bool
        returns:
            p_k:        torch.Tensor (n, k)
            (or)
            y:          torch.LongTensor (n)
        """
        x = check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

        if probs:
            p_k = torch.exp(weighted_log_prob)
            return torch.squeeze(p_k / (p_k.sum(1, keepdim=True)))
        else:
            return torch.squeeze(torch.max(weighted_log_prob, 1)[1].type(torch.LongTensor))

    def predict_proba(self, x):
        """
        Returns normalized probabilities of class membership.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
        returns:
            y:          torch.LongTensor (n)
        """
        return self.predict(x, probs=True)

    def score_samples(self, x):
        """
        Computes log-likelihood of samples under the current model.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
        returns:
            score:      torch.LongTensor (n)
        """
        x = check_size(x)

        score = self.__score(x, sum_data=False)
        return score

    def _estimate_log_prob(self, x):
        """
        Returns a tensor with dimensions (n, k, 1), which indicates the log-likelihood that samples belong to the
        k-th Gaussian. args: x:            torch.Tensor (n, d) or (n, 1, d) returns: log_prob:     torch.Tensor (n,
        k, 1)
        """
        x = check_size(x)

        mu = self.mu
        prec = torch.rsqrt(self.var)

        log_p = torch.sum((mu * mu + x * x - 2 * x * mu) * (prec ** 2), dim=2, keepdim=True)
        log_det = torch.sum(torch.log(prec), dim=2, keepdim=True)

        return -.5 * (self.n_features * np.log(2. * pi) + log_p) + log_det

    def _e_step(self, x):
        """
        Computes log-responses that indicate the (logarithmic) posterior belief (sometimes called responsibilities)
        that a dat point was generated by one of the k mixture components. Also returns the mean of the mean of the
        logarithms of the probabilities (as is done in sklearn). This is the so-called expectation step of the
        EM-algorithm. args: x:              torch.Tensor (n,d) or (n, 1, d) returns: log_prob_norm:  torch.Tensor (1)
        log_resp:       torch.Tensor (n, k, 1)
        """
        x = check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)
        log_resp = weighted_log_prob - log_prob_norm

        return torch.mean(log_prob_norm), log_resp

    def _m_step(self, x, log_resp):
        """
        From the log-probabilities, computes new parameters pie, mu, var (that maximize the log-likelihood). This is
        the maximization step of the EM-algorithm. args: x:          torch.Tensor (n, d) or (n, 1, d) log_resp:
        torch.Tensor (n, k, 1) returns: pie:         torch.Tensor (1, k, 1) mu:         torch.Tensor (1, k,
        d) var:        torch.Tensor (1, k, d)
        """
        x = check_size(x)

        resp = torch.exp(log_resp)

        pie = torch.sum(resp, dim=0, keepdim=True) + self.eps
        mu = torch.sum(resp * x, dim=0, keepdim=True) / pie

        x2 = (resp * x * x).sum(0, keepdim=True) / pie
        mu2 = mu * mu
        xmu = (resp * mu * x).sum(0, keepdim=True) / pie
        var = x2 - 2 * xmu + mu2 + self.eps

        pie = pie / x.shape[0]

        return pie, mu, var

    def __em(self, x):
        """
        Performs one iteration of the expectation-maximization algorithm by calling the respective subroutines.
        args:
            x:          torch.Tensor (n, 1, d)
        """
        _, log_resp = self._e_step(x)
        pie, mu, var = self._m_step(x, log_resp)

        self.__update_pi(pie)
        self.__update_mu(mu)
        self.__update_var(var)

    def __score(self, x, sum_data=True):
        """
        Computes the log-likelihood of the dat under the model.
        args:
            x:                  torch.Tensor (n, 1, d)
            sum_data:           bool
        returns:
            score:              torch.Tensor (1)
            (or)
            per_sample_score:   torch.Tensor (n)

        """
        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)
        per_sample_score = torch.logsumexp(weighted_log_prob, dim=1)

        if sum_data:
            return per_sample_score.sum()
        else:
            return torch.squeeze(per_sample_score)

    def __update_mu(self, mu):
        """
        Updates mean to the provided value.
        args:
            mu:         torch.FloatTensor
        """

        assert mu.size() in [(self.n_components, self.n_features), (1, self.n_components,
                                                                    self.n_features)], "Input mu does not have " \
                                                                                       "required tensor dimensions (" \
                                                                                       "%i, %i) or (1, %i, %i)" % (
                                                                                           self.n_components,
                                                                                           self.n_features,
                                                                                           self.n_components,
                                                                                           self.n_features)

        if mu.size() == (self.n_components, self.n_features):
            self.mu = mu.unsqueeze(0)
        elif mu.size() == (1, self.n_components, self.n_features):
            self.mu.dat = mu

    def __update_var(self, var):
        """
        Updates variance to the provided value.
        args:
            var:        torch.FloatTensor
        """

        assert var.size() in [(self.n_components, self.n_features), (1, self.n_components,
                                                                     self.n_features)], "Input var does not have " \
                                                                                        "required tensor dimensions (" \
                                                                                        "%i, %i) or (1, %i, %i)" % (
                                                                                            self.n_components,
                                                                                            self.n_features,
                                                                                            self.n_components,
                                                                                            self.n_features)

        if var.size() == (self.n_components, self.n_features):
            self.var = var.unsqueeze(0)
        elif var.size() == (1, self.n_components, self.n_features):
            self.var.dat = var

    def __update_pi(self, pie):
        """
        Updates pie to the provided value.
        args:
            pie:         torch.FloatTensor
        """

        assert pie.size() in [
            (1, self.n_components, 1)], "Input pie does not have required tensor dimensions (%i, %i, %i)" % (
            1, self.n_components, 1)

        self.pi.dat = pie
