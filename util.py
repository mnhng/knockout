import torch
import torch.nn as nn


class CondGaussian():
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

    def estimate(self, idx1, idx2, data, return_covariance=False):
        mu1, mu2 = self.mean[idx1], self.mean[idx2]

        up, down = self.cov[idx1], self.cov[idx2]
        s11, s12 = up[:, idx1], up[:, idx2]
        s21, s22 = down[:, idx1], down[:, idx2]

        iv = torch.linalg.inv(s11)
        cmu = mu2 + s21 @ iv @ (data[:, idx1] - mu1).T
        if return_covariance:
            return cmu.T, s22 - s21 @ iv @ s12

        return cmu.T


def copy_to(param_dict, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in param_dict.items()}


def empirical_marginals(X, Y, *ranges):
    assert X.shape[1] == len(ranges)
    marginals = []
    for i, values in enumerate(ranges):
        coord = X[:, i]
        segments = zip(values, values[1:])
        Y_equal_1_prob = [Y[(s < coord) & (coord <= e)].mean() for s, e in segments]
        marginals.append(torch.tensor(Y_equal_1_prob))

    return marginals


def js_div(P, Q):
    assert P.shape == Q.shape
    M = (P + Q) / 2
    return (nn.functional.kl_div(P.log(), M, reduction='mean') + nn.functional.kl_div(M.log(), Q, reduction='mean')) / 2