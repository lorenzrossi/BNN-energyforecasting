"""Bayes by backprop (Blundell et al., 2015) for a fully connected network.

Every weight and bias has a Gaussian variational posterior N(mu, sigma^2) with
sigma = softplus(rho). The prior is a scale mixture of two zero-mean Gaussians;
with pi = 1 it reduces to a single Gaussian. The KL term is estimated by Monte
Carlo from the weights sampled in the forward pass, which is what the original
paper and the blitz library do.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

LOG_SQRT_2PI = math.log(math.sqrt(2 * math.pi))


def _log_normal(x, mu, sigma):
    return -LOG_SQRT_2PI - torch.log(sigma) - (x - mu) ** 2 / (2 * sigma**2)


class BayesianLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_sigma_1: float = 0.1,
        prior_sigma_2: float = 0.4,
        prior_pi: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -7.0,
    ):
        super().__init__()
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        shape = (out_features, in_features)
        self.weight_mu = nn.Parameter(torch.empty(shape).normal_(posterior_mu_init, 0.1))
        self.weight_rho = nn.Parameter(torch.empty(shape).normal_(posterior_rho_init, 0.1))
        self.bias_mu = nn.Parameter(torch.empty(out_features).normal_(posterior_mu_init, 0.1))
        self.bias_rho = nn.Parameter(torch.empty(out_features).normal_(posterior_rho_init, 0.1))
        # Set by forward(): log q(w) - log p(w) for the last sampled weights.
        self.kl = torch.tensor(0.0)

    def _sample(self, mu, rho):
        sigma = F.softplus(rho)
        w = mu + sigma * torch.randn_like(mu)
        return w, _log_normal(w, mu, sigma).sum() - self._log_prior(w)

    def _log_prior(self, w):
        log_p1 = _log_normal(w, 0.0, torch.tensor(self.prior_sigma_1, device=w.device))
        if self.prior_pi >= 1.0:
            return log_p1.sum()
        log_p2 = _log_normal(w, 0.0, torch.tensor(self.prior_sigma_2, device=w.device))
        mix = self.prior_pi * torch.exp(log_p1) + (1 - self.prior_pi) * torch.exp(log_p2)
        return torch.log(mix + 1e-8).sum()

    def forward(self, x):
        w, kl_w = self._sample(self.weight_mu, self.weight_rho)
        b, kl_b = self._sample(self.bias_mu, self.bias_rho)
        self.kl = kl_w + kl_b
        return F.linear(x, w, b)


class BayesianMLP(nn.Module):
    def __init__(self, n_inputs: int, hidden: int = 10, n_layers: int = 2, **prior):
        super().__init__()
        layers, width = [], n_inputs
        for _ in range(n_layers):
            layers += [BayesianLinear(width, hidden, **prior), nn.ReLU()]
            width = hidden
        layers.append(BayesianLinear(width, 1, **prior))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def kl_divergence(self):
        """KL estimate from the most recent forward pass."""
        return sum(m.kl for m in self.modules() if isinstance(m, BayesianLinear))


def elbo_loss(model, x, y, n_samples: int = 5, kl_weight: float = 1.0):
    """Average over n_samples weight draws of MSE + kl_weight * KL.

    Returns (loss, mse, kl); only loss carries gradients that matter.
    """
    mse_total, kl_total = 0.0, 0.0
    for _ in range(n_samples):
        mse_total = mse_total + F.mse_loss(model(x), y)
        kl_total = kl_total + model.kl_divergence()
    mse = mse_total / n_samples
    kl = kl_total / n_samples
    return mse + kl_weight * kl, mse, kl


@torch.no_grad()
def predict_samples(model, x, n_samples: int = 100) -> torch.Tensor:
    """Stack n_samples stochastic forward passes: shape (n_samples, batch, 1)."""
    return torch.stack([model(x) for _ in range(n_samples)])
