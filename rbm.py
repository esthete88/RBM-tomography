import torch
import torch.nn as nn
from torch.distributions import Bernoulli


class RBM(nn.Module):
    """RBM model.

    Parameters
    ----------
    vis_size : int
        Number of visible neurons (number of qubits).
    hid_size : int
        Number of hidden neurons.
    init_sigma : float
        Standard deviation for initialization of RBM parameters.

    """

    def __init__(self, vis_size, hid_size, init_sigma=1, dtype=torch.float32):
        super().__init__()

        self.vis_size = vis_size
        self.hid_size = hid_size

        self._W = nn.Parameter(init_sigma * torch.randn(vis_size, hid_size, dtype=dtype))
        self._bv = nn.Parameter(init_sigma * torch.randn(vis_size, dtype=dtype))
        self._bh = nn.Parameter(init_sigma * torch.randn(hid_size, dtype=dtype))

        self.dtype = dtype

    def forward_pass(self, vis):
        """Computes samples of hidden neurons."""
        hid_logits = vis @ self._W + self._bh
        bernoulli = Bernoulli(logits=hid_logits)
        hid = bernoulli.sample()
        return hid

    def backward_pass(self, hid):
        """Computes samples of visible neurons."""
        vis_logits = hid @ self._W.T + self._bv
        bernoulli = Bernoulli(logits=vis_logits)
        vis = bernoulli.sample()
        return vis

    def sample(self, vis, n_gibbs_steps=1):
        """Samples from RBM using Gibbs sampling with `n_gibbs_steps` iterations and
        `vis` as a starting point for Markov chain."""
        vis = vis.to(dtype=self.dtype)
        for _ in range(n_gibbs_steps):
            hid = self.forward_pass(vis)
            vis = self.backward_pass(hid)
        return vis

    def prob(self, vis):
        """Returns p(v) for each v in `vis`."""
        return torch.exp(vis @ self._bv + torch.sum(torch.log(1 + torch.exp(vis @ self._W + self._bh)), dim=1))
