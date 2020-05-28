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
        
#         self._bv = nn.Parameter(torch.zeros(vis_size, dtype=dtype))
#         self._bh = nn.Parameter(torch.zeros(hid_size, dtype=dtype))

        self.dtype = dtype

    def forward_pass(self, vis, temperature=1):
        """Computes samples of hidden neurons."""
        hid_logits = vis @ self._W + self._bh
        hid_probs = torch.sigmoid(hid_logits / temperature)
        bernoulli = Bernoulli(hid_probs)
        hid = bernoulli.sample()
        return hid

    def backward_pass(self, hid, temperature=1):
        """Computes samples of visible neurons."""
        vis_logits = hid @ self._W.T + self._bv
        vis_probs = torch.sigmoid(vis_logits / temperature)
        bernoulli = Bernoulli(vis_probs)
        vis = bernoulli.sample()
        return vis

    def sample(self, vis, n_gibbs_steps=1, temperature=1):
        """Samples from RBM using Gibbs sampling with `n_gibbs_steps` iterations and
        `vis` as a starting point for Markov chain."""
        vis = vis.to(dtype=self.dtype)
        for _ in range(n_gibbs_steps):
            hid = self.forward_pass(vis, temperature)
            vis = self.backward_pass(hid, temperature)
        return vis

    def prob(self, vis):
        """Returns p(v) for each v in `vis`."""
        return torch.exp(vis @ self._bv + torch.sum(torch.log(1 + torch.exp(vis @ self._W + self._bh)), dim=1))
