import torch
import torch.nn as nn
from torch.distributions import RelaxedBernoulli


class RBM(nn.Module):
    def __init__(self, vis_size, hid_size, init_sigma=1, dtype=torch.float32):
        super().__init__()

        self.vis_size = vis_size
        self.hid_size = hid_size

        self._W = nn.Parameter(init_sigma * torch.randn(vis_size, hid_size), dtype=dtype)
        self._bv = nn.Parameter(init_sigma * torch.randn(vis_size, dtype=dtype))
        self._bh = nn.Parameter(init_sigma * torch.randn(hid_size, dtype=dtype))

        self.dtype=dtype

    def forward_pass(self, vis):
        n_samples = vis.shape[0]
        vis = vis.to(dtype=self.dtype)
        hid_logits = vis @ self._W + self._bh
        bernoulli = RelaxedBernoulli(1, logits=hid_logits)
        hid = bernoulli.rsample(n_samples)
        return hid

    def backward_pass(self, hid):
        n_samples = hid.shape[0]
        vis_logits = self._W @ hid + self._bv
        bernoulli = RelaxedBernoulli(1, logits=vis_logits)
        vis = bernoulli.rsample(n_samples)
        return vis

    def sample(self, vis, n_steps=1):
        for _ in range(n_steps):
            hid = self.forward_pass(self, vis)
            vis = self.backward_pass(self, hid)
        return vis

    def prob(self, vis):
        return torch.exp(vis @ self._bv + torch.sum(torch.log(1 + torch.exp(vis @ self._W + self._bh)), dim=1))
