import math
import numpy as np
import torch
import torch.nn as nn

from numpy.polynomial.hermite import hermval

from rbm import RBM


class Tomograph(nn.Module):
    def __init__(self, vis_size, hid_size, n_steps=1, init_sigma=1, dtype=torch.float32, eps=1e-8):

        self.vis_size = vis_size
        self.hid_size = hid_size

        self.amplitude_rbm = RBM(vis_size, hid_size, init_sigma=init_sigma, dtype=dtype)
        self.phase_rbm = RBM(vis_size, hid_size, init_sigma=init_sigma, dtype=dtype)

        self.dtype = dtype
        self._eps = eps

    def forward(self, fock_indices):
        device = next(self.parameters()).device
        vis = idx2vis(fock_indices, self.vis_size, dtype=self.dtype, device=device)
        sampled_vis = self.amplitude_rbm.sample(vis, n_steps=self.n_steps)

        amplitude_prob = self.amplitude_rbm.prob(sampled_vis)
        amplitude = torch.sqrt(amplitude_prob / amplitude_prob.sum())

        phase_prob = self.phase_rbm.prob(sampled_vis)
        phase = torch.log(phase_prob + self._eps) / 2

        return amplitude, phase

    def loss(self, encoded_data, fock_indices):
        state_amplitude, state_phase = self(fock_indices)  # [n_indices,]
        data_amplitude, data_phase = encoded_data  # [batch_size, n_indices]
        amplitude = data_amplitude * state_amplitude
        phase = data_phase - state_amplitude

        likelihood = torch.sum(amplitude * torch.cos(phase), dim=0) ** 2 \
                   + torch.sum(amplitude * torch.sin(phase), dim=0) ** 2

        return -torch.mean(torch.log(likelihood + self._eps))


def encode_data(fock_indices, x, theta, dtype=torch.float32, device=torch.device('cpu')):
    """Computes <n|X, theta> using Hermitian polynomes."""
    amplitude = count_hermvals(fock_indices, x) \
              * torch.exp(-x ** 2 / 2).unsqueeze(1) \
              * torch.pow(2, -fock_indices / 2).unsqueeze(0) \
              / torch.sqrt(factorial(fock_indices)).unsqueeze(0) \
              / math.pi ** 0.25

    phase = theta.unsqueeze(1) * fock_indices.unsqueeze(0)
    return amplitude, phase


def idx2vis(idx, dim, dtype=torch.float32, device=torch.device('cpu')):
    vis = torch.zeros(idx.shape[0], dim, dtype=dtype, device=device)
    for i in range(idx.shape[0]):
        id_bin = torch.as_tensor([int(c) for c in bin(int(idx[i]))[2:]], dtype=dtype, device=device)
        vis[i, -id_bin.shape[0]:] = id_bin
    return vis


def vis2idx(vis):
    return torch.sum(2 ** reversed(torch.arange(0, vis.shape[1])) * vis, dim=1)


def count_hermvals(n, x, dtype=torch.float32, device=torch.device('cpu')):
    batch_size = len(x)
    hermvals = torch.zeros(batch_size, n.shape[0])
    for i, nval in enumerate(n):
        coef = np.zeros(int(n[-1]) + 1)
        coef[nval] = 1
        hermvals[:, i] = hermval(x, coef)
    return hermvals


def factorial(x):
    return torch.exp(torch.lgamma(x + 1))
