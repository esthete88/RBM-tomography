import math
import numpy as np
import torch
import torch.nn as nn

from numpy.polynomial.hermite import hermval

from rbm import RBM


class Tomograph(nn.Module):
    def __init__(self, vis_size, hid_size, gibbs=True, n_samples=1, n_gibbs_steps=1,
                 init_sigma=1, dtype=torch.float32, eps=1e-8):
        super().__init__()

        self.vis_size = vis_size
        self.hid_size = hid_size

        self.gibbs = gibbs
        self.n_gibbs_steps = n_gibbs_steps
        self.n_samples = n_samples

        self.amplitude_rbm = RBM(vis_size, hid_size, init_sigma=init_sigma, dtype=dtype)
        self.phase_rbm = RBM(vis_size, hid_size, init_sigma=init_sigma, dtype=dtype)

        self.dtype = dtype
        self._eps = eps

    def forward(self, vis):
        if self.gibbs:
            vis = self.amplitude_rbm.sample(vis, n_gibbs_steps=self.n_gibbs_steps)
        else:
            fock_indices = torch.arange(2 ** self.vis_size)
            vis = idx2vis(fock_indices, self.vis_size)

        amplitude_prob = self.amplitude_rbm.prob(vis)
        amplitude = torch.sqrt(amplitude_prob / amplitude_prob.sum())

        phase_prob = self.phase_rbm.prob(vis)
        phase = torch.log(phase_prob + self._eps) / 2

        predicted_state = amplitude, phase

        return predicted_state, vis

    def predict(self):
        fock_indices = torch.arange(2 ** self.vis_size)
        vis = idx2vis(fock_indices, self.vis_size)

        amplitude_prob = self.amplitude_rbm.prob(vis)
        amplitude = (torch.sqrt(amplitude_prob / amplitude_prob.sum())).cpu().detach().numpy()

        phase_prob = self.phase_rbm.prob(vis)
        phase = (torch.log(phase_prob + self._eps) / 2).cpu().detach().numpy()

        predicted_state = amplitude * np.exp(1j * phase)

        return predicted_state

    def loss(self, data_states, predicted_state):
        data_amplitudes, data_phases = data_states  # [batch_size, n_indices]
        predicted_amplitude, predicted_phase = predicted_state  # [n_indices,]

        amplitudes = data_amplitudes * predicted_amplitude
        phases = data_phases - predicted_phase

        likelihood = torch.sum(amplitudes * torch.cos(phases), dim=1) ** 2 \
                   + torch.sum(amplitudes * torch.sin(phases), dim=1) ** 2

        return -torch.mean(torch.log(likelihood + self._eps))

    def fit(self, x, theta, n_epochs=1000, lr=1e-3, callbacks=None):
        device = next(self.parameters()).device

        fock_indices = torch.arange(2 ** self.vis_size)
        encoded_data = encode_data(fock_indices, x, theta, self.dtype, device)  # [batch_size, n_indices]

        vis = idx2vis(fock_indices[torch.randint(len(fock_indices), (self.n_samples,))], self.vis_size)

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for e in range(n_epochs):
            predicted_state, vis = self.forward(vis)
            sampled_indices = vis2idx(vis)

            data_amplitudes = encoded_data[0][:, sampled_indices]
            data_phases = encoded_data[1][:, sampled_indices]
            data_states = data_amplitudes, data_phases

            loss = self.loss(data_states, predicted_state)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_log = {
                'epoch': e,
                'n_epochs': n_epochs,
                'loss': loss.cpu().detach().numpy(),
            }

            for callback in callbacks:
                callback(epoch_log)


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
        id_bin = torch.as_tensor([int(c) for c in bin(idx[i])[2:]], dtype=dtype, device=device)
        vis[i, -id_bin.shape[0]:] = id_bin
    return vis


def vis2idx(vis):
    return torch.sum(2 ** reversed(torch.arange(0, vis.shape[1])) * vis, dim=1, dtype=torch.long)


def count_hermvals(n, x, dtype=torch.float32, device=torch.device('cpu')):
    batch_size = len(x)
    hermvals = torch.zeros(batch_size, n.shape[0])
    for i, nval in enumerate(n):
        coef = np.zeros(n[-1] + 1)
        coef[nval] = 1
        hermvals[:, i] = hermval(x, coef)
    return hermvals


def factorial(x):
    return torch.exp(torch.lgamma(x + 1.))
