import math
import numpy as np
import torch
import torch.nn as nn

from numpy.polynomial.hermite import hermval

from rbm import RBM


class Tomograph(nn.Module):
    """Class for RBM tomography.

    Parameters
    ----------
    vis_size : int
        Number of visible neurons (number of qubits).
    hid_size : int
        Number of hidden neurons.
    gibbs : bool
        If `True`, uses contrastive divergence with Gibbs sampling.
    n_samples : int
        Number of samples to use in one iteration.
    n_gibbs_steps : int
        Number of steps in each Gibbs chain.
    init_sigma : float
        Standard deviation for initialization of RBM parameters.

    Attributes
    ----------
    amplitude_rbm : RBM
        RBM for computing amplitude.
    phase_rbm : RBM
        RBM for computing phase.

    """

    def __init__(self, vis_size, hid_size, gibbs=True, n_samples=2, n_gibbs_steps=1,
                 init_sigma=1, dtype=torch.float32, eps=1e-10):
        super().__init__()

        self.vis_size = vis_size
        self.hid_size = hid_size

        self.gibbs = gibbs
        self.n_gibbs_steps = n_gibbs_steps
        self.n_samples = n_samples

        self.amplitude_rbm = RBM(vis_size, hid_size, init_sigma=init_sigma, dtype=dtype)
        self.phase_rbm = RBM(vis_size, hid_size, init_sigma=init_sigma, dtype=dtype)
        
        self.temperature = 1

        self.dtype = dtype
        self._eps = eps

    def forward(self, unique_vis):            
        amplitude_prob = self.amplitude_rbm.prob(unique_vis)
        amplitude = (torch.sqrt(amplitude_prob / amplitude_prob.sum()))

        phase_prob = self.phase_rbm.prob(unique_vis)
        phase = torch.log(phase_prob + self._eps) / 2
        
        predicted_state = amplitude, phase

        return predicted_state

    def predict(self):
        """Returns tuple of amplitudes and phases of the reconstructed state in Fock basis."""
        device = next(self.parameters()).device
        fock_indices = torch.arange(2 ** self.vis_size)
        vis = idx2vis(fock_indices, self.vis_size, dtype=self.dtype, device=device)

        amplitude_prob = self.amplitude_rbm.prob(vis)
        amplitude = (torch.sqrt(amplitude_prob / amplitude_prob.sum()))

        phase_prob = self.phase_rbm.prob(vis)
        phase = torch.log(phase_prob + self._eps) / 2

        return amplitude, phase

    def llh_loss(self, data_states, predicted_state):
        """Computes negative log-likelihood of reconstruction."""
        data_amplitudes, data_phases = data_states  # [batch_size, n_indices]
        predicted_amplitude, predicted_phase = predicted_state  # [n_indices,]

        amplitudes = data_amplitudes * predicted_amplitude
        phases = data_phases + predicted_phase

        likelihood = torch.sum(amplitudes * torch.cos(phases), dim=1) ** 2 \
                   + torch.sum(amplitudes * torch.sin(phases), dim=1) ** 2

        llh = torch.mean(torch.log(likelihood + self._eps))

        return -llh

    def fit(self, x, theta, n_epochs=1000, lr=1e-1, callbacks=None):
        """Fits RBM to the data."""
        device = next(self.parameters()).device

        fock_indices = torch.arange(2 ** self.vis_size)
        
        if self.gibbs:
            vis = idx2vis(fock_indices[torch.randint(len(fock_indices), (self.n_samples,))], self.vis_size, 
                          dtype=self.dtype, device=device)
        else:
            encoded_data = encode_data(fock_indices, x, theta, self.dtype, device)  # [batch_size, n_indices]
            unique_vis = idx2vis(fock_indices, self.vis_size, dtype=self.dtype, device=device)

        opt = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.99))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, [1000, 3000])
        
        for e in range(n_epochs): 
            
            epoch_log = {
                'epoch': e,
                'n_epochs': n_epochs,
            }
            
            if self.gibbs:
                vis = idx2vis(fock_indices[torch.randint(len(fock_indices), (self.n_samples,))], self.vis_size, 
                              dtype=self.dtype, device=device)
                vis = self.amplitude_rbm.sample(vis, n_gibbs_steps=self.n_gibbs_steps, temperature=self.temperature)
                unique_vis = torch.unique(vis, dim=0)
                sampled_indices = vis2idx(unique_vis)
                epoch_log['sampled_indices'] = sampled_indices
                encoded_data = encode_data(sampled_indices, x, theta, self.dtype, device)
                
            predicted_state = self.forward(unique_vis)
            loss = self.llh_loss(encoded_data, predicted_state)
            epoch_log['loss'] = loss.cpu().detach().numpy()

            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            
            if callbacks is not None:
                for callback in callbacks:
                    callback(self, epoch_log)


def encode_data(fock_indices, x, theta, dtype=torch.float32, device=torch.device('cpu')):
    """Computes <n|X, theta> using Hermitian polynomes.

    Parameters
    ----------
    fock_indices : torch.Tensor
        Indices of fock vectors to use in decomposition.
    x : torch.Tensor
        X quadratures of the data.
    theta : torch.Tensor
        Theta quadratures of the data.

    Returns
    -------
    Tuple of amplitudes and phases.

    """

    amplitude = count_hermvals(fock_indices, x, dtype=dtype, device=device) \
              * torch.exp(-x ** 2 / 2).unsqueeze(1) \
              / 2 ** (fock_indices / 2.).unsqueeze(0) \
              / torch.sqrt(factorial(fock_indices)).unsqueeze(0) \
              / math.pi ** 0.25

    phase = theta.unsqueeze(1) * fock_indices.unsqueeze(0)
    return amplitude, phase


def idx2vis(idx, dim, dtype=torch.float64, device=torch.device('cpu')):
    """Encodes Fock vectors."""
    vis = torch.zeros(idx.shape[0], dim, dtype=dtype, device=device)
    for i in range(idx.shape[0]):
        id_bin = torch.as_tensor([int(c) for c in bin(idx[i])[2:]], dtype=dtype, device=device)
        vis[i, -id_bin.shape[0]:] = id_bin
    return vis


def vis2idx(vis):
    """Turns encodings of Fock vectors to corresponding indices."""
    return torch.sum(2 ** reversed(torch.arange(0, vis.shape[1])) * vis, dim=1, dtype=torch.long)


def count_hermvals(n, x, dtype=torch.float64, device=torch.device('cpu')):
    """Returns tensor of shape [len(x), len(n)] with hermitian polynomes values H_n(x)."""
    hermvals = torch.zeros(len(x), len(n), dtype=dtype, device=device)
    for i, nval in enumerate(n):
        coef = torch.zeros(n[-1] + 1, dtype=dtype, device=device)
        coef[nval] = 1
        hermvals[:, i] = hermval(x, coef)
    return hermvals


def factorial(x):
    """Returns element-wise factorial."""
    return torch.exp(torch.lgamma(x + 1.))
