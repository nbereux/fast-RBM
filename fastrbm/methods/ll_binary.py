import h5py
import numpy as np
import torch
from typing import Tuple

Tensor = torch.Tensor


@torch.jit.script
def sample_hiddens(
    v: Tensor, hbias: Tensor, weight_matrix: Tensor, beta: float = 1.0
) -> Tuple[Tensor, Tensor]:
    """Samples the hidden layer conditioned on the state of the visible layer.

    Args:
        v (Tensor): Visible layer.
        hbias (Tensor): Hidden bias.
        weight_matrix (Tensor): Weight matrix.
        beta (float, optional): Inverse temperature. Defaults to 1..

    Returns:
        Tuple[Tensor, Tensor]: Hidden units, hidden magnetizations.
    """
    mh = torch.sigmoid(beta * (hbias + (v @ weight_matrix)))
    h = torch.bernoulli(mh)
    return (h, mh)


@torch.jit.script
def sample_visibles(
    h: Tensor, vbias: Tensor, weight_matrix: Tensor, beta: float = 1.0
) -> Tuple[Tensor, Tensor]:
    """Samples the visible layer conditioned on the hidden layer.

    Args:
        h (Tensor): Hidden layer.
        vbias (Tensor): Visible bias.
        weight_matrix (Tensor): Weight matrix.
        beta (float, optional): Inverse temperature. Defaults to 1..

    Returns:
        Tuple[Tensor, Tensor]: Visible units, visible magnetizations.
    """
    mv = torch.sigmoid(beta * (vbias + (h @ weight_matrix.T)))
    v = torch.bernoulli(mv)
    return (v, mv)


def compute_energy(
    v: Tensor, h: Tensor, params: Tuple[Tensor, Tensor, Tensor]
) -> Tensor:
    """Computes the Hamiltonian on the visible (v) and hidden (h) variables.

    Args:
        v (Tensor): Visible units.
        h (Tensor): Hidden units.
        params (Tuple[Tensor, Tensor, Tensor])): (vbias, hbias, weight_matrix) Parameters of the model.

    Returns:
        Tensor: Energy of the data points.
    """
    vbias, hbias, weight_matrix = params
    fields = torch.tensordot(vbias, v, dims=[[0], [1]]) + torch.tensordot(
        hbias, h, dims=[[0], [1]]
    )
    interaction = torch.multiply(
        v, torch.tensordot(h, weight_matrix, dims=[[1], [1]])
    ).sum(1)

    return -fields - interaction


def compute_partition_function_AIS(
    num_chains: int,
    num_beta: int,
    params: Tuple[Tensor, Tensor, Tensor],
    device: torch.device,
) -> float:
    """Estimates the partition function of the model using Annealed Importance Sampling.

    Args:
        num_chains (int): Number of parallel chains.
        num_beta (int): Number of inverse temperatures that define the trajectories.
        params (Tuple[Tensor, Tensor, Tensor])): (vbias, hbias, weight_matrix) Parameters of the model.
        device (device): device.

    Returns:
        float: Estimate of the log-partition function.
    """
    vbias, hbias, weight_matrix = params
    num_visibles = vbias.shape[0]
    num_hiddens = hbias.shape[0]
    E = torch.zeros(num_chains, device=device, dtype=torch.float64)
    beta_list = np.linspace(0.0, 1.0, num_beta)
    dB = 1.0 / num_beta

    # initialize the chains
    vbias0 = torch.zeros(size=(num_visibles,), device=device)
    hbias0 = torch.zeros(size=(num_hiddens,), device=device)
    energy0 = torch.zeros(num_chains, device=device, dtype=torch.float64)
    v = torch.bernoulli(torch.sigmoid(vbias0)).repeat(num_chains, 1)
    h = torch.bernoulli(torch.sigmoid(hbias0)).repeat(num_chains, 1)
    energy1 = compute_energy(v, h, params).type(torch.float64)
    E += energy1 - energy0
    for beta in beta_list:
        h, _ = sample_hiddens(v, hbias, weight_matrix, beta=beta)
        v, _ = sample_visibles(h, vbias, weight_matrix, beta=beta)
        E += compute_energy(v, h, params).type(torch.float64)

    # Subtract the average for avoiding overflow
    W = -dB * E
    W_ave = W.mean()
    logZ0 = (num_visibles + num_hiddens) * np.log(2)
    logZ = logZ0 + torch.log(torch.mean(torch.exp(W - W_ave))) + W_ave
    return logZ


def compute_energy_visibles(v: Tensor, params: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
    """Returns the energy of the model computed on the input data.

    Args:
        v (Tensor): Visible data.
        params (Tuple[Tensor, Tensor, Tensor]): (vbias, hbias, weight_matrix) Parameters of the model.

    Returns:
        Tensor: Energies of the data points.
    """
    vbias, hbias, weight_matrix = params
    field = v @ vbias
    exponent = hbias + (v @ weight_matrix)
    log_term = torch.where(
        exponent < 10, torch.log(1.0 + torch.exp(exponent)), exponent
    )
    return -field - log_term.sum(1)


def compute_log_likelihood_AIS_PT(filename, epochs, chains, dataset, device):
    ep = epochs[0]
    with h5py.File(filename, "r") as f:
        params = tuple(
            torch.from_numpy(f[f"epoch_{ep}"][f"{par}"][()]).to(device)
            for par in ["vbias", "hbias", "weight_matrix"]
        )
    logZ0 = compute_partition_function_AIS(
        params=params, num_chains=5000, num_beta=1000, device=device
    )
    logz = logZ0
    logZ = torch.zeros(len(epochs) - 1, device=device)
    init_energy_data = compute_energy_visibles(dataset.data, params)
    init_L = -init_energy_data.mean() - logz
    L = torch.zeros(len(epochs) - 1, device=device)
    for idx in range(len(epochs) - 1):
        ep = epochs[idx]
        with h5py.File(filename, "r") as f:
            params0 = tuple(
                torch.from_numpy(f[f"epoch_{ep}"][f"{par}"][()]).to(device)
                for par in ["vbias", "hbias", "weight_matrix"]
            )
            params1 = tuple(
                torch.from_numpy(f[f"epoch_{epochs[idx+1]}"][f"{par}"][()]).to(device)
                for par in ["vbias", "hbias", "weight_matrix"]
            )
        E0 = compute_energy_visibles(chains[idx], params0)
        E1 = compute_energy_visibles(chains[idx], params1)
        Ed = compute_energy_visibles(dataset.data, params1)
        # avoid exp overflow
        E_ave = (-E1 + E0).mean()
        c0 = torch.log(torch.mean(torch.exp(-E1 + E0 - E_ave))) + E_ave
        logz += c0
        logZ[idx] = logz
        # print(torch.mean(Ed),logZ[idx],logz)
        L[idx] = -torch.mean(Ed) - logZ[idx]
    L = torch.concatenate([torch.tensor([init_L], device=L.device), L])
    return L
