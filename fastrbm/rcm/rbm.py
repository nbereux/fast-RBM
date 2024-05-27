import numpy as np
import torch
from typing import Tuple

from fastrbm.rcm.utils import log2cosh


def get_energy_rbm(
    m: torch.Tensor,
    Edm: torch.Tensor,
    vbias: torch.Tensor,
    configurational_entropy: torch.Tensor,
) -> torch.Tensor:
    return Edm - m @ vbias - configurational_entropy


def get_ll_rbm(
    configurational_entropy: torch.Tensor,
    data: torch.Tensor,
    m: torch.Tensor,
    W: torch.Tensor,
    hbias: torch.Tensor,
    vbias: torch.Tensor,
    U: torch.Tensor,
    num_visibles: int,
    return_logZ: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    num_samples = data.shape[0]
    proj_W = U @ W
    proj_vbias = (U @ vbias) / num_visibles**0.5
    Edm = -log2cosh((num_visibles**0.5) * (m @ proj_W) - hbias).sum(1) / num_visibles
    Fs = (data @ proj_W) * (num_visibles**0.5) - hbias.unsqueeze(0)
    Eds = -log2cosh(Fs).sum() / num_visibles
    Eds /= num_samples

    sample_energy = Eds - data @ proj_vbias
    # compute Z
    LL = -num_visibles * torch.mean(sample_energy)
    m_energy = get_energy_rbm(
        m=m,
        Edm=Edm,
        vbias=proj_vbias,
        configurational_entropy=configurational_entropy,
    )
    F = m_energy
    F0 = -configurational_entropy
    logZ = torch.logsumexp(-num_visibles * F, 0)
    logZ0 = torch.logsumexp(-num_visibles * F0, 0)
    logZ00 = num_visibles * np.log(2)
    logZ -= logZ0 - logZ00
    if return_logZ:
        return LL - logZ, logZ
    return LL - logZ


def get_proba_rbm(
    m: torch.Tensor,
    configurational_entropy: torch.Tensor,
    U: torch.Tensor,
    vbias: torch.Tensor,
    hbias: torch.Tensor,
    W: torch.Tensor,
    return_logZ: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    num_visibles = vbias.shape[0]
    num_points = m.shape[0]
    pdm = torch.zeros(num_points)
    proj_W = U @ W
    proj_vbias = (U @ vbias) / num_visibles**0.5
    energy_m = (
        -log2cosh(num_visibles**0.5 * (m @ proj_W) - hbias).sum(1) / num_visibles
        - m @ proj_vbias
        - configurational_entropy
    )
    Fmin = energy_m.min()

    Z = torch.exp(-num_visibles * (energy_m - Fmin)).sum()
    pdm = torch.exp(-num_visibles * (energy_m - Fmin)) / Z
    if return_logZ:
        return pdm, torch.log(Z)
    return pdm


def sample_rbm(
    p_m: torch.Tensor,
    m: torch.Tensor,
    mu: torch.Tensor,
    U: torch.Tensor,
    num_samples: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    num_points, intrinsic_dimension = m.shape
    num_visibles = U.shape[1]
    cdf = torch.zeros(num_points + 1, device=device, dtype=dtype)
    cdf[1:] = torch.cumsum(p_m, 0)
    x = torch.rand(num_samples, device=device, dtype=dtype)
    idx = torch.searchsorted(sorted_sequence=cdf, input=x) - 1
    mu_full = (mu[idx] @ U) * num_visibles**0.5  # n_samples x Nv
    x = torch.rand((num_samples, num_visibles), device=device, dtype=dtype)
    p = 1 / (1 + torch.exp(-2 * mu_full))  # n_samples x Nv
    s_gen = 2 * (x < p) - 1
    return s_gen


def sample_potts_rcm(p_m, mu, U, num_samples, num_colors, device, dtype):
    num_visibles = U.shape[1]
    num_sites = num_visibles // num_colors
    num_points = mu.shape[0]
    cdf = torch.zeros(num_points + 1, device=device, dtype=dtype)
    cdf[1:] = torch.cumsum(p_m, 0)
    x = torch.rand(num_samples, device=device, dtype=dtype)
    idx = torch.searchsorted(sorted_sequence=cdf, input=x) - 1
    mu_full = (mu[idx] @ U) * num_visibles**0.5  # n_samples x Nv

    p = torch.nn.functional.softmax(2 * mu_full.reshape(-1, num_colors), dim=-1)
    s_gen = torch.multinomial(p.reshape(-1, num_colors), 1).reshape(
        num_samples, num_sites
    )
    return s_gen
