import numpy as np
import torch
from typing import Tuple

from fastrbm.rcm.features import evaluate_features
from fastrbm.rcm.utils import log2cosh


def compute_p_rcm(
    m: torch.Tensor,
    configurational_entropy: torch.Tensor,
    features_rcm: torch.Tensor,
    vbias: torch.Tensor,
    q: torch.Tensor,
    num_visibles: int,
) -> torch.Tensor:
    """Computes an estimation of the density learned by the RCM on the discretization points.

    Parameters
    ----------
    m : torch.Tensor
        Discretization points. (n_points, n_dim)
    configurational_entropy : torch.Tensor
        Configurational entropy for the discretization points. (n_points,)
    features_rcm : torch.Tensor
        Evaluation of the features of the RCM on the discretization points. (n_points,)
    vbias : torch.Tensor
        Visible bias of the RCM. (n_dim,)
    q : torch.Tensor
        Hyperplanes weights. (n_feat,)
    num_visibles : int
        Dimension of the original dataset.

    Returns
    -------
    torch.Tensor
        Estimation of the density. (n_points,)
    """
    F = -configurational_entropy - m @ vbias - features_rcm.T @ q
    p_m = torch.exp(-num_visibles * (F - F.min()))
    Z = p_m.sum()
    return p_m / Z


def get_energy_coulomb(
    sample: torch.Tensor,
    features: torch.Tensor,
    q: torch.Tensor,
    vbias: torch.Tensor,
) -> torch.Tensor:
    """Compute the energy of the RCM for each of the sample points.

    Parameters
    ----------
    sample : torch.Tensor
        Sample points. (n_points, n_dim)
    features : torch.Tensor
        Features of the RCM. (n_feat, n_dim+1)
    q : torch.Tensor
        Hyperplanes weights. (n_feat,)
    vbias : torch.Tensor
        Visible bias of the RCM.

    Returns
    -------
    torch.Tensor
        Energy of the RCM. (n_points,)
    """
    eval_features = evaluate_features(features=features, sample=sample)
    return -q @ eval_features - vbias @ sample.T


def get_ll_coulomb(
    configurational_entropy: torch.Tensor,
    data: torch.Tensor,
    m: torch.Tensor,
    features: torch.Tensor,
    q: torch.Tensor,
    vbias: torch.Tensor,
    U: torch.Tensor,
    num_visibles: int,
    return_logZ: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """Compute the log-likelihood of the RCM on the data.

    Parameters
    ----------
    configurational_entropy : torch.Tensor
        Configurational_entropy of the discretization points. (n_points)
    data : torch.Tensor
        Data points. (n_data, n_dim)
    m : torch.Tensor
        Discretization points. (n_points, n_dim)
    features : torch.Tensor
        Features of the RCM. (n_feat, n_dim+1)
    q : torch.Tensor
        Hyperplanes weights. (n_feat,)
    vbias : torch.Tensor
        Visible bias of the RCM. (n_dim)
    num_visibles : int
        Dimension of the original dataset,

    Returns
    -------
    torch.Tensor
        Log-likelihood
    """
    sample_energy = get_energy_coulomb(
        sample=data,
        features=features,
        q=q,
        vbias=vbias,
    )

    # compute Z
    LL = -num_visibles * torch.mean(sample_energy)
    # print(LL)
    m_energy = get_energy_coulomb(sample=m, features=features, q=q, vbias=vbias)
    F = m_energy - configurational_entropy
    F0 = -configurational_entropy
    # print(F0)

    # print("F0:", F0)
    logZ = torch.logsumexp(-num_visibles * F, 0)
    logZ0 = torch.logsumexp(-num_visibles * F0, 0)
    logZ00 = num_visibles * np.log(2)
    # print("logZ: ",logZ)
    # print("logZ0: ",logZ0)
    # print("logZ00: ",logZ00)
    logZ -= logZ0 - logZ00
    if return_logZ:
        return LL - logZ, logZ
    return LL - logZ
