import torch
from typing import Tuple


def rcm_to_rbm(
    q: torch.Tensor, proj_vbias: torch.Tensor, features: torch.Tensor, U: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Converts the parameters of the RCM to a Bernoulli-Bernoulli RBM with {-1,1} random variables.

    Parameters
    ----------
    q : torch.Tensor
        Hyperplanes weigths. (n_feat,)
    proj_vbias : torch.Tensor
        Visible bias of the RCM. (n_dim,)
    features : torch.Tensor
        Features of the RCM. (n_feat, n_dim+1)
    U : torch.Tensor
        Projection matrix of the PCA. (n_dim, n_visible)

    Returns
    -------
    torch.Tensor
        Visible bias of the RBM. (n_visible,)
    torch.Tensor
        Hidden bias of the RBM. (n_feat,)
    torch.Tensor
        Weight matrix of the RBM. (n_visible, n_feat)
    """
    _, num_visibles = U.shape
    num_features = features.shape[0]
    n = features[:, :-1] @ U
    vbias = (U.T @ proj_vbias) * num_visibles**0.5
    hbias = num_visibles * (features[:, -1] * q) / num_features
    W = (n * q.unsqueeze(1)) * (num_visibles**0.5) / num_features
    return vbias, hbias, W.T
