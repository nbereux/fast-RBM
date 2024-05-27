import numpy as np
import torch
from tqdm import tqdm

from typing import Tuple

from fastrbm.rcm.utils import unravel_index


@torch.jit.script
def compute_cov(mat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the covariance for mat with zeros on the diagonal

    Parameters
    ----------
    mat : torch.Tensor
        Samples matrix. (n_dim, n_sample)

    Returns
    -------
    torch.Tensor
        Covariance matrix. (n_dim, n_dim)
    torch.Tensor
        Diagonal elements of the covariance matrix. (n_dim,)
    """
    cov = mat @ mat.T / (mat.shape[1] - 1)
    mat = torch.sqrt(torch.diag(cov).unsqueeze(1) @ torch.diag(cov).unsqueeze(0))
    cov /= mat
    diagonal = torch.diag(cov).clone()
    cov.fill_diagonal_(0.0)
    return cov, diagonal


def merge_features(
    features: torch.Tensor,
    idx: torch.Tensor,
    idy: torch.Tensor,
    tmp: torch.Tensor,
    m: torch.Tensor,
    cov: torch.Tensor,
    diagonal: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge features given two indexes and compute the new covariance matrix.

    Parameters
    ----------
    features : torch.Tensor
        Features of the RCM. (n_feat, n_dim+1)
    idx : torch.Tensor
        Index of the first feature. (1,)
    idy : torch.Tensor
        Index of the second feature. (1,)
    tmp : torch.Tensor
        #TODO
    m : torch.Tensor
        Discretization points. (n_points, n_dim)
    cov : torch.Tensor
        Covariance matrix of the features with zero diagonal. (n_feat, n_feat)
    diagonal : torch.Tensor
        Diagonal of the covariance matrix. (n_feat)
    device : torch.device
        Device.
    dtype : torch.dtype
        Dtype.

    Returns
    -------
    torch.Tensor,
        New features. (n_feat-1, n_dim+1)
    torch.Tensor,
        #TODO
    torch.Tensor,
        New covariance matrix. (n_feat-1, n_feat-1)
    torch.Tensor]
        Diagonal of the new covariance matrix. (n_feat-1, n_feat-1)
    """
    intrinsic_dimension = m.shape[1]
    new_feature = (features[idx] + features[idy]) / 2

    indices = torch.arange(features.shape[0], device=device, dtype=dtype)
    mask = (indices != idx) & (indices != idy)

    features = features[mask]
    tmp = tmp[mask]
    diagonal = diagonal[mask]
    cov = cov[mask][:, mask]

    features = torch.vstack([features, new_feature.unsqueeze(0)])
    to_add_tmp = torch.abs(
        new_feature[:(intrinsic_dimension)] @ m.T - new_feature[intrinsic_dimension]
    )
    to_add_tmp -= to_add_tmp.mean()
    tmp = torch.vstack([tmp, to_add_tmp.unsqueeze(0)])
    new_line = tmp @ to_add_tmp.T / (tmp.shape[1] - 1)
    new_line[:-1] /= torch.sqrt(diagonal * new_line[-1])
    diagonal = torch.cat([diagonal, new_line[-1].unsqueeze(0)])
    cov = torch.hstack([cov, new_line[:-1].unsqueeze(1)])
    cov = torch.vstack([cov, new_line.unsqueeze(0)])
    cov[-1, -1] = 0
    return features, tmp, cov, diagonal


def eliminate_features(
    features: torch.Tensor,
    m: torch.Tensor,
    num_features: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Merge features with the highest correlation coef until the number of rows reaches num_features.

    Parameters
    ----------
    features : torch.Tensor
        Features. (n_feat, n_dim+1)
    m : torch.Tensor
        Discretization points. (n_points, n_dim)
    num_features : int
        Desired number of features
    device : torch.device
        Device
    dtype : torch.dtype
        Data type

    Returns
    -------
    torch.Tensor
        Features. (num_features, n_dim+1)
    """
    if features.shape[0] <= num_features:
        print("Not enough features to perform feature merging. Skipping.")
        return features

    intrinsic_dimension = m.shape[1]
    tmp = (
        torch.abs(
            features[:, :(intrinsic_dimension)] @ m.T
            - features[:, intrinsic_dimension].unsqueeze(1)
        )
        / features.shape[0]
    )
    tmp = tmp - torch.mean(tmp, dim=1).unsqueeze(1)

    pbar = tqdm(range(features.shape[0] - num_features))
    pbar.set_description("Merging features")
    cov, diagonal = compute_cov(tmp)
    for i in pbar:
        idx, idy = unravel_index(torch.argmax(cov), cov.shape)
        features, tmp, cov, diagonal = merge_features(
            features=features,
            idx=idx,
            idy=idy,
            tmp=tmp,
            m=m,
            cov=cov,
            diagonal=diagonal,
            device=device,
            dtype=dtype,
        )
    return features


@torch.jit.script
def evaluate_features(features: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
    """Compute the distance from each point to the hyperplanes defined by the features

    Parameters
    ----------
    features : torch.Tensor
        Features of the RCM. (n_feat, n_dim+1)
    sample : torch.Tensor
        Points. (n_sample, n_dim)

    Returns
    -------
    torch.Tensor
        Distance for each point to each feature. (n_feat, n_sample)
    """
    intrinsic_dimension = sample.shape[1]
    return (
        torch.abs(
            features[:, :(intrinsic_dimension)] @ sample.T
            - features[:, -1].unsqueeze(1)
        )
        / features.shape[0]
    )


def decimation(
    features: torch.Tensor, q: torch.Tensor, feature_threshold: int, num_visibles: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Removes features with an importance q < feature_threshold / num_visibles.

    Parameters
    ----------
    features : torch.Tensor
        Features of the RCM. (n_feat, n_dim+1)
    q : torch.Tensor
        Feature importance. (n_feat,)
    feature_threshold : int
        #TODO
    num_visibles : int
        Dimension of the original space

    Returns
    -------
    torch.Tensor
        Filtered features. (new_n_feat, n_dim+1)
    torch.Tensor
        Filtered features importance. (new_n_feat,)
    """
    mask = torch.where(q > feature_threshold / num_visibles)[0]
    features = features[mask]
    q = q[mask]
    return features, q


def compute_loss(data, features, gamma):
    first_term = 0.5 * ((data @ features.T) ** 2).mean(0).sum()
    corr = features.T @ features - torch.eye(
        features.shape[1], device=features.device, dtype=features.dtype
    )
    second_term = torch.norm(corr, p="fro") ** 2
    return first_term, gamma * second_term


def compute_grad(data, features, gamma):
    M = features.T @ features - torch.eye(
        features.shape[1], device=features.device, dtype=features.dtype
    )
    first_term = (data.unsqueeze(1) * (data @ features.T).unsqueeze(2)).mean(0)
    second_term = features @ (M + M.T) / features.shape[0]
    return first_term + gamma * second_term


def gen_features_v2(data, num_features):
    data = torch.hstack(
        [data, torch.ones((data.shape[0], 1), device=data.device, dtype=data.dtype)]
    )
    features = torch.rand(
        (num_features, data.shape[1]), device=data.device, dtype=data.dtype
    )
    features /= features.norm(dim=1).unsqueeze(1)

    all_train_loss = []
    all_test_loss = []
    learning_rate = 0.1
    num_epochs = 10000
    gamma = 2

    pbar = tqdm(range(num_epochs))

    thr = 1e-8
    prev_loss = 0
    for i in pbar:
        grad = compute_grad(data, features, gamma)
        features -= learning_rate * grad
        features /= features.norm(dim=1).unsqueeze(1)
        train_loss, reg_train = compute_loss(data, features, gamma)
        all_train_loss.append(train_loss.item())

        if np.abs(train_loss.item() - prev_loss) < thr:
            break
        prev_loss = train_loss.item()
        # pbar.write(f"{train_loss:.3f} | {test_loss:.3f}")
    return features


def get_features(num_features, intrinsic_dimension):
    features = np.zeros((num_features, intrinsic_dimension + 1))
    # spherical distribution of features: this should not be used for intrinsic_dimension-with_bias > 2
    intrinsic_dimension = features.shape[1] - 1
    n_hidden = features.shape[0]

    if intrinsic_dimension == 1:
        features[:, 0] = 1.0
        features[:, intrinsic_dimension] = np.linspace(-1, 1, n_hidden)

    elif intrinsic_dimension == 2:
        n = np.arange(n_hidden) // 4
        m = np.arange(n_hidden) % 4
        features[:, 0] = np.cos(m * np.pi / 4.0)
        features[:, 1] = np.sin(m * np.pi / 4.0)
        features[:, intrinsic_dimension] = np.linspace(-1, 1, n_hidden // 4)[n]

    elif intrinsic_dimension == 3:
        n = np.arange(n_hidden) // 13
        m = np.arange(n_hidden) % 13
        # print((n_hidden // 13))
        # print(n)
        features[:, intrinsic_dimension] = np.linspace(-1, 1, n_hidden // 13 + 1)[n]
        for m_val in range(13):
            m_mask = m == m_val

            if m_val < 3:
                features[m_mask, m_val] = 1.0

            elif m_val < 6:
                features[m_mask, m_val - 3] = np.sqrt(0.5)
                features[m_mask, (m_val - 2) % 3] = np.sqrt(0.5)

            elif m_val < 9:
                features[m_mask, m_val - 6] = -np.sqrt(0.5)
                features[m_mask, (m_val - 5) % 3] = np.sqrt(0.5)

            else:
                s1 = (12 - m_val) // 2
                s2 = (12 - m_val) % 2
                features[m_mask, 0] = (2 * s1 - 1) / np.sqrt(3.0)
                features[m_mask, 1] = (2 * s2 - 1) / np.sqrt(3.0)
                features[m_mask, 2] = 1 / np.sqrt(3.0)

    elif intrinsic_dimension == 4:
        n = np.arange(n_hidden) // 40
        m = np.arange(n_hidden) % 40
        features[:, intrinsic_dimension] = np.linspace(-1, 1, n_hidden // 40)[n]

        for m_val in range(40):
            m_mask = m == m_val

            if m_val < 4:
                features[m_mask, m_val] = 1.0

            elif m_val < 8:
                features[m_mask, m_val - 4] = np.sqrt(0.5)
                features[m_mask, (m_val - 3) % 4] = np.sqrt(0.5)

            elif m_val < 10:
                features[m_mask, m_val - 8] = np.sqrt(0.5)
                features[m_mask, (m_val - 6) % 4] = np.sqrt(0.5)

            elif m_val < 14:
                features[m_mask, m_val - 10] = -np.sqrt(0.5)
                features[m_mask, (m_val - 9) % 4] = np.sqrt(0.5)

            elif m_val < 16:
                features[m_mask, m_val - 14] = -np.sqrt(0.5)
                features[m_mask, (m_val - 12) % 4] = np.sqrt(0.5)

            elif m_val < 32:
                s1 = m_val - 16 > 3 and m_val - 16 < 8
                s2 = m_val - 16 > 7
                features[m_mask, (m_val - 16) % 4] = (1 - 2 * s1) / np.sqrt(3.0)
                features[m_mask, (m_val - 15) % 4] = (1 - 2 * s2) / np.sqrt(3.0)
                features[m_mask, (m_val - 14) % 4] = 1 / np.sqrt(3.0)

            else:
                s = [
                    (m_val > 32 and m_val < 35),
                    (m_val > 33 and m_val < 37),
                    (m_val > 35 and m_val < 39),
                    (m_val > 37),
                ]
                for a in range(4):
                    features[m_mask, a] = 0.5 * (1 - 2 * s[a])
    else:
        raise ValueError("Too many intrinsic dimensions")
    return features
