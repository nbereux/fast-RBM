import numpy as np
import torch
from tqdm import tqdm


def compute_phi(m, mu, U, num_sites, num_colors):
    num_visibles = num_sites * num_colors
    sqrn = num_visibles**0.5
    p1 = -(np.log(num_colors) / num_colors)
    p2 = (
        1
        / num_visibles
        * torch.logsumexp(
            2 * sqrn * (mu @ U).reshape(-1, num_sites, num_colors), dim=-1
        ).sum(-1)
    )
    p3 = 1 / sqrn * (mu @ U).sum(-1)
    return p1 + p2 - p3


def get_configurational_entropy(m, mu, U, num_sites, num_colors):
    return compute_phi(m, mu, U, num_sites, num_colors) - torch.einsum(
        "bi, bi->b", m, mu
    )


def eff_grad_phi(mu, U, num_colors, num_sites):
    num_visibles = num_colors * num_sites
    sqrn = np.sqrt(num_visibles)
    num_points = mu.shape[0]
    tmp = (mu @ U).reshape(num_points, num_sites, num_colors)
    U = U.reshape(U.shape[0], num_sites, num_colors)
    tmp = torch.nn.functional.softmax(2 * sqrn * tmp, dim=2) - 0.5

    U = U.unsqueeze(0)
    tmp = tmp.unsqueeze(1)
    grad = torch.einsum("abiq, cdiq -> cb", U, tmp)
    grad *= 2 / sqrn
    return grad


def eff_hessian(mu, U, num_sites, num_colors):
    num_visibles = num_colors * num_sites
    sqrn = torch.sqrt(torch.tensor([num_visibles], device=mu.device))
    num_points = mu.shape[0]
    tmp = (mu @ U).reshape(num_points, num_sites, num_colors)
    U = U.reshape(mu.shape[1], num_sites, num_colors)
    tmp = torch.nn.functional.softmax(2 * sqrn * tmp, dim=2)
    ## First term
    n_dim = U.shape[0]

    first_term = torch.zeros(
        (num_points, U.shape[0], U.shape[0]), device=mu.device, dtype=mu.dtype
    )
    second_term = torch.zeros(
        (num_points, U.shape[0], U.shape[0]), device=mu.device, dtype=mu.dtype
    )

    # Pre-compute some values:
    curr_U = U.unsqueeze(0)
    curr_tmp = tmp.unsqueeze(1)
    tmp_second_term = torch.einsum("abiq, cdiq -> cbi", curr_U, curr_tmp)

    # First term
    for beta in range(n_dim):
        for gamma in range(n_dim):
            curr_U = U[beta] * U[gamma]
            first_term[:, beta, gamma] = torch.einsum("iq, biq -> b", curr_U, tmp)

    # Second term
    second_term = torch.bmm(tmp_second_term, tmp_second_term.permute(0, 2, 1))

    hessian = 4 * (first_term - second_term)
    return hessian


def get_lagrange_multipliers_potts(
    m: torch.Tensor,
    U: torch.Tensor,
    err_threshold: float = 1e-10,
    max_iter: int = 10000,
    num_colors: int = 21,
):
    num_points, intrinsic_dimension = m.shape
    n_visible = U.shape[1]
    num_sites = n_visible // num_colors
    err = 10

    mu = torch.zeros((num_points, intrinsic_dimension), device=m.device, dtype=m.dtype)
    done_m = []
    done_mu = []
    H = torch.eye(U.shape[0], device=U.device, dtype=U.dtype).repeat(m.shape[0], 1, 1)
    for n_iter in tqdm(range(max_iter)):
        grad_phi = eff_grad_phi(mu, U, num_colors, num_sites)

        H = 0.9 * H + 0.1 * eff_hessian(mu, U, num_sites, num_colors)
        to_keep = torch.where(torch.linalg.matrix_rank(H) == U.shape[0])
        m = m[to_keep]
        mu = mu[to_keep]
        grad_phi = grad_phi[to_keep]
        H = H[to_keep]

        dmu = torch.linalg.solve(H, (m - grad_phi))
        mu += dmu
        err = torch.linalg.vector_norm(dmu, ord=float("inf"), dim=1)

        # If the gradient explodes, the point won't converge
        to_keep = err < 1e5
        m = m[to_keep]
        mu = mu[to_keep]
        grad_phi = grad_phi[to_keep]
        H = H[to_keep]
        err = err[to_keep]

        to_store = err <= err_threshold
        if to_store.sum() > 0:
            done_m.append(m[to_store])
            done_mu.append(mu[to_store])
            m = m[torch.logical_not(to_store)]
            mu = mu[torch.logical_not(to_store)]
            H = H[torch.logical_not(to_store)]
        if len(m) == 0:
            break
    if len(done_m) > 0:
        done_m = torch.vstack(done_m)
        done_mu = torch.vstack(done_mu)
    else:
        done_m = torch.tensor([], device=m.device, dtype=m.dtype)
        done_mu = torch.tensor([], device=m.device, dtype=m.dtype)

    num_converged_points = done_m.shape[0]
    configurational_entropy = torch.zeros(
        num_converged_points, device=done_m.device, dtype=done_m.dtype
    )

    configurational_entropy = get_configurational_entropy(
        done_m, done_mu, U, num_sites=num_sites, num_colors=num_colors
    )
    print(f"Number of valid points: {num_converged_points}/{num_points}")
    return (done_m, done_mu, configurational_entropy)
