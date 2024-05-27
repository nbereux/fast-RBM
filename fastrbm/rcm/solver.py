import torch
from tqdm import tqdm

from typing import Tuple
from fastrbm.rcm.features import evaluate_features
from fastrbm.rcm.proba import get_ll_coulomb, compute_p_rcm
from fastrbm.rcm.log import build_log_string_train


@torch.jit.script
def compute_pos_grad(
    proj_train: torch.Tensor, features: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the positive term of the gradient for the visible bias and the hyperplanes weights.

    Parameters
    ----------
    proj_train : torch.Tensor
        The projected training dataset. (n_samples, n_dim)
    features : torch.Tensor
        The features of the RCM. (n_feat, n_dim+1)

    Returns
    -------
    torch.Tensor
        Positive term of the gradient for the visible bias. (n_dim,)

    torch.Tensor
        Positive term of the gradient for the hyperplanes weights. (n_feat,)
    """
    grad_vbias_pos = proj_train.mean(0)
    grad_q_pos = evaluate_features(features=features, sample=proj_train).mean(1)
    return grad_vbias_pos, grad_q_pos


@torch.jit.script
def compute_neg_grad(
    m: torch.Tensor,
    mu: torch.Tensor,
    features_rcm: torch.Tensor,
    p_m: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the negative term of the gradient for the visible bias and the hyperplanes weights.

    Parameters
    ----------
    m : torch.Tensor
        The discretization points. (n_points, n_dim)
    features_rcm : torch.Tensor
        The features of the model evaluated on the discretization points. (n_points,)
    p_m : torch.Tensor
        The probability density estimated by the RCM in each of the discretization points. (n_points,)

    Returns
    -------
    torch.Tensor
        Negative term of the gradient for the visible bias. (n_dim,)

    torch.Tensor
        Negative term of the gradient for the hyperplanes weights. (n_feat,)
    """
    grad_vbias_neg = m.T @ p_m
    mu_av = mu.T @ p_m
    grad_q_neg = features_rcm @ p_m
    return grad_vbias_neg, grad_q_neg, mu_av


def compute_hessian(
    prev_hessian: torch.Tensor,
    m: torch.Tensor,
    p_m: torch.Tensor,
    grad_vbias_neg: torch.Tensor,
    grad_q_neg: torch.Tensor,
    features_rcm: torch.Tensor,
    smooth_rate: float,
) -> torch.Tensor:
    """Compute an estimate of the Hessian for the whole set of parameters. The Hessian is then interpolated with previous one based on smooth rate.

    Parameters
    ----------
    prev_hessian : torch.Tensor
        Previous Hessian. (n_dim+n_feat, n_dim+n_feat)
    m : torch.Tensor
        The discretization points. (n_points, n_dim)
    p_m : torch.Tensor
        The probability density estimated by the RCM in each of the discretization points. (n_points,)
    grad_vbias_neg : torch.Tensor
        The negative term of the visible bias gradient. (n_dim,)
    grad_q_neg : torch.Tensor
        The negative term of the hyperplanes weights gradient. (n_feat,)
    features_rcm : torch.Tensor
        The features of the model evaluated on the discretization points. (n_points)
    smooth_rate : float
        The interpolation strength. `1` means no interpolation and `0` keep the previous Hessian.

    Returns
    -------
    torch.Tensor
        The new Hessian (n_dim+n_feat, n_dim+n_feat)
    """
    device = p_m.device
    R = torch.hstack([m, features_rcm.T])
    xr = torch.cat([grad_vbias_neg, grad_q_neg])
    # Need to use sparse matrices to avoid memory overflow while keeping a vectorized computation.
    tmp = torch.sparse.mm(
        torch.sparse.spdiags(
            p_m.cpu(), offsets=torch.tensor([0]), shape=(p_m.shape[0], p_m.shape[0])
        ).to(device),
        R,
    ).T
    new_hessian = tmp @ R - xr.unsqueeze(1) @ xr.unsqueeze(0)
    return (1 - smooth_rate) * prev_hessian + smooth_rate * new_hessian


@torch.jit.script
def update_parameters(
    vbias: torch.Tensor,
    q: torch.Tensor,
    grad_vbias_neg: torch.Tensor,
    grad_vbias_pos: torch.Tensor,
    grad_q_neg: torch.Tensor,
    grad_q_pos: torch.Tensor,
    mu_av: torch.Tensor,
    inverse_hessian: torch.Tensor,
    learning_rate: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Updates the parameters of the RCM using a quasi-Newton method.

    Parameters
    ----------
    vbias : torch.Tensor
        Visible bias. (n_dim,)
    q : torch.Tensor
        Hyperplanes weights (n_feat,)
    grad_vbias_neg : torch.Tensor
        Negative term of the visible bias gradient. (n_dim,)
    grad_vbias_pos : torch.Tensor
        Positive term of the visible bias gradient. (n_dim,)
    grad_q_neg : torch.Tensor
        Negative term of the hyperplanes weights gradient. (n_dim,)
    grad_q_pos : torch.Tensor
        Positive term of the hyperplanes weigths gradient. (n_dim,)
    inverse_hessian : torch.Tensor
        Inverse of the estimate of the Hessian for the trainable parameters. (n_dim+n_feat, n_dim+n_feat)
    learning_rate : float
        Learning rate.

    Returns
    -------
    torch.Tensor
        Updated visible bias. (n_dim,)
    torch.Tensor
        Updated hyperplanes weigths. (n_feat,)
    """
    intrinsic_dimension = vbias.shape[0]
    params = torch.cat([vbias, q])
    grad_params = torch.cat([grad_vbias_pos - grad_vbias_neg, grad_q_pos - grad_q_neg])

    # print(params.shape)
    # print(inverse_hessian.shape)
    # print(grad_params.shape)
    params += inverse_hessian @ grad_params * learning_rate
    vbias = params[:intrinsic_dimension]
    q = params[intrinsic_dimension:]
    q[q < 0] = 0
    return vbias, q


def train_rcm(
    proj_train: torch.Tensor,
    proj_test: torch.Tensor,
    m: torch.Tensor,
    mu: torch.Tensor,
    configurational_entropy: torch.Tensor,
    features: torch.Tensor,
    U: torch.Tensor,
    max_iter: int,
    num_visibles: int,
    learning_rate: float,
    adapt: bool,
    min_learning_rate: float,
    smooth_rate: float,
    stop_ll: float,
    total_iter: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    num_features = features.shape[0]
    intrinsic_dimension = m.shape[1]
    q = torch.zeros(num_features, device=device, dtype=dtype)
    vbias = torch.zeros(intrinsic_dimension, device=device, dtype=dtype)

    features_rcm = evaluate_features(features=features, sample=m)

    prev_test_ll = 0
    curr_ll = get_ll_coulomb(
        configurational_entropy=configurational_entropy,
        data=proj_train,
        m=m,
        features=features,
        q=q,
        vbias=vbias,
        U=U,
        num_visibles=num_visibles,
        return_logZ=False,
    )
    best_ll = curr_ll
    best_q = q.clone()
    best_vbias = vbias.clone()
    grad_vbias_pos, grad_q_pos = compute_pos_grad(
        proj_train=proj_train, features=features
    )
    print(f"LL start: {curr_ll}")

    count_lr = 0
    hessian = torch.eye(num_features + intrinsic_dimension, device=device, dtype=dtype)
    inverse_hessian = torch.eye(
        num_features + intrinsic_dimension, device=device, dtype=dtype
    )

    pbar = tqdm(range(max_iter))
    header = " num iter | train ll | test ll  |  mean q  | |\u2207vbias| | curr lr  | count +lr"
    pbar.write(header)
    all_train_ll = []
    all_test_ll = []
    for n_iter in pbar:
        p_m = compute_p_rcm(
            m=m,
            configurational_entropy=configurational_entropy,
            features_rcm=features_rcm,
            vbias=vbias,
            q=q,
            num_visibles=num_visibles,
        )
        grad_vbias_neg, grad_q_neg, mu_av = compute_neg_grad(
            m=m, mu=mu, features_rcm=features_rcm, p_m=p_m
        )
        vbias, q = update_parameters(
            vbias=vbias,
            q=q,
            grad_vbias_neg=grad_vbias_neg,
            grad_vbias_pos=grad_vbias_pos,
            grad_q_neg=grad_q_neg,
            grad_q_pos=grad_q_pos,
            mu_av=mu_av,
            inverse_hessian=inverse_hessian,
            learning_rate=learning_rate,
        )
        if n_iter % 100 == 0:
            new_ll = get_ll_coulomb(
                configurational_entropy=configurational_entropy,
                data=proj_train,
                m=m,
                features=features,
                q=q,
                vbias=vbias,
                U=U,
                num_visibles=num_visibles,
            )
            if new_ll > curr_ll:
                learning_rate *= 1.0 + 0.02 * adapt
                count_lr += 1
            else:
                learning_rate *= 1.0 - 0.1 * adapt
            learning_rate = max(min_learning_rate, learning_rate)
            if new_ll > best_ll:
                best_q = torch.clone(q)
                best_vbias = torch.clone(vbias)
                best_ll = new_ll
            curr_ll = new_ll
            hessian = compute_hessian(
                prev_hessian=hessian,
                m=m,
                p_m=p_m,
                grad_vbias_neg=grad_vbias_neg,
                grad_q_neg=grad_q_neg,
                features_rcm=features_rcm,
                smooth_rate=smooth_rate,
            )
            # Lstsq is not precise enough -> the training diverges at some point
            # inverse_hessian = torch.linalg.lstsq(
            #     hessian, torch.eye(hessian.shape[0], device=device, dtype=dtype)
            # ).solution

            # pinv is slow
            # inverse_hessian = torch.linalg.pinv(hessian, rtol=1e-4)

            # Regularized inversion seems to remain the best
            inverse_hessian = torch.linalg.inv(
                hessian
                + torch.diag(
                    torch.ones(hessian.shape[0], device=device, dtype=dtype) * 1e-5
                )
            )
            if n_iter % 1000 == 0:
                fill = " "
                new_test_ll = get_ll_coulomb(
                    configurational_entropy=configurational_entropy,
                    data=proj_test,
                    m=m,
                    features=features,
                    q=q,
                    vbias=vbias,
                    U=U,
                    num_visibles=num_visibles,
                )
                all_train_ll.append(new_ll.item())
                all_test_ll.append(new_test_ll.item())
                grad_vbias_norm = torch.norm(grad_vbias_pos - grad_vbias_neg)
                log_string = build_log_string_train(
                    train_ll=new_ll.item(),
                    test_ll=new_test_ll.item(),
                    n_iter=n_iter,
                    mean_q=q.mean().item(),
                    grad_vbias_norm=grad_vbias_norm.item(),
                    curr_lr=learning_rate,
                    count=count_lr,
                )
                count_lr = 0
                pbar.write(log_string)
                if torch.abs(prev_test_ll - new_test_ll) < stop_ll:
                    break
                prev_test_ll = prev_test_ll * 0.05 + new_test_ll * 0.95
    total_iter += n_iter
    all_train_ll = torch.tensor(all_train_ll)
    all_test_ll = torch.tensor(all_test_ll)
    return best_vbias, best_q, all_train_ll, all_test_ll, total_iter
