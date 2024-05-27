import h5py
from typing import Tuple
import torch

Tensor = torch.Tensor


def init_parameters(
    num_visibles: int, num_hiddens: int, dataset: Tensor, device: torch.device
) -> Tuple[Tensor, Tensor, Tensor]:
    """Initialize the parameters of the RBM.

    Args:
        num_visibles (int): Number of visible units.
        num_hiddens (int): Number of hidden units.
        dataset (Tensor): Matrix of data.
        device (torch.device): Device.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: visible bias, hidden bias, weight matrix
    """
    eps = 1e-4
    init_std = 1e-4
    frequencies = dataset.mean(0)
    frequencies = torch.clamp(frequencies, min=eps, max=(1.0 - eps))
    vbias = (torch.log(frequencies) - torch.log(1.0 - frequencies)).to(device)
    hbias = torch.zeros(num_hiddens, device=device)
    weight_matrix = (
        torch.randn(size=(num_visibles, num_hiddens), device=device) * init_std
    )
    return (vbias, hbias, weight_matrix)


def init_parallel_chains(
    num_chains: int, num_visibles: int, num_hiddens: int, device: torch.device
) -> Tuple[Tensor, Tensor]:
    """Initialize the parallel chains of the RBM.

    Args:
        num_chains (int): Number of parallel chains.
        num_visibles (int): Number of visible units.
        num_hiddens (int): Number of hidden units.
        device (torch.device): Device.

    Returns:
        Tuple[Tensor, Tensor]: Initial visible and hidden units.
    """
    v = torch.randint(0, 2, size=(num_chains, num_visibles), device=device).type(
        torch.float32
    )
    h = torch.randint(0, 2, size=(num_chains, num_hiddens), device=device).type(
        torch.float32
    )
    return (v, h)


def compute_energy(
    v: Tensor, h: Tensor, vbias: Tensor, hbias: Tensor, weight_matrix: Tensor
) -> Tensor:
    """Returns the energy of the model computed on the input data.

    Args:
        v (Tensor): Visible data.
        h (Tensor): Hidden data.
        vbias (Tensor): Visible bias.
        hbias (Tensor): Hidden bias.
        weight_matrix (Tensor): Weight matrix.

    Returns:
        Tensor: Energies of the data points.
    """
    fields = (v @ vbias) + (h @ hbias)
    interaction = ((v @ weight_matrix) * h).sum(1)
    return -fields - interaction


def compute_gradient(
    data: Tuple[Tensor, Tensor, Tensor], parallel_chains: Tuple[Tensor, Tensor]
) -> Tuple[Tensor, Tensor, Tensor]:
    """Computes the gradient of the log-likelihood. Implements the centered version of the gradient,
    which normally improveds the quality of the learning.

    Args:
        data (Tuple[Tensor, Tensor, Tensor]): (v, h, data_weights) Observed data.
        parallel_chains (Tuple[Tensor, Tensor]): (v, h) Parallel chains.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: (grad_vbias, grad_hbias, grad_weight_matrix).
    """
    # Unpacking the arguments
    v_data, h_data, w_data = data
    v_gen, h_gen = parallel_chains
    num_chains = v_gen.shape[0]

    # Averages over data and generated samples
    v_data_mean = (v_data * w_data).sum(0) / w_data.sum()
    torch.clamp_(v_data_mean, min=1e-4, max=(1.0 - 1e-4))
    h_data_mean = (h_data * w_data).sum(0) / w_data.sum()
    v_gen_mean = v_gen.mean(0)
    torch.clamp_(v_gen_mean, min=1e-4, max=(1.0 - 1e-4))
    h_gen_mean = h_gen.mean(0)

    # Centered variables
    v_data_centered = v_data - v_data_mean
    h_data_centered = h_data - h_data_mean
    v_gen_centered = v_gen - v_data_mean
    h_gen_centered = h_gen - h_data_mean

    # Gradient
    grad_weight_matrix = (
        (v_data_centered * w_data).T @ h_data_centered
    ) / w_data.sum() - (v_gen_centered.T @ h_gen_centered) / num_chains
    grad_vbias = v_data_mean - v_gen_mean - (grad_weight_matrix @ h_data_mean)
    grad_hbias = h_data_mean - h_gen_mean - (v_data_mean @ grad_weight_matrix)
    return (grad_vbias, grad_hbias, grad_weight_matrix)


def sample_hiddens(
    v: Tensor, hbias: Tensor, weight_matrix: Tensor
) -> Tuple[Tensor, Tensor]:
    """Samples the hidden layer conditioned on the state of the visible layer.

    Args:
        v (Tensor): Visible layer.
        hbias (Tensor): Hidden bias.
        weight_matrix (Tensor): Weight matrix.

    Returns:
        Tuple[Tensor, Tensor]: Hidden units and magnetizations.
    """
    mh = torch.sigmoid(hbias + v @ weight_matrix)
    h = torch.bernoulli(mh)
    return (h, mh)


def sample_visibles(h: Tensor, vbias: Tensor, weight_matrix: Tensor) -> Tensor:
    """Samples the visible layer conditioned on the hidden layer.

    Args:
        h (Tensor): Hidden layer.
        vbias (Tensor): Visible bias.
        weight_matrix (Tensor): Weight matrix.

    Returns:
        Tensor: Visible units.
    """
    mv = torch.sigmoid(vbias + h @ weight_matrix.T)
    v = torch.bernoulli(mv)
    return v


def sample_state(
    parallel_chains: Tuple[Tensor, Tensor],
    params: Tuple[Tensor, Tensor, Tensor],
    gibbs_steps: int,
) -> Tuple[Tensor, Tensor]:
    """Generates data sampled from the model by performing at most "gibbs_steps" Monte Carlo updates.

    Args:
        parallel_chains (Tuple[Tensor, Tensor]): (v, h) Initial state.
        params (Tuple[Tensor, Tensor, Tensor]): (vbias, hbias, weight_matrix) Parameters of the model.
        gibbs_steps (int): Number of Monte Carlo updates.

    Returns:
        Tuple[Tensor, Tensor]: Updated parallel chains.
    """
    v, h = parallel_chains
    vbias, hbias, weight_matrix = params
    for _ in range(gibbs_steps):
        h, _ = sample_hiddens(v=v, hbias=hbias, weight_matrix=weight_matrix)
        v = sample_visibles(h=h, vbias=vbias, weight_matrix=weight_matrix)
    parallel_chains = (v, h)
    return parallel_chains


def compute_eps(
    parallel_chains: Tuple[Tensor, Tensor], params: Tuple[Tensor, Tensor, Tensor]
) -> Tensor:
    """Computes the effective population size.

    Args:
        parallel_chains (Tuple[Tensor, Tensor]): (v, h) Parallel chains.
        params (Tuple[Tensor, Tensor, Tensor]): (vbias, hbias, weight_matrix) Parameters of the model.

    Returns:
        Tensor: Effective population size
    """
    block_size = 50
    num_chains = len(parallel_chains[0])
    v, h = parallel_chains
    vbias, hbias, weight_matrix = params
    num_blocks = num_chains // block_size
    block_energies = torch.stack(
        [
            compute_energy(
                v[i * block_size : (i + 1) * block_size],
                h[i * block_size : (i + 1) * block_size],
                vbias,
                hbias,
                weight_matrix,
            ).mean()
            for i in range(num_blocks)
        ]
    )
    mean_energy = compute_energy(v, h, vbias, hbias, weight_matrix).mean()
    var_blocks_e = torch.sum(torch.square(block_energies - mean_energy)) / (
        num_blocks * (num_blocks - 1)
    )
    var_energy = compute_energy(v, h, vbias, hbias, weight_matrix).var()
    eps = var_energy / var_blocks_e / num_chains
    return eps


def compute_ess(logit_weights: Tensor) -> Tensor:
    lwc = logit_weights - logit_weights.min()
    numerator = torch.square(torch.mean(torch.exp(-lwc)))
    denominator = torch.mean(torch.exp(-2.0 * lwc))
    return numerator / denominator


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
