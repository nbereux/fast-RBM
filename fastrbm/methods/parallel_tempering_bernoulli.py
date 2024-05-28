import h5py
import numpy as np
import torch
from tqdm import tqdm
from typing import Tuple

Tensor = torch.Tensor


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
    return v, mv


def sample_state(
    chains: Tensor, params: Tuple[Tensor, Tensor, Tensor], gibbs_steps: int
) -> Tensor:
    """Generates data sampled from the model by performing gibbs_steps Monte Carlo updates.

    Args:
        parallel_chains (Tensor): Initial visible state.
        params (Tuple[Tensor, Tensor, Tensor]): (vbias, hbias, weight_matrix) Parameters of the model.
        gibbs_steps (int): Number of Monte Carlo updates.

    Returns:
        Tuple[Tensor, Tensor]: Generated visibles, generated hiddens
    """
    # Unpacking the arguments
    v = chains
    vbias, hbias, weight_matrix = params

    for _ in torch.arange(gibbs_steps):
        h, _ = sample_hiddens(v=v, hbias=hbias, weight_matrix=weight_matrix)
        v, _ = sample_visibles(h=h, vbias=vbias, weight_matrix=weight_matrix)
    return v


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


def init_sampling(
    fname_rcm: str,
    fname_rbm: str,
    n_gen: int,
    it_mcmc: int = None,
    epochs: list = None,
    device: torch.device = "cpu",
    show_pbar: bool = True,
) -> Tensor:
    use_rcm = fname_rcm is not None
    with h5py.File(fname_rbm, "r") as f:
        if epochs is None:
            epochs = []
            for key in f.keys():
                if "epoch" in key:
                    ep = int(key.replace("epoch_", ""))
                    epochs.append(ep)
            epochs = np.sort(epochs)  # epoch_0 is the rcm
        num_visibles = f["parallel_chains"][()].shape[1]
    n_models = len(epochs)
    if it_mcmc is None:
        v_init = torch.randint(0, 2, size=(n_gen, num_visibles), device=device).float()
        chains = v_init.repeat(n_models, 1).reshape(n_models, n_gen, num_visibles)
    else:
        # Sample from the rcm
        chains = torch.zeros(size=(n_models, n_gen, num_visibles), device=device)
        if use_rcm:
            with h5py.File(fname_rcm, "r") as f:
                tmp_name = "best_trial"
                U = torch.from_numpy(f["const"]["U"][()]).to(device)
                m = torch.from_numpy(f["const"]["m"][()]).to(device)
                mu = torch.from_numpy(f["const"]["mu"][()]).to(device)
                p_m = torch.from_numpy(f[tmp_name]["pdm"][()]).to(device)
            gen0 = sample_rcm(
                p_m=p_m, m=m, mu=mu, U=U, num_samples=n_gen, device=device
            )
            chains[0] = (gen0 + 1) / 2
        else:
            v_init = torch.randint(
                0, 2, size=(n_gen, num_visibles), device=device
            ).float()
            with h5py.File(fname_rbm, "r") as f:
                params = tuple(
                    torch.from_numpy(f[f"epoch_{epochs[0]}"][f"{par}"][()]).to(device)
                    for par in ["vbias", "hbias", "weight_matrix"]
                )

            chains[0] = sample_state(v_init, params=params, gibbs_steps=it_mcmc)
        # initialize all models by performing it_mcmc steps starting from the state of the previous model
        if show_pbar:
            pbar = tqdm(total=(len(epochs) - int(use_rcm)) * it_mcmc)
        for idx, ep in enumerate(epochs[1:]):
            # print(f[f"epoch_{ep}"].keys())
            with h5py.File(fname_rbm, "r") as f:
                params = tuple(
                    torch.from_numpy(f[f"epoch_{ep}"][f"{par}"][()]).to(device)
                    for par in ["vbias", "hbias", "weight_matrix"]
                )
            chains[idx + 1] = sample_state(
                chains[idx], params=params, gibbs_steps=it_mcmc
            )
            if show_pbar:
                pbar.update(it_mcmc)
    return chains


def swap_configurations(chains_rcm, chains_rbm, params_rcm, params_rbm) -> Tensor:
    """Tries to swap the chains of adjacient models using the Metropolis rule.

    Args:
        chains (Tensor): Previous configuration of the chains.
        show_acc_rate (bool, optional): Print the acceptance rate. Defaults to False.
        epochs (list): List of epochs to be used for the sampling.

    Returns:
        Tensor: Swapped chains.
    """
    # device = chains_rbm.device
    chains_rbm = chains_rbm[0]
    n_chains, L = chains_rbm.shape

    # deltaE =  - E(j, s(j+1)) + E(j, s(j)) + E(j+1, s(j+1)) - E(j+1, s(j))
    delta_E = (
        -compute_energy_visibles(chains_rbm, params_rcm)
        + compute_energy_visibles(chains_rcm, params_rcm)
        + compute_energy_visibles(chains_rbm, params_rbm)
        - compute_energy_visibles(chains_rcm, params_rbm)
    )

    swap_chain = torch.bernoulli(torch.clamp(torch.exp(delta_E), max=1.0)).bool()
    acc_rate = (swap_chain.sum() / n_chains).cpu().numpy()
    swapped_chains_rbm = torch.where(
        swap_chain.unsqueeze(1).repeat(1, L), chains_rcm, chains_rbm
    )
    # print(f"Acc rate:", acc_rate)
    return swapped_chains_rbm, acc_rate


def sample_rcm(
    p_m: torch.Tensor,
    m: torch.Tensor,
    mu: torch.Tensor,
    U: torch.Tensor,
    num_samples: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Exact sampling from the RCM"""

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


def pt_sampling(rcm, params_rcm, params_rbm, num_samples, it_mcmc, increment):
    device = params_rcm[0].device
    chains_rcm = sample_rcm(**rcm, device=device, num_samples=num_samples)
    chains_rcm = (chains_rcm + 1) / 2
    chains_rbm = chains_rcm.clone()
    counts = 0
    all_acc_rate = []
    while counts < it_mcmc:
        counts += increment
        chains_rcm = sample_rcm(**rcm, device=device, num_samples=num_samples)
        chains_rcm = (chains_rcm + 1) / 2
        chains_rbm = sample_state(
            parallel_chains=(chains_rbm, torch.zeros_like(chains_rbm)),
            params=params_rbm,
            gibbs_steps=increment,
        )
        chains_rbm, curr_acc_rate = swap_configurations(
            chains_rbm=chains_rbm,
            chains_rcm=chains_rcm,
            params_rbm=params_rbm,
            params_rcm=params_rcm,
        )
        all_acc_rate.append(curr_acc_rate)
    print(f"Mean acc rate: {np.mean(all_acc_rate)}")
    return chains_rbm


# @torch.jit.script
def sampling_step(
    rcm: dict, fname_rbm: str, chains: Tensor, it_mcmc: int, epochs: list
) -> Tensor:
    """Performs it_mcmc sampling steps with all the models and samples a new configurartion for the RCM.

    Args:
        rcm (dict): Dict with rcm parameters
        fname_rbm (str): Path to rbm.
        chains (Tensor): Previous configuration of the chains.
        it_mcmc (int): Number of steps to perform.
        epochs (list): List of epochs to be used for the sampling.

    Returns:
        Tensor: Updated chains.
    """
    n_chains = len(chains[0])
    device = chains.device

    use_rcm = rcm is not None
    # Sample from the rcm
    if use_rcm:
        gen0 = sample_rcm(**rcm, device=device, num_samples=n_chains)
        gen0 = (gen0 + 1) / 2
        with h5py.File(fname_rbm, "r") as f:
            params = tuple(
                torch.from_numpy(f[f"epoch_0"][f"{par}"][()]).to(device)
                for par in ["vbias", "hbias", "weight_matrix"]
            )
            # print(params)
        gen0 = sample_state(chains=gen0, params=params, gibbs_steps=10)
        chains[0] = gen0
    # Sample from rbm
    for idx, ep in enumerate(epochs[int(use_rcm) :], int(use_rcm)):
        with h5py.File(fname_rbm, "r") as f:
            params = tuple(
                torch.from_numpy(f[f"epoch_{ep}"][f"{par}"][()]).to(device)
                for par in ["vbias", "hbias", "weight_matrix"]
            )
        chains[idx] = sample_state(
            chains=chains[idx], params=params, gibbs_steps=it_mcmc
        )
    return chains


def swap_configurations_multi(
    chains, fname_rbm, epochs: list, show_acc_rate: bool = False
) -> Tensor:
    """Tries to swap the chains of adjacient models using the Metropolis rule.

    Args:
        chains (Tensor): Previous configuration of the chains.
        fname_rbm (str): Path to rbm.
        show_acc_rate (bool, optional): Print the acceptance rate. Defaults to False.
        epochs (list): List of epochs to be used for the sampling.

    Returns:
        Tensor: Swapped chains.
    """
    device = chains.device
    n_chains, L = chains[0].shape
    f = h5py.File(fname_rbm, "r")
    n_rbms = len(chains)
    acc_rate = []
    for idx in range(n_rbms - 1):
        params_0 = tuple(
            torch.from_numpy(f[f"epoch_{epochs[idx]}"][f"{par}"][()]).to(device)
            for par in ["vbias", "hbias", "weight_matrix"]
        )
        params_1 = tuple(
            torch.from_numpy(f[f"epoch_{epochs[idx + 1]}"][f"{par}"][()]).to(device)
            for par in ["vbias", "hbias", "weight_matrix"]
        )
        # deltaE = -E(j, s(j+1)) + E(j, s(j)) + E(j+1, s(j+1)) - E(j+1, s(j))
        delta_E = (
            -compute_energy_visibles(chains[idx + 1], params_0)
            + compute_energy_visibles(chains[idx], params_0)
            + compute_energy_visibles(chains[idx + 1], params_1)
            - compute_energy_visibles(chains[idx], params_1)
        )
        swap_chain = torch.exp(delta_E) > torch.rand(size=(n_chains,), device=device)
        acc_rate.append((swap_chain.sum() / n_chains).cpu().numpy())
        swapped_chains_0 = torch.where(
            swap_chain.unsqueeze(1).repeat(1, L), chains[idx + 1], chains[idx]
        )
        swapped_chains_1 = torch.where(
            swap_chain.unsqueeze(1).repeat(1, L), chains[idx], chains[idx + 1]
        )
        chains[idx] = swapped_chains_0
        chains[idx + 1] = swapped_chains_1
    if show_acc_rate:
        print(f"acc_rate: {np.array(acc_rate)[:]}")
    f.close()
    return chains, acc_rate


def PTsampling(
    rcm: dict,
    fname_rbm: str,
    init_chains: Tensor,
    it_mcmc: int = None,
    epochs: list = None,
    increment: int = 10,
    show_pbar: bool = True,
    show_acc_rate: bool = True,
) -> Tensor:
    chains = init_chains
    f = h5py.File(fname_rbm, "r")

    if epochs is None:
        epochs = []
        for key in f.keys():
            if "epoch" in key:
                ep = int(key.replace("epoch_", ""))
                epochs.append(ep)
        epochs = np.sort(epochs)  # epoch_0 is the rcm
    f.close()
    assert len(epochs) == len(
        chains
    ), f"epochs and chains must have the same length, but got {len(epochs)} and {len(chains)}"
    n_models = len(epochs) - 1
    steps = 0
    if show_pbar:
        pbar = tqdm(total=it_mcmc)
    save_chains = []

    while steps < it_mcmc:
        if show_pbar:
            pbar.update(increment)  # * n_models)
        chains, acc_rate = swap_configurations_multi(
            chains=chains,
            fname_rbm=fname_rbm,
            epochs=epochs,
            show_acc_rate=show_acc_rate,
        )
        chains = sampling_step(
            rcm=rcm,
            fname_rbm=fname_rbm,
            chains=chains,
            it_mcmc=increment,
            epochs=epochs,
        )
        if steps % 10 == 0:
            save_chains.append(chains[-1].cpu().numpy())
        steps += increment  # * n_models
    return chains, save_chains, acc_rate

