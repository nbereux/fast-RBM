from typing import Tuple

import torch
import numpy as np

from fastrbm.methods.methods_binary import compute_energy, init_parallel_chains

Tensor = torch.Tensor
def sample_hiddens(
        v: Tensor, hbias: Tensor, weight_matrix: Tensor, beta: float
) -> Tuple[Tensor, Tensor]:
    """Samples the hidden layer conditioned on the state of the visible layer.

    Args:
        v (Tensor): Visible layer.
        hbias (Tensor): Hidden bias.
        weight_matrix (Tensor): Weight matrix.

    Returns:
        Tuple[Tensor, Tensor]: Hidden units and magnetizations.
    """
    mh = torch.sigmoid(beta*(hbias + v @ weight_matrix))
    h = torch.bernoulli(mh)
    return (h, mh)


def sample_visibles(h: Tensor, vbias: Tensor, weight_matrix: Tensor, beta: float) -> Tensor:
    """Samples the visible layer conditioned on the hidden layer.

    Args:
        h (Tensor): Hidden layer.
        vbias (Tensor): Visible bias.
        weight_matrix (Tensor): Weight matrix.

    Returns:
        Tensor: Visible units.
    """
    mv = torch.sigmoid(beta*(vbias + h @ weight_matrix.T))
    v = torch.bernoulli(mv)
    return v

def sample_state(
        parallel_chains: Tuple[Tensor, Tensor],
        params: Tuple[Tensor, Tensor, Tensor],
        gibbs_steps: int,
        beta: float,
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
        h, _ = sample_hiddens(v=v, hbias=hbias, weight_matrix=weight_matrix, beta=beta)
        v = sample_visibles(h=h, vbias=vbias, weight_matrix=weight_matrix, beta=beta)
    parallel_chains = (v, h)
    return parallel_chains

def find_inverse_temperatures(
        target_acc_rate: float,
        params:Tuple[Tensor,Tensor,Tensor]
) -> Tensor:
    vbias, hbias, weight_matrix = params
    inverse_temperatures = torch.linspace(0,1,1000)
    selected_temperatures = [0]
    prev_chains = init_parallel_chains(
        num_chains=100,
        num_hiddens=weight_matrix.shape[1],
        num_visibles=weight_matrix.shape[0],
        device=weight_matrix.device
    )
    new_chains = init_parallel_chains(
        num_chains=100,
        num_hiddens=weight_matrix.shape[1],
        num_visibles=weight_matrix.shape[0],
        device=weight_matrix.device
    )
    for i in range(len(inverse_temperatures)-1):
        prev_chains = sample_state(
            parallel_chains=prev_chains,
            params=params,
            beta=selected_temperatures[-1],
            gibbs_steps=10
        )
        new_chains = sample_state(
            parallel_chains=new_chains,
            params=params,
            beta=inverse_temperatures[i],
            gibbs_steps=10
        )
        
        _, acc_rate = swap_configurations(
            chains=(
                torch.vstack([
                    prev_chains[0].unsqueeze(0),
                    new_chains[0].unsqueeze(0)
                ]),
                torch.vstack([
                    prev_chains[1].unsqueeze(0),
                    new_chains[1].unsqueeze(0)
                ])
            ),
            params=params,
            inverse_temperatures=torch.tensor([
                selected_temperatures[-1],
                inverse_temperatures[i]
            ]),
            show_acc_rate=False,
        )
        if acc_rate[-1] < target_acc_rate+0.1:
            selected_temperatures.append(inverse_temperatures[i])
            prev_chains = (new_chains[0].clone(), new_chains[1].clone())
    return torch.tensor(selected_temperatures)
    
def swap_configurations(
        chains:Tuple[Tensor, Tensor],
        params: Tuple[Tensor, Tensor, Tensor],
        inverse_temperatures: Tensor,
        show_acc_rate: bool
):
    vbias, hbias, weight_matrix = params
    parallel_chains_v, parallel_chains_h = chains
    _, n_chains, L = parallel_chains_v.shape
    device = weight_matrix.device
    acc_rate = []
    for idx in range(inverse_temperatures.shape[0]-1):
        energy_0 = compute_energy(v=parallel_chains_v[idx], h=parallel_chains_h[idx], vbias=vbias, hbias=hbias, weight_matrix=weight_matrix)
        energy_1 = compute_energy(v=parallel_chains_v[idx+1], h=parallel_chains_h[idx+1], vbias=vbias, hbias=hbias, weight_matrix=weight_matrix)
        delta_E = (
            - energy_1*inverse_temperatures[idx]
            + energy_0*inverse_temperatures[idx]
            + energy_1*inverse_temperatures[idx+1]
            - energy_0*inverse_temperatures[idx+1]
        )
        swap_chain = torch.exp(delta_E) > torch.rand(size=(n_chains,), device=device)
        acc_rate.append((swap_chain.sum() / n_chains).cpu().numpy())

        swapped_chains_v_0 = torch.where(
            swap_chain.unsqueeze(1).repeat(1, L), parallel_chains_v[idx + 1], parallel_chains_v[idx]
        )
        swapped_chains_h_0 = torch.where(
            swap_chain.unsqueeze(1).repeat(1, L), parallel_chains_h[idx + 1], parallel_chains_h[idx]
        )
        swapped_chains_v_1 = torch.where(
            swap_chain.unsqueeze(1).repeat(1, L), parallel_chains_v[idx], parallel_chains_v[idx + 1]
        )
        swapped_chains_h_1 = torch.where(
            swap_chain.unsqueeze(1).repeat(1, L), parallel_chains_h[idx], parallel_chains_h[idx + 1]
        )
        parallel_chains_v[idx] = swapped_chains_v_0
        parallel_chains_h[idx] = swapped_chains_h_0
        parallel_chains_v[idx + 1] = swapped_chains_v_1
        parallel_chains_h[idx + 1] = swapped_chains_h_1
    if show_acc_rate:
        print(f"acc_rate: {np.array(acc_rate)[:]}")
    return (parallel_chains_v, parallel_chains_h), np.array(acc_rate)


def PTSampling(it_mcmc: int, increment: int, target_acc_rate: float, num_chains: int, params: Tuple[Tensor,Tensor,Tensor]):
    vbias, hbias, weight_matrix = params
    
    inverse_temperatures = find_inverse_temperatures(target_acc_rate, params)
    parallel_chains_v, parallel_chains_h = init_parallel_chains(
        num_chains*inverse_temperatures.shape[0],
        num_visibles=weight_matrix.shape[0],
        num_hiddens=weight_matrix.shape[1],
        device=weight_matrix.device
    )
    parallel_chains_v = parallel_chains_v.reshape(inverse_temperatures.shape[0], num_chains, weight_matrix.shape[0])
    parallel_chains_h = parallel_chains_v.reshape(inverse_temperatures.shape[0], num_chains, weight_matrix.shape[1])

    # Annealing to initialize the chains
    for i in range(inverse_temperatures.shape[0]):
        for j in range(i,inverse_temperatures.shape[0]):
            parallel_chains_v[j], parallel_chains_h[j] = sample_state(
                parallel_chains=(parallel_chains_v[j], parallel_chains_h[j]),
                params=params,
                gibbs_steps=increment,
                beta=inverse_temperatures[i]
            )
            
    
    counts = 0
    while counts < it_mcmc:
        counts += increment
        # Iterate chains
        for i in range(parallel_chains_v.shape[0]):
            parallel_chains_v[i], parallel_chains_h[i] = sample_state(
                parallel_chains=(parallel_chains_v[i], parallel_chains_h[i]),
                params=params,
                gibbs_steps=increment,
                beta=inverse_temperatures[i]
            )
        # Swap chains
        (parallel_chains_v, parallel_chains_h), acc_rate = swap_configurations(
            chains=(parallel_chains_v, parallel_chains_h),
            params=params,
            inverse_temperatures=inverse_temperatures,
            show_acc_rate=True,
        )
        
    return (parallel_chains_v, parallel_chains_h), inverse_temperatures
    

