from typing import Tuple
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from fastrbm.dataset.dataset_class import RBMDataset
from fastrbm.methods.methods_binary import (
    init_parallel_chains,
    init_parameters,
    compute_energy,
    sample_hiddens,
    sample_state,
    compute_eps,
    compute_ess,
)

# from methods.methods_pop import systematic_resampling

Tensor = torch.Tensor


def update_free_energy(free_energy: Tensor, logit_weights: Tensor) -> float:
    """Estimates the free energy of the model using pop-MC.

    Args:
        free_energy (Tensor): Free energy baseline.
        logit_weights (Tensor): Unnormalized log probability of the parallel chains.

    Returns:
        float: Estimate of the free energy.
    """
    free_energy -= torch.log((torch.exp(-logit_weights)).mean())
    return free_energy.item()


def update_parameters(
    data: Tuple[Tensor, Tensor, Tensor],
    parallel_chains: Tuple[Tensor, Tensor],
    params: Tuple[Tensor, Tensor, Tensor],
    learning_rate: float,
    logit_weights: Tensor,
    grad: Tuple[Tensor, Tensor, Tensor],
) -> Tuple[Tuple[Tensor, Tensor, Tensor], Tensor, Tuple[Tensor, Tensor, Tensor]]:

    # Compute the gradient of the log-likelihood
    v_data, h_data, w_data = data
    v_gen, h_gen = parallel_chains

    lwc = logit_weights - logit_weights.min()
    w_gen = (torch.exp(-lwc) / torch.sum(torch.exp(-lwc))).unsqueeze(-1)

    # Averages over data and generated samples
    v_data_mean = (v_data * w_data).sum(0) / w_data.sum()
    torch.clamp_(v_data_mean, min=1e-4, max=(1.0 - 1e-4))
    h_data_mean = (h_data * w_data).sum(0) / w_data.sum()
    v_gen_mean = (v_gen * w_gen).sum(0)
    torch.clamp_(v_gen_mean, min=1e-4, max=(1.0 - 1e-4))
    h_gen_mean = (h_gen * w_gen).sum(0)

    # Centered variables
    v_data_centered = v_data - v_data_mean
    h_data_centered = h_data - h_data_mean
    v_gen_centered = v_gen - v_data_mean
    h_gen_centered = h_gen - h_data_mean

    # Gradient
    grad_weight_matrix = (
        (v_data_centered * w_data).T @ h_data_centered
    ) / w_data.sum() - ((v_gen_centered * w_gen).T @ h_gen_centered)
    grad_vbias = v_data_mean - v_gen_mean - (grad_weight_matrix @ h_data_mean)
    grad_hbias = h_data_mean - h_gen_mean - (v_data_mean @ grad_weight_matrix)
    grad = (grad_vbias, grad_hbias, grad_weight_matrix)

    # Weights of the chains # BEA: I changed to use the new gradient for the logit
    dw = learning_rate * compute_energy(*parallel_chains, *grad)
    logit_weights += dw

    # Update parameters
    vbias_new = params[0] + learning_rate * grad[0]
    hbias_new = params[1] + learning_rate * grad[1]
    weight_matrix_new = params[2] + learning_rate * grad[2]
    params_new = (vbias_new, hbias_new, weight_matrix_new)
    # Computes the new chain weights
    return (params_new, logit_weights, grad)


def multinomial_resampling(
    parallel_chains: Tuple[Tensor, Tensor], logit_weights: Tensor, device: torch.device
) -> Tuple[Tensor, Tensor]:
    """Performs the systematic resampling of the parallel chains according to their relative weight.

    Args:
        parallel_chains (Tuple[Tensor, Tensor]): (v, h) parallel chains.
        logit_weights (Tensor): Unnormalized log probability of the parallel chains.
        device (torch.device): Device.

    Returns:
        Tuple[Tensor, Tensor]: Resampled parallel chains.
    """
    v, h = parallel_chains
    num_chains = v.shape[0]
    logit_weights_shifted = logit_weights - logit_weights.min()
    weights = torch.exp(-logit_weights_shifted) / torch.sum(
        torch.exp(-logit_weights_shifted)
    )
    bootstrap_idxs = weights.multinomial(num_samples=num_chains, replacement=True)
    v_resampled = v[bootstrap_idxs]
    h_resampled = h[bootstrap_idxs]
    return (v_resampled, h_resampled)


@torch.jit.script
def systematic_resampling(
    parallel_chains: Tuple[Tensor, Tensor], logit_weights: Tensor, device: torch.device
) -> Tuple[Tensor, Tensor]:
    """Performs the systematic resampling of the parallel chains according to their relative weight.

    Args:
        parallel_chains (Tuple[Tensor, Tensor]): (v, h) parallel chains.
        logit_weights (Tensor): Unnormalized log probability of the parallel chains.
        index_list (Tensor): List with the indices of the chains. Used for computing the number of families in the chains population.
        device (torch.device): Device.

    Returns:
        Tuple[Tensor, Tensor]: Resampled parallel chains.
    """
    v, h = parallel_chains
    num_chains = v.shape[0]
    logit_weights_shifted = logit_weights - logit_weights.min()
    weights = torch.exp(-logit_weights_shifted) / torch.sum(
        torch.exp(-logit_weights_shifted)
    )
    weights_span = torch.cumsum(weights, dim=0)
    weights_span = torch.cumsum(weights.double(), dim=0).float()
    rand_unif = torch.rand(size=(1,), device=device)
    arrow_span = (torch.arange(num_chains, device=device) + rand_unif) / num_chains
    mask = (weights_span.reshape(num_chains, 1) >= arrow_span).sum(1)
    counts = torch.diff(mask, prepend=torch.tensor([0], device=device))
    v_resampled = torch.repeat_interleave(v, counts, dim=0)
    h_resampled = torch.repeat_interleave(h, counts, dim=0)
    return (v_resampled, h_resampled)


@torch.jit.script
def fit_batch(
    batch: Tuple[Tensor, Tensor],
    parallel_chains: Tuple[Tensor, Tensor],
    params: Tuple[Tensor, Tensor, Tensor],
    learning_rate: float,
    gibbs_steps: int,
    logit_weights: Tensor,
    grad: Tuple[Tensor, Tensor, Tensor],
    min_eps: float,
    device: torch.device,
    allow_resample: bool = True,
) -> Tuple[
    Tensor,
    Tuple[Tensor, Tensor],
    Tuple[Tensor, Tensor, Tensor],
    Tuple[Tensor, Tensor, Tensor],
    Tuple[Tensor, Tensor],
]:

    _, hbias, weight_matrix = params
    v_data, w_data = batch
    _, h_data = sample_hiddens(v=v_data, hbias=hbias, weight_matrix=weight_matrix)
    data = (v_data, h_data, w_data)

    # BEA: I have changed the order of the sampling
    # Update the state of the model
    parallel_chains = sample_state(
        parallel_chains=parallel_chains, params=params, gibbs_steps=gibbs_steps
    )
    eps = compute_eps(parallel_chains=parallel_chains, params=params)

    # Update the parameters
    params, logit_weights, grad = update_parameters(
        data=data,
        parallel_chains=parallel_chains,
        params=params,
        logit_weights=logit_weights,
        learning_rate=learning_rate,
        grad=grad,
    )

    # Resample the chains if the eps is smaller than the threshold
    ess = compute_ess(logit_weights=logit_weights)
    # print(ess)
    if ess < min_eps and allow_resample:
        # print("resample")
        # parallel_chains = multinomial_resampling(
        #     parallel_chains=parallel_chains, logit_weights=logit_weights, device=device
        # )
        # prev_parallel_chains = parallel_chains[0].clone()
        parallel_chains = systematic_resampling(
            parallel_chains=parallel_chains, logit_weights=logit_weights, device=device
        )  # BEA: I have changed the resampling method
        # print(f"norm_diff: {(prev_parallel_chains - parallel_chains[0]).norm()}")
        logit_weights = torch.zeros(size=(len(parallel_chains[0]),), device=device)
        ess_ps = compute_ess(logit_weights=logit_weights)
        # print(f"ess post sampling: {ess_ps}")
    # Write the logs
    logs = (ess, eps)
    return logit_weights, parallel_chains, params, grad, logs


def fit(
    dataset: RBMDataset,
    epochs: int,
    checkpoints: np.ndarray,
    num_hiddens: int = 100,
    num_chains: int = 500,
    batch_size: int = 500,
    gibbs_steps: int = 50,
    filename: str = "JarJar.h5",
    record_log: bool = True,
    *args,
    **kwargs,
) -> None:
    """Fits an RBM model on the training data and saves the results in a file.

    Args:
        dataset (Dataset): Training data.
        epochs (int): Number of epochs to be performed.
        num_hiddens (int, optional): Number of hidden units. Defaults to 100.
        num_chains (int, optional): Number of parallel chains. Defaults to 500.
        batch_size (int, optional): Batch size. Defaults to 500.
        max_gibbs_steps (int, optional): Maximum number of Monte Carlo steps to update the state of the model. Defaults to 50.
        num_fit_lr (int, optional): Number of points to be considered when computing the moving average of the learning rates. Defaults to 5.
        filename (str, optional): Path of the file where to store the trained model. Defaults to "RBM.h5".
        record_log (bool, optional): Whether to record the information of the training in a .csv file or not. Defaults to True.
        checkpoints (list, optional): List of epochs at which storing the model. Defaults to None.
    """

    # Setup the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logit_weights = torch.zeros(size=(num_chains,), device=device)
    learning_rate = kwargs["learning_rate"]

    # Load the data
    # dataset.data = torch.from_numpy(dataset.data).to(device).float()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        # pin_memory=False,
        # num_workers=1,
        drop_last=True,
    )
    # Initialize the learning rate history
    num_visibles = dataset.get_num_visibles()

    # Validate checkpoints
    # if checkpoints is None:
    #     checkpoints = [epochs]
    # checkpoints = list(checkpoints)  # Initialize the deviation parameter

    min_eps = kwargs["min_eps"]

    # Create file for saving the model
    # timestamp = str(".".join(list(str(time.localtime()[i]) for i in range(5))))
    filename = Path(filename)
    filename = filename.parent / Path(f"{filename.name}")
    # Create log file for recording the important observables
    if record_log:
        log_filename = filename.parent / Path(f"log-{filename.stem}.csv")
        log_file = open(log_filename, "w")
        log_file.write("ess,eps\n")
        log_file.close()
    params = init_parameters(
        num_visibles=num_visibles,
        num_hiddens=num_hiddens,
        dataset=dataset.data,
        device=device,
    )
    params = [p.float() for p in params]
    grad = (torch.zeros_like(p) for p in params)
    parallel_chains = init_parallel_chains(
        num_chains=num_chains,
        num_visibles=num_visibles,
        num_hiddens=num_hiddens,
        device=device,
    )
    # Save the hyperparameters of the model
    with h5py.File(filename, "w") as file_model:
        hyperparameters = file_model.create_group("hyperparameters")
        hyperparameters["epochs"] = epochs
        hyperparameters["num_hiddens"] = num_hiddens
        hyperparameters["num_visibles"] = num_visibles
        hyperparameters["num_chains"] = num_chains
        hyperparameters["batch_size"] = batch_size
        hyperparameters["min_eps"] = min_eps
        hyperparameters["gibbs_steps"] = gibbs_steps
        hyperparameters["filename"] = str(filename)
        hyperparameters["learning_rate"] = learning_rate
        file_model["parallel_chains"] = parallel_chains[0].cpu().numpy()

    # Training the model
    log_file = open(log_filename, "a")
    pbar = tqdm(initial=0, total=epochs, colour="red", dynamic_ncols=True, ascii="-#")
    pbar.set_description("Training RBM")
    num_updates = 0
    allow_resample = False
    for epoch in range(epochs + 1):
        if epoch > 300:
            allow_resample = True
        for batch in dataloader:
            batch = (
                batch["data"],  # .to(device, non_blocking=True).float(),
                batch[
                    "weights"
                ],  # .to(device, non_blocking=True).unsqueeze(1).float(),
            )
            results = fit_batch(
                batch=batch,
                parallel_chains=parallel_chains,
                params=params,
                gibbs_steps=gibbs_steps,
                learning_rate=learning_rate,
                min_eps=min_eps,
                logit_weights=logit_weights,
                grad=grad,
                device=device,
                allow_resample=allow_resample,
            )
            logit_weights, parallel_chains, params, grad, logs = results
            if record_log:
                log_file.write(
                    ",".join(
                        [
                            (
                                str(l.detach().cpu().numpy())
                                if type(l) == Tensor
                                else str(l)
                            )
                            for l in logs
                        ]
                    )
                    + "\n"
                )
            num_updates += 1

        # Save the model if a checkpoint is reached
        if epoch in checkpoints:
            vbias, hbias, weight_matrix = params
            with h5py.File(filename, "r+") as file_model:
                checkpoint = file_model.create_group(f"epoch_{epoch}")
                checkpoint["gradient_updates"] = num_updates
                checkpoint["vbias"] = vbias.cpu().numpy()
                checkpoint["hbias"] = hbias.cpu().numpy()
                checkpoint["weight_matrix"] = weight_matrix.cpu().numpy()
                checkpoint["logit_weights"] = logit_weights.cpu().numpy()
                checkpoint["torch_rng_state"] = torch.get_rng_state()
                checkpoint["numpy_rng_arg0"] = np.random.get_state()[0]
                checkpoint["numpy_rng_arg1"] = np.random.get_state()[1]
                checkpoint["numpy_rng_arg2"] = np.random.get_state()[2]
                checkpoint["numpy_rng_arg3"] = np.random.get_state()[3]
                checkpoint["numpy_rng_arg4"] = np.random.get_state()[4]
                del file_model["parallel_chains"]
                file_model["parallel_chains"] = parallel_chains[0].cpu().numpy()

        pbar.update(1)
    log_file.close()


def restore_training(
    filename: str, dataset: RBMDataset, epochs: int, checkpoints: np.ndarray
):
    # Setup the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Retrieve the the number of training epochs already performed on the model
    with h5py.File(filename, "r+") as file_model:
        num_epochs = 0
        for file_key in file_model.keys():
            if "epoch" in file_key:
                epoch = int(file_key.split("_")[1])
                if num_epochs < epoch:
                    num_epochs = epoch
    if epochs <= num_epochs:
        raise RuntimeError(
            f"The parameter /'epochs/' ({epochs}) must be greater than the previous number of epochs ({num_epochs})."
        )
    last_file_key = f"epoch_{num_epochs}"

    # Open the log file if it exists
    log_filename = filename.parent / Path(f"log-{filename.stem}.csv")
    record_log = log_filename.exists()
    print(record_log)
    if record_log:
        log_file = open(log_filename, "a")

    # Load the last checkpoint
    with h5py.File(filename, "r+") as file_model:
        torch.set_rng_state(
            torch.tensor(np.array(file_model[last_file_key]["torch_rng_state"]))
        )
        np_rng_state = tuple(
            [
                file_model[last_file_key]["numpy_rng_arg0"][()].decode("utf-8"),
                file_model[last_file_key]["numpy_rng_arg1"][()],
                file_model[last_file_key]["numpy_rng_arg2"][()],
                file_model[last_file_key]["numpy_rng_arg3"][()],
                file_model[last_file_key]["numpy_rng_arg4"][()],
            ]
        )
        np.random.set_state(np_rng_state)
        weight_matrix = torch.tensor(
            file_model[last_file_key]["weight_matrix"][()],
            device=device,
            dtype=torch.float32,
        )
        vbias = torch.tensor(
            file_model[last_file_key]["vbias"][()], device=device, dtype=torch.float32
        )
        hbias = torch.tensor(
            file_model[last_file_key]["hbias"][()], device=device, dtype=torch.float32
        )
        num_updates = int(file_model[last_file_key]["gradient_updates"][()])
        parallel_chains_v = torch.tensor(
            file_model["parallel_chains"][()], device=device, dtype=torch.float32
        )
        batch_size = int(file_model["hyperparameters"]["batch_size"][()])
        gibbs_steps = int(file_model["hyperparameters"]["gibbs_steps"][()])
        min_eps = float(file_model["hyperparameters"]["min_eps"][()])
        learning_rate = file_model["hyperparameters"]["learning_rate"][()]

    # Initialize the chains, import the data
    params = (vbias, hbias, weight_matrix)
    grad = (torch.zeros_like(p) for p in params)
    _, parallel_chains_h = sample_hiddens(parallel_chains_v, hbias, weight_matrix)
    parallel_chains = (parallel_chains_v, parallel_chains_h)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    logit_weights = torch.zeros(size=(len(parallel_chains_v),), device=device)

    # Validate checkpoints
    # if checkpoints is None:
    #     checkpoints = [epochs]
    # checkpoints = list(checkpoints)

    # Continue the training
    pbar = tqdm(
        initial=num_epochs, total=epochs, colour="red", dynamic_ncols=True, ascii="-#"
    )
    pbar.set_description("Training RBM")
    for epoch in range(num_epochs + 1, epochs + 1):
        for batch in dataloader:
            batch = (
                batch["data"],  # .to(device),
                batch["weights"],  # .to(device).unsqueeze(1).float(),
            )
            results = fit_batch(
                batch=batch,
                parallel_chains=parallel_chains,
                params=params,
                gibbs_steps=gibbs_steps,
                learning_rate=learning_rate,
                min_eps=min_eps,
                logit_weights=logit_weights,
                grad=grad,
                device=device,
            )
            logit_weights, parallel_chains, params, grad, logs = results
            if record_log:
                log_file.write(
                    ",".join(
                        [
                            (
                                str(l.detach().cpu().numpy())
                                if type(l) == Tensor
                                else str(l)
                            )
                            for l in logs
                        ]
                    )
                    + "\n"
                )
            num_updates += 1
        # Save the model if a checkpoint is reached
        if epoch in checkpoints:
            vbias, hbias, weight_matrix = params
            with h5py.File(filename, "r+") as file_model:
                checkpoint = file_model.create_group(f"epoch_{epoch}")
                checkpoint["gradient_updates"] = num_updates
                checkpoint["vbias"] = vbias.cpu().numpy()
                checkpoint["hbias"] = hbias.cpu().numpy()
                checkpoint["weight_matrix"] = weight_matrix.cpu().numpy()
                checkpoint["logit_weights"] = logit_weights.cpu().numpy()
                checkpoint["torch_rng_state"] = torch.get_rng_state()
                checkpoint["numpy_rng_arg0"] = np.random.get_state()[0]
                checkpoint["numpy_rng_arg1"] = np.random.get_state()[1]
                checkpoint["numpy_rng_arg2"] = np.random.get_state()[2]
                checkpoint["numpy_rng_arg3"] = np.random.get_state()[3]
                checkpoint["numpy_rng_arg4"] = np.random.get_state()[4]
                del file_model["parallel_chains"]
                file_model["parallel_chains"] = parallel_chains[0].cpu().numpy()
        pbar.update(1)

    with h5py.File(filename, "r+") as file_model:
        del file_model["hyperparameters"]["epochs"]
        file_model["hyperparameters"]["epochs"] = epoch
