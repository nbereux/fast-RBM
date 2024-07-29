import h5py
import numpy as np
from pathlib import Path
import time
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Tuple, Optional

from fastrbm.dataset.dataset_class import RBMDataset

from fastrbm.methods.methods_binary import (
    init_parallel_chains,
    init_parameters,
    sample_hiddens,
    sample_state,
    compute_gradient,
    compute_eps,
)

Tensor = torch.Tensor


def update_parameters(
    data: Tuple[Tensor, Tensor, Tensor],
    parallel_chains: Tuple[Tensor, Tensor],
    params: Tuple[Tensor, Tensor, Tensor],
    learning_rate: float,
) -> Tuple[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:

    grad = compute_gradient(data, parallel_chains)

    # Update parameters
    vbias_new = params[0] + learning_rate * grad[0]
    hbias_new = params[1] + learning_rate * grad[1]
    weight_matrix_new = params[2] + learning_rate * grad[2]
    params_new = (vbias_new, hbias_new, weight_matrix_new)
    # Computes the new chain weights
    return (params_new, grad)


@torch.jit.script
def fit_batch(
    batch: Tuple[Tensor, Tensor],
    parallel_chains: Tuple[Tensor, Tensor],
    params: Tuple[Tensor, Tensor, Tensor],
    learning_rate: float,
    gibbs_steps: int,
    grad: Tuple[Tensor, Tensor, Tensor],
) -> Tuple[
    Tuple[Tensor, Tensor],
    Tuple[Tensor, Tensor, Tensor],
    Tuple[Tensor, Tensor, Tensor],
    Tuple[Tensor],
]:

    _, hbias, weight_matrix = params
    v_data, w_data = batch
    _, h_data = sample_hiddens(v=v_data, hbias=hbias, weight_matrix=weight_matrix)
    data = (v_data, h_data, w_data)

    # Update the state of the model
    parallel_chains = sample_state(
        parallel_chains=parallel_chains, params=params, gibbs_steps=gibbs_steps
    )
    eps = compute_eps(parallel_chains=parallel_chains, params=params)

    # Update the parameters
    params, grad = update_parameters(
        data=data,
        parallel_chains=parallel_chains,
        params=params,
        learning_rate=learning_rate,
    )

    # Write the logs
    logs = (eps,)
    return parallel_chains, params, grad, logs


def fit(
    dataset: RBMDataset,
    epochs: int,
    checkpoints: np.ndarray,
    num_hiddens: int = 100,
    num_chains: int = 500,
    batch_size: int = 500,
    gibbs_steps: int = 50,
    filename: str = "popRBM.h5",
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

    learning_rate = kwargs["learning_rate"]

    # Load the data
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    num_visibles = dataset.get_num_visibles()

    # Create file for saving the model
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
        hyperparameters["gibbs_steps"] = gibbs_steps
        hyperparameters["filename"] = str(filename)
        hyperparameters["learning_rate"] = learning_rate
        file_model["parallel_chains"] = parallel_chains[0].cpu().numpy()

    # Training the model
    log_file = open(log_filename, "a")
    pbar = tqdm(initial=0, total=epochs, colour="red", dynamic_ncols=True, ascii="-#")
    pbar.set_description("Training RBM")
    start = time.time()
    num_updates = 0
    for epoch in range(epochs + 1):
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
                grad=grad,
            )
            parallel_chains, params, grad, logs = results
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
                checkpoint["torch_rng_state"] = torch.get_rng_state()
                checkpoint["numpy_rng_arg0"] = np.random.get_state()[0]
                checkpoint["numpy_rng_arg1"] = np.random.get_state()[1]
                checkpoint["numpy_rng_arg2"] = np.random.get_state()[2]
                checkpoint["numpy_rng_arg3"] = np.random.get_state()[3]
                checkpoint["numpy_rng_arg4"] = np.random.get_state()[4]
                del file_model["parallel_chains"]
                file_model["parallel_chains"] = parallel_chains[0].cpu().numpy()
                checkpoint["time"] = time.time() - start

        pbar.update(1)
    log_file.close()


def restore_training(
    filename: str, dataset: Dataset, epochs: int, checkpoints: np.ndarray
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
        learning_rate = file_model["hyperparameters"]["learning_rate"][()]
        time_elapsed = file_model[last_file_key]["time"][()]

    # Initialize the chains, import the data
    params = (vbias, hbias, weight_matrix)
    grad = (torch.zeros_like(p) for p in params)
    _, parallel_chains_h = sample_hiddens(parallel_chains_v, hbias, weight_matrix)
    parallel_chains = (parallel_chains_v, parallel_chains_h)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        # pin_memory=True,
        # num_workers=1,
        drop_last=True,
    )

    # Validate checkpoints
    if checkpoints is None:
        checkpoints = [epochs]
    checkpoints = list(checkpoints)

    # Continue the training
    pbar = tqdm(
        initial=num_epochs, total=epochs, colour="red", dynamic_ncols=True, ascii="-#"
    )
    start = time.time()
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
                grad=grad,
            )
            parallel_chains, params, grad, logs = results
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
                checkpoint["torch_rng_state"] = torch.get_rng_state()
                checkpoint["numpy_rng_arg0"] = np.random.get_state()[0]
                checkpoint["numpy_rng_arg1"] = np.random.get_state()[1]
                checkpoint["numpy_rng_arg2"] = np.random.get_state()[2]
                checkpoint["numpy_rng_arg3"] = np.random.get_state()[3]
                checkpoint["numpy_rng_arg4"] = np.random.get_state()[4]
                del file_model["parallel_chains"]
                file_model["parallel_chains"] = parallel_chains[0].cpu().numpy()
                file_model["time"] = time.time() - start + time_elapsed
                file_model.close()
        pbar.update(1)

    with h5py.File(filename, "r+") as file_model:
        del file_model["hyperparameters"]["epochs"]
        file_model["hyperparameters"]["epochs"] = epoch
