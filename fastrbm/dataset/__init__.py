import numpy as np
import torch
from typing import Tuple

from fastrbm.dataset.dataset_class import RBMDataset
from fastrbm.dataset.definitions import ROOT_DIR_DATASET_PACKAGE
from fastrbm.dataset.load_MNIST import load_MNIST
from fastrbm.dataset.load_GENE import load_dat, load_GENE


def load_dataset(
    dataset_name: str,
    subset_labels=None,
    variable_type: str = "Bernoulli",
    path_clu=None,
    train_size: float = 0.6,
    test_size: float = None,
    seed: int = 19023741073419046239412739401234901,
    binary_threshold=0.3,
    use_torch: bool = False,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> Tuple[RBMDataset, RBMDataset | None]:
    rng = np.random.default_rng(seed)
    data = None
    weights = None
    names = None
    labels = None
    match dataset_name:
        case "MNIST":
            data, labels = load_MNIST(
                variable_type=variable_type,
                subset_labels=subset_labels,
                binary_threshold=binary_threshold,
            )
        case "GENE":
            data = load_GENE(variable_type=variable_type)
        case "MICKEY":
            data = np.load(ROOT_DIR_DATASET_PACKAGE / "data/mickey.npy").T
            if np.unique(data).shape[0] == 2:
                if variable_type == "Bernoulli" and np.allclose(
                    np.unique(data), np.array([-1.0, 1.0])
                ):
                    data = (data + 1) / 2
                elif variable_type == "Ising" and np.allclose(
                    np.unique(data), np.array([0.0, 1.0])
                ):
                    data = data * 2 - 1
        case _:
            if dataset_name[-4:] == ".npy":
                data = np.load(dataset_name).T
                if np.unique(data).shape[0] == 2:
                    if variable_type == "Bernoulli" and np.allclose(
                        np.unique(data), np.array([-1.0, 1.0])
                    ):
                        data = (data + 1) / 2
                    elif variable_type == "Ising" and np.allclose(
                        np.unique(data), np.array([0.0, 1.0])
                    ):
                        data = data * 2 - 1
            elif dataset_name[-4:] == ".dat" or dataset_name[-2:] == ".d":
                data = load_dat(dataset_name, variable_type=variable_type)
            elif dataset_name[-3:] == ".pt":
                data = torch.load(dataset_name)
            else:
                raise ValueError(
                    """
                    Dataset could not be loaded as the type is not recognized.
                    It should be either: 
                     - '.d',
                     - '.dat'
                     - '.npy' 
                     - '.pt'
                    """
                )

    if weights is None:
        weights = np.ones(data.shape[0])
    if names is None:
        names = np.arange(data.shape[0])
    if labels is None:
        labels = -np.ones(data.shape[0])
    permutation_index = rng.permutation(data.shape[0])
    train_size = int(train_size * data.shape[0])
    if test_size is not None:
        test_size = int(test_size * data.shape[0])
    else:
        test_size = data.shape[0] - train_size

    train_dataset = RBMDataset(
        data=data[permutation_index[:train_size]],
        variable_type=variable_type,
        labels=labels[permutation_index[:train_size]],
        weights=weights[permutation_index[:train_size]],
        names=names[permutation_index[:train_size]],
        dataset_name=dataset_name,
        use_torch=use_torch,
        device=device,
        dtype=dtype,
    )
    test_dataset = None
    if test_size > 0:
        test_dataset = RBMDataset(
            data=data[permutation_index[train_size : train_size + test_size]],
            variable_type=variable_type,
            labels=labels[permutation_index[train_size : train_size + test_size]],
            weights=weights[permutation_index[train_size : train_size + test_size]],
            names=names[permutation_index[train_size : train_size + test_size]],
            dataset_name=dataset_name,
            use_torch=use_torch,
            device=device,
            dtype=dtype,
        )
    return train_dataset, test_dataset
