import textwrap
import gzip
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Dict, Optional, Union


class RBMDataset(Dataset):

    def __init__(
        self,
        data: np.ndarray,
        variable_type: str,
        labels: np.ndarray,
        weights: np.ndarray,
        names: np.ndarray,
        dataset_name: str,
        use_torch: bool = False,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.data = data
        self.variable_type = variable_type
        self.labels = labels
        self.weights = weights[:, None]
        self.names = names
        self.dataset_name = dataset_name
        self.use_torch = use_torch
        self.device = device
        self.dtype = dtype
        if use_torch:
            self.data = torch.from_numpy(self.data).to(self.device).to(self.dtype)
            self.weights = torch.from_numpy(self.weights).to(self.device).to(self.dtype)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        return {
            "data": self.data[index],
            "labels": self.labels[index],
            "weights": self.weights[index],
            "names": self.names[index],
        }

    def __str__(self) -> str:
        return textwrap.dedent(
            f"""
        Dataset: {self.dataset_name}
        Variable type: {self.variable_type}
        Number of samples: {self.data.shape[0]}
        Number of features: {self.data.shape[1]}
        """
        )

    def get_num_visibles(self) -> int:
        return self.data.shape[1]

    def get_num_states(self) -> int:
        return int(self.data.max() + 1)

    def get_effective_size(self) -> int:
        return int(self.weights.sum())

    def get_covariance_matrix(
        self,
        num_data: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Returns the covariance matrix of the data. If path_clu was specified, the weighted covariance matrix is computed.

        Args:
            device (torch.device, optional): Device. Defaults to torch.device("cpu").
            num_data (int, optional): Number of data to extract for computing the covariance matrix.

        Returns:
            Tensor: Covariance matrix of the dataset.
        """
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        num_states = self.get_num_states()
        num_visibles = self.get_num_visibles()
        if self.use_torch:
            data = self.data.to(device).to(dtype)
            weights = self.weights.to(device).to(dtype)
        else:
            data = torch.tensor(self.data, device=device, dtype=dtype)
            weights = torch.tensor(self.weights, device=device, dtype=dtype)
        if num_data is not None:
            idxs = torch.multinomial(
                input=(torch.ones(self.__len__(), device=device) / self.__len__()),
                num_samples=num_data,
                replacement=False,
            )
            data = data[idxs]
            weights = weights[idxs]
        if num_states != 2:
            data_oh = (
                torch.eye(num_states, device=device)[data]
                .reshape(-1, num_states * num_visibles)
                .float()
            )
        else:
            data_oh = data
        norm_weights = weights.reshape(-1, 1) / weights.sum()
        data_mean = (data_oh * norm_weights).sum(0, keepdim=True)
        cov_matrix = ((data_oh * norm_weights).mT @ data_oh) - (
            data_mean.mT @ data_mean
        )
        return cov_matrix

    def get_gzip_entropy(self, mean_size: int = 50, num_samples: int = 100):
        pbar = tqdm(range(mean_size))
        pbar.set_description("Compute entropy gzip")
        en = np.zeros(mean_size)
        for i in pbar:
            en[i] = len(
                gzip.compress(
                    (
                        self.data[torch.randperm(self.data.shape[0])[:num_samples]]
                    ).astype(int)
                )
            )
        return np.mean(en)
