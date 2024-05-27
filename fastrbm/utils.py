import h5py
import numpy as np
import scipy
import torch


# Select the list of training times (ages) at which saving the model.
def get_checkpoints(args) -> np.ndarray:
    if args["spacing"] == "exp":
        checkpoints = []
        xi = args["epochs"]
        for _ in range(args["n_save"]):
            checkpoints.append(xi)
            xi = xi / args["epochs"] ** (1 / args["n_save"])
        checkpoints = np.unique(np.array(checkpoints, dtype=np.int32))
    elif args["spacing"] == "linear":
        checkpoints = np.linspace(1, args["epochs"], args["n_save"]).astype(np.int32)
    checkpoints = np.unique(np.append(checkpoints, args["epochs"]))
    return checkpoints


def get_all_epochs(filename):
    with h5py.File(filename, "r") as f:
        epochs = []
        for key in f.keys():
            if "epoch" in key:
                ep = int(key.replace("epoch_", ""))
                epochs.append(ep)

        epochs = np.sort(epochs)
    return epochs


def load_model(filename, age, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with h5py.File(filename, "r") as f:
        vbias = torch.tensor(
            f[f"epoch_{age}"]["vbias"][()], device=device, dtype=torch.float32
        )
        hbias = torch.tensor(
            f[f"epoch_{age}"]["hbias"][()], device=device, dtype=torch.float32
        )
        weight_matrix = torch.tensor(
            f[f"epoch_{age}"]["weight_matrix"][()],
            device=device,
            dtype=torch.float32,
        )
    return vbias, hbias, weight_matrix


def get_eigenvalues_history(filename):
    with h5py.File(filename, "r") as f:
        gradient_updates = []
        eigenvalues = []
        for key in f.keys():
            if "epoch" in key:
                weight_matrix = f[key]["weight_matrix"][()]
                weight_matrix = weight_matrix.reshape(-1, weight_matrix.shape[-1])
                eig = scipy.linalg.svd(weight_matrix, compute_uv=False)
                eigenvalues.append(eig.reshape(*eig.shape, 1))
                gradient_updates.append(f[key]["gradient_updates"][()])

    # Sort the results
    sorting = np.argsort(gradient_updates)
    gradient_updates = np.array(gradient_updates)[sorting]
    eigenvalues = np.array(np.hstack(eigenvalues).T)[sorting]

    return gradient_updates, eigenvalues
