import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from fastrbm.rcm.rbm import get_ll_rbm


def plot_LL_time(fname, num_trials, train=True):
    last_train_ll = []
    last_test_ll = []
    num_features = []

    str_train = "train" if train else "test"

    for i in range(num_trials):
        with h5py.File(fname, "r") as f:
            last_train_ll.append(np.array(f[f"trial_{i}"][f"{str_train}_ll"]))
            last_test_ll.append(np.array(f[f"trial_{i}"][f"{str_train}_ll"]))
            num_features.append(np.array(f[f"trial_{i}"]["q"]).shape[0])
    colors = plt.cm.coolwarm(np.linspace(0, 1, num_trials))
    fig, ax = plt.subplots(1, 1)
    for i in range(num_trials):
        ax.plot(
            np.arange(len(last_train_ll[i])) * 1000,
            last_train_ll[i],
            label=num_features[i],
            c=colors[i],
        )
    ax.legend()
    ax.set_xlabel("Training iterations")
    ax.set_ylabel(f"{str_train} Log-Likelihood")


def plot_LL_rbm(fname, num_trials, proj_data, U):
    all_ll = []
    num_features = []
    num_visibles = U.shape[0]
    for trial in range(num_trials):
        with h5py.File(fname, "r") as f:
            m = torch.from_numpy(np.array(f["const"]["m"]))
            configurational_entropy = torch.from_numpy(
                np.array(f["const"]["configurational_entropy"])
            )
            W_rbm = torch.from_numpy(np.array(f[f"trial_{trial}"]["W_rbm"]))
            hbias_rbm = torch.from_numpy(np.array(f[f"trial_{trial}"]["hbias_rbm"]))
            vbias_rbm = torch.from_numpy(np.array(f[f"trial_{trial}"]["vbias_rbm"]))
            num_features.append(np.array(f[f"trial_{trial}"]["q"]).shape[0])

        all_ll.append(
            get_ll_rbm(
                configurational_entropy,
                proj_data,
                m,
                W_rbm,
                hbias_rbm,
                vbias_rbm,
                torch.from_numpy(U).T,
                num_visibles,
                False,
            )
        )
    fig, ax = plt.subplots(1, 1)
    ax.plot(num_features, all_ll)
    ax.set_xlabel("Number of features")
    ax.set_ylabel("LL")
    fig.suptitle("LL of the recovered RBM")
    return np.argmax(np.array(all_ll))


def plot_p_rcm(fname, data, dir1=0, dir2=None, trial=0):
    with h5py.File(fname, "r") as f:
        m = np.array(f["const"]["m"])
        pdm = np.array(f[f"trial_{trial}"]["pdm"])
    if dir2 is None:
        if len(m.shape) == 2:
            curr_m = m[:, dir1]
        elif len(m.shape) == 1:
            curr_m = m
        else:
            print(f"m should be a 1d or 2d array, got {len(m.shape)}d instead")
            return
        x_plot, counts = np.unique(curr_m, return_counts=True)
        y_plot = np.zeros_like(x_plot)
        for i, idx in enumerate(x_plot):
            y_plot[i] += pdm[curr_m == idx].sum()
        fig, ax = plt.subplots(1, 1)
        ax.plot(x_plot, y_plot, label="P(m)")
        ax.set_xlabel(f"m")
        ax.set_ylabel(f"p(m)")
        fig.suptitle(f"Density learned by the model along PC{dir1}")
    else:
        if len(m.shape) != 2:
            print(f"m should be a 2d array, got {len(m.shape)}d instead")
        x_unique = np.unique(m[:, dir1])
        y_unique = np.unique(m[:, dir2])
        x_plot, y_plot, z_plot = [], [], []
        for i, idx in enumerate(x_unique):
            for j, idy in enumerate(y_unique):
                mask1 = m[:, dir1] == idx
                mask2 = m[mask1][:, dir2] == idy
                if mask2.sum() > 0:
                    x_plot.append(idx)
                    y_plot.append(idy)
                    z_plot.append(pdm[mask1][mask2].sum())
        fig, ax = plt.subplots(1, 1)

        ax.scatter(data[:, dir1], data[:, dir2], s=1)
        ax.tricontour(x_plot, y_plot, z_plot)
        ax.set_xlabel(f"PC{dir1}")
        ax.set_ylabel(f"PC{dir2}")
