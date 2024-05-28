import argparse
import h5py
import torch
from tqdm import tqdm

from fastrbm.dataset import load_dataset
from fastrbm.dataset.parser import add_args_dataset

from fastrbm.rcm.lagrange import get_lagrange_multipliers
from fastrbm.rcm.pca import compute_U


def create_parser():
    parser = argparse.ArgumentParser(
        "Discretize the space along a projection of the dataset."
    )
    parser = add_args_dataset(parser)
    parser.add_argument(
        "--dimension",
        nargs="+",
        help="The dimensions on which to do RCM",
        required=True,
    )
    parser.add_argument("--border_length", type=float, default=0.04)
    parser.add_argument("--n_pts_dim", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("-o", "--filename", type=str, default="mesh.h5")
    return parser


def batched_lagrange_mult(
    m,
    U,
    err_threshold=1e-10,
    max_iter=10_000,
    batch_size=10_000,
):
    # To avoid using too much memory
    if m.shape[0] > batch_size:
        n_batch = m.shape[0] // batch_size
        all_m = []
        all_mu = []
        all_configurational_entropy = []
        for i in range(n_batch):
            curr_m, curr_mu, curr_configurational_entropy = get_lagrange_multipliers(
                m[batch_size * i : batch_size * (i + 1)],
                U,
                err_threshold=err_threshold,
                max_iter=max_iter,
            )
            if len(curr_m > 0):
                all_m.append(curr_m)
                all_mu.append(curr_mu)
                all_configurational_entropy.append(curr_configurational_entropy)
        curr_m = torch.vstack(all_m)
        curr_mu = torch.vstack(all_mu)
        curr_configurational_entropy = torch.hstack(all_configurational_entropy)
    else:
        curr_m, curr_mu, curr_configurational_entropy = get_lagrange_multipliers(
            m,
            U,
            err_threshold=err_threshold,
            max_iter=max_iter,
        )
    return curr_m, curr_mu, curr_configurational_entropy


def main(args: dict):
    dtype = torch.float64
    train_dataset, _ = load_dataset(
        dataset_name=args["data"],
        variable_type=args["variable_type"],
        subset_labels=args["subset_labels"],
        train_size=args["train_size"],
        test_size=args["test_size"],
        binary_threshold=args["binary_threshold"],
        use_torch=True,
        device=args["device"],
        dtype=dtype,
    )
    intrinsic_dimension = len(args["dimension"])
    train_set = train_dataset.data
    print(train_dataset)
    U = compute_U(
        M=train_set,
        intrinsic_dimension=intrinsic_dimension,
        device=args["device"],
        dtype=dtype,
    ).T

    # We do a first iteration on the unit ball to get the limits of the domain
    n_pts_dim = 50
    mesh = torch.meshgrid(
        *[
            torch.linspace(
                -1,
                1,
                n_pts_dim,
                device=args["device"],
                dtype=dtype,
            )
            for i in range(U.shape[0])
        ],
        indexing="ij",
    )
    m = torch.vstack([elt.flatten() for elt in mesh]).T
    m, _, _ = batched_lagrange_mult(m=m, U=U)
    torch.cuda.empty_cache()

    # We do it again with a better discretization
    n_pts_dim = args["n_pts_dim"]
    dim_min = m.min(0).values
    dim_max = m.max(0).values
    mesh = torch.meshgrid(
        *[
            torch.linspace(
                dim_min[i] - args["border_length"],
                dim_max[i] + args["border_length"],
                n_pts_dim,
                device=args["device"],
                dtype=dtype,
            )
            for i in range(U.shape[0])
        ],
        indexing="ij",
    )
    m = torch.vstack([elt.flatten() for elt in mesh]).T
    m, mu, configurational_entropy = batched_lagrange_mult(m=m, U=U)
    torch.cuda.empty_cache()

    num_visibles = U.shape[1]
    device = m.device
    counting = torch.zeros(len(m), device=device)

    for id, m0 in tqdm(enumerate(m), total=m.shape[0]):
        num_samples = 1000
        iidx = (id * torch.ones(num_samples)).int()
        mu_full = (mu[iidx] @ U) * num_visibles**0.5
        x = torch.rand((num_samples, num_visibles), device=device, dtype=dtype)
        p = 1 / (1 + torch.exp(-2 * mu_full))
        s_gen = 2 * (x < p) - 1
        m_gen = s_gen.float() @ U.T.float() / num_visibles**0.5
        indexes = torch.norm(m_gen - m0, dim=1) < 0.01
        counting[id] = torch.sum(indexes) / num_samples
    # Save the results
    with h5py.File(args["filename"], "w") as f:
        f["m"] = m.cpu().numpy()
        f["mu"] = mu.cpu().numpy()
        f["configurational_entropy"] = (
            configurational_entropy.cpu().numpy()
            + torch.log(counting).cpu().numpy() / num_visibles
        )
        f["U"] = U.cpu().numpy()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    args = vars(args)
    main(args)
