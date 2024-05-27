import argparse
import numpy as np
from pathlib import Path


def add_args_rcm(parser: argparse.ArgumentParser):
    rcm_args = parser.add_argument_group("RCM")
    parser.add_argument(
        "-o",
        "--filename",
        type=Path,
        default="RCM.h5",
        help="(Defaults to RCM.h5). Path to the file where to save the model.",
    )
    rcm_args.add_argument(
        "--save_all_trial",
        default=False,
        help="(Defaults to False). Save all trial of RCM. Useful when decimation is True.",
        action="store_true",
    )
    rcm_args.add_argument(
        "--num_hidden",
        type=int,
        default=100,
        help="(Defaults to 100). Number of hidden units.",
    )
    rcm_args.add_argument(
        "--dimension",
        nargs="+",
        help="The dimensions on which to do RCM",
        required=True,
    )
    rcm_args.add_argument(
        "--num_points",
        type=int,
        default=None,
        help="(Defaults to 100*id). Number of points used for discretization of the intrinsic space.",
    )
    rcm_args.add_argument(
        "--num_sample_gen",
        type=int,
        default=2_000,
        help="(Defaults to 2 000). Number of sample to generate post-training",
    )
    rcm_args.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="(Defaults to 0.01). Learning rate.",
    )
    rcm_args.add_argument(
        "--max_iter",
        type=int,
        default=10_000,
        help="(Defaults to 10 000). Maximum number of iteration per epoch.",
    )
    rcm_args.add_argument(
        "--smooth_rate",
        type=float,
        default=0.1,
        help="(Defaults to 0.1). Smoothing rate for the update of the hessian.",
    )
    rcm_args.add_argument(
        "--min_learning_rate",
        type=float,
        default=1e-6,
        help="(Defaults to 1e-6). r_min.",
    )
    rcm_args.add_argument(
        "--adapt",
        default=False,
        help="(Defaults to True). Use an adaptive learning rate strategy.",
        action="store_true",
    )
    rcm_args.add_argument(
        "--stop_ll",
        type=float,
        default=1e-2,
        help="(Defaults to 1e-2). Log-likelihood precision for early stopping.",
    )
    rcm_args.add_argument(
        "--feature_threshold",
        type=int,
        default=500,
        help="(Defaults to 500). Feature threshold for feature decimation.",
    )
    rcm_args.add_argument(
        "--eigen_threshold",
        type=float,
        default=1e-4,
        help="(Defaults to 1e-4). Minimum eigenvalue for the hessian.",
    )
    rcm_args.add_argument(
        "--decimation",
        default=False,
        help="(Defaults to False). Decimate features.",
        action="store_true",
    )
    parser.add_argument(
        "--mesh_file",
        type=str,
        default=None,
        help="(Defaults to None). Path to a precomputed mesh.",
    )
    rcm_args.add_argument(
        "--seed",
        type=int,
        default=8127394031293,
        help="(Defaults to None). Seed for the RCM method. Does not change the seed for dataset splitting.",
    )
    return parser


def create_parser():
    parser = argparse.ArgumentParser(
        description="Train a RCM and convert it to Bernoulli-Bernoulli RBM."
    )
    parser.add_argument(
        "-d",
        "--data",
        type=Path,
        required=True,
        help="Filename of the dataset to be used for training the model. (should be a npy file with array of shape (Nv x Ns))",
    )
    parser.add_argument(
        "-o",
        "--filename",
        type=Path,
        default="RCM.h5",
        help="(Defaults to RCM.h5). Path to the file where to save the model.",
    )
    parser.add_argument(
        "--save_all_trial",
        default=False,
        help="(Defaults to False). Save all trial of RCM. Useful when decimation is True.",
        action="store_true",
    )
    parser.add_argument(
        "--num_hidden",
        type=int,
        default=100,
        help="(Defaults to 100). Number of hidden units.",
    )
    parser.add_argument(
        "--dimension",
        nargs="+",
        help="The dimensions on which to do RCM",
        required=True,
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=None,
        help="(Defaults to 100*id). Number of points used for discretization of the intrinsic space.",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=None,
        help="(Defaults to 60 percent of the dataset). Number of samples in the training set.",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=None,
        help="(Defaults to the samples not in the training set)",
    )
    parser.add_argument(
        "--num_sample_gen",
        type=int,
        default=2_000,
        help="(Defaults to 2 000). Number of sample to generate post-training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="(Defaults to 0.01). Learning rate.",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=10_000,
        help="(Defaults to 10 000). Maximum number of iteration per epoch.",
    )
    parser.add_argument(
        "--smooth_rate",
        type=float,
        default=0.1,
        help="(Defaults to 0.1). Smoothing rate for the update of the hessian.",
    )
    parser.add_argument(
        "--min_learning_rate",
        type=float,
        default=1e-6,
        help="(Defaults to 1e-6). r_min.",
    )
    parser.add_argument(
        "--adapt",
        default=False,
        help="(Defaults to True). Use an adaptive learning rate strategy.",
        action="store_true",
    )
    parser.add_argument(
        "--stop_ll",
        type=float,
        default=1e-2,
        help="(Defaults to 1e-2). Log-likelihood precision for early stopping.",
    )
    parser.add_argument(
        "--feature_threshold",
        type=int,
        default=500,
        help="(Defaults to 500). Feature threshold for feature decimation.",
    )
    parser.add_argument(
        "--eigen_threshold",
        type=float,
        default=1e-4,
        help="(Defaults to 1e-4). Minimum eigenvalue for the hessian.",
    )
    parser.add_argument(
        "--decimation",
        default=False,
        help="(Defaults to False). Decimate features.",
        action="store_true",
    )
    parser.add_argument(
        "--corr_features_threshold",
        type=float,
        default=0.9,
        help="(Defaults to 0.9). Threshold for feature merging.",
    )
    parser.add_argument(
        "--save_rbm",
        default=False,
        help="(Defaults to False). Save RBMs along RCM training.",
    )
    return parser


def check_args(dataset: np.ndarray, args: dict):
    if isinstance(dataset, np.ndarray):
        pass
    else:
        raise TypeError("The dataset should be a numpy array.")

    if len(dataset.shape) != 2:
        raise ValueError(
            f"The dataset should be a 2-dimensional array, here it has {len(dataset.shape)} dimensions."
        )
    # if not (dataset.dtype == np.int32):
    #     print("-----")
    #     print(f"The dataset has dtype {dataset.dtype} and should be np.int32.")
    #     print("Casting the dataset to np.int32")
    #     print("-----")

    #     dataset = dataset.astype(np.int32)

    # unique_values = np.unique(dataset)
    # print(unique_values)
    # if np.allclose(unique_values, np.array([0, 1], dtype=np.int32)):
    #     print("Converting the dataset to {-1, 1}...")
    #     # Convert [0, 1] values to [-1, 1]
    #     dataset = 2 * dataset - 1

    # elif not (np.allclose(unique_values, np.array([-1, 1], dtype=np.int32))):
    #     raise ValueError(
    #         f"The dataset should be composed either of binary variables [0, 1] or binary variables [-1, 1]. Got {unique_values}"
    #     )

    if args.num_hidden <= 0:
        raise ValueError(f"num_hidden should be >=1, got {args.num_hidden}.")

    if (len(args.dimension) <= 0) or (len(args.dimension) > 3):
        raise ValueError(
            f"Number of provided dimensions should be in range [1, 3], got {len(args.dimension)}."
        )
    args.dimension = np.array([int(elt) for elt in args.dimension])
    args.intrinsic_dimension = len(args.dimension)
    # if (args.intrinsic_dimension <= 0) or (args.intrinsic_dimension >= 5):
    #     raise ValueError(
    #         f"n_intrinsic_dimensions should be in range [1, 5], got {args.intrinsic_dimension}."
    #     )

    if args.num_points is None:
        args.num_points = 100 ** (args.intrinsic_dimension)
    if args.num_points <= 0:
        raise ValueError(f"n_points should be >= 1, got {args.num_points}.")

    print(dataset.shape)
    if args.train_size is None:
        args.train_size = int(dataset.shape[1] * 0.6)
    if args.train_size <= 0:
        raise ValueError(f"train_size should be >= 1, got {args.train_size}")
    if args.test_size is None:
        args.test_size = dataset.shape[1] - args.train_size
    if args.test_size <= 0:
        raise ValueError(f"test_size should be >= 1, got {args.test_size}")

    if args.train_size + args.test_size > dataset.shape[1]:
        raise ValueError(
            f"The total number of sample from the dataset ({dataset.shape[1]}) exceeds train_size ({args.train_size}) + test_size ({args.test_size}): {args.train_size+args.test_size}"
        )

    # The dataset should be of shape n_visible x n_sample
    print(f"Dataset:  {dataset.shape[0]} dimensions x {dataset.shape[1]} samples.")
    n_visible = dataset.shape[0]

    # The RCM program needs the dataset as n_sample x n_visible
    dataset = np.copy(dataset, order="F")

    return dataset, args
