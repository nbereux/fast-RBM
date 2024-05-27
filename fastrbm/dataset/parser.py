import argparse


def add_args_dataset(parser: argparse.ArgumentParser):
    dataset_args = parser.add_argument_group("Dataset")
    dataset_args.add_argument(
        "-d",
        "--data",
        type=str,
        required=True,
        help="Name of the dataset ('GENE', 'MNIST', 'MICKEY'), or path to a data file (type should be .d, .dat, .pt or .npy)",
    )
    dataset_args.add_argument(
        "--subset_labels",
        nargs="*",
        default=None,
        type=int,
        help="(Defaults to None). The subset of labels to use during training. None means all dataset.",
    )
    dataset_args.add_argument(
        "--train_size",
        type=float,
        default=0.6,
        help="(Defaults to 0.6). The proportion of the dataset to use as training set.",
    )
    dataset_args.add_argument(
        "--test_size",
        type=float,
        default=None,
        help="(Defaults to None). The proportion of the dataset ot use as testing set.",
    )
    dataset_args.add_argument(
        "--variable_type",
        type=str,
        default="Bernoulli",
        help="(Defaults to 'Bernoulli'). The type of the variables of the dataset.",
        choices=["Bernoulli", "Ising", "Continuous", "Potts"],
    )
    dataset_args.add_argument(
        "--binary_threshold",
        type=float,
        default=0.3,
        help="(Defaults to 0.3). The threshold to binarize the dataset.",
    )
    dataset_args.add_argument(
        "--use_torch",
        default=False,
        action="store_true",
        help="Load the dataset as torch.Tensor",
    )
    return parser
