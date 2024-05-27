import argparse

from fastrbm.dataset import load_dataset
from fastrbm.dataset.parser import add_args_dataset
from fastrbm.rcm.parser import add_args_rcm
from fastrbm.rcm.training import train


def create_parser():
    parser = argparse.ArgumentParser("Train a Restricted Coulomb Machine")
    parser = add_args_dataset(parser)
    parser = add_args_rcm(parser)
    pytorch_args = parser.add_argument_group("PyTorch")
    pytorch_args.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="(Defaults to cuda). The device to use in PyTorch.",
    )
    pytorch_args.add_argument(
        "--dtype",
        type=str,
        choices=["int", "float", "double"],
        default="float",
        help="(Defaults to double). The dtype to use in PyTorch.",
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    args = vars(args)
    if args["num_points"] is None:
        args["num_points"] = 100 ** (len(args["dimension"]))
    train_dataset, test_dataset = load_dataset(
        dataset_name=args["data"],
        subset_labels=args["subset_labels"],
        variable_type=args["variable_type"],
        path_clu=None,
        train_size=args["train_size"],
        test_size=args["test_size"],
        binary_threshold=args["binary_threshold"],
    )
    print(train_dataset)
    train(
        train_dataset=train_dataset.data,
        test_dataset=test_dataset.data,
        args=args,
        mesh_file=args["mesh_file"],
    )
