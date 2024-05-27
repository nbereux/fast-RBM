import argparse
import torch

from fastrbm.dataset import load_dataset
from fastrbm.dataset.parser import add_args_dataset
from fastrbm.models.parser import add_args_rbm, add_args_saves
from fastrbm.models.BernoulliBernoulliPCDRBM import fit, restore_training
from fastrbm.utils import get_checkpoints


def create_parser():
    parser = argparse.ArgumentParser(description="Train a Restricted Boltzmann Machine")
    parser = add_args_dataset(parser)
    parser = add_args_rbm(parser)
    parser = add_args_saves(parser)
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


def train_rbm(args):
    train_dataset, test_dataset = load_dataset(
        dataset_name=args["data"],
        subset_labels=args["subset_labels"],
        variable_type=args["variable_type"],
        path_clu=None,
        train_size=args["train_size"],
        test_size=args["test_size"],
        binary_threshold=args["binary_threshold"],
        use_torch=args["use_torch"],
        device=args["device"],
        dtype=args["dtype"],
    )

    print(train_dataset)

    checkpoints = get_checkpoints(args)
    if args["batch_size"] > train_dataset.__len__():
        print(
            f"Warning: batch_size ({args.batch_size}) is bigger than the size of the training set ({train_dataset.__len__()}). Setting batch_size to {train_dataset.__len__()}."
        )
        args["batch_size"] = train_dataset.__len__()

    match args["model"]:
        case "BernoulliBernoulliPCDRBM":
            from fastrbm.models.BernoulliBernoulliPCDRBM import fit, restore_training
        case "BernoulliBernoulliJarJarRBM":
            from fastrbm.models.BernoulliBernoulliJarJarRBM import fit, restore_training
        case _:
            raise ValueError(f"Unrecognized model: {args['model']}")
    if args["restore"]:
        restore_training(
            args["filename"],
            dataset=train_dataset,
            epochs=args["epochs"],
            checkpoints=checkpoints,
        )
    else:
        fit(
            dataset=train_dataset,
            epochs=args["epochs"],
            num_hiddens=args["num_hiddens"],
            num_chains=args["num_chains"],
            batch_size=args["batch_size"],
            gibbs_steps=args["gibbs_steps"],
            filename=args["filename"],
            learning_rate=args["learning_rate"],
            record_log=args["log"],
            checkpoints=checkpoints,
            min_eps=args["min_eps"],  # Only used in JarJar
        )


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    args = vars(args)
    match args["dtype"]:
        case "int":
            args["dtype"] = torch.int64
        case "float":
            args["dtype"] = torch.float32
        case "double":
            args["dtype"] = torch.float64
    train_rbm(args)
