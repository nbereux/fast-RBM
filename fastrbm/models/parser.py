import argparse
from pathlib import Path


def add_args_rbm(parser: argparse.ArgumentParser):
    rbm_args = parser.add_argument_group("RBM")
    rbm_args.add_argument(
        "--num_hiddens",
        type=int,
        default=100,
        help="(Defaults to 100). Number of hidden units.",
    )
    rbm_args.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="(Defaults to 0.01). Learning rate.",
    )
    rbm_args.add_argument(
        "--gibbs_steps",
        type=int,
        default=100,
        help="(Defaults to 100). Number of gibbs steps to perform for each gradient update.",
    )
    rbm_args.add_argument(
        "--batch_size",
        type=int,
        default=2000,
        help="(Defaults to 2000). Minibatch size.",
    )
    rbm_args.add_argument(
        "--num_chains",
        type=int,
        default=2000,
        help="(Defaults to 2000). Number of parallel chains.",
    )
    rbm_args.add_argument(
        "--epochs",
        default=1000,
        type=int,
        help="(Defaults to 1000). Number of training epochs.",
    )
    rbm_args.add_argument(
        "--model",
        default="BernoulliBernoulliPCDRBM",
        type=str,
        choices=["BernoulliBernoulliPCDRBM", "BernoulliBernoulliJarJarRBM"],
        help="(Defaults to BernoulliBernoulliPCDRBM). The algorithm used to train the RBM.",
    )
    rbm_args.add_argument(
        "--restore",
        default=False,
        help="(Defaults to False) To restore an old training.",
        action="store_true",
    )
    jarjar_args = parser.add_argument_group("Jar-RBM")
    jarjar_args.add_argument(
        "--min_eps",
        type=float,
        default=0.7,
        help="(Defaults to 0.7). Minimum effective population size allowed.",
    )
    return parser


def add_args_saves(parser: argparse.ArgumentParser):
    save_args = parser.add_argument_group("Save")
    save_args.add_argument(
        "--filename",
        type=Path,
        default="RBM.h5",
        help="(Defaults to RBM.h5). Path to the file where to save the model or load if training is restored.",
    )
    save_args.add_argument(
        "--n_save",
        type=int,
        default=50,
        help="(Defaults to 50). Number of models to save during the training.",
    )
    save_args.add_argument(
        "--spacing",
        type=str,
        default="exp",
        help="(Defaults to exp). Spacing to save models.",
        choices=["exp", "linear"],
    )
    save_args.add_argument(
        "--log", default=False, action="store_true", help="Log metrics during training."
    )
    return parser
