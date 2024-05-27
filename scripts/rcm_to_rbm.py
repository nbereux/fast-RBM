import argparse
import h5py
import numpy as np
from pathlib import Path
import time
import torch

from fastrbm.rcm.rbm import sample_rbm
from fastrbm.dataset.parser import add_args_dataset


def add_args_convert(parser: argparse.ArgumentParser):
    convert_args = parser.add_argument_group("Convert")
    convert_args.add_argument(
        "--path",
        "-i",
        type=Path,
        required=True,
        help="Path to the folder h5 archive of the RCM.",
    )
    convert_args.add_argument(
        "--output",
        "-o",
        type=Path,
        default="RBM.h5",
        help="(Defaults to RBM.h5). Path to the file where to save the model in RBM format.",
    )
    convert_args.add_argument(
        "--num_hiddens",
        type=int,
        default=50,
        help="(Defaults to 50). Target number of hidden nodes for the RBM.",
    )
    convert_args.add_argument(
        "--therm_steps",
        type=int,
        default=10000,
        help="(Defaults to 1e4). Number of steps to be performed to thermalize the chains.",
    )
    convert_args.add_argument(
        "--trial",
        type=int,
        default=None,
        help="(Defaults to the best trial). RCM trial to use",
    )

    rbm_args = parser.add_argument_group("RBM")
    rbm_args.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="(Defaults to 0.01). Learning rate.",
    )
    rbm_args.add_argument(
        "--gibbs_steps",
        type=int,
        default=20,
        help="(Defaults to 10). Number of Gibbs steps for each gradient estimation.",
    )
    rbm_args.add_argument(
        "--batch_size",
        type=int,
        default=2000,
        help="(Defaults to 1000). Minibatch size.",
    )
    rbm_args.add_argument(
        "--seed",
        type=int,
        default=945723295,
        help="(Defaults to 9457232957489). Seed for the experiments.",
    )
    rbm_args.add_argument(
        "--num_chains",
        default=2000,
        type=int,
        help="(Defaults to 2000). The number of permanent chains.",
    )
    jarjar_args = parser.add_argument_group("Jar-RBM")
    jarjar_args.add_argument(
        "--min_eps",
        type=float,
        default=0.7,
        help="(Defaults to 0.7). Minimum effective population size allowed.",
    )
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


def create_parser():
    parser = argparse.ArgumentParser(
        description="Convert RCM into an RBM readable format."
    )
    parser = add_args_dataset(parser)
    parser = add_args_convert(parser)
    return parser


@torch.jit.script
class BBParams:
    def __init__(
        self, weight_matrix: torch.Tensor, vbias: torch.Tensor, hbias: torch.Tensor
    ):
        self.weight_matrix = weight_matrix
        self.vbias = vbias
        self.hbias = hbias


@torch.jit.script
class Chain:
    def __init__(
        self,
        visible: torch.Tensor,
        hidden: torch.Tensor,
        mean_visible: torch.Tensor,
        mean_hidden: torch.Tensor,
    ) -> None:
        self.visible = visible
        self.hidden = hidden
        self.mean_visible = mean_visible
        self.mean_hidden = mean_hidden


def ising_to_bernoulli(params: BBParams) -> BBParams:
    params.vbias = 2.0 * (params.vbias - params.weight_matrix.sum(1))
    params.hbias = 2.0 * (-params.hbias - params.weight_matrix.sum(0))
    params.weight_matrix = 4.0 * params.weight_matrix
    return params


def convert(args: dict, device: torch.device, dtype: torch.dtype):
    # Set the random seed
    if args["seed"] is not None:
        np.random.seed(args["seed"])
        torch.manual_seed(args["seed"])

    if args["trial"] is None:
        trial_name = "best_trial"
    else:
        trial_name = f"trial_{args['trial']}"
    # Import parameters
    print(f"Trial selected: {trial_name}")
    with h5py.File(args["path"], "r") as f:
        vbias_rcm = (
            torch.from_numpy(np.array(f[trial_name]["vbias_rbm"])).to(device).to(dtype)
        )
        hbias_rcm = (
            torch.from_numpy(np.array(f[trial_name]["hbias_rbm"])).to(device).to(dtype)
        )
        weight_matrix_rcm = (
            torch.from_numpy(np.array(f[trial_name]["W_rbm"])).to(device).to(dtype)
        )
        parallel_chains_v = (
            torch.from_numpy(np.array(f[trial_name]["samples_gen"]))
            .to(device)
            .to(dtype)
        )
        p_m = torch.from_numpy(np.array(f[trial_name]["pdm"])).to(device).to(dtype)
        m = torch.from_numpy(np.array(f["const"]["m"])).to(device).to(dtype)
        mu = torch.from_numpy(np.array(f["const"]["mu"])).to(device).to(dtype)
        U = torch.from_numpy(np.array(f["const"]["U"])).to(device).to(dtype)
        if "time" in f.keys():
            total_time = np.array(f["time"]).item()
        else:
            total_time = 0
    start = time.time()
    params = BBParams(weight_matrix=weight_matrix_rcm, vbias=vbias_rcm, hbias=hbias_rcm)
    params = ising_to_bernoulli(params=params)
    num_visibles, num_hiddens_rcm = params.weight_matrix.shape

    num_hiddens_add = args["num_hiddens"] - num_hiddens_rcm
    if num_hiddens_add < 0:
        print("The target number of hidden nodes is lower than the RCMs one.")
        num_hiddens_add = 0
    print(f"Adding {num_hiddens_add} hidden nodes.")

    hbias_add = torch.zeros(size=(num_hiddens_add,), device=device)
    weight_matrix_add = (
        torch.randn(size=(num_visibles, num_hiddens_add), device=device) * 1e-4
    )
    params.hbias = torch.cat([params.hbias, hbias_add])
    params.weight_matrix = torch.cat([params.weight_matrix, weight_matrix_add], dim=1)
    num_hiddens = num_hiddens_rcm + num_hiddens_add

    # if args["batch_size"] > parallel_chains_v.shape[0]:
    parallel_chains_v = sample_rbm(p_m, m, mu, U, args["num_chains"], device, dtype)

    # Convert parallel chains into (0, 1) format
    parallel_chains_v = (parallel_chains_v + 1) / 2
    print(parallel_chains_v)
    # Thermalize chains
    print("Thermalizing the parallel chains...")
    num_chains = len(parallel_chains_v)

    @torch.jit.script
    def sample_hiddens(chains: Chain, params: BBParams, beta: float = 1.0) -> Chain:
        chains.mean_hidden = torch.sigmoid(
            beta * (params.hbias + (chains.visible @ params.weight_matrix))
        )
        chains.hidden = torch.bernoulli(chains.mean_hidden)
        return chains

    @torch.jit.script
    def sample_visibles(chains: Chain, params: BBParams, beta: float = 1.0) -> Chain:
        chains.mean_visible = torch.sigmoid(
            beta * (params.vbias + (chains.hidden @ params.weight_matrix.T))
        )
        chains.visible = torch.bernoulli(chains.mean_visible)
        return chains

    def sample_state(
        gibbs_steps: int, chains: Chain, params: BBParams, beta: float = 1.0
    ) -> Chain:
        for _ in range(gibbs_steps):
            chains = sample_hiddens(chains=chains, params=params, beta=beta)
            chains = sample_visibles(chains=chains, params=params, beta=beta)
        return chains

    def init_chains(
        num_samples: int,
        params: BBParams,
        start_v: torch.Tensor = None,
    ) -> Chain:
        num_visibles, num_hiddens = params.weight_matrix.shape
        print(params.weight_matrix.dtype)
        mean_visible = (
            torch.ones(
                size=(num_samples, num_visibles), device=params.weight_matrix.device
            )
            / 2
        )
        if start_v is None:
            visible = torch.bernoulli(mean_visible)
        else:
            visible = start_v
        mean_hidden = torch.sigmoid((params.hbias + (visible @ params.weight_matrix)))
        hidden = torch.bernoulli(mean_hidden)
        return Chain(
            visible=visible,
            hidden=hidden,
            mean_visible=mean_visible,
            mean_hidden=mean_hidden,
        )

    parallel_chains = init_chains(num_chains, params=params, start_v=parallel_chains_v)
    print(parallel_chains.visible)

    parallel_chains = sample_state(
        chains=parallel_chains,
        params=params,
        gibbs_steps=args["therm_steps"],
    )

    print(parallel_chains.visible)

    # Generate output file
    with h5py.File(args["output"], "w") as f:
        hyperparameters = f.create_group("hyperparameters")
        hyperparameters["num_hiddens"] = num_hiddens
        hyperparameters["num_visibles"] = num_visibles
        hyperparameters["training_mode"] = "PCD"
        hyperparameters["batch_size"] = args["batch_size"]
        hyperparameters["gibbs_steps"] = args["gibbs_steps"]
        hyperparameters["min_eps"] = args["min_eps"]
        hyperparameters["epochs"] = 0
        hyperparameters["filename"] = str(args["output"])
        hyperparameters["learning_rate"] = args["learning_rate"]
        hyperparameters["beta"] = 1.0  # TODO: parser
        hyperparameters["variable_type"] = args["variable_type"]
        hyperparameters["binary_threshold"] = 0.3
        hyperparameters["dataset_name"] = args["data"]

        # hyperparameters["num_chains"] = num_chains
        # hyperparameters["gibbs_steps_history"] = (
        #     torch.full((20,), 10, device=device).cpu().numpy()
        # )
        # hyperparameters["eps_history"] = (
        #     torch.full((20,), 1, device=device, dtype=torch.int).cpu().numpy()
        # )

        checkpoint = f.create_group(f"epoch_0")
        checkpoint["torch_rng_state"] = torch.get_rng_state()
        checkpoint["numpy_rng_arg0"] = np.random.get_state()[0]
        checkpoint["numpy_rng_arg1"] = np.random.get_state()[1]
        checkpoint["numpy_rng_arg2"] = np.random.get_state()[2]
        checkpoint["numpy_rng_arg3"] = np.random.get_state()[3]
        checkpoint["numpy_rng_arg4"] = np.random.get_state()[4]
        checkpoint["weight_matrix"] = params.weight_matrix.cpu().float().numpy()
        checkpoint["vbias"] = params.vbias.cpu().float().numpy()
        checkpoint["hbias"] = params.hbias.cpu().float().numpy()
        checkpoint["gradient_updates"] = 0
        checkpoint["free_energy"] = 0.0
        checkpoint["time"] = time.time() - start + total_time
        checkpoint["learning_rate"] = args["learning_rate"]

        # checkpoint["deviation"] = 0.01
        f["parallel_chains"] = parallel_chains.visible.cpu().float().numpy()
    # Generate log output file
    log_filename = args["output"].parent / Path(f"log-{args['output'].stem}.csv")
    with open(log_filename, "w") as log_file:
        log_file.write("eps,steps,deviation,lr_vbias,lr_hbias,lr_weight_matrix\n")


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
    convert(args, args["device"], args["dtype"])
