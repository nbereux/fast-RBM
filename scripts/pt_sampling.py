import argparse
import h5py
import numpy as np

from fastrbm.methods.parallel_tempering_temperature import PTSampling

from fastrbm.utils import load_model, get_all_epochs

def main(filename, out_file, num_samples, it_mcmc, target_acc_rate):
    with h5py.File(filename, "r") as f:
        epochs = []
        for key in f.keys():
            if "epoch" in key:
                ep = int(key.replace("epoch_", ""))
                epochs.append(ep)
    epochs = np.sort(epochs)
    device = "cuda"
    age = get_all_epochs(filename)[-1]
    params = load_model(filename, age, device=device)
    (chains_v, chains_h), inverse_temperatures, index = PTSampling(it_mcmc=it_mcmc, increment=1, target_acc_rate=target_acc_rate, num_chains=num_samples, params=params)
    
    with h5py.File(out_file, "w") as f:
        f["gen_chains"] = chains_v.cpu().numpy()
        f["sel_beta"] = inverse_temperatures


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PT sampling on the provided model")
    parser.add_argument("-i", "--filename", type=str, help="Model to use for sampling")
    parser.add_argument(
        "-o", "--out_file", type=str, help="Path to save the samples after generation"
    )
    parser.add_argument(
        "--num_samples",
        default=1000,
        type=int,
        help="(Defaults to 1000). Number of generated samples.",
    )
    parser.add_argument(
        "--target_acc_rate",
        default=0.3,
        type=float,
        help="(Defaults to 0.3). Target acceptance rate between two consecutive models.",
    )
    parser.add_argument(
        "--it_mcmc",
        default=1000,
        type=int,
        help="(Defaults to 1000). Number of MCMC steps to perform.",
    )
    args = parser.parse_args()
    main(
        filename=args.filename,
        out_file=args.out_file,
        num_samples=args.num_samples,
        it_mcmc=args.it_mcmc,
        target_acc_rate=args.target_acc_rate,
    )
