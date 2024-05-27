import argparse
import h5py
import numpy as np
import torch
from tqdm import tqdm

from fastrbm.methods.parallel_tempering_bernoulli import (
    init_sampling,
    PTsampling,
    compute_energy_visibles,
    sample_rcm,
    sample_state,
)


def get_epochs_pt_sampling(
    filename, rcm, filename_rcm, epochs, device, target_acc_rate
):
    use_rcm = filename_rcm is not None
    sel_epochs = [epochs[0]]
    pbar = tqdm(enumerate(epochs[1:], 1), total=len(epochs[1:]))
    num_chains = 10000
    it_mcmc = 1000
    # Init first chain
    if use_rcm:
        with h5py.File(filename_rcm, "r") as f:
            tmp_name = "best_trial"
            U = torch.from_numpy(f["const"]["U"][()]).to(device)
            m = torch.from_numpy(f["const"]["m"][()]).to(device)
            mu = torch.from_numpy(f["const"]["mu"][()]).to(device)
            p_m = torch.from_numpy(f[tmp_name]["pdm"][()]).to(device)
        chains = sample_rcm(
            p_m=p_m, m=m, mu=mu, U=U, num_samples=num_chains, device=device
        )
        chains = (chains + 1) / 2
    else:
        with h5py.File(filename, "r") as f:
            num_visibles = f["parallel_chains"][()].shape[1]
            params = tuple(
                torch.from_numpy(f[f"epoch_{epochs[0]}"][f"{par}"][()]).to(device)
                for par in ["vbias", "hbias", "weight_matrix"]
            )
        v_init = torch.randint(
            0, 2, size=(num_chains, num_visibles), device=device
        ).float()
        chains = sample_state(v_init, params=params, gibbs_steps=it_mcmc)

    chains = chains.unsqueeze(0)
    for i, ep in pbar:
        saved_chains = chains.cpu()
        torch.cuda.empty_cache()
        pbar.update(1)
        sel_epochs.append(ep)
        with h5py.File(filename, "r") as f:
            params = tuple(
                torch.from_numpy(f[f"epoch_{ep}"][f"{par}"][()]).to(device)
                for par in ["vbias", "hbias", "weight_matrix"]
            )
        new_chains = sample_state(chains[-1], params=params, gibbs_steps=it_mcmc)

        chains = torch.vstack([chains, new_chains.unsqueeze(0)])
        chains, _, acc_rate = PTsampling(
            rcm,
            filename,
            chains,
            1,
            sel_epochs,
            1,
            show_pbar=False,
            show_acc_rate=False,
        )
        sel_epochs.pop(-1)
        pbar.write(str(acc_rate[-1]))
        if acc_rate[-1] < target_acc_rate:
            sel_epochs.append(epochs[i - 1])
            pbar.write(str(sel_epochs))
        else:
            chains = saved_chains.cuda()
    if epochs[-1] not in sel_epochs:
        sel_epochs.append(epochs[-1])
    return sel_epochs


def match_acc_rate(
    epoch_ref, all_epochs, fname_model, chains_probe, acc_rate_target, device="cuda"
):
    n_chains = len(chains_probe[0])
    idx_ref = np.where(all_epochs == epoch_ref)[0][0]
    f = h5py.File(fname_model, "r")
    params_ref = tuple(
        torch.from_numpy(f[f"epoch_{all_epochs[idx_ref]}"][f"{par}"][()]).to(device)
        for par in ["vbias", "hbias", "weight_matrix"]
    )
    chains_ref = chains_probe[idx_ref]
    for i in range(idx_ref):
        idx_test = idx_ref - i - 1
        params_test = tuple(
            torch.from_numpy(f[f"epoch_{all_epochs[idx_test]}"][f"{par}"][()]).to(
                device
            )
            for par in ["vbias", "hbias", "weight_matrix"]
        )
        chains_test = chains_probe[idx_test]
        delta_E = (
            -compute_energy_visibles(chains_ref, params_test)
            + compute_energy_visibles(chains_test, params_test)
            + compute_energy_visibles(chains_ref, params_ref)
            - compute_energy_visibles(chains_test, params_ref)
        )
        swap_chain = torch.bernoulli(torch.clamp(torch.exp(delta_E), max=1.0)).bool()
        acc_rate = (swap_chain.sum() / n_chains).cpu().numpy()
        if (acc_rate < acc_rate_target + 0.1) or (all_epochs[idx_test] == 0):
            print(
                f"Epochs match: {all_epochs[idx_ref]} -> {all_epochs[idx_test]} with acc_rate = {acc_rate}"
            )
            return all_epochs[idx_test]
    if 0 in all_epochs:
        return 0
    return 1


def main(filename, filename_rcm, out_file, num_samples, it_mcmc, target_acc_rate):
    with h5py.File(filename, "r") as f:
        epochs = []
        for key in f.keys():
            if "epoch" in key:
                ep = int(key.replace("epoch_", ""))
                epochs.append(ep)
    epochs = np.sort(epochs)
    device = "cuda"
    if filename_rcm is not None:
        tmp_name = "best_trial"
        with h5py.File(filename_rcm, "r") as f:
            print(f.keys())
            print(f[tmp_name].keys())
            U = torch.from_numpy(np.array(f["const"]["U"])).to(device)
            m = torch.from_numpy(np.array(f["const"]["m"])).to(device)
            mu = torch.from_numpy(np.array(f["const"]["mu"])).to(device)
            p_m = torch.from_numpy(np.array(f[tmp_name]["pdm"])).to(device)
        rcm = {"U": U, "m": m, "mu": mu, "p_m": p_m}
    else:
        rcm = None

    sel_epochs = get_epochs_pt_sampling(
        filename=filename,
        rcm=rcm,
        filename_rcm=filename_rcm,
        epochs=epochs,
        device=device,
        target_acc_rate=target_acc_rate,
    )

    chains = init_sampling(
        fname_rcm=filename_rcm,
        fname_rbm=filename,
        n_gen=num_samples,
        it_mcmc=1000,
        epochs=sel_epochs,
        device=device,
    )
    chains, _, _ = PTsampling(
        rcm=rcm,
        fname_rbm=filename,
        init_chains=chains,
        it_mcmc=it_mcmc,
        epochs=sel_epochs,
        increment=1,
    )

    with h5py.File(out_file, "w") as f:
        f["gen_chains"] = chains.cpu().numpy()
        f["sel_epochs"] = sel_epochs


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PTT sampling on the provided model")
    parser.add_argument("-i", "--filename", type=str, help="Model to use for sampling")
    parser.add_argument(
        "-o", "--out_file", type=str, help="Path to save the samples after generation"
    )
    parser.add_argument("--filename_rcm", type=str, default=None)
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
        filename_rcm=args.filename_rcm,
        out_file=args.out_file,
        num_samples=args.num_samples,
        it_mcmc=args.it_mcmc,
        target_acc_rate=args.target_acc_rate,
    )
