import numpy as np

from fastrbm.dataset.definitions import ROOT_DIR_DATASET_PACKAGE


def load_dat(filename, variable_type="Bernoulli"):
    dataset = np.genfromtxt(filename)
    match variable_type:
        case "Bernoulli":
            pass
        case "Ising":
            dataset = dataset * 2 - 1
            pass
        case "Ising":
            dataset = dataset * 2 - 1
    return dataset.T


def load_GENE(
    filename=ROOT_DIR_DATASET_PACKAGE / "data/1kg_xtrain.d",
    variable_type="Bernoulli",
):
    return load_dat(filename=filename, variable_type=variable_type)
