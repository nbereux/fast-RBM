import gzip
import pickle
import numpy as np

from fastrbm.dataset.definitions import ROOT_DIR_DATASET_PACKAGE


def load_MNIST(
    filename=ROOT_DIR_DATASET_PACKAGE / "data/mnist.pkl.gz",
    variable_type="Bernoulli",
    subset_labels=None,
    binary_threshold=0.3,
):
    with gzip.open(filename, "rb") as f:
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    dataset = []
    labels = []

    # Extract only the labels of interest
    if subset_labels is not None:
        for label in subset_labels:
            dataset.append(
                np.array(training_data[0][training_data[1] == label], dtype=float)
            )
            labels.append(np.array(training_data[1][training_data[1] == label]))
        dataset = np.concatenate(dataset)
        labels = np.concatenate(labels)
    else:
        dataset = np.array(training_data[0])
        labels = np.array(training_data[1])

    # Transform the dataset to the requested variable type
    match variable_type:
        case "Bernoulli":
            dataset = (dataset > binary_threshold).astype("float")
        case "Ising":
            dataset = (dataset > binary_threshold).astype("float") * 2 - 1
        case "Continuous":
            pass

    return dataset, labels
