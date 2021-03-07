import pathlib
from typing import Tuple

import matplotlib.pyplot as plt
import numpy
import tensorflow as tf

from neural_network_compression.common import utility
from neural_network_compression.common.trainer import LeNetDataset
from neural_network_compression.le_net_300_100_trainer import LeNet300100Trainer


def reset_seed() -> None:
    numpy.random.seed(0)
    tf.random.set_seed(0)


def get_train_and_test_dataset() -> Tuple[LeNetDataset, LeNetDataset]:
    utility.download_mnist("data/mnist")

    [(x_train, y_train), (x_validation, y_validation), (x_test, y_test)] = utility.read_mnist(
        "data/mnist", flatten=True, num_train=-1
    )
    y_test = tf.argmax(y_test, axis=1)

    train_dataset = LeNetDataset(x_train, y_train)
    test_dataset = LeNetDataset(x_test, y_test)
    return train_dataset, test_dataset


def run_experiment_with_lenet300100(
    train_epochs: int,
    prune_train_epochs: int,
    semi_prune_train_epochs: int,
    maximum_centroid_bits: int,
    k_means_initialization_mode: str,
    with_cumulative_weight_distribution: bool,
    experiment_name: str,
) -> None:
    reset_seed()
    train_dataset, test_dataset = get_train_and_test_dataset()
    trainer = LeNet300100Trainer()
    report_directory = f"{trainer.model_name}_{experiment_name}"

    train_accuracies = trainer.train(
        train_dataset=train_dataset, test_dataset=test_dataset, epochs=train_epochs
    )

    pruned_train_accuracies = trainer.pruned_train(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        epochs=prune_train_epochs,
        with_standard_deviation_smoothing=True,
    )
    semi_pruned_train_accuracies = trainer.semi_pruned_train(
        train_dataset=train_dataset, test_dataset=test_dataset, epochs=semi_prune_train_epochs
    )

    trainer.store_report(report_directory)

    after_quantization_accuracy = trainer.quantize(
        test_dataset=test_dataset,
        with_cumulative_weight_distribution=with_cumulative_weight_distribution,
        maximum_centroid_bits=maximum_centroid_bits,
        k_means_initialization_mode=k_means_initialization_mode,
    )

    fig, ax = plt.subplots()
    plt.legend(["Epochs"])
    ax.plot(range(train_epochs), train_accuracies, color="r", label="train")
    ax.plot(
        range(train_epochs, train_epochs + prune_train_epochs),
        pruned_train_accuracies,
        color="g",
        label="pruned train",
    )
    ax.plot(
        range(
            train_epochs + prune_train_epochs,
            train_epochs + semi_prune_train_epochs + prune_train_epochs,
        ),
        semi_pruned_train_accuracies,
        color="b",
        label="semi pruned train",
    )

    plt.axhline(y=after_quantization_accuracy, color="y", linestyle="-", label="after quantization")

    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    pathlib.Path(report_directory).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{report_directory}/accuracy_plot.png")


if __name__ == "__main__":
    run_experiment_with_lenet300100(
        train_epochs=2,
        prune_train_epochs=2,
        semi_prune_train_epochs=2,
        maximum_centroid_bits=2,
        k_means_initialization_mode="density",
        with_cumulative_weight_distribution=True,
        experiment_name="2BitsDensityQuantization",
    )
