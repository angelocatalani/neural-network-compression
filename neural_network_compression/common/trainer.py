import pathlib
from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple, Tuple

import numpy
import tensorflow as tf
from tqdm import tqdm

from neural_network_compression.common import utility


class LeNetDataset(NamedTuple):
    input_data: numpy.ndarray
    output_data: numpy.ndarray


class Trainer(ABC):
    # the neural network to train
    neural_network: tf.keras.Model

    # the optimizer algorithm for training
    optimizer: tf.keras.optimizers.Optimizer

    # internal cache with zero weight indexes
    pruned_indexes_by_layer: Dict[tf.keras.layers.Layer, Tuple[List[int], List[int]]] = {}

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        The model name.
        """

    @property
    @abstractmethod
    def _layers_to_prune_with_threshold(self) -> Dict[tf.keras.layers.Layer, Tuple[float, float]]:
        """
        The layers to be pruned with the associated two thresholds to set respectively
        the params and the biases to zero.
        """

    def quantize(
        self,
        test_dataset: LeNetDataset,
        with_cumulative_weight_distribution: bool,
        maximum_centroid_bits: int,
        k_means_initialization_mode: str,
    ) -> numpy.float:

        for layer_name, layer in tqdm(self.neural_network.get_config().items(), desc="quantize"):
            quantized_weights_and_bias = []
            for params in layer.get_weights():
                cdfs = None
                if with_cumulative_weight_distribution:
                    flattened_params = params.flatten()
                    (non_zero_indexes,) = numpy.nonzero(flattened_params == 0)
                    params_different_from_zero = numpy.delete(
                        flattened_params, non_zero_indexes, axis=0
                    )
                    cdfs = utility.get_weight_distribution(params_different_from_zero)

                quantized_weights_and_bias.append(
                    utility.get_quantized_weight(
                        params,
                        bits=maximum_centroid_bits,
                        mode=k_means_initialization_mode,
                        cdfs=cdfs,
                    )[0]
                )
            layer.set_weights(quantized_weights_and_bias)

        return self._get_accuracy(test_dataset)

    def train(
        self, train_dataset: LeNetDataset, test_dataset: LeNetDataset, epochs: int
    ) -> List[float]:
        """
        Optimize the loss function (normal training) for the current neural network for the specified epochs.
        """
        accuracies = []
        train_dataset = (
            tf.data.Dataset.from_tensor_slices(
                (train_dataset.input_data, train_dataset.output_data)
            )
            .shuffle(1000)
            .batch(512, drop_remainder=True)
        )
        progress_bar = tqdm(range(epochs), desc="train")
        for _ in progress_bar:
            for x_batch, y_batch in train_dataset:
                gradient = self._get_gradient(x_batch, y_batch)
                self._apply_gradient(gradient)
            accuracy = self._get_accuracy(test_dataset)
            progress_bar.set_postfix({"accuracy": accuracy})
            accuracies.append(accuracy)
        return accuracies

    def semi_pruned_train(
        self, train_dataset: LeNetDataset, test_dataset: LeNetDataset, epochs: int
    ) -> List[float]:
        """
        Train the neural network while keeping the previously pruned weights to zero.
        """
        accuracies = []
        train_dataset = (
            tf.data.Dataset.from_tensor_slices(
                (train_dataset.input_data, train_dataset.output_data)
            )
            .shuffle(1000)
            .batch(512, drop_remainder=True)
        )
        progress_bar = tqdm(range(epochs), desc="semi pruned train")
        for _ in progress_bar:
            for x_batch, y_batch in train_dataset:
                gradient = self._get_gradient(x_batch, y_batch)
                self._apply_gradient(gradient)
                self._reset_pruned_parameters()
            accuracy = self._get_accuracy(test_dataset)
            progress_bar.set_postfix({"accuracy": accuracy})
            accuracies.append(accuracy)
        return accuracies

    def pruned_train(
        self,
        train_dataset: LeNetDataset,
        test_dataset: LeNetDataset,
        epochs: int,
        with_standard_deviation_smoothing: bool,
    ) -> List[float]:
        """
        Train the neural network while pruning the parameters.
        Compute all the accuracies at the end of each epoch.
        """
        accuracies = []
        train_dataset = (
            tf.data.Dataset.from_tensor_slices(
                (train_dataset.input_data, train_dataset.output_data)
            )
            .shuffle(1000)
            .batch(512, drop_remainder=True)
        )
        progress_bar = tqdm(range(epochs), desc="pruned train")
        for _ in progress_bar:
            for x_batch, y_batch in train_dataset:
                self._prune_parameters(with_standard_deviation_smoothing)
                gradient = self._get_gradient(x_batch, y_batch)
                self._apply_gradient(gradient)
                self._reset_pruned_parameters()
            accuracy = self._get_accuracy(test_dataset)
            progress_bar.set_postfix({"accuracy": accuracy})
            accuracies.append(accuracy)
        return accuracies

    def store_report(self, directory: str) -> None:
        """
        Store the report about the parameters distribution for the current neural network.
        """
        report = ""
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        for layer_name, layer in tqdm(
            self.neural_network.get_config().items(), desc="store report"
        ):
            report += f"layer: {layer_name}\n"
            weight_layer, bias_layer = layer.get_weights()
            report += f"zeroed weights: {numpy.count_nonzero(weight_layer == 0)}\ntotal weights: {weight_layer.shape[0] * weight_layer.shape[1]}\n"
            report += f"zeroed biases: {numpy.count_nonzero(bias_layer == 0)}\ntotal weights: {bias_layer.shape[0]}\n\n"

            utility.show_weight_distribution(
                weight_layer, f"{directory}/{layer_name}_weights.png", plot_std=True, plot_cdf=True
            )
            utility.show_weight_distribution(
                bias_layer, f"{directory}/{layer_name}_biases.png", plot_std=False, plot_cdf=True
            )
        with open(f"{directory}/report.txt", "w") as f:
            f.write(report)

    def _prune_parameters(self, with_standard_deviation_smoothing: bool) -> None:
        """
        Prune the neural network parameters.
        """
        for layer_to_prune, (
            weigh_threshold,
            bias_threshold,
        ) in self._layers_to_prune_with_threshold.items():
            (weights, biases) = layer_to_prune.get_weights()
            zero_weight_indexes = utility.prune_weigth(
                weights, threshold=weigh_threshold, std_smooth=with_standard_deviation_smoothing
            )
            zero_bias_indexes = utility.prune_weigth(
                biases, threshold=bias_threshold, std_smooth=with_standard_deviation_smoothing
            )
            self.pruned_indexes_by_layer[layer_to_prune] = (zero_weight_indexes, zero_bias_indexes)
            layer_to_prune.set_weights([weights, biases])

    def _reset_pruned_parameters(self) -> None:
        """
        Set to zero the parameters previously pruned.
        """
        for pruned_layer, (
            zero_weight_indexes,
            zero_bias_indexes,
        ) in self.pruned_indexes_by_layer.items():
            (weights, biases) = pruned_layer.get_weights()
            weights[zero_weight_indexes] = 0
            biases[zero_bias_indexes] = 0
            pruned_layer.set_weights([weights, biases])

    def _get_gradient(
        self, input_data: tf.Tensor, expected_output: tf.TensorArraySpec
    ) -> tf.Tensor:
        """
        Compute the gradient.
        """
        with tf.GradientTape() as tape:
            loss = self._get_error(input_data, expected_output)
        return tape.gradient(loss, self.neural_network.trainable_variables)

    def _apply_gradient(self, gradient: tf.Tensor) -> None:
        """
        Update the neural network weights according to the optimizer algorithm.
        """
        self.optimizer.apply_gradients(zip(gradient, self.neural_network.trainable_variables))

    def _get_accuracy(self, dataset: LeNetDataset) -> numpy.float:
        """
        Compute the accuracy for the current neural network given the input data.
        """
        y_predicted = tf.nn.softmax(self.neural_network(tf.constant(dataset.input_data)))
        y_predicted = tf.argmax(y_predicted, axis=1)
        return tf.reduce_mean(
            tf.cast(tf.equal(y_predicted, dataset.output_data), tf.float32)
        ).numpy()

    @abstractmethod
    def _get_error(self, input_data: tf.Tensor, expected_output: tf.TensorArraySpec) -> tf.Tensor:
        """
        Compute the error for a given loss function from input data and expected output.
        """
