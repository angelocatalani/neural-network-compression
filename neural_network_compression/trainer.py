from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple, Tuple

import numpy
import tensorflow as tf
from tqdm import tqdm

from neural_network_compression import utility


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
    def input_width(self) -> int:
        """
        The width of a single input image.
        """

    @property
    @abstractmethod
    def input_height(self) -> int:
        """
        The height of a single input image.
        """

    @property
    @abstractmethod
    def neural_network_layers(self) -> List[tf.keras.layers.Layer]:
        """
        The layers to be pruned with the associated two thresholds to set respectively
        the params and the biases to zero.
        """

    @property
    @abstractmethod
    def layers_to_prune_with_threshold(self) -> Dict[tf.keras.layers.Layer, Tuple[float, float]]:
        """
        The layers to be pruned with the associated two thresholds to set respectively
        the params and the biases to zero.
        """

    @abstractmethod
    def get_error(self, input_data: tf.Tensor, expected_output: tf.TensorArraySpec) -> tf.Tensor:
        """
        Compute the error for a given loss function from input data and expected output.
        """

    def quantize(
        self,
        test_dataset: LeNetDataset,
        cumulative_weight_distribution: numpy.ndarray,
        maximum_centroid_bits: int,
        kmeans_initialization_mode: str,
    ) -> numpy.float:
        i = 0
        for layer in self.neural_network_layers:
            (params, bias) = layer.get_weights()
            quantized_params = utility.get_quantized_weight(
                params,
                bits=maximum_centroid_bits,
                mode=kmeans_initialization_mode,
                cdfs=(cumulative_weight_distribution[i], cumulative_weight_distribution[i + 1]),
            )
            quantized_bias = utility.get_quantized_weight(
                bias,
                bits=maximum_centroid_bits,
                mode=kmeans_initialization_mode,
                cdfs=(cumulative_weight_distribution[i], cumulative_weight_distribution[i + 1]),
            )
            layer.set_weights([quantized_params, quantized_bias])
            i += 1
        return self.get_accuracy(test_dataset)

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
        for _ in tqdm(range(epochs), desc="train"):
            for x_batch, y_batch in train_dataset:
                gradient = self.get_gradient(x_batch, y_batch)
                self.apply_gradient(gradient)
            accuracies.append(self.get_accuracy(test_dataset))
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
        for _ in tqdm(range(epochs), desc="semi pruned train"):
            for x_batch, y_batch in train_dataset:
                gradient = self.get_gradient(x_batch, y_batch)
                self.apply_gradient(gradient)
                self.reset_pruned_parameters()
            accuracies.append(self.get_accuracy(test_dataset))
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
        for _ in tqdm(range(epochs), desc="pruned train"):
            for x_batch, y_batch in train_dataset:
                self.prune_parameters(with_standard_deviation_smoothing)
                gradient = self.get_gradient(x_batch, y_batch)
                self.apply_gradient(gradient)
                self.reset_pruned_parameters()
            accuracies.append(self.get_accuracy(test_dataset))
        return accuracies

    def prune_parameters(self, with_standard_deviation_smoothing: bool) -> None:
        """
        Prune the neural network parameters.
        """
        for layer_to_prune, (
            weigh_threshold,
            bias_threshold,
        ) in self.layers_to_prune_with_threshold.items():
            (weights, biases) = layer_to_prune.get_weights()
            zero_weight_indexes = utility.prune_weigth(
                weights, threshold=weigh_threshold, std_smooth=with_standard_deviation_smoothing
            )
            zero_bias_indexes = utility.prune_weigth(
                biases, threshold=bias_threshold, std_smooth=with_standard_deviation_smoothing
            )
            self.pruned_indexes_by_layer[layer_to_prune] = [zero_weight_indexes, zero_bias_indexes]
            layer_to_prune.set_weights([weights, biases])

    def reset_pruned_parameters(self) -> None:
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

    def get_gradient(self, input_data: tf.Tensor, expected_output: tf.TensorArraySpec) -> tf.Tensor:
        """
        Compute the gradient.
        """
        with tf.GradientTape() as tape:
            loss = self.get_error(input_data, expected_output)
        return tape.gradient(loss, self.neural_network.trainable_variables)

    def apply_gradient(self, gradient: tf.Tensor) -> None:
        """
        Update the neural network weights according to the optimizer algorithm.
        """
        self.optimizer.apply_gradients(zip(gradient, self.neural_network.trainable_variables))

    def get_accuracy(self, dataset: LeNetDataset) -> numpy.float:
        """
        Compute the accuracy for the current neural network given the input data.
        """
        y_predicted = tf.nn.softmax(self.neural_network(tf.constant(dataset.input_data)))
        y_predicted = tf.argmax(y_predicted, axis=1)
        return tf.reduce_mean(
            tf.cast(tf.equal(y_predicted, dataset.output_data), tf.float32)
        ).numpy()
