from typing import List, Tuple

import tensorflow as tf

from abc import ABC, abstractmethod, abstractproperty

from tensorflow.python.data.ops.dataset_ops import DatasetV1

from neural_network_compression import utility


class Trainer(ABC):
    pruned_idexes_by_layer = {}

    @abstractmethod
    @property
    def neural_network(self) -> tf.keras.Model:
        """
        The neural network to train.
        """

    @abstractmethod
    @property
    def optimizer(self) -> tf.keras.optimizers.Optimizer:
        """
        The optimizer algorithm for training.
        """

    @abstractmethod
    @property
    def input_width(self) -> int:
        """
        The width of a single input image.
        """

    @abstractmethod
    @property
    def input_height(self) -> int:
        """
        The height of a single input image.
        """

    @abstractmethod
    @property
    def layers_to_prune_with_threshold(self) -> List[Tuple[tf.keras.layers.Layer, float]]:
        """
        The neural network layers to be pruned.
        We do not always want to prune all the layers
        to avoid generating irrecoverable loss.
        """

    @abstractmethod
    def get_error(self, input_data: tf.Tensor, expected_output: tf.TensorArraySpec) -> tf.Tensor:
        """
        Compute the error for a given loss function from input data and expected output.
        """

    def pruned_train(self, train_dataset: DatasetV1, epochs: int) -> None:
        for _ in range(epochs):
            for x_batch, y_batch in train_dataset:
                self.prune_parameters()
                gradient = self.get_gradient(
                    tf.reshape(x_batch, shape=[-1, self.input_width, self.input_height, 1]), y_batch
                )
                self.apply_gradient(gradient)
                self.reset_pruned_parameters()

    def prune_parameters(self) -> None:
        for layer_to_prune, threshold in self.layers_to_prune_with_threshold:
            (weights, biases) = layer_to_prune.get_weights()
            zero_weight_indexes = utility.prune_weigth(
                weights, threshold=threshold, std_smooth=True
            )
            zero_bias_indexes = utility.prune_weigth(biases, threshold=threshold, std_smooth=True)
            self.pruned_idexes_by_layer[layer_to_prune] = [zero_weight_indexes, zero_bias_indexes]
            layer_to_prune.set_weights([weights, biases])

    def reset_pruned_parameters(self) -> None:
        for pruned_layer, (
            zero_weight_indexes,
            zero_bias_indexes,
        ) in self.pruned_idexes_by_layer.items():
            (weights, biases) = pruned_layer.get_weights()
            weights[zero_weight_indexes] = 0
            biases[zero_bias_indexes] = 0
            pruned_layer.set_weights([weights, biases])

    def train(self, train_dataset: DatasetV1, epochs: int) -> None:
        """
        Optimize the loss function (normal training) for the current neural network for the specified epochs.
        """
        for _ in range(epochs):
            for x_batch, y_batch in train_dataset:
                gradient = self.get_gradient(
                    tf.reshape(x_batch, shape=[-1, self.input_width, self.input_height, 1]), y_batch
                )
                self.apply_gradient(gradient)

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
