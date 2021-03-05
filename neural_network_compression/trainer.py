from typing import List

import tensorflow as tf

from abc import ABC, abstractmethod, abstractproperty

from tensorflow.python.data.ops.dataset_ops import DatasetV1


class Trainer(ABC):
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
    def layers_to_be_pruned(self)->List[tf.keras.layers.Layer]:
        """
        The neural network layers to be pruned.
        We do not always want to prune all the layers
        to avoid generating irrecoverable loss.
        """

    @abstractmethod
    def get_error(self, input_data:tf.Tensor, expected_output: tf.TensorArraySpec)->tf.Tensor:
        """
        Compute the error for a given loss function from input data and expected output.
        """

    def prune_and_train(self):
        raise NotImplemented()

    def train(self,train_dataset:DatasetV1, epochs: int) -> None:
        """
        Optimize the loss function (normal training) for the current neural network for the specified epochs.
        """
        for _ in range(epochs):
            for x_batch, y_batch in train_dataset:
                gradient = self.get_gradient(x_batch, y_batch)
                self.apply_gradient(gradient)


    def get_gradient(self,input_data:tf.Tensor, expected_output: tf.TensorArraySpec)->tf.Tensor:
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

