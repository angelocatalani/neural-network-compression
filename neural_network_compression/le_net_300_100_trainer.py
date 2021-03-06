from typing import List, Tuple, Dict

import tensorflow as tf

from neural_network_compression.neural_networks import LeNet300100
from neural_network_compression.trainer import Trainer


class LeNet300100Trainer(Trainer):
    """
    The trainer for the LeNet300100 neural network.
    """

    neural_network = LeNet300100()
    optimizer = tf.keras.optimizers.Adam(0.001)

    @property
    def input_width(self) -> int:
        return 28

    @property
    def input_height(self) -> int:
        return 28

    @property
    def layers_to_prune_with_threshold(self) -> Dict[tf.keras.layers.Layer, Tuple[float, float]]:
        return {
            self.neural_network.dense1: (1, 0.1),
            self.neural_network.dense2: (1, 0.1),
            self.neural_network.out: (0.5, 1),
        }

    def get_error(self, input_data: tf.Tensor, expected_output: tf.TensorArraySpec) -> tf.Tensor:
        predictions = self.neural_network(input_data)
        cross_entropy = tf.reduce_mean(tf.losses.BinaryCrossentropy()(expected_output, predictions))

        w1 = self.neural_network.dense1.get_weights()[0]
        w2 = self.neural_network.dense2.get_weights()[0]
        w3 = self.neural_network.out.get_weights()[0]
        l2_regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3)
        return tf.reduce_mean(cross_entropy + 0.01 * l2_regularization)
