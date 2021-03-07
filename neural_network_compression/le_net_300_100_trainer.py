from typing import Dict, Tuple

import tensorflow as tf

from neural_network_compression.common.trainer import Trainer
from neural_network_compression.neural_networks import LeNet300100


class LeNet300100Trainer(Trainer):
    """
    The trainer for the LeNet300100 neural network.
    """

    @property
    def model_name(self) -> str:
        return "LeNet300100"

    neural_network = LeNet300100()
    optimizer = tf.keras.optimizers.Adam(0.001)

    @property
    def _layers_to_prune_with_threshold(self) -> Dict[tf.keras.layers.Layer, Tuple[float, float]]:
        return {
            self.neural_network.dense1: (1, 0.1),
            self.neural_network.dense2: (1, 0.1),
            self.neural_network.out: (0.5, 0),
        }

    def _get_error(self, input_data: tf.Tensor, expected_output: tf.TensorArraySpec) -> tf.Tensor:
        logits_predictions = self.neural_network(input_data)
        cross_entropy = tf.reduce_mean(
            tf.losses.BinaryCrossentropy(from_logits=True)(expected_output, logits_predictions)
        )

        w1 = self.neural_network.dense1.get_weights()[0]
        w2 = self.neural_network.dense2.get_weights()[0]
        w3 = self.neural_network.out.get_weights()[0]
        l2_regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3)
        return tf.reduce_mean(cross_entropy + 0.01 * l2_regularization)
