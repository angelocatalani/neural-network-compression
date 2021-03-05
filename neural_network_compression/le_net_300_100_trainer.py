import tensorflow as tf

from neural_network_compression.neural_networks import LeNet300100
from neural_network_compression.trainer import Trainer


class LeNet300100Trainer(Trainer):
    """
    The trainer for the LeNet300100 neural network.
    """

    @property
    def neural_network(self) -> tf.keras.Model:
        return LeNet300100()

    def get_error(self, input_data:tf.Tensor, expected_output: tf.TensorArraySpec)->tf.Tensor:
        predictions = self.neural_network(input_data)
        cross_entropy = tf.reduce_mean(tf.losses.BinaryCrossentropy()(expected_output, predictions))

        w1 = self.neural_network.layer1.get_weights()[0]
        w2 = self.neural_network.layer2.get_weights()[0]
        w3 = self.neural_network.out.get_weights()[0]
        l2_regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3)
        return tf.reduce_mean(cross_entropy + 0.01 * l2_regularization)





