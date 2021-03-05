from typing import Any, Dict

import tensorflow as tf


class LeNet300100(tf.keras.Model):
    """
    LeNet300100 is a convolutional neural network structure proposed by Yann LeCun et al. in 1989 [1].

    [1]: `LeCun, Yann, Bottou, Leon, Bengio, Yoshua, and Haffner, Patrick.
    Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.`.
    """

    def __init__(self) -> None:
        super(LeNet300100, self).__init__()

        # weights: (input_size: 28*28, 300) = 235200 +(300, )
        self.dense1 = tf.keras.layers.Dense(300, activation=tf.nn.relu)

        # weights: (235200, 100) = 23520000 +(100, )
        self.dense2 = tf.keras.layers.Dense(100, activation=tf.nn.relu)

        # weights: (23520000, 10) = 235200000 +(10, )
        self.out = tf.keras.layers.Dense(10)

    def call(self, x: tf.Tensor, **kwargs: Dict[str, Any]) -> tf.Tensor:
        return self.out(self.dense2(self.dense1(x)))

    def get_config(self) -> Dict[str, tf.keras.layers]:
        config: Dict[str, tf.keras.layers] = super().get_config().copy()
        config.update(
            {
                "dense1": self.dense1,
                "dense2": self.dense2,
                "out": self.out,
            }
        )
        return config
