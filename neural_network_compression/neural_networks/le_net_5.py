from typing import Any, Dict

import tensorflow as tf


class LeNet_5(tf.keras.Model):
    """
    LeNet_5 is a convolutional neural network structure proposed by Yann LeCun et al. in 1989 [1].

    [1]: `LeCun, Yann, Bottou, Leon, Bengio, Yoshua, and Haffner, Patrick.
    Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.`.
    """

    def __init__(self) -> None:
        super(LeNet_5, self).__init__()

        # weights: (5,5,1,20) = 500 + (20,)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=20, kernel_size=[5, 5], padding="same", activation=tf.nn.relu
        )
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2)

        # weights: (5,5,20,50) = 25000 + (50,)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=50, kernel_size=[5, 5], padding="same", activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2)

        #  weights: (2450, 256) = 627200 + (256,)
        self.dense = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.dropout = tf.keras.layers.Dropout(0.2)

        # weights: (256, 10) = 2560 + (10,)
        self.logits = tf.keras.layers.Dense(units=10)

    def call(self, x: tf.Tensor, **kwargs: Dict[str, Any]) -> tf.Tensor:
        training = kwargs.get("training", True)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        feature_dim = x.shape[1] * x.shape[2] * x.shape[3]
        x = tf.reshape(x, (-1, feature_dim))
        x = self.dense(x)
        x = self.dropout(x, training=training)
        return self.logits(x)
