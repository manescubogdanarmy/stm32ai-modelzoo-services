"""
Custom STM-MNIST model with sigmoid activation for multi-label classification.
"""
import tensorflow as tf
from tensorflow.keras import layers
from typing import Tuple


def get_stmnist_multilabel(num_classes: int = None, input_shape: Tuple[int, int, int] = None,
                          dropout: float = None) -> tf.keras.models.Model:
    """
    Returns a stmnist model modified for multi-label classification.
    
    This version uses sigmoid activation instead of softmax to allow multiple classes
    to be active simultaneously (e.g., multiple sensors triggering alerts).

    Args:
    - num_classes: integer, the number of output classes.
    - input_shape: tuple of integers, the shape of the input tensor (height, width, channels).
    - dropout: float, dropout rate.

    Returns:
    - tf.keras.models.Model object, the modified stmnist model.
    """
    # Define the input tensor
    inputs = tf.keras.Input(shape=input_shape)

    # Define the number of filters
    num_filters = 16

    # Block 1
    x = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Block 2
    num_filters = 2 * num_filters
    x = layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Block 3
    num_filters = 2 * num_filters
    x = layers.Conv2D(num_filters, kernel_size=1, strides=1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Global average pooling layer
    x = layers.GlobalAveragePooling2D()(x)

    if dropout:
        x = layers.Dropout(rate=dropout, name="dropout")(x)

    # Output layer with sigmoid activation for multi-label classification
    x = layers.Dense(num_classes, activation="sigmoid", name="predictions")(x)

    # Create the model
    model = tf.keras.models.Model(inputs, x, name="stmnist_multilabel")

    return model
