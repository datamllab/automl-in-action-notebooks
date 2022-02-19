"""shell
pip install -r https://raw.githubusercontent.com/datamllab/automl-in-action-notebooks/master/requirements.txt
"""

import tensorflow as tf

tf.random.set_seed(42)

"""
## Load data
"""

from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

"""
## Explore data
"""

train_images.shape, test_images.shape
"""invisible
"""

len(train_labels), len(test_labels)
"""invisible
"""

train_labels, test_labels
"""invisible
"""

import matplotlib.pyplot as plt

plt.figure()
plt.imshow(train_images[0])  # , cmap='gray'
plt.colorbar()
plt.title("Label is {label}".format(label=train_labels[0]))
plt.show()

"""
## Data Preparation: scaling
"""

train_images = train_images / 255.0
test_images = test_images / 255.0
"""inline
matplotlib inline
"""
import matplotlib
import matplotlib.pyplot as plt

# plot first 20 images
n = 20
_, axes = plt.subplots(2, 10, figsize=(10, 2))
plt.tight_layout()
for i in range(n):
    row, col = i // 10, i % 10
    axes[row, col].set_axis_off()
    axes[row, col].imshow(
        train_images[
            i,
        ],
        cmap=plt.cm.binary,
        interpolation="nearest",
    )  # plt.cm.gray_r
    axes[row, col].set_title("Label: %i" % train_labels[i])

"""
## Build MLP
"""

from tensorflow import keras
from tensorflow.keras import layers

mlp_model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=train_images.shape[1:]),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ]
)
mlp_model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

"""invisible
"""
mlp_model.summary()
"""invisible
"""

mlp_model.fit(train_images, train_labels, epochs=5, batch_size=64, verbose=1)
"""invisible
"""

test_loss, test_acc = mlp_model.evaluate(test_images, test_labels, verbose=0)
test_acc

"""
## Build CNN
"""


def build_cnn():
    model = keras.Sequential(
        [
            keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=train_images.shape[1:] + (1,)
            ),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation="relu"),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


"""invisible
"""

cnn_model = build_cnn()

"""invisible
"""
cnn_model.summary()

"""invisible
"""
train_images_4d = train_images[..., tf.newaxis]
test_images_4d = test_images[..., tf.newaxis]
train_images_4d.shape, test_images_4d.shape

"""invisible
"""
cnn_model.fit(train_images_4d, train_labels, epochs=5, batch_size=64, verbose=1)
"""invisible
"""

test_loss, test_acc = cnn_model.evaluate(test_images_4d, test_labels, verbose=0)
test_acc

"""
## Make predictions
"""

test_predictions = cnn_model.predict(test_images_4d)
test_predictions[:5]

"""invisible
"""
import numpy as np

np.argmax(test_predictions[0])
"""invisible
"""

test_labels[0]
