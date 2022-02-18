"""shell
pip install -r https://raw.githubusercontent.com/datamllab/automl-in-action-notebooks/master/requirements.txt
"""

from tensorflow.keras.datasets import fashion_mnist

(x_train, _), (x_test, _) = fashion_mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses


class AutoencoderModel(Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_layer = layers.Dense(latent_dim, activation="relu")
        self.decoder_layer = layers.Dense(784, activation="sigmoid")

    def encode(self, encoder_input):
        encoder_output = layers.Flatten()(encoder_input)
        encoder_output = self.encoder_layer(encoder_output)
        return encoder_output

    def decode(self, decoder_input):
        decoder_output = decoder_input
        decoder_output = self.decoder_layer(decoder_output)
        decoder_output = layers.Reshape((28, 28))(decoder_output)
        return decoder_output

    def call(self, x):
        return self.decode(self.encode(x))


import numpy as np

tf.random.set_seed(5)
np.random.seed(5)
autoencoder = AutoencoderModel(64)
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.fit(
    x_train, x_train, epochs=10, shuffle=True, validation_data=(x_test, x_test)
)

autoencoder.evaluate(x_test, x_test)

autoencoder.encode(x_test[:1])

import matplotlib.pyplot as plt


def show_images(model, images):
    encoded_imgs = model.encode(images).numpy()
    decoded_imgs = model.decode(encoded_imgs).numpy()

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(images[i])
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i])
        plt.title("reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


show_images(autoencoder, x_test)

import keras_tuner
from tensorflow import keras
from keras_tuner import RandomSearch


class AutoencoderBlock(keras.Model):
    def __init__(self, latent_dim, hp):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_layers = []
        for i in range(
            hp.Int("encoder_layers", min_value=0, max_value=2, step=1, default=0)
        ):
            self.encoder_layers.append(
                layers.Dense(
                    units=hp.Choice("encoder_layers_{i}".format(i=i), [64, 128, 256]),
                    activation="relu",
                )
            )
        self.encoder_layers.append(layers.Dense(latent_dim, activation="relu"))
        self.decoder_layers = []
        for i in range(
            hp.Int("decoder_layers", min_value=0, max_value=2, step=1, default=0)
        ):
            self.decoder_layers.append(
                layers.Dense(
                    units=hp.Choice("decoder_layers_{i}".format(i=i), [64, 128, 256]),
                    activation="relu",
                )
            )
        self.decoder_layers.append(layers.Dense(784, activation="sigmoid"))

    def encode(self, encoder_input):
        encoder_output = layers.Flatten()(encoder_input)
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output)
        return encoder_output

    def decode(self, decoder_input):
        decoder_output = decoder_input
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output)
        decoder_output = layers.Reshape((28, 28))(decoder_output)
        return decoder_output

    def call(self, x):
        return self.decode(self.encode(x))


def build_model(hp):
    latent_dim = 20
    autoencoder = AutoencoderBlock(latent_dim, hp)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


tuner = RandomSearch(
    build_model,
    objective="val_loss",
    max_trials=10,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld",
)

tuner.search(x_train, x_train, epochs=10, validation_data=(x_test, x_test))

autoencoder = tuner.get_best_models(num_models=1)[0]
tuner.results_summary(1)
autoencoder.evaluate(x_test, x_test)

show_images(autoencoder, x_test)
