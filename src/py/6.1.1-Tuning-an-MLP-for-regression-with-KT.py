"""shell
pip install -r https://raw.githubusercontent.com/datamllab/automl-in-action-notebooks/master/requirements.txt
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(hp):
    input_node = keras.Input(shape=(20,))
    units = hp.Int("units", min_value=32, max_value=512, step=32)
    output_node = layers.Dense(units=units, activation="relu")(input_node)
    output_node = layers.Dense(units=units, activation="relu")(output_node)
    output_node = layers.Dense(units=1, activation="sigmoid")(output_node)
    model = keras.Model(input_node, output_node)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


from keras_tuner import RandomSearch

tuner = RandomSearch(
    build_model,
    objective="val_mae",
    max_trials=5,
    executions_per_trial=3,
    directory="my_dir",
    project_name="helloworld",
)

tuner.search_space_summary()

import numpy as np

x_train = np.random.rand(100, 20)
y_train = np.random.rand(100, 1)
x_val = np.random.rand(20, 20)
y_val = np.random.rand(20, 1)

tuner.search(x_train, y_train, epochs=1, validation_data=(x_val, y_val))

tuner.results_summary(5)

from tensorflow import keras

best_models = tuner.get_best_models(num_models=2)
best_model = best_models[0]
best_model.save("path_to_best_model")
best_model = keras.models.load_model("path_to_best_model")
print(best_model.predict(x_val))
best_model.summary()


def build_model(hp):
    input_node = keras.Input(shape=(20,))
    units = hp.Int("units", min_value=32, max_value=512, step=32)
    output_node = layers.Dense(units=units, activation="relu")(input_node)
    output_node = layers.Dense(units=units, activation="relu")(output_node)
    output_node = layers.Dense(units=1, activation="sigmoid")(output_node)
    model = keras.Model(input_node, output_node)
    optimizer_name = hp.Choice("optimizer", ["adam", "adadelta"])
    learning_rate = hp.Float(
        "learning_rate", min_value=1e-5, max_value=0.1, sampling="log"
    )
    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


tuner = RandomSearch(
    build_model,
    objective="val_mae",
    max_trials=5,
    executions_per_trial=3,
    directory="my_dir",
    project_name="helloworld",
)

import keras_tuner as kt


class Regressor(kt.HyperModel):
    def build(self, hp):
        input_node = keras.Input(shape=(20,))
        units = hp.Int("units", min_value=32, max_value=512, step=32)
        output_node = layers.Dense(units=units, activation="relu")(input_node)
        output_node = layers.Dense(units=units, activation="relu")(output_node)
        output_node = layers.Dense(units=1, activation="sigmoid")(output_node)
        model = keras.Model(input_node, output_node)
        optimizer_name = hp.Choice("optimizer", ["adam", "adadelta"])
        learning_rate = hp.Float(
            "learning_rate", min_value=1e-5, max_value=0.1, sampling="log"
        )
        if optimizer_name == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        return model

    def fit(self, hp, model, **kwargs):
        return model.fit(
            batch_size=hp.Int("batch_size"), shuffle=hp.Boolean("shuffle"), **kwargs
        )


tuner = RandomSearch(
    build_model,
    objective="val_mae",
    max_trials=5,
    executions_per_trial=3,
    directory="my_dir",
    project_name="helloworld",
)

from tensorflow.keras.layers.experimental.preprocessing import Normalization

layer = Normalization(input_shape=(20,))
layer.adapt(x_train)

model = tf.keras.Sequential([layer, tf.keras.layers.Dense(1)])
model.compile(optimizer="adam", loss="mse")
model.fit(x_train, y_train)

normalized_x_train = layer(x_train)
dataset_x_train = tf.data.Dataset.from_tensor_slices(x_train).batch(32)
normalized_dataset = dataset_x_train.map(layer)

from keras_tuner import HyperModel


class Regressor(HyperModel):
    def build(self, hp):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(1))
        model.compile(optimizer="adam", loss="mse")
        return model

    def fit(self, hp, model, x, y, **kwargs):
        if hp.Boolean("normalize"):
            layer = Normalization(input_shape=(20,))
            layer.adapt(x)
            x = layer(x)
        return model.fit(x=x, y=y, **kwargs)


tuner = RandomSearch(Regressor(), objective="val_loss", max_trials=2)
tuner.search(x_train, y_train, validation_data=(x_val, y_val))
