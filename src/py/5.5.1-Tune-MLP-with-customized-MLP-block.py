"""shell
pip install -r https://raw.githubusercontent.com/datamllab/automl-in-action-notebooks/master/requirements.txt
"""

"""
## Customize a block for tuning the number of units
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_mlp():
    input_node = keras.Input(shape=(20,))

    output_node = layers.Dense(units=32, activation="relu")(input_node)
    output_node = layers.Dense(units=32, activation="relu")(output_node)
    output_node = layers.Dense(units=1, activation="sigmoid")(output_node)

    model = keras.Model(input_node, output_node)
    return model


mlp_model = build_mlp()
"""invisible
"""

import autokeras as ak
import tensorflow as tf
from tensorflow.keras import layers


class MlpBlock(ak.Block):
    def build(self, hp, inputs):

        input_node = tf.nest.flatten(inputs)[0]

        units = hp.Int(name="units", min_value=32, max_value=512, step=32)

        output_node = layers.Dense(units=units, activation="relu")(input_node)
        output_node = layers.Dense(units=units, activation="relu")(output_node)

        return output_node


"""invisible
"""

import autokeras as ak
import tensorflow as tf
from tensorflow.keras import layers


class MlpBlock(ak.Block):
    def build(self, hp, inputs):

        input_node = tf.nest.flatten(inputs)[0]

        units_1 = hp.Int(name="units_1", min_value=32, max_value=512, step=32)

        units_2 = hp.Int(name="units_2", min_value=32, max_value=512, step=32)

        output_node = layers.Dense(units=units_1, activation="relu")(input_node)

        output_node = layers.Dense(units=units_2, activation="relu")(output_node)

        return output_node


"""
## Customize a block for tuning different types of hyperparameters
"""

import autokeras as ak
import tensorflow as tf
from tensorflow.keras import layers


class MlpBlock(ak.Block):
    def build(self, hp, inputs):
        output_node = tf.nest.flatten(inputs)[0]
        for i in range(hp.Choice("num_layers", [1, 2, 3])):
            output_node = layers.Dense(
                units=hp.Int("units_" + str(i), min_value=32, max_value=512, step=32),
                activation="relu",
            )(output_node)
        return output_node


"""invisible
"""

import autokeras as ak
import tensorflow as tf
from tensorflow.keras import layers


class MlpBlock(ak.Block):
    def build(self, hp, inputs):
        output_node = tf.nest.flatten(inputs)[0]
        for i in range(hp.Choice("num_layers", [1, 2, 3])):
            output_node = layers.Dense(
                units=hp.Int("units_" + str(i), min_value=32, max_value=512, step=32),
                activation="relu",
            )(output_node)
        if hp.Boolean("dropout"):
            output_node = layers.Dropout(
                rate=hp.Float("dropout_rate", min_value=0, max_value=1)
            )(output_node)
        return output_node


"""
## Using the customized block to create an AutoML pipeline
"""

import keras_tuner as kt

hp = kt.HyperParameters()
inputs = tf.keras.Input(shape=(20,))
MlpBlock().build(hp, inputs)
"""invisible
"""

import numpy as np

x_train = np.random.rand(100, 20)
y_train = np.random.rand(100, 1)
x_test = np.random.rand(100, 20)

input_node = ak.StructuredDataInput()
output_node = MlpBlock()(input_node)
output_node = ak.RegressionHead()(output_node)
auto_model = ak.AutoModel(input_node, output_node, max_trials=3, overwrite=True)
auto_model.fit(x_train, y_train, epochs=1)
"""invisible
"""

auto_model.predict(x_test).shape
"""invisible
"""

auto_model.tuner.search_space_summary()
