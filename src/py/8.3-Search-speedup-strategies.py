"""shell
pip install -r https://raw.githubusercontent.com/datamllab/automl-in-action-notebooks/master/requirements.txt
"""

"""
## 8.3.1 Model scheduling with Hyperband
"""

import tensorflow as tf
from tensorflow.keras.datasets import mnist
import autokeras as ak
import keras_tuner as kt

(x_train, y_train), (x_test, y_test) = mnist.load_data()


def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(
        tf.keras.layers.Dense(
            units=hp.Int("units", min_value=32, max_value=512, step=32),
            activation="relu",
        )
    )
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


tuner = kt.Hyperband(
    build_model,
    objective="val_loss",
    max_epochs=10,
    factor=3,
    hyperband_iterations=2,
    directory="result_dir",
    project_name="helloworld",
)

tuner.search(x_train, y_train, epochs=1, validation_data=(x_test, y_test))

"""
## 8.3.2 Faster convergence with pretrained weights in search space
"""

import tensorflow as tf
import autokeras as ak

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
input_node = ak.ImageInput()
output_node = ak.Normalization()(input_node)
output_node = ak.ImageAugmentation()(output_node)
output_node = ak.ResNetBlock(pretrained=True)(output_node)
output_node = ak.ClassificationHead()(output_node)
model = ak.AutoModel(
    inputs=input_node, outputs=output_node, max_trials=2, overwrite=True
)
model.fit(x_train[:100], y_train[:100], epochs=1)
model.evaluate(x_test, y_test)

"""invisible
"""
import tensorflow as tf

resnet = tf.keras.applications.ResNet50(include_top=False, weights="imagenet")
resnet.summary()

"""invisible
"""
import tensorflow as tf
import kerastuner as kt


def build_model(hp):
    if hp.Boolean("pretrained"):
        weights = "imagenet"
    else:
        weights = None
    resnet = tf.keras.applications.ResNet50(include_top=False, weights=weights)
    if hp.Boolean("freeze"):
        resnet.trainable = False

    input_node = tf.keras.Input(shape=(32, 32, 3))
    output_node = resnet(input_node)
    output_node = tf.keras.layers.Dense(10, activation="softmax")(output_node)
    model = tf.keras.Model(inputs=input_node, outputs=output_node)
    model.compile(loss="sparse_categorical_crossentropy")
    return model


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

tuner = kt.RandomSearch(
    build_model,
    objective="val_loss",
    max_trials=4,
    overwrite=True,
    directory="result_dir",
    project_name="pretrained",
)

tuner.search(
    x_train[:100], y_train[:100], epochs=1, validation_data=(x_test[:100], y_test[:100])
)
