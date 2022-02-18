"""shell
!pip install -r https://raw.githubusercontent.com/datamllab/automl-in-action-notebooks/master/requirements.txt
"""

"""
## 8.2.1 Data parallelism
"""

import tensorflow as tf
from tensorflow.keras.datasets import mnist
import autokeras as ak

(x_train, y_train), (x_test, y_test) = mnist.load_data()
clf = ak.ImageClassifier(
    overwrite=True, max_trials=1, distribution_strategy=tf.distribute.MirroredStrategy()
)
clf.fit(x_train, y_train, epochs=1)

import keras_tuner as kt


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


tuner = kt.RandomSearch(
    build_model,
    objective="val_loss",
    max_trials=1,
    directory="my_dir",
    distribution_strategy=tf.distribute.MirroredStrategy(),
    project_name="helloworld",
)

tuner.search(x_train, y_train, epochs=1, validation_data=(x_test, y_test))

"""
## 8.2.3 Parallel tuning

Note: The code below are only for the convenience of copying. It should be setup on your
machines and run locally.

```shell
export KERASTUNER_TUNER_ID="chief"
export KERASTUNER_ORACLE_IP="127.0.0.1"
export KERASTUNER_ORACLE_PORT="8000"
python run_tuning.py
```

```shell
export KERASTUNER_TUNER_ID="tuner0"
export KERASTUNER_ORACLE_IP="127.0.0.1"
export KERASTUNER_ORACLE_PORT="8000"
python run_tuning.py
```
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


tuner = kt.RandomSearch(
    build_model,
    objective="val_loss",
    max_trials=1,
    directory="result_dir",
    project_name="helloworld",
)

tuner.search(x_train, y_train, epochs=1, validation_data=(x_test, y_test))
