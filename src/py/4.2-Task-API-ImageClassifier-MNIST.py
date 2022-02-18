"""shell
pip install -r
https://raw.githubusercontent.com/datamllab/automl-in-action-notebooks/master/requirements.txt
https://raw.githubusercontent.com/datamllab/automl-in-action-notebooks/master/requirements.txt
"""

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_visible_devices(gpus[-1], "GPU")

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Training image shape:", x_train.shape)  # (60000, 28, 28)
print("Training label shape:", y_train.shape)  # (60000,)
print("First five training labels:", y_train[:5])  # array([5 0 4 1 9], dtype=uint8)

import autokeras as ak

# Initialize the image classifier.
clf = ak.ImageClassifier(max_trials=2)  # It tries two different models.

# Feed the image classifier with training data
# 20% of the data is used as validation data by default for tuning
# the process may run for a bit long time, please try to use GPU
clf.fit(x_train, y_train, epochs=3)  # each model is trained for three epochs

test_loss, test_acc = clf.evaluate(x_test, y_test, verbose=0)
print("Test accuracy: ", test_acc)

predicted_y = clf.predict(x_test)
print(predicted_y)

best_model = clf.export_model()
best_model.summary()

from tensorflow.keras.models import load_model

best_model.save("model_autokeras")

loaded_model = load_model("model_autokeras")  # , custom_objects=ak.CUSTOM_OBJECTS

predicted_y = loaded_model.predict(tf.expand_dims(x_test, -1))
print(predicted_y)

test_loss, test_acc = clf.evaluate(x_test, y_test, verbose=0)
print("Test accuracy: ", test_acc)

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

clf = ak.ImageClassifier(
    max_trials=2,
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    objective="val_accuracy",
)

clf.fit(
    x_train,
    y_train,
    validation_split=0.15,
    epochs=3,
    verbose=2,
)

import keras_tuner


def my_metric(y_true, y_pred):
    correct_labels = tf.cast(y_true == y_pred, tf.float32)
    return tf.reduce_mean(correct_labels, axis=-1)


clf = ak.ImageClassifier(
    seed=42,
    max_trials=2,
    loss="categorical_crossentropy",
    # Wrap the function into a Keras Tuner Objective
    # and pass it to AutoKeras.
    # Direction can be 'min' or 'max'
    # meaning we want to minimize or maximize the metric.
    # 'val_my_metric' is just add a 'val_' prefix
    # to the function name or the metric name.
    objective=keras_tuner.Objective("val_my_metric", direction="max"),
    # Include it as one of the metrics.
    metrics=[my_metric],
)

clf.fit(x_train, y_train, validation_split=0.15, epochs=3)
