"""shell
!pip install -r https://raw.githubusercontent.com/datamllab/automl-in-action-notebooks/master/requirements.txt
"""

import copy
import os

import tensorflow as tf
import keras_tuner as kt


class DeepTuner(kt.Tuner):
    def run_trial(self, trial, X, y, validation_data, **fit_kwargs):
        model = self.hypermodel.build(trial.hyperparameters)

        model.fit(
            X,
            y,
            batch_size=trial.hyperparameters.Choice("batch_size", [16, 32]),
            **fit_kwargs
        )

        X_val, y_val = validation_data  # get the validation data
        eval_scores = model.evaluate(X_val, y_val)
        self.save_model(trial.trial_id, model)  # save the model to disk
        return {
            name: value for name, value in zip(model.metrics_names, eval_scores)
        }  # inform the oracle of the eval result, the result is a dictionary with the metric names as the keys.

    def save_model(self, trial_id, model, step=0):
        fname = os.path.join(self.get_trial_dir(trial_id), "model")
        model.save(fname)

    def load_model(self, trial):
        fname = os.path.join(self.get_trial_dir(trial.trial_id), "model")
        model = tf.keras.models.load_model(fname)
        return model


from sklearn.datasets import load_digits

# Load the hand-written digits dataset
digits = load_digits()

# Get the images and corresponding labels
images, labels = digits.images, digits.target
images.shape, labels.shape

# reshape images to vectors
n_samples = len(images)
X = images.reshape((n_samples, -1))

# Split data into train and test subsets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, shuffle=False
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, shuffle=False
)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_train[:10])

import kerastuner as kt


def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(64,)))
    for i in range(hp.Int("num_layers", min_value=1, max_value=4)):
        model.add(
            tf.keras.layers.Dense(
                hp.Int("units_{i}".format(i=i), min_value=32, max_value=128, step=32),
                activation="relu",
            )
        )
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


random_tuner = DeepTuner(
    oracle=kt.oracles.RandomSearch(
        objective=kt.Objective("accuracy", "max"), max_trials=10, seed=42
    ),
    hypermodel=build_model,
    overwrite=True,
    project_name="random_tuner",
)

random_tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=10)
random_tuner.search_space_summary()

best_model = random_tuner.get_best_models(1)[0]
y_pred_test = best_model.evaluate(X_test, y_test)
print(best_model.metrics_names)
print(y_pred_test)
