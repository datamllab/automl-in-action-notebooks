"""shell
pip install -r https://raw.githubusercontent.com/datamllab/automl-in-action-notebooks/master/requirements.txt
"""

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import keras_tuner as kt


def build_model(hp):
    model_type = hp.Choice("model_type", ["svm", "random_forest", "mlp"], default="mlp")
    if model_type == "svm":
        with hp.conditional_scope("model_type", "svm"):
            model = SVC(
                C=hp.Float("C", 1e-3, 10, sampling="linear", default=1),
                kernel=hp.Choice("kernel_type", ["linear", "rbf"], default="linear"),
                random_state=42,
            )
    elif model_type == "random_forest":
        with hp.conditional_scope("model_type", "random_forest"):
            model = RandomForestClassifier(
                n_estimators=hp.Int("n_estimators", 10, 200, step=10),
                max_depth=hp.Int("max_depth", 3, 10),
            )
    elif model_type == "mlp":
        with hp.conditional_scope("model_type", "mlp"):
            model = tf.keras.Sequential()
            model.add(tf.keras.Input(shape=(64,)))
            for i in range(hp.Int("num_layers", min_value=1, max_value=4)):
                model.add(
                    tf.keras.layers.Dense(
                        hp.Int(
                            "units_{i}".format(i=i),
                            min_value=32,
                            max_value=128,
                            step=32,
                        ),
                        activation="relu",
                    )
                )
            model.add(tf.keras.layers.Dense(10, activation="softmax"))
            model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    else:
        raise ValueError("Unrecognized model_type")
    return model


"""invisible
"""
import pickle
import os
import tensorflow as tf


class ShallowDeepTuner(kt.Tuner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trial_id_to_type = {}

    def run_trial(self, trial, x, y, validation_data, epochs=None, **fit_kwargs):
        model = self.hypermodel.build(trial.hyperparameters)
        x_val, y_val = validation_data  # get the validation data
        if isinstance(model, tf.keras.Model):
            model.fit(
                x,
                y,
                validation_data=validation_data,
                batch_size=trial.hyperparameters.Choice("batch_size", [16, 32]),
                epochs=epochs,
                **fit_kwargs
            )
            accuracy = {
                name: value
                for name, value in zip(
                    model.metrics_names, model.evaluate(x_val, y_val)
                )
            }["accuracy"]
            self.trial_id_to_type[trial.trial_id] = "keras"
        else:
            model = self.hypermodel.build(trial.hyperparameters)  # build the model
            model.fit(x, y)  # fit the model
            accuracy = model.score(x_val, y_val)
            self.trial_id_to_type[trial.trial_id] = "sklearn"
        self.save_model(trial.trial_id, model)  # save the model to disk
        return {"accuracy": accuracy}

    def save_model(self, trial_id, model):
        fname = os.path.join(self.get_trial_dir(trial_id), "model")
        if isinstance(model, tf.keras.Model):
            model.save(fname)
        else:
            with tf.io.gfile.GFile(fname, "wb") as f:
                pickle.dump(model, f)

    def load_model(self, trial):
        fname = os.path.join(self.get_trial_dir(trial.trial_id), "model")
        if self.trial_id_to_type[trial.trial_id] == "keras":
            model = tf.keras.models.load_model(fname)
        else:
            with tf.io.gfile.GFile(fname, "rb") as f:
                model = pickle.load(f)
        return model


"""invisible
"""

from sklearn.datasets import load_digits

# Load the hand-written digits dataset
digits = load_digits()

# Get the images and corresponding labels
images, labels = digits.images, digits.target
images.shape, labels.shape

# reshape images to vectors
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))

# Split data into train and test subsets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, shuffle=False
)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, shuffle=False
)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_train[:10])

"""invisible
"""
random_tuner = ShallowDeepTuner(
    oracle=kt.oracles.RandomSearch(
        objective=kt.Objective("accuracy", "max"), max_trials=30, seed=42
    ),
    hypermodel=build_model,
    overwrite=True,
    project_name="random_tuner",
)

random_tuner.search(x_train, y_train, validation_data=(x_val, y_val), epochs=10)

"""invisible
"""
random_tuner.search_space_summary()
random_tuner.results_summary(30)

"""invisible
"""
best_model = random_tuner.get_best_models(1)[0]
print(type(best_model))

"""invisible
"""
# Evaluate the best discovered model
from sklearn.metrics import accuracy_score

y_pred_test = best_model.predict(x_test)
test_acc = accuracy_score(y_test, y_pred_test)
print("The prediction accuracy on test set: {:.2f} %".format(test_acc * 100))
