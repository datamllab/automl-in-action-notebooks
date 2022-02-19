"""shell
pip install -r https://raw.githubusercontent.com/datamllab/automl-in-action-notebooks/master/requirements.txt
"""

import keras_tuner as kt


class MyTuner(kt.RandomSearch):
    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        x = hp.Float("x", -1.0, 1.0)
        return x * x + 1


tuner = MyTuner(max_trials=20)
tuner.search()
tuner.results_summary()

"""invisible
"""
import os
import pickle
import tensorflow as tf
import kerastuner as kt


class ShallowTuner(kt.Tuner):
    def __init__(self, oracle, hypermodel, **kwargs):
        super(ShallowTuner, self).__init__(
            oracle=oracle, hypermodel=hypermodel, **kwargs
        )

    def search(self, X, y, validation_data):
        """performs hyperparameter search."""
        return super(ShallowTuner, self).search(X, y, validation_data)

    def run_trial(self, trial, X, y, validation_data):
        model = self.hypermodel.build(trial.hyperparameters)  # build the model
        model.fit(X, y)  # fit the model
        X_val, y_val = validation_data  # get the validation data
        eval_score = model.score(X_val, y_val)  # evaluate the model
        self.save_model(trial.trial_id, model)  # save the model to disk
        return {"score": eval_score}

    def save_model(self, trial_id, model, step=0):
        """save the model with pickle"""
        fname = os.path.join(self.get_trial_dir(trial_id), "model.pickle")
        with tf.io.gfile.GFile(fname, "wb") as f:
            pickle.dump(model, f)

    def load_model(self, trial):
        """load the model with pickle"""
        fname = os.path.join(self.get_trial_dir(trial.trial_id), "model.pickle")
        with tf.io.gfile.GFile(fname, "rb") as f:
            return pickle.load(f)


"""invisible
"""

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

"""invisible
"""

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from kerastuner.engine import hyperparameters as hp


def build_model(hp):
    model_type = hp.Choice("model_type", ["svm", "random_forest"])
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
    else:
        raise ValueError("Unrecognized model_type")
    return model


my_sklearn_tuner = ShallowTuner(
    oracle=kt.oracles.RandomSearch(
        objective=kt.Objective("score", "max"), max_trials=10, seed=42
    ),
    hypermodel=build_model,
    overwrite=True,
    project_name="my_sklearn_tuner",
)

my_sklearn_tuner.search(X_train, y_train, validation_data=(X_val, y_val))

"""invisible
"""
# Evaluate the best discovered model
from sklearn.metrics import accuracy_score

best_model = my_sklearn_tuner.get_best_models(1)[0]
best_model.fit(X_train, y_train)
y_pred_test = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred_test)
print("The prediction accuracy on test set: {:.2f} %".format(test_acc * 100))
