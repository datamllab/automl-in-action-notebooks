"""shell
pip install -r https://raw.githubusercontent.com/datamllab/automl-in-action-notebooks/master/requirements.txt
"""

"""
### Load the California housing price prediction dataset
"""

from sklearn.datasets import fetch_california_housing

house_dataset = fetch_california_housing()

# Import pandas package to format the data
import pandas as pd

# Extract features with their names into the a dataframe format
data = pd.DataFrame(house_dataset.data, columns=house_dataset.feature_names)

# Extract target with their names into a pd.Series object with name MEDV
target = pd.Series(house_dataset.target, name="MEDV")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, shuffle=False
)


"""
### Use LightGBM GBDT model to do regression
"""

import lightgbm as lgb
from sklearn.metrics import mean_squared_error

gbdt_model = lgb.LGBMRegressor(
    boosting_type="gbdt", num_leaves=31, learning_rate=0.05, n_estimators=10
)  # create model

validation_data = (X_val, y_val)
gbdt_model.fit(
    X_train,
    y_train,
    eval_set=[validation_data],
    eval_metric="mse",
    early_stopping_rounds=5,
)  # fit the model

# evalaute model
y_pred_gbdt = gbdt_model.predict(X_test, num_iteration=gbdt_model.best_iteration_)
test_mse_1 = mean_squared_error(y_test, y_pred_gbdt)
print("The GBDT prediction MSE on test set: {}".format(test_mse_1))

# save, load, and evaluate the model
fname = "gbdt_model.txt"
gbdt_model.booster_.save_model(fname, num_iteration=gbdt_model.best_iteration_)

gbdt_model_2 = lgb.Booster(model_file=fname)
gbdt_model_2.predict(X_test)
test_mse_2 = mean_squared_error(y_test, y_pred_gbdt)
print("The reloaded GBDT prediction MSE on test set: {}".format(test_mse_2))


"""
### Create the LightGBM model building function
"""


def build_model(hp):
    model = lgb.LGBMRegressor(
        boosting_type="gbdt",
        num_leaves=hp.Choice("num_leaves", [15, 31, 63], default=31),
        learning_rate=hp.Float("learning_rate", 1e-3, 10, sampling="log", default=0.05),
        n_estimators=hp.Int("n_estimators", 10, 200, step=10),
    )

    return model


"""
### Customize the LightGBM tuner
"""

import os
import pickle
import tensorflow as tf
import kerastuner as kt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error


class LightGBMTuner(kt.Tuner):
    def run_trial(self, trial, X, y, validation_data):
        model = self.hypermodel.build(trial.hyperparameters)  # build the model
        model.fit(
            X_train,
            y_train,
            eval_set=[validation_data],
            eval_metric="mse",
            early_stopping_rounds=5,
        )  # fit the model
        X_val, y_val = validation_data
        y_pred = model.predict(
            X_val, num_iteration=model.best_iteration_
        )  # evaluate the model
        eval_mse = mean_squared_error(y_val, y_pred)
        self.save_model(trial.trial_id, model)  # save the model to disk
        return {"mse": eval_mse}

    def save_model(self, trial_id, model, step=0):
        fname = os.path.join(self.get_trial_dir(trial_id), "model.txt")
        model.booster_.save_model(fname, num_iteration=model.best_iteration_)

    def load_model(self, trial):
        fname = os.path.join(self.get_trial_dir(trial.trial_id), "model.txt")
        model = lgb.Booster(model_file=fname)
        return model


"""
### Run the tuner to select a LightGBM models for the housing price prediction
"""

my_lightgbm_tuner = LightGBMTuner(
    oracle=kt.oracles.RandomSearch(
        objective=kt.Objective("mse", "min"), max_trials=10, seed=42
    ),
    hypermodel=build_model,
    overwrite=True,
    project_name="my_lightgbm_tuner",
)

my_lightgbm_tuner.search(X_train, y_train, validation_data=(X_val, y_val))

"""
### Evaluate the best discovered model
"""

from sklearn.metrics import mean_squared_error

best_model = my_lightgbm_tuner.get_best_models(1)[0]
y_pred_test = best_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred_test)
print("The prediction MSE on test set: {}".format(test_mse))


"""
### Check the best model
"""

my_lightgbm_tuner.get_best_models(1)
"""invisible
"""

my_lightgbm_tuner.results_summary(1)
