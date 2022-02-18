"""shell
!pip install -r https://raw.githubusercontent.com/datamllab/automl-in-action-notebooks/master/requirements.txt
"""

import tensorflow as tf

tf.random.set_seed(42)

"""
## Load data
"""

# Import the dataset loading function from sklearn
from sklearn.datasets import fetch_california_housing

# Load the California housing dataset
house_dataset = fetch_california_housing()

# Display the oringal data
house_dataset.keys()

# Import pandas package to format the data
import pandas as pd

# Extract features with their names into the a dataframe format
data = pd.DataFrame(house_dataset.data, columns=house_dataset.feature_names)

# Extract target with their names into a pd.Series object with name MEDV
target = pd.Series(house_dataset.target, name="MEDV")

# Visualize the first 5 samples of the data
data.head(5)

# Split data into training and test dataset
from sklearn.model_selection import train_test_split

train_data, test_data, train_targets, test_targets = train_test_split(
    data, target, test_size=0.2, random_state=42
)

# Check the shape of whole dataset and the splited training and test set
print("--Shape of the whole data--\n {}".format(data.shape))
print("\n--Shape of the target vector--\n {}".format(target.shape))
print("\n--Shape of the training data--\n {}".format(train_data.shape))
print("\n--Shape of the testing data--\n {}".format(test_data.shape))

train_data.shape, test_data.shape

train_data

"""
## Data Preparation: normalization
"""


def norm(x, mean, std):
    return (x - mean) / std


mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

normed_train_data = norm(train_data, mean, std)
normed_test_data = norm(test_data, mean, std)

"""
## Build up an MLP
"""

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential(
    [
        layers.Dense(64, activation="relu", input_shape=[8]),
        layers.Dense(64, activation="relu"),
        layers.Dense(1),
    ]
)

# Now try out the model. Take a batch of 5 examples from the training data and call model.predict on it.
example_batch = normed_train_data[:5]
example_result = model.predict(example_batch)
example_result

model.summary()

# Customize the optimizer configuration (learning rate here)
optimizer = tf.keras.optimizers.RMSprop(0.01)
model.compile(loss="mse", optimizer=optimizer, metrics=["mae", "mse"])

"""
### Train & test the model
"""

model.fit(normed_train_data, train_targets, epochs=300, batch_size=1024, verbose=1)

loss, mae, mse = model.evaluate(normed_test_data, test_targets, verbose=0)
mse

"""
### Tune the number of epochs
"""

# Train the model (in silent mode, verbose=0)


def build_model():
    model = keras.Sequential(
        [
            layers.Dense(
                64, activation="relu", input_shape=[normed_train_data.shape[1]]
            ),
            layers.Dense(64, activation="relu"),
            layers.Dense(1),
        ]
    )
    optimizer = tf.keras.optimizers.RMSprop(0.01)
    model.compile(loss="mse", optimizer=optimizer, metrics=["mae", "mse"])
    return model


model = build_model()

EPOCHS = 500
history = model.fit(
    normed_train_data,
    train_targets,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=1024,
    verbose=1,
)

import pandas as pd

hist = pd.DataFrame(history.history)
hist["epoch"] = history.epoch
hist.tail()

import matplotlib.pyplot as plt

plt.plot(hist["epoch"], hist["mse"], label="train mse")
plt.plot(hist["epoch"], hist["val_mse"], label="val mse")
plt.xlabel("Epochs")
plt.ylabel("MSE")
# Set a title of the current axes.
plt.title("Training and Validation MSE by Epoch")
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()

import numpy as np


def smooth_curve(values, std=5):
    # gaussian smoothing: Smooths a list of values by convolving with a gussian.
    width = std * 4
    x = np.linspace(-width, width, 2 * width + 1)
    kernel = np.exp(-((x / 5) ** 2))

    values = np.array(values)
    weights = np.ones_like(values)

    smoothed_values = np.convolve(values, kernel, mode="same")
    smoothed_weights = np.convolve(weights, kernel, mode="same")

    return smoothed_values / smoothed_weights


import matplotlib.pyplot as plt

plt.plot(hist["epoch"], smooth_curve(hist["mse"]), label="train mse")
plt.plot(hist["epoch"], smooth_curve(hist["val_mse"]), label="val mse")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.ylim((0, 0.5))
# Set a title of the current axes.
plt.title("Training and Validation MSE by Epoch (smoothed)")
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()

"""
## Final fit model with full data & test model
"""

model = build_model()
model.fit(normed_train_data, train_targets, epochs=150, batch_size=1024, verbose=1)

loss, mae, mse = model.evaluate(normed_test_data, test_targets, verbose=0)
mse

test_predictions = model.predict(normed_test_data).flatten()

a = plt.axes(aspect="equal")
plt.scatter(test_targets, test_predictions)
plt.xlabel("True Values [MEDV]")
plt.ylabel("Predictions [MEDV]")
lims = [0, 5]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
