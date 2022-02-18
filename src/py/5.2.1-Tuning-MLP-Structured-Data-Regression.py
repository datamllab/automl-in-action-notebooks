"""shell
pip install -r https://raw.githubusercontent.com/datamllab/automl-in-action-notebooks/master/requirements.txt
"""

import tensorflow as tf
import autokeras as ak

"""
### Tuning MLP for structured-data regression  (Normalization + DenseBlock)
"""

input_node = ak.StructuredDataInput()
output_node = ak.Normalization()(input_node)
output_node = ak.DenseBlock(use_batchnorm=False, dropout=0.0)(output_node)
output_node = ak.RegressionHead(dropout=0.0)(output_node)
auto_model = ak.AutoModel(
    inputs=input_node, outputs=output_node, max_trials=10, overwrite=True, seed=42
)

from sklearn.datasets import fetch_california_housing

house_dataset = fetch_california_housing()

# Import pandas package to format the data
import pandas as pd

# Extract features with their names into the a dataframe format
data = pd.DataFrame(house_dataset.data, columns=house_dataset.feature_names)

# Extract target with their names into a pd.Series object with name MEDV
target = pd.Series(house_dataset.target, name="MEDV")

from sklearn.model_selection import train_test_split

train_data, test_data, train_targets, test_targets = train_test_split(
    data, target, test_size=0.2, random_state=42
)

auto_model.fit(train_data, train_targets, batch_size=1024, epochs=150)

"""
### Visualize the best pipeline
"""

best_model = auto_model.export_model()
tf.keras.utils.plot_model(
    best_model, show_shapes=True, expand_nested=True
)  # rankdir='LR'

"""
### Evaluate best pipeline
"""

test_loss, test_acc = auto_model.evaluate(test_data, test_targets, verbose=0)
print("Test accuracy: ", test_acc)

"""
### Show best trial
"""

auto_model.tuner.results_summary(num_trials=1)
best_model = auto_model.export_model()
tf.keras.utils.plot_model(best_model, show_shapes=True, expand_nested=True)

from tensorflow import keras

best_model.save("saved_model")
best_model = keras.models.load_model("saved_model")

"""
### Customize the search space for tuning MLP
"""

from keras_tuner.engine import hyperparameters as hp

input_node = ak.StructuredDataInput()
output_node = ak.Normalization()(input_node)
output_node = ak.DenseBlock(
    num_layers=1,
    num_units=hp.Choice("num_units", [128, 256, 512, 1024]),
    use_batchnorm=False,
    dropout=0.0,
)(output_node)
output_node = ak.DenseBlock(
    num_layers=1,
    num_units=hp.Choice("num_units", [16, 32, 64]),
    use_batchnorm=False,
    dropout=0.0,
)(output_node)
output_node = ak.RegressionHead()(output_node)
auto_model = ak.AutoModel(
    inputs=input_node, outputs=output_node, max_trials=10, overwrite=True, seed=42
)

auto_model.fit(train_data, train_targets, batch_size=1024, epochs=150)


"""
### Display the best pipeline
"""

best_model = auto_model.export_model()
tf.keras.utils.plot_model(
    best_model, show_shapes=True, expand_nested=True
)  # rankdir='LR'

test_loss, test_acc = auto_model.evaluate(test_data, test_targets, verbose=0)
print("Test accuracy: ", test_acc)

auto_model.tuner.results_summary(num_trials=1)

best_model.summary()
