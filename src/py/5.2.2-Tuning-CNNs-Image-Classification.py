"""shell
pip install -r https://raw.githubusercontent.com/datamllab/automl-in-action-notebooks/master/requirements.txt
"""

import tensorflow as tf
import autokeras as ak

"""
### Load MNIST dataset
"""

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Training image shape:", x_train.shape)  # (60000, 28, 28)
print("Training label shape:", y_train.shape)  # (60000,)
print("First five training labels:", y_train[:5])  # array([5 0 4 1 9], dtype=uint8)

"""
### Run the ImageClassifier
"""

from kerastuner.engine import hyperparameters as hp


input_node = ak.ImageInput()
output_node = ak.Normalization()(input_node)
output_node = ak.ConvBlock(
    num_blocks=2, max_pooling=True, separable=False, dropout=0.0
)(output_node)
output_node = ak.ClassificationHead(dropout=0.0)(output_node)

auto_model = ak.AutoModel(
    inputs=input_node, outputs=output_node, max_trials=10, overwrite=True, seed=42
)

# Use the first 100 training samples as a quick demo.
# You may run with the full dataset, but expect a longer training time.
auto_model.fit(x_train[:100], y_train[:100], epochs=3)
test_loss, test_acc = auto_model.evaluate(x_test, y_test, verbose=0)
print("Test accuracy: ", test_acc)

"""
### Get the summarized results during the tuning process (return the best 10 models if
existed)
"""

auto_model.tuner.results_summary(1)

"""
### Retrieve & Display best model
"""

best_model = auto_model.export_model()
best_model.summary()

tf.keras.utils.plot_model(best_model, show_shapes=True, expand_nested=True)

"""

### Predict with the best model.

"""

predicted_y = auto_model.predict(x_test)
print(predicted_y)
