"""shell
!pip install -r https://raw.githubusercontent.com/datamllab/automl-in-action-notebooks/master/requirements.txt
"""

"""
## Load MNIST dataset
"""

from tensorflow.keras.datasets import cifar10, fashion_mnist, mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Training image shape:", x_train.shape)  # (60000, 28, 28)
print("Training label shape:", y_train.shape)  # (60000,)
print("First five training labels:", y_train[:5])  # array([5 0 4 1 9], dtype=uint8)

"""
## ResNetBlock
"""

import timeit
import autokeras as ak

input_node = ak.ImageInput()
output_node = ak.Normalization()(input_node)
output_node = ak.ResNetBlock()(output_node)
output_node = ak.ClassificationHead()(output_node)

resnet_auto_model = ak.AutoModel(
    inputs=input_node, outputs=output_node, max_trials=3, overwrite=True, seed=42
)

start_time = timeit.default_timer()
# Use the first 100 training samples for 1 epoch with batch_size=8 as a quick demo.
# You may run with the full dataset with 10 epochs and a larger batch size, but expect a longer training time.
resnet_auto_model.fit(x_train[:100], y_train[:100], epochs=1, batch_size=8)
stop_time = timeit.default_timer()
print("Total time: {time} seconds.".format(time=round(stop_time - start_time, 2)))

"""
### Get the summarized results during the tuning process
"""

resnet_auto_model.tuner.results_summary()

"""
### Display best model
"""

best_resnet_model = resnet_auto_model.export_model()
best_resnet_model.summary()

"""
### Evaluate the best resnet model on the test data.
"""

# Only evaluating the first 100 samples as a quick demo
test_loss, test_acc = resnet_auto_model.evaluate(
    x_test[:100], y_test[:100], batch_size=8
)
print("Accuracy: {accuracy}%".format(accuracy=round(test_acc * 100, 2)))

"""
## XceptionBlock
"""

import autokeras as ak

input_node = ak.ImageInput()
output_node = ak.Normalization()(input_node)
output_node = ak.XceptionBlock()(output_node)
output_node = ak.ClassificationHead()(output_node)

xception_auto_model = ak.AutoModel(
    inputs=input_node, outputs=output_node, max_trials=3, overwrite=True, seed=42
)

start_time = timeit.default_timer()
# Use the first 100 training samples for 1 epoch with batch_size=8 as a quick demo.
# You may run with the full dataset with 10 epochs and a larger batch size, but expect a longer training time.
xception_auto_model.fit(x_train[:100], y_train[:100], epochs=1, batch_size=8)
stop_time = timeit.default_timer()
print("Total time: {time} seconds.".format(time=round(stop_time - start_time, 2)))

"""
### Display the best xception model
"""

import tensorflow as tf

best_xception_model = xception_auto_model.export_model()
tf.keras.utils.plot_model(
    best_xception_model, show_shapes=True, expand_nested=True
)  # rankdir='LR'

best_xception_model.summary()

"""
### Evaluate the best xception model on the test data.
"""

# Only evaluating the first 100 samples as a quick demo
test_loss, test_acc = resnet_auto_model.evaluate(x_test[:100], y_test[:100])
print("Accuracy: {accuracy}%".format(accuracy=round(test_acc * 100, 2)))

"""
## HyperBlock for image classification (ImageBlock)
"""

import timeit
import autokeras as ak

input_node = ak.ImageInput()
output_node = ak.ImageBlock(
    # Normalize the dataset.
    normalize=True,
    # Do not do data augmentation.
    augment=False,
)(input_node)
output_node = ak.ClassificationHead(dropout=0.0)(output_node)

auto_model = ak.AutoModel(
    inputs=input_node, outputs=output_node, max_trials=3, overwrite=True, seed=42
)

start_time = timeit.default_timer()
# Use the first 100 training samples for 1 epoch and batch_size=8 as a quick demo.
# You may run with the full dataset with 10 epochs with a larger batch size, but expect a longer training time.
auto_model.fit(x_train[:100], y_train[:100], epochs=1, batch_size=8)
stop_time = timeit.default_timer()
print("Total time: {time} seconds.".format(time=round(stop_time - start_time, 2)))

auto_model.tuner.results_summary(num_trials=1)

best_model = auto_model.export_model()
best_model.summary()

# Only evaluating the first 100 samples as a quick demo
test_loss, test_acc = auto_model.evaluate(x_test[:100], y_test[:100], batch_size=8)
print("Accuracy: {accuracy}%".format(accuracy=round(test_acc * 100, 2)))
