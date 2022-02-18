"""shell
pip install -r https://raw.githubusercontent.com/datamllab/automl-in-action-notebooks/master/requirements.txt
"""

import tensorflow as tf
import autokeras as ak

"""
## Load Cifar10 dataset
"""

from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("Training image shape:", x_train.shape)  # (60000, 28, 28)
print("Training label shape:", y_train.shape)  # (60000,)
print("First five training labels:", y_train[:5])  # array([5 0 4 1 9], dtype=uint8)
"""invisible
"""

airplane_automobile_indices_train = (y_train[:, 0] == 0) | (y_train[:, 0] == 1)
airplane_automobile_indices_test = (y_test[:, 0] == 0) | (y_test[:, 0] == 1)
x_train, y_train = (
    x_train[airplane_automobile_indices_train],
    y_train[airplane_automobile_indices_train],
)
x_test, y_test = (
    x_test[airplane_automobile_indices_test],
    y_test[airplane_automobile_indices_test],
)
print("Training image shape:", x_train.shape)  # (60000, 28, 28)
print("Training label shape:", y_train.shape)  # (60000,)
print("First five training labels:", y_train[:5])  # array([5 0 4 1 9], dtype=uint8)
"""invisible
"""

# plot first few images
from matplotlib import pyplot as plt

for i in range(9):
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    plt.imshow(x_train[i])
# show the figure
plt.show()

"""
## Jointly selecting image augmentation and normalization methods for ResNet models
(ImageBlock)
"""

input_node = ak.ImageInput()
output_node = ak.Normalization()(input_node)
output_node = ak.ImageAugmentation()(output_node)  # horizontal_flip=False
output_node = ak.ResNetBlock(version="v2")(output_node)
output_node = ak.ClassificationHead(dropout=0.0)(output_node)
clf = ak.AutoModel(
    inputs=input_node, outputs=output_node, overwrite=True, max_trials=10
)
clf.fit(x_train, y_train, epochs=10)

"""invisible
"""
import autokeras as ak
import timeit

input_node = ak.ImageInput()
output_node = ak.ImageBlock(
    # do not specify if we want to use normalization and let it to search automatically
    normlize=None,
    # do not specify if we want to use adata ugmentation method and let it to search automatically
    augment=None,
    # Only search resnet architectures.
    block_type="resnet",
)(input_node)
output_node = ak.ClassificationHead(dropout=0.0)(output_node)

auto_model = ak.AutoModel(
    inputs=input_node, outputs=output_node, max_trials=10, overwrite=True, seed=42
)

start_time = timeit.default_timer()
auto_model.fit(x_train, y_train, epochs=10, batch_size=64)
stop_time = timeit.default_timer()
print("Total time: {time} seconds.".format(time=round(stop_time - start_time, 2)))

"""invisible
"""
auto_model.tuner.results_summary()
"""invisible
"""

best_model = auto_model.export_model()
best_model.summary()
"""invisible
"""

test_loss, test_acc = auto_model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: {accuracy}%".format(accuracy=round(test_acc * 100, 2)))
