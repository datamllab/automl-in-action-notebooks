"""shell
pip install -r https://raw.githubusercontent.com/datamllab/automl-in-action-notebooks/master/requirements.txt
"""

import tensorflow as tf
import numpy as np

model = tf.keras.applications.DenseNet121(include_top=False, weights=None)
print(model(np.random.rand(100, 32, 32, 3)).shape)

"""
### Build an AutoML block to select among different DenseNet models
"""

import autokeras as ak
import tensorflow as tf


class DenseNetBlock(ak.Block):
    def build(self, hp, inputs):
        version = hp.Choice("version", ["DenseNet121", "DenseNet169", "DenseNet201"])
        if version == "DenseNet121":
            dense_net_func = tf.keras.applications.DenseNet121
        elif version == "DenseNet169":
            dense_net_func = tf.keras.applications.DenseNet169
        elif version == "DenseNet201":
            dense_net_func = tf.keras.applications.DenseNet201
        return dense_net_func(include_top=False, weights=None)(inputs)


"""
### Build a HyperBlock to select between DenseNet and ResNet
"""

# Model selection block
class SelectionBlock(ak.Block):
    def build(self, hp, inputs):
        if hp.Choice("model_type", ["densenet", "resnet"]) == "densenet":
            outputs = DenseNetBlock().build(hp, inputs)
        else:
            outputs = ak.ResNetBlock().build(hp, inputs)
        return outputs


"""invisible
"""

# Model selection block with conditional scope
class SelectionBlock(ak.Block):
    def build(self, hp, inputs):
        if hp.Choice("model_type", ["densenet", "resnet"]) == "densenet":
            with hp.conditional_scope("model_type", ["densenet"]):
                outputs = DenseNetBlock().build(hp, inputs)
        else:
            with hp.conditional_scope("model_type", ["resnet"]):
                outputs = ak.ResNetBlock().build(hp, inputs)
        return outputs


"""
### Build model with the customized HyperBlock and conduct search
"""

input_node = ak.ImageInput()
output_node = SelectionBlock()(input_node)
output_node = ak.ClassificationHead()(output_node)
auto_model = ak.AutoModel(input_node, output_node, max_trials=5, overwrite=True)

from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
auto_model.fit(x_train[:100], y_train[:100], epochs=1)

"""invisible
"""
auto_model.tuner.search_space_summary()
