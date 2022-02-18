"""shell
!pip install -r https://raw.githubusercontent.com/datamllab/automl-in-action-notebooks/master/requirements.txt
"""

import tensorflow as tf
import autokeras as ak

"""
### Load MNIST dataset
"""

import numpy as np

num_instances = 1000

# Generate image data.
image_data = np.random.rand(num_instances, 32, 32, 3).astype(np.float32)
image_train, image_test = image_data[:800], image_data[800:]

# Generate structured data.
structured_data = np.random.choice(["a", "b", "c", "d", "e"], size=(num_instances, 3))
structured_train, structured_test = structured_data[:800], structured_data[800:]


# Generate classification labels of five classes.
classification_target = np.random.randint(5, size=num_instances)
clf_target_train, clf_target_test = (
    classification_target[:800],
    classification_target[800:],
)

# Generate regression targets.
regression_target = np.random.rand(num_instances, 1).astype(np.float32)
reg_target_train, reg_target_test = regression_target[:800], regression_target[800:]

structured_train[:5]

"""
### Run the ImageClassifier
"""

import autokeras as ak

input_node1 = ak.ImageInput()
branch1 = ak.Normalization()(input_node1)
branch1 = ak.ConvBlock()(branch1)

input_node2 = ak.StructuredDataInput()
branch2 = ak.CategoricalToNumerical()(input_node2)
branch2 = ak.DenseBlock()(branch2)

merge_node = ak.Merge()([branch1, branch2])
output_node1 = ak.ClassificationHead()(merge_node)
output_node2 = ak.RegressionHead()(merge_node)


auto_model = ak.AutoModel(
    inputs=[input_node1, input_node2],
    outputs=[output_node1, output_node2],
    max_trials=3,
    overwrite=True,
    seed=42,
)

auto_model.fit(
    [image_train, structured_train],
    [clf_target_train, reg_target_train],
    epochs=3,
)

"""
### Get the summarized results during the tuning process (return the best 10 models if
existed)
"""

auto_model.tuner.results_summary()

"""
### Retrieve best model
"""

best_model = auto_model.export_model()
best_model.summary()

tf.keras.utils.plot_model(
    best_model, show_shapes=True, expand_nested=True
)  # rankdir='LR'

"""
### Evaluate the best model on the test data.
"""


total_loss, clf_loss, reg_loss, clf_acc, reg_mse = auto_model.evaluate(
    [image_test, structured_test],
    [clf_target_test, reg_target_test],
)
print("\nTotal testing loss: ", total_loss)
print("Classification testing cross-entropy loss: ", clf_loss)
print("Regression testing MSE loss: ", reg_loss)
print("Classification testing accuracy: ", clf_acc)
print("Regression testing MSE: ", reg_mse)
