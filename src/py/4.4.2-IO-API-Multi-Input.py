"""shell
pip install -r https://raw.githubusercontent.com/datamllab/automl-in-action-notebooks/master/requirements.txt
"""

import tensorflow as tf

"""
### Create synthetic image - attributes dataset
"""

import numpy as np

num_instances = 100

# Generate image data.
image_data = np.random.rand(num_instances, 32, 32, 3).astype(np.float32)
image_train, image_test = image_data[:80], image_data[80:]

# Generate structured data.
structured_data = np.random.rand(num_instances, 20).astype(np.float32)
structured_train, structured_test = structured_data[:80], structured_data[80:]

# Generate classification labels of five classes.
classification_target = np.random.randint(5, size=num_instances)
target_train, target_test = classification_target[:80], classification_target[80:]

"""
### IO API for multi-input learning
"""

import autokeras as ak

# Initialize the IO API.
multi_input_learner = ak.AutoModel(
    inputs=[ak.ImageInput(), ak.StructuredDataInput()],
    outputs=ak.ClassificationHead(),
    max_trials=3,
    #     project_name='io_api_multimodal',
)

# Fit the model with prepared data.
multi_input_learner.fit(
    [image_train, structured_train], target_train, epochs=10, verbose=2
)

"""
### Evaluate the best model on the test data.
"""

test_loss, test_acc = multi_input_learner.evaluate(
    [image_test, structured_test], target_test, verbose=0
)
print("Test accuracy: ", test_acc)
