"""shell
!pip install -r https://raw.githubusercontent.com/datamllab/automl-in-action-notebooks/master/requirements.txt
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
clf_target_train, clf_target_test = (
    classification_target[:80],
    classification_target[80:],
)

# Generate regression targets.
regression_target = np.random.rand(num_instances, 1).astype(np.float32)
reg_target_train, reg_target_test = regression_target[:80], regression_target[80:]

"""
### IO API for multi-task learning
"""

import autokeras as ak

# Initialize the IO API.
multi_output_learner = ak.AutoModel(
    inputs=[ak.ImageInput(), ak.StructuredDataInput()],
    outputs=[ak.ClassificationHead(), ak.RegressionHead()],
    max_trials=3,
    #     project_name='io_api_multitask'
)

# Fit the model with prepared data.
multi_output_learner.fit(
    [image_train, structured_train],
    [clf_target_train, reg_target_train],
    epochs=10,
    verbose=2,
)

"""
### Retrieve best model
"""

best_model = multi_output_learner.export_model()
best_model.summary()

tf.keras.utils.plot_model(best_model, show_shapes=True, expand_nested=True)
