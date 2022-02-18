"""shell
!pip install -r https://raw.githubusercontent.com/datamllab/automl-in-action-notebooks/master/requirements.txt
"""

"""
### Load MNIST dataset
"""

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Training image shape:", x_train.shape)  # (60000, 28, 28)
print("Training label shape:", y_train.shape)  # (60000,)
print("First five training labels:", y_train[:5])  # array([5 0 4 1 9], dtype=uint8)

"""
### IO API for image classification
"""

import autokeras as ak

# Initialize the IO API.
io_model = ak.AutoModel(
    inputs=ak.ImageInput(),
    outputs=ak.ClassificationHead(
        loss="categorical_crossentropy", metrics=["accuracy"]
    ),
    objective="val_loss",
    tuner="random",
    max_trials=3,
    overwrite=True,
)

# Fit the model with prepared data.
# Use the first 100 training samples for 1 epoch as a quick demo.
# You may run with the full dataset with 10 epochs, but expect a longer training time.
io_model.fit(x_train[:100], y_train[:100], epochs=1)

"""
### Get the summarized results during the tuning process (return the best 10 models if
existed)
"""

io_model.tuner.results_summary()

"""
### Retrieve best model
"""

best_model = io_model.export_model()
best_model.summary()

"""

### Predict with the best model.
"""

predicted_y = io_model.predict(x_test[:100])
print(predicted_y)

"""
### Evaluate the best model on the test data.
"""

test_loss, test_acc = io_model.evaluate(x_test[:100], y_test[:100])
print("Test accuracy: ", test_acc)
