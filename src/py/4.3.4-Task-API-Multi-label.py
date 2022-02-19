"""shell
pip install -r https://raw.githubusercontent.com/datamllab/automl-in-action-notebooks/master/requirements.txt
"""

import tensorflow as tf
import autokeras as ak

"""
### Create synthetic multi-label dataset
"""

from sklearn.datasets import make_multilabel_classification

X, Y = make_multilabel_classification(
    n_samples=100,
    n_features=64,
    n_classes=3,
    n_labels=2,
    allow_unlabeled=False,
    random_state=1,
)
X = X.reshape((100, 8, 8))
X.shape, Y.shape
"""invisible
"""

x_train, x_test, y_train, y_test = X[:80], X[80:], Y[:80], Y[80:]

"""
### Run the ImageClassifier for multi-label classification
"""

# Initialize the image classifier.
clf = ak.ImageClassifier(
    max_trials=10, multi_label=True, overwrite=True
)  # It tries two different pipelines.

# Feed the image classifier with training data
# 20% of the data is used as validation data by default for tuning
# the process may run for a bit long time, please try to use GPU
clf.fit(x_train, y_train, epochs=3, verbose=2)  # each model is trained for three epochs

"""

### Predict with the best model.

"""

predicted_y = clf.predict(x_test)
print("The prediction shape is: {}".format(predicted_y.shape))
print(
    "The predicted labels of the first five instances are:\n {}".format(
        predicted_y[:5, :]
    )
)

"""invisible
"""
test_loss, test_acc = clf.evaluate(x_test, y_test, verbose=0)
print("Test accuracy: ", test_acc)
