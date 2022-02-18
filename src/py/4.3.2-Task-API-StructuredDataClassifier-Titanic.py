"""shell
!pip install -r https://raw.githubusercontent.com/datamllab/automl-in-action-notebooks/master/requirements.txt
"""

import tensorflow as tf
import autokeras as ak

"""
## Titanic data downloaded with csv files
"""

"""
### Download training and testing csv files
"""

import tensorflow as tf

TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

"""
### Run `StructuredDataClassifier` API
"""

import autokeras as ak

# Initialize the structured data classifier.
clf = ak.StructuredDataClassifier(max_trials=10)  # Try 10 different pipelines.

# Feed the structured data classifier with training data.
clf.fit(
    # The path to the train.csv file.
    train_file_path,
    # The name of the label column.
    "survived",
    verbose=2,
)


clf = ak.StructuredDataClassifier(
    column_names=[
        "sex",
        "age",
        "n_siblings_spouses",
        "parch",
        "fare",
        "class",
        "deck",
        "embark_town",
        "alone",
    ],
    column_types={"sex": "categorical", "fare": "numerical"},
    max_trials=10,
)

clf.fit(
    train_file_path,
    "survived",
    verbose=2,
)

"""
### Predict with the best model.
"""

predicted_y = clf.predict(test_file_path)
print(predicted_y[:5])

"""
### Evaluate the best pipeline with the testing csv file.
"""

test_loss, test_acc = clf.evaluate(test_file_path, "survived", verbose=0)
print("Test accuracy: ", test_acc)
