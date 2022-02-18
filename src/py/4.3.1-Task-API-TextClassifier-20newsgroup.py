"""shell
!pip install -r https://raw.githubusercontent.com/datamllab/automl-in-action-notebooks/master/requirements.txt
"""

import tensorflow as tf
import autokeras as ak

gpus = tf.config.experimental.list_physical_devices("GPU")
print(gpus)

"""
### Load 20newsgroup dataset
"""

import numpy as np
from sklearn.datasets import fetch_20newsgroups

categories = ["rec.autos", "rec.motorcycles"]

news_train = fetch_20newsgroups(
    subset="train", shuffle=True, random_state=42, categories=categories
)
news_test = fetch_20newsgroups(
    subset="test", shuffle=True, random_state=42, categories=categories
)

doc_train, label_train = np.array(news_train.data), np.array(news_train.target)
doc_test, label_test = np.array(news_test.data), np.array(news_test.target)

print(
    "Unique labels {}. \nNumber of unique labels: {}.\n\n".format(
        np.unique(label_train), len(np.unique(label_train))
    )
)

print("The number of documents for training: {}.".format(len(doc_train)))
print("The number of documents for testing: {}.".format(len(doc_test)))

type(doc_train[0]), doc_train[0]

"""
### Run the TextClassifier
"""

# Initialize the text classifier.
clf = ak.TextClassifier(
    max_trials=2, overwrite=True
)  # It tries 3 different models. overwrite the preious history

# Feed the text classifier with training data.
clf.fit(doc_train, label_train, verbose=2)

test_loss, test_acc = clf.evaluate(doc_test, label_test, verbose=0)
print("Test accuracy: ", test_acc)
