"""shell
pip install -r https://raw.githubusercontent.com/datamllab/automl-in-action-notebooks/master/requirements.txt
"""
"""
## Load data
"""

from sklearn.datasets import fetch_20newsgroups

news_train = fetch_20newsgroups(subset="train", shuffle=True, random_state=42)
news_test = fetch_20newsgroups(subset="test", shuffle=True, random_state=42)

"""
## Exploratory data analysis
"""

doc_train, label_train = news_train.data, news_train.target
doc_test, label_test = news_test.data, news_test.target

print("The number of documents for training: {}.".format(len(doc_train)))
print("The number of documents for testing: {}.\n".format(len(doc_test)))

import numpy as np

print(
    "Unique labels {}. \nNumber of unique labels: {}.\n\n".format(
        np.unique(label_train), len(np.unique(label_train))
    )
)

print(type(doc_train[0]))
print("\nThe first training document:\n\n{}".format(doc_train[0]))

"""
## Data preprocessing &  feature engineering
"""

# Tokenization
from sklearn.feature_extraction.text import CountVectorizer

count_vec = CountVectorizer()
X_train_counts = count_vec.fit_transform(doc_train)
X_train_counts.shape

"""invisible
"""
# TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

"""
## Build up a logistic regression classfier & a Naive Bayes classifier
"""

"""
### logistic regression classfier 
"""

# Train
from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(multi_class="ovr", random_state=42)
lr_clf.fit(X_train_tfidf, label_train)

"""invisible
"""
# Test
from sklearn.metrics import accuracy_score

X_test_counts = count_vec.transform(doc_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
label_pred_test = lr_clf.predict(X_test_tfidf)

lr_acc = accuracy_score(label_test, label_pred_test)
print("Test accuracy: {:.2f} %".format(lr_acc * 100))

from sklearn.metrics import plot_confusion_matrix

# Display the testing results
"""inline
matplotlib inline
"""
import matplotlib.pyplot as plt

disp = plot_confusion_matrix(lr_clf, X_test_tfidf, label_test)
disp.figure_.suptitle("Confusion Matrix")
disp.figure_.set_size_inches(20, 10)

plt.show()

"""
### Naive Bayes classifier
"""

# Train
from sklearn.naive_bayes import MultinomialNB

nb_clf = MultinomialNB().fit(X_train_tfidf, label_train)

"""invisible
"""
# Test
X_test_counts = count_vec.transform(doc_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
label_pred_test = nb_clf.predict(X_test_tfidf)

lr_acc = accuracy_score(label_test, label_pred_test)
print("Test accuracy: {:.2f} %".format(lr_acc * 100))

# Display the testing results
disp = plot_confusion_matrix(nb_clf, X_test_tfidf, label_test)
disp.figure_.suptitle("Confusion Matrix")
disp.figure_.set_size_inches(20, 10)

plt.show()

"""
### Build a pipeline
"""

"""
## Fine-Tuning: jointly tune three hyperparameters of the whole pipepline
"""

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline


# Build up the decision tree regressor
text_clf = Pipeline(
    [
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", MultinomialNB()),
    ]
)

# Create a dictionary for all the hyperparameters
hps = {
    "vect__ngram_range": [(1, 1), (1, 2)],
    "tfidf__use_idf": (True, False),
    "clf__alpha": (1, 1e-1, 1e-2),
}

# Transform the performance_metric into a scoring function using 'make_scorer'.
scoring_fnc = make_scorer(accuracy_score)

# Create the grid search cv object (3-fold cross-validation)
grid_search = GridSearchCV(
    estimator=text_clf, param_grid=hps, scoring=scoring_fnc, cv=3, verbose=5, n_jobs=-1
)

# Fit the grid search object to the training data to search the optimal model
grid_search = grid_search.fit(doc_train, label_train)

"""invisible
"""
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)

"""
## Retrive the best model
"""

grid_search.best_params_
best_pipeline = grid_search.best_estimator_

# Produce the value for 'max_depth'
print("The best combination of hyperparameters are:")

for hp_name in sorted(hps.keys()):
    print("%s: %r" % (hp_name, grid_search.best_params_[hp_name]))

best_pipeline.fit(doc_train, label_train)

# Model prediction on training & test data
label_pred_train = best_pipeline.predict(doc_train)
label_pred_test = best_pipeline.predict(doc_test)

# Display the testing results
train_acc = accuracy_score(label_train, label_pred_train)
test_acc = accuracy_score(label_test, label_pred_test)
print("\nThe prediction accuracy on training set: {:.2f} %".format(train_acc * 100))
print("The prediction accuracy on test set: {:.2f} %".format(test_acc * 100))

"""invisible
"""

train_acc = accuracy_score(label_train, label_pred_train)
test_acc = accuracy_score(label_test, label_pred_test)
print("Training Accuracy: {:.2f} %".format(train_acc * 100))
print("Test Accuracy: {:.2f} %".format(test_acc * 100))
