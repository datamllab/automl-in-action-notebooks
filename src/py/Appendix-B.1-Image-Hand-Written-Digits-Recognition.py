"""shell
pip install -r https://raw.githubusercontent.com/datamllab/automl-in-action-notebooks/master/requirements.txt
"""

"""
## Load data
"""

# Import the dataset loading function from sklearn
from sklearn.datasets import load_digits

# Load the hand-written digits dataset
digits = load_digits()

# Get the images and corresponding labels
images, labels = digits.images, digits.target
images.shape, labels.shape

import numpy as np

np.max(images), np.max(labels), np.min(images), np.min(labels)

"""
## Exploratory data analysis & Data preprocessing &  feature engineering
"""

"""inline
matplotlib inline
"""
import matplotlib
import matplotlib.pyplot as plt

# plot first 20 images
n = 20
_, axes = plt.subplots(2, 10, figsize=(10, 2))
plt.tight_layout()
for i in range(n):
    row, col = i // 10, i % 10
    axes[row, col].set_axis_off()
    axes[row, col].imshow(
        images[
            i,
        ],
        cmap=plt.cm.gray_r,
        interpolation="nearest",
    )
    axes[row, col].set_title("Label: %i" % labels[i])

# reshape images to vectors
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
X.shape

# Split data into train and test subsets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, shuffle=False
)

print("Shape of the training data: {}".format(X_train.shape))
print("Shape of the testing data: {}".format(X_test.shape))

X_train.shape, X_test.shape

"""
### PCA
"""

from sklearn.decomposition import PCA

n_components = 10
pca = PCA(n_components=n_components).fit(
    X_train
)  # , svd_solver='randomized',whiten=True
X_train_pca = pca.transform(X_train)

print(X_train.shape)
print(X_train_pca.shape)

X_train.shape, X_train_pca.shape

plt.hist(pca.explained_variance_ratio_, bins=10, log=True)
pca.explained_variance_ratio_.sum()

plt.figure(figsize=(8, 6))
plt.scatter(
    X_train_pca[:, 0],
    X_train_pca[:, 1],
    c=y_train,
    edgecolor="none",
    alpha=0.5,
    cmap=plt.cm.get_cmap("Spectral", 10),
)
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("PCA 2D Embedding")
plt.colorbar()

"""
### TSNE
"""

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)

X_train_tsne = tsne.fit_transform(X_train)

plt.figure(figsize=(8, 6))
plt.scatter(
    X_train_tsne[:, 0],
    X_train_tsne[:, 1],
    c=y_train,
    edgecolor="none",
    alpha=0.5,
    cmap=plt.cm.get_cmap("Spectral", 10),
)
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("t-SNE 2D Embedding")
plt.colorbar()

"""
## Build up SVM classifier
"""

"""
### Training
"""

# Import library for support vector machine classifier
from sklearn.svm import SVC

# Create a support vector classifier
clf = SVC(C=1, kernel="linear", random_state=42)

# Train the model using the training sets
clf.fit(X_train, y_train)


"""
### Testing
"""

from sklearn.metrics import accuracy_score

# Now predict the value of the digit on the test set:
y_pred_test = clf.predict(X_test)


# Display the testing results
acc = accuracy_score(y_test, y_pred_test)
print("The prediction accuracy: {:.2f} %".format(acc * 100))

from sklearn.metrics import plot_confusion_matrix

disp = plot_confusion_matrix(clf, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix (linear SVC)")

plt.show()

"""
### PCA (10 components) + SVC
"""

"""
### Build a pipeline
"""

from sklearn.pipeline import Pipeline

image_clf = Pipeline(
    [
        ("pca", PCA(n_components=10)),
        ("clf", SVC(C=1, kernel="linear", random_state=42)),
    ]
)

image_clf.fit(X_train, y_train)

# Test
y_pred_test = image_clf.predict(X_test)

# Display the testing results
acc = accuracy_score(y_test, y_pred_test)
print("The prediction accuracy: {:.2f} %".format(acc * 100))

"""
## Fine-Tuning: jointly tune the PCA components and SVC
"""

# Hp tuning with Sklearn

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer


# Create a dictionary for all the hyperparameters
hps = {
    "pca__n_components": [2, 5, 10, 20],
    "clf__C": [0.05, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15],
    "clf__kernel": ["linear", "rbf"],
}

# Construct a scoring function for performance estimation.
scoring_fnc = make_scorer(accuracy_score)

# Create the grid search cv object (5-fold cross-validation)
grid_search = GridSearchCV(
    estimator=image_clf, param_grid=hps, scoring=scoring_fnc, cv=3, verbose=5, n_jobs=-1
)

# Fit the grid search object to the training data to search the optimal model
grid_search = grid_search.fit(X_train, y_train)

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

# Model prediction on training & test data
y_pred_train = best_pipeline.predict(X_train)
y_pred_test = best_pipeline.predict(X_test)

# Display the testing results
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)
print("The prediction accuracy on training set: {:.2f} %".format(train_acc * 100))
print("The prediction accuracy on test set: {:.2f} %".format(test_acc * 100))

disp = plot_confusion_matrix(best_pipeline, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix (PCA + RBF SVC)")

plt.show()
