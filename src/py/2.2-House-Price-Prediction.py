"""shell
pip install -r https://raw.githubusercontent.com/datamllab/automl-in-action-notebooks/master/requirements.txt
"""

"""
## Load data
"""

# Import the dataset loading function from sklearn
from sklearn.datasets import fetch_california_housing

# Load the California housing dataset
house_dataset = fetch_california_housing()

# Display the oringal data
house_dataset.keys()

"""invisible
"""

# Import pandas package to format the data
import pandas as pd

# Extract features with their names into the a dataframe format
data = pd.DataFrame(house_dataset.data, columns=house_dataset.feature_names)

# Extract target with their names into a pd.Series object with name MedPrice
target = pd.Series(house_dataset.target, name="MedPrice")

# Visualize the first 5 samples of the data
data.head(5)

"""
### Split the dataset into training and test set
"""

# Split data into training and test dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=42
)

# Check the shape of whole dataset and the splited training and test set
print("--Shape of the whole data--\n {}".format(data.shape))
print("\n--Shape of the target vector--\n {}".format(target.shape))
print("\n--Shape of the training data--\n {}".format(X_train.shape))
print("\n--Shape of the testing data--\n {}".format(X_test.shape))

"""invisible
"""

(data.shape, target.shape), (X_train.shape, y_train.shape), (X_test.shape, y_test.shape)

"""
## Exploratory data analysis  &  data preprocessing
"""

"""
### Q1: What are the data type of the values in each feature?
"""

data.dtypes
"""invisible
"""

# Check for feature value type
print("-- Feature type --\n{}".format(data.dtypes))
print("\n-- Target type --\n{}".format(target.dtypes))

"""
### Q2: How many distinct values each feature has in the dataset?
"""

# Check for unique feature values
print("\n-- # of unique feature values --\n{}".format(data.nunique()))

"""
### Q3: What are the scale and basic statistics of each feature?
"""

# Viewing the data statistics
pd.options.display.float_format = "{:,.2f}".format
data.describe()

"""
### Q4: Are there missing values contained in the data?
"""

# Copy data to avoid inplace
train_data = X_train.copy()

# Add a column "MedPrice" for the target house price
train_data["MedPrice"] = y_train

# Check if there're missing values
print(
    "\n-- check missing values in training data --\n{}".format(
        train_data.isnull().any()
    )
)
print("\n-- check missing values in test data --\n{}".format(X_test.isnull().any()))

"""
##   Feature engineering: feature selection
"""

# Can we gain some insights by visualizing the distribution of them or correlationship between them?
# Import libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns


# Pretty display for notebooks (only in a Jupyter notebook)
"""inline
matplotlib inline
"""


# Plot the correlation across all the features and the target
plt.figure(figsize=(30, 10))

correlation_matrix = train_data.corr().round(2)
sns.heatmap(
    data=correlation_matrix, square=True, annot=True, cmap="Blues"
)  # fmt='.1f', annot_kws={'size':15},
"""invisible
"""

# Select high correlation features & display the pairplot

selected_feature_set = ["MedInc", "AveRooms"]  # 'PTRATIO', , 'Latitude', 'HouseAge'
sub_train_data = train_data[selected_feature_set + ["MedPrice"]]

# Extract the new training features
X_train = sub_train_data.drop(["MedPrice"], axis=1)

# Select same feature sets for test data
X_test = X_test[selected_feature_set]


sns.pairplot(sub_train_data, height=3.5, plot_kws={"alpha": 0.4})
plt.tight_layout()


"""
## Build up a linear regressor & a decision tree regressor
"""

"""
### Linear regression
"""

# Import library for linear regression
from sklearn.linear_model import LinearRegression

# Training
# Create a Linear regressor
linear_regressor = LinearRegression()

# Train the model using the training sets
linear_regressor.fit(X_train, y_train)

# Display the learned parameters
# Convert the coefficient values to a dataframe
coeffcients = pd.DataFrame(
    linear_regressor.coef_, X_train.columns, columns=["Coefficient"]
)

# Display the intercept value
print("Learned intercept: {:.2f}".format(linear_regressor.intercept_))

print("\n--The learned coefficient value learned by the linear regression model--")
print(coeffcients)
"""invisible
"""


# Import the built-in MSE metric
from sklearn.metrics import mean_squared_error

# Model prediction on training data
y_pred_train = linear_regressor.predict(X_train)
print("\n--Train MSE--\n{}".format(mean_squared_error(y_train, y_pred_train)))


# Testing
y_pred_test = linear_regressor.predict(X_test)

print("Test MSE: {:.2f}".format(mean_squared_error(y_test, y_pred_test)))
"""invisible
"""

# Visualizing the differences between actual prices and predicted values
plt.scatter(y_test, y_pred_test)
plt.xlabel("MedPrice")
plt.ylabel("Predicted MedPrice")
plt.title("MedPrice vs Predicted MedPrice")
plt.show()
"""invisible
"""

# Checking Normality of errors
sns.distplot(y_test - y_pred_test)
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

"""
### Decision tree
"""

# Import library for decision tree
from sklearn.tree import DecisionTreeRegressor

tree_regressor = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_regressor.fit(X_train, y_train)
"""invisible
"""

# Model prediction on training & test data
y_pred_train = tree_regressor.predict(X_train)
y_pred_test = tree_regressor.predict(X_test)

print("Train MSE: {:.2f}".format(mean_squared_error(y_train, y_pred_train)))

print("Test MSE: {:.2f}".format(mean_squared_error(y_test, y_pred_test)))

"""invisible
"""
# Plot outputs
# Visualizing the differences between actual prices and predicted values
plt.scatter(y_test, y_pred_test)
plt.xlabel("MedPrice")
plt.ylabel("Predicted MedPrice")
plt.title("MedPrice vs Predicted MedPrice")
plt.show()

"""invisible
"""
# Visualizing the decision tree
from six import StringIO
import sklearn.tree as tree
import pydotplus

from IPython.display import Image

dot_data = StringIO()
tree.export_graphviz(
    tree_regressor,
    out_file=dot_data,
    class_names=["MedPrice"],  # the target names.
    feature_names=selected_feature_set,  # the feature names.
    filled=True,  # Whether to fill in the boxes with colours.
    rounded=True,  # Whether to round the corners of the boxes.
    special_characters=True,
)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

"""
## Fine-Tuning: tune the tree depth hyperparameter in the tree regressor
"""

import numpy as np
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)  # sample indices of datasets for 5-fold cv

cv_sets = []
for train_index, test_index in kf.split(X_train):
    cv_sets.append(
        (
            X_train.iloc[train_index],
            y_train.iloc[train_index],
            X_train.iloc[test_index],
            y_train.iloc[test_index],
        )
    )  # construct 5-fold cv datasets

"""invisible
"""
max_depths = list(range(1, 11))  # candidate max_depth hyperparamters

for max_depth in max_depths:
    cv_results = []
    regressor = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
for (
    x_tr,
    y_tr,
    x_te,
    y_te,
) in cv_sets:  # loop through all the cv sets and average the validation results
    regressor.fit(x_tr, y_tr)
    cv_results.append(mean_squared_error(regressor.predict(x_te), y_te))
print("Tree depth: {}, Avg. MSE: {}".format(max_depth, np.mean(cv_results)))

"""invisible
"""
# Hp tuning with Sklearn

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# Build up the decision tree regressor
regressor = DecisionTreeRegressor(random_state=42)

# Create a dictionary for the hyperparameter 'max_depth' with a range from 1 to 10
hps = {"max_depth": list(range(1, 11))}

# Transform 'performance_metric' into a scoring function using 'make_scorer'.
# The default scorer function is the greater the better, here MSE is the lower the better,
# so we set ``greater_is_better'' to be False.
scoring_fnc = make_scorer(mean_squared_error, greater_is_better=False)

# Create the grid search cv object (5-fold cross-validation)
grid_search = GridSearchCV(
    estimator=regressor, param_grid=hps, scoring=scoring_fnc, cv=5
)

# Fit the grid search object to the training data to search the optimal model
grid_search = grid_search.fit(X_train, y_train)


"""invisible
"""
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(-mean_score, params)

plt.plot(hps["max_depth"], -cvres["mean_test_score"])
plt.title("5-fold CV MSE change with tree max depth")
plt.xlabel("max_depth")
plt.ylabel("MSE")
plt.show()

"""
## Retrive the best model
"""

grid_search.best_params_
best_tree_regressor = grid_search.best_estimator_

# Produce the value for 'max_depth'
print("Best hyperparameter is {}.".format(grid_search.best_params_))

# Model prediction on training & test data
y_pred_train = best_tree_regressor.predict(X_train)
y_pred_test = best_tree_regressor.predict(X_test)

print("\n--Train MSE--\n{}".format(mean_squared_error(y_train, y_pred_train)))

print("\n--Test MSE--\n{}\n".format(mean_squared_error(y_test, y_pred_test)))

"""
## Real test curve V.S. cross-validation curve
"""

test_results = []
for max_depth in hps["max_depth"]:
    tmp_results = []
    regressor = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    regressor.fit(X_train, y_train)
    test_results.append(mean_squared_error(regressor.predict(X_test), y_test))
    print("Tree depth: {}, Test MSE: {}".format(max_depth, test_results[-1]))

plt.plot(hps["max_depth"], -cvres["mean_test_score"])
plt.plot(hps["max_depth"], test_results)
plt.title("Comparison of the changing curve of the CV results and real test results")
plt.legend(["CV", "Test"])
plt.xlabel("max_depth")
plt.ylabel("MSE")
plt.show()
