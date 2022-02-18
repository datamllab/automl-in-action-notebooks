"""shell
pip install -r https://raw.githubusercontent.com/datamllab/automl-in-action-notebooks/master/requirements.txt
"""
"""
## Load data
"""

# Import the dataset loading function from sklearn
from sklearn.datasets import fetch_openml

# Load the titanic dataset from openml
titanic = fetch_openml(name="titanic", version=1, as_frame=True)

"""invisible
"""
data, label = titanic.data.copy(), titanic.target.copy()
data.head(5)

"""invisible
"""
data.dtypes

"""invisible
"""
# look at the data
print("\n-- | Shape of the data -> (n_sample * n_feature) |--\n {}".format(data.shape))

"""invisible
"""
label

"""
## Feature engineering: deal with missing values
"""

data.isnull().sum()

"""invisible
"""
# Check for missing values
print("-- # missing values --\n{}".format(data.isnull().sum()))

"""
### Simple drop out the features with too many missing values
"""

# Remove 'cabin', 'boat', 'body' features
data = data.drop(["cabin", "boat", "body", "home.dest"], axis=1)

"""invisible
"""
data.head()

"""
### Imputation missing fare and embarked with sensible feature correlation
"""

import seaborn as sns

boxplot = sns.boxplot(x="embarked", y="fare", data=data, hue="pclass")
boxplot.axhline(80)
boxplot.set_title("Boxplot of fare grouped by embarked and pclass")
boxplot.text(x=2.6, y=80, s="fare = $80", size="medium", color="blue", weight="bold")

"""invisible
"""
data[data["embarked"].isnull()]

"""invisible
"""
# Impute missing value in embarked
data["embarked"][[168, 284]] = "C"

"""invisible
"""
data[data["fare"].isnull()]

"""invisible
"""
# Impute missing value in fare
data["fare"][1225] = (
    data.groupby(["embarked", "pclass"]).get_group(("S", 3))["fare"].median()
)

"""
### Imputation missing age with naive statistical information
"""

# Use median age to fill the missing ages
data["age"].fillna(data["age"].median(skipna=True), inplace=True)

"""
##### NOTE: We could also use sensible value information such as the relationship betwee
embarked and fare predictive relationship to fill the missing values of the age variable
rather than directly use statistical median. 
"""

print("\n-- # of missing values --\n{}".format(data.isnull().sum()))

"""
## Feature engineering: name title extraction
"""

data.head()

"""invisible
"""
data["title"] = data["name"].str.extract(" ([A-Za-z]+)\.", expand=False)
data["title"].value_counts()

"""invisible
"""
data["title"] = data["title"].replace(
    [
        "Lady",
        "Countess",
        "Capt",
        "Col",
        "Don",
        "Dr",
        "Major",
        "Rev",
        "Sir",
        "Jonkheer",
        "Dona",
    ],
    "Rare",
)
data["title"] = data["title"].replace("Mlle", "Miss")
data["title"] = data["title"].replace("Ms", "Miss")
data["title"] = data["title"].replace("Mme", "Mrs")

data = data.drop(["name"], axis=1)

"""invisible
"""
data.head()

"""
## Feature engineering: categorical feature enoding
"""

data["ticket"].describe()

"""invisible
"""
import pandas as pd

encode_col_list = ["sex", "embarked", "title"]
for i in encode_col_list:
    data = pd.concat([data, pd.get_dummies(data[i], prefix=i)], axis=1)
    data.drop(i, axis=1, inplace=True)

# direct drop the ticket feature here since it is a categorical feature with too high
levels
data["ticket"].describe()
data.drop("ticket", axis=1, inplace=True)

"""invisible
"""
data.shape

"""invisible
"""
data.head()

"""invisible
"""
data.dtypes

"""
## Split Training / Testing
"""

"""
#### NOTE: Here we do feature engineering first and then do the split.
"""

# Split data into training and test dataset
X_train, X_test, y_train, y_test = data[:891], data[891:], label[:891], label[891:]

print("--Shape of the training data--\n {}".format(X_train.shape))
print("\n--Shape of the testing data--\n {}".format(X_test.shape))

"""
## Build up a Decision Tree, a Random Forest & a GBDT classifier
"""

from sklearn.metrics import accuracy_score, plot_confusion_matrix
import matplotlib.pyplot as plt

"""
### decision tree
"""

from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
dt_clf.fit(X_train, y_train)

# Now predict the value of the digit on the test set:
y_pred_test = dt_clf.predict(X_test)

"""invisible
"""
# Display the testing results
acc = accuracy_score(y_test, y_pred_test)
print("Test accuracy: {:.2f} %".format(acc * 100))

disp = plot_confusion_matrix(dt_clf, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix of DT CLF")

plt.show()

"""invisible
"""
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Train and test Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_test = rf_clf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_test)

# Train and test GBDT
gbdt_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbdt_clf.fit(X_train, y_train)
y_pred_test = gbdt_clf.predict(X_test)
acc_gbdt = accuracy_score(y_test, y_pred_test)

# Pring the results
print("Random forest test accuracy: {:.2f} %".format(acc_rf * 100))
print("GBDT test accuracy: {:.2f} %".format(acc_gbdt * 100))

"""invisible
"""
disp = plot_confusion_matrix(rf_clf, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix of RF CLF")

plt.show()

"""invisible
"""
disp = plot_confusion_matrix(gbdt_clf, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix of GBDT CLF")

plt.show()
