"""shell
pip install -r https://raw.githubusercontent.com/datamllab/automl-in-action-notebooks/master/requirements.txt
"""

"""
### Load California housing price prediction dataset
"""

from sklearn.datasets import fetch_california_housing

house_dataset = fetch_california_housing()

# Import pandas package to format the data
import pandas as pd

# Extract features with their names into the a dataframe format
data = pd.DataFrame(house_dataset.data, columns=house_dataset.feature_names)

# Extract target with their names into a pd.Series object with name MEDV
target = pd.Series(house_dataset.target, name="MEDV")

from sklearn.model_selection import train_test_split

train_data, test_data, train_targets, test_targets = train_test_split(
    data, target, test_size=0.2, random_state=42
)
"""invisible
"""

# Check the shape of whole dataset and the splited training and test set
print("--Shape of the whole data--\n {}".format(data.shape))
print("\n--Shape of the target vector--\n {}".format(target.shape))
print("\n--Shape of the training data--\n {}".format(train_data.shape))
print("\n--Shape of the testing data--\n {}".format(test_data.shape))

"""
### Run the StructuredDataRegressor
"""

import autokeras as ak

regressor = ak.StructuredDataRegressor(max_trials=10, overwrite=True)
regressor.fit(x=train_data, y=train_targets, batch_size=1024)

"""

### Predict with the best model.

"""

predicted_y = regressor.predict(test_data)
print(predicted_y)

"""
### Evaluate the best model on the test data.
"""


test_loss, test_mse = regressor.evaluate(test_data, test_targets, verbose=0)
print("Test MSE: ", test_mse)
