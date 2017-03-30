# this model scored 8.53631 on kaggle public score
import pandas as pd
import os
import helpers.settings as settings
import numpy as np
from sklearn.linear_model import Ridge


# load the data
train = pd.read_csv(os.path.join(
    os.getcwd(), settings.DATA_DIR, 'train_processed.csv'))
test = pd.read_csv(os.path.join(
    os.getcwd(), settings.DATA_DIR, 'test_processed.csv'))

y_train = train["SalePrice"]
X_train = train.drop(["SalePrice"], axis=1)
X_test = test

# create a ridge regression model
model2 = Ridge(alpha=10, fit_intercept=True)
model2.fit(X_train, y_train)
y_test = model2.predict(X_test)

# submission
test_df = pd.read_csv(os.path.join(os.getcwd(), settings.DATA_DIR, 'test.csv'))
submission2 = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": y_test
})
submission2.to_csv(os.path.join(os.getcwd(), settings.OUT_DIR,
                                'ridge_regression_sklearn.csv'), index=False)
