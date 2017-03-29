# this model scored 7.25042 on kaggle public score
import pandas as pd
import os
import helpers.settings as settings
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

train = pd.read_csv(os.path.join(
    os.getcwd(), settings.DATA_DIR, 'train_processed.csv'))
test = pd.read_csv(os.path.join(
    os.getcwd(), settings.DATA_DIR, 'test_processed.csv'))

y_train = train["SalePrice"]
X_train = train.drop(["SalePrice"], axis=1)
X_test = test

model1 = LinearRegression()
model1.fit(X_train, y_train)
y_test = model1.predict(X_test)

print(cross_val_score(model1, X_train, y_train))

test_df = pd.read_csv(os.path.join(os.getcwd(), settings.DATA_DIR, 'test.csv'))
submission1 = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": y_test
})
submission1.to_csv(os.path.join(os.getcwd(), settings.OUT_DIR,
                                'least_square_regression_sklearn.csv'), index=False)
