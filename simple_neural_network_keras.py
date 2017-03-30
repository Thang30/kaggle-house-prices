# this model scored 7.40729 on kaggle public score
import pandas as pd
import os
import helpers.settings as settings
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense

# setting paths
train = pd.read_csv(os.path.join(
    os.getcwd(), settings.DATA_DIR, 'train_processed.csv'))
test = pd.read_csv(os.path.join(
    os.getcwd(), settings.DATA_DIR, 'test_processed.csv'))

# load data
y_train = train["SalePrice"].values
X_train = train.drop(["SalePrice"], axis=1).values
X_test = test.values

# create a simple 1 hidden-layer neural network model
model = Sequential()
model.add(Dense(X_train.shape[1], input_dim=X_train.shape[
          1], activation='relu'))
model.add(Dense(1))

# compile model
model.compile(loss='mean_squared_error', optimizer='adam')

# fix random seed for reproducibility
seed = 10
np.random.seed(seed)

# train the model
model.fit(X_train, y_train, epochs=10, batch_size=100)

# prediction
y_test = model.predict(X_test, batch_size=100, verbose=0)
y_test = y_test.flatten()

# check for missing value
print(np.isnan(y_test)[np.isnan(y_test) == True].size)

# submission
test_df = pd.read_csv(os.path.join(os.getcwd(), settings.DATA_DIR, 'test.csv'))
submission1 = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": y_test
})
submission1.to_csv(os.path.join(os.getcwd(), settings.OUT_DIR,
                                'simple_neural_network_keras.csv'), index=False)
