import pandas as pd
import numpy as np
ds = pd.read_csv("final_train.csv")
df = pd.read_csv("final_test.csv")
# Assuming your dataframe is named df
df = df.drop("Unnamed: 0", axis=1)

# Assuming your dataframe is named df
ds = ds.drop("Unnamed: 0", axis=1)

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Assuming your dataframe is named ds
# Features (X): date, onpromotion, oil_prices
X = ds[['date', 'onpromotion', 'oil_prices']].values
# Target variable (y): sales
y = ds['sales'].values

# Convert the date to numerical representation using label encoding
X[:, 0] = pd.to_datetime(X[:, 0]).astype(np.int64) // 10**9

# Normalize the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).reshape(-1)

# Reshape the data for CNN (assuming you want to use a window of size 3)
window_size = 3
X_cnn = np.array([X_scaled[i:i+window_size, :] for i in range(len(X_scaled) - window_size)])
y_cnn = y_scaled[window_size:]

# Create CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(window_size, X.shape[1])))
model.add(Flatten())
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_cnn, y_cnn, epochs=80, batch_size=32, validation_split=0.1, verbose=1)

import pickle
pickle.dump(model, open('CNN-model-pkl','wb'))