import pandas as pd
import numpy as np

# Assuming your dataset is a DataFrame named 'df'
# Replace 'your_dataset.csv' with the actual path or name of your dataset file
df = pd.read_csv('final_train.csv')

# Convert the 'date' column to datetime type
df['date'] = pd.to_datetime(df['date'])

# Extract numeric features from the 'date' column
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek  # Monday is 0 and Sunday is 6
df.drop('date', axis=1, inplace=True)


import pandas as pd

# Load the dataset
file_path = 'final_train.csv'
df = pd.read_csv(file_path)

# Separate 'sales' column (y) and other columns (X)
y = df['sales']
X = df.drop(['sales'], axis=1)
X['date'] = pd.to_datetime(X['date'])

# Extract numeric features from the 'date' column
X['year'] = X['date'].dt.year
X['month'] = X['date'].dt.month
X['day'] = X['date'].dt.day
X['day_of_week'] = X['date'].dt.dayofweek  # Monday is 0 and Sunday is 6
X.drop('date', axis=1, inplace=True)

# Drop the 'Unnamed' column from X
X = X.drop(['Unnamed: 0'], axis=1)
from sklearn.model_selection import train_test_split

# Split X into training and testing sets (80:20 ratio)
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

x = np.array(df)

from sklearn.model_selection import train_test_split

# Split y into training and testing sets (80:20 ratio)
y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)

y = np.array(df['sales'])

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from datetime import datetime

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the Feedforward Neural Network (FNN) model
model_fnn = Sequential()
model_fnn.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model_fnn.add(Dense(32, activation='relu'))
model_fnn.add(Dense(1, activation='linear'))  # Assuming you have a regression task

# Compile the FNN model
model_fnn.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

# Train the FNN model
model_fnn.fit(X_train_scaled, y_train, epochs=250, batch_size=32, validation_split=0.2)

import pickle
pickle.dump(model_fnn, open('FNN-model-pkl','wb'))