import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout
from util.utils import logger as logger

# Load the CSV file
file_path = os.env('DATA_FILE_PATH')

if file_path is None:
    logger.error('Missing the environment variable \'DATA_FILE_PATH\'')
    sys.exit(1)

data = pd.read_csv(file_path)

# Convert measTimeStampRf to datetime
data['measTimeStampRf'] = pd.to_datetime(data['measTimeStampRf'])

# Set the timestamp as the index
data.set_index('measTimeStampRf', inplace=True)

# Select relevant features for forecasting
features = ['RRU.PrbUsedDl', 'targetTput', 'DRB.UEThpDl', 'x', 'y', 
            'RF.serving.RSRP', 'RF.serving.RSRQ', 'RF.serving.RSSINR']

# Extract the feature data
X = data[features].values

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the target variable (let's forecast 'targetTput')
y = data['targetTput'].values

# Create sequences for CNN input
def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i+time_steps])
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)

# Create sequences with a specified time step
time_steps = 10
X_seq, y_seq = create_sequences(X_scaled, y, time_steps)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(time_steps, len(features))))
model.add(Dropout(0.2))
model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model
loss = model.evaluate(X_test, y_test)

# Output the training history and evaluation loss
print(history.history)
print("Test Loss:", loss)