# ==========================================================
# Apple Stock Price Prediction using LSTM
# Train Script
# ==========================================================

import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

df = pd.read_csv("apple stock.csv")

print(df.head())

# ----------------------------------------------------------
# Select Closing Price
# ----------------------------------------------------------

data = df[['AAPL.Close']]

print("\nDataset Shape:", data.shape)

# ----------------------------------------------------------
# Scale Data
# ----------------------------------------------------------

scaler = MinMaxScaler(feature_range=(0,1))

scaled_data = scaler.fit_transform(data)

# Save scaler

with open("scaler.pkl","wb") as f:
    pickle.dump(scaler,f)

print("Scaler Saved Successfully")

# ----------------------------------------------------------
# Create Sequences
# ----------------------------------------------------------

sequence_length = 60

X = []
y = []

for i in range(sequence_length,len(scaled_data)):

    X.append(scaled_data[i-sequence_length:i,0])

    y.append(scaled_data[i,0])

X = np.array(X)
y = np.array(y)

# Reshape for LSTM

X = np.reshape(X,(X.shape[0],X.shape[1],1))

print("X Shape :",X.shape)
print("Y Shape :",y.shape)

# ----------------------------------------------------------
# Train Test Split
# ----------------------------------------------------------

split = int(len(X)*0.80)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

print()

print("Training Samples :",len(X_train))
print("Testing Samples :",len(X_test))

# ----------------------------------------------------------
# Build LSTM Model
# ----------------------------------------------------------

model = Sequential()

model.add(

    LSTM(
        units=64,
        return_sequences=True,
        input_shape=(X_train.shape[1],1)
    )

)

model.add(Dropout(0.2))

model.add(

    LSTM(
        units=64
    )

)

model.add(Dropout(0.2))

model.add(

    Dense(
        units=25,
        activation='relu'
    )

)

model.add(

    Dense(
        units=1
    )

)

# ----------------------------------------------------------
# Compile Model
# ----------------------------------------------------------

model.compile(

    optimizer='adam',

    loss='mean_squared_error'

)

model.summary()

# ----------------------------------------------------------
# Early Stopping
# ----------------------------------------------------------

early_stop = EarlyStopping(

    monitor='val_loss',

    patience=5,

    restore_best_weights=True

)

# ----------------------------------------------------------
# Train Model
# ----------------------------------------------------------

history = model.fit(

    X_train,

    y_train,

    validation_data=(X_test,y_test),

    epochs=30,

    batch_size=32,

    callbacks=[early_stop]

)

# ----------------------------------------------------------
# Save Model
# ----------------------------------------------------------

model.save("model.keras")

print()

print("Model Saved Successfully")

print("Scaler Saved Successfully")

print()

print("Training Completed Successfully")