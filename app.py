# ==========================================================
# Apple Stock Price Prediction using LSTM
# Streamlit App
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------

st.set_page_config(
    page_title="Apple Stock Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Apple Stock Price Prediction using LSTM")

# ----------------------------------------------------------
# Load Model
# ----------------------------------------------------------

model = load_model("model.keras")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

df = pd.read_csv("apple stock.csv")

st.subheader("Dataset Preview")

st.dataframe(df.head())

# ----------------------------------------------------------
# Stock Price Graph
# ----------------------------------------------------------

st.subheader("Apple Closing Price")

fig = plt.figure(figsize=(12,5))

plt.plot(df["AAPL.Close"])

plt.xlabel("Days")

plt.ylabel("Closing Price")

plt.grid(True)

st.pyplot(fig)

# ----------------------------------------------------------
# Prediction
# ----------------------------------------------------------

close_data = df["AAPL.Close"].values.reshape(-1,1)

scaled_data = scaler.transform(close_data)

last_60_days = scaled_data[-60:]

X_test = np.array([last_60_days])

X_test = np.reshape(X_test,(1,60,1))

prediction = model.predict(X_test,verbose=0)

prediction = scaler.inverse_transform(prediction)

predicted_price = prediction[0][0]

# ----------------------------------------------------------
# Display Prediction
# ----------------------------------------------------------

st.subheader("📌 Predicted Next Day Closing Price")

st.success(f"${predicted_price:.2f}")

# ----------------------------------------------------------
# Last 60 Days Graph
# ----------------------------------------------------------

st.subheader("Last 60 Closing Prices")

fig2 = plt.figure(figsize=(12,5))

plt.plot(close_data[-60:],label="Previous 60 Days")

plt.scatter(
    len(close_data[-60:]),
    predicted_price,
    color="red",
    label="Prediction"
)

plt.legend()

plt.grid(True)

st.pyplot(fig2)

# ----------------------------------------------------------
# Statistics
# ----------------------------------------------------------

st.subheader("Dataset Statistics")

st.write(df["AAPL.Close"].describe())

# ----------------------------------------------------------
# Footer
# ----------------------------------------------------------

st.markdown("---")

st.write("Developed using LSTM, TensorFlow and Streamlit")