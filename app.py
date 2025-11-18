# app.py — Your AI Stock Predictor Web App!
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

# Page title
st.title("🚀 AI Stock Price Predictor")
st.write("Predict tomorrow's stock price with ML!")

# Sidebar for stock input
st.sidebar.header("Choose a Stock")
symbol = st.sidebar.text_input("Enter stock symbol (e.g., AAPL, TSLA)", value="AAPL").upper()

if st.sidebar.button("Predict!"):
    with st.spinner("Downloading data & predicting..."):
        
        # Download data
        data = yf.download(symbol, period="5y")
        if data.empty:
            st.error("Invalid ticker!!! Try stocks on yahoofinance only!!!")
        else:
            # Load or train model (use your saved one if AAPL)
            model_file = f"{symbol}_model.pkl"
            if os.path.exists(model_file):
                model = joblib.load(model_file)
                st.success(f"Loaded saved model for {symbol}!")
            else:
                # Train new model (same as before)
                def create_features(data, window=5):
                    prices = data['Close'].values.flatten()
                    X, y = [], []
                    for i in range(window, len(prices)):
                        X.append(prices[i-window:i])
                        y.append(prices[i])
                    return np.array(X), np.array(y)
                
                X, y = create_features(data)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                model = LinearRegression()
                model.fit(X_train, y_train)
                joblib.dump(model, model_file)
                st.info(f"Trained new model for {symbol}!")
            
            # Predict next 10 days
            last_5 = data['Close'].values[-5:]
            future_preds = []
            current_window = last_5.copy()
            for _ in range(10):
                pred = model.predict(current_window.reshape(1, -1))[0]
                future_preds.append(pred)
                current_window = np.append(current_window[1:], pred)
            
            # Future dates (trading days)
            last_date = data.index[-1]
            future_dates = []
            current_date = last_date
            for _ in range(10):
                current_date += timedelta(days=1)
                while current_date.weekday() >= 5:
                    current_date += timedelta(days=1)
                future_dates.append(current_date)
            
            # Plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data.index[-100:], data['Close'].values[-100:], 
                    label="Actual (Last 100 Days)", color="blue", linewidth=2)
            ax.plot(future_dates, future_preds, 
                    label="AI Prediction (Next 10 Days)", color="red", 
                    linestyle="--", marker="o", linewidth=2)
            ax.set_title(f"{symbol} Stock: Past + AI Future")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (USD)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Show predictions in table
            st.subheader("Next 10 Trading Days Predictions")
            pred_df = pd.DataFrame({
                "Date": [d.strftime("%Y-%m-%d (%a)") for d in future_dates],
                "Predicted Price": [f"${float(p):.2f}" for p in future_preds]
            })
            st.table(pred_df)
            
            # Last price
            last_price = float(data['Close'].values[-1])
            st.metric("Last Closing Price", f"${last_price:.2f}")
            
            st.balloons()  # Fun celebration!