# =========================================
# LSTM STOCK PREDICTION API (FINAL VERSION)
# =========================================

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import joblib
import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
import uvicorn

# =========================================
# CONFIG
# =========================================
LOOKBACK = 60
MODEL_PATH = "lstm_model.h5"
SCALER_PATH = "scaler.save"

# =========================================
# FETCH DATA
# =========================================
def fetch_data(ticker):
    try:
        data = yf.download(ticker, period="5y")
        if data.empty:
            raise ValueError("No data found")

        # Ensure correct format
        data = data[['Close']].dropna()
        return data

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# =========================================
# PREPROCESS
# =========================================
def preprocess(data):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(LOOKBACK, len(scaled)):
        X.append(scaled[i-LOOKBACK:i])
        y.append(scaled[i])

    return np.array(X), np.array(y), scaler

# =========================================
# BUILD MODEL
# =========================================
def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# =========================================
# LOAD / TRAIN MODEL
# =========================================
def get_model(X, y, scaler):
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print("✅ Loading saved model...")
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    else:
        print("🚀 Training new model...")
        model = build_model((X.shape[1], 1))
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)

        model.save(MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)

    return model, scaler

# =========================================
# PREDICTION
# =========================================
def predict_next(model, data, scaler):
    last_60 = data['Close'].values[-LOOKBACK:].reshape(-1, 1)

    scaled = scaler.transform(last_60)
    X_input = np.reshape(scaled, (1, LOOKBACK, 1))

    pred_scaled = model.predict(X_input, verbose=0)
    pred = scaler.inverse_transform(pred_scaled)

    return float(pred[0][0])

# =========================================
# PIPELINE
# =========================================
def run_pipeline(ticker):
    data = fetch_data(ticker)

    X, y, scaler = preprocess(data)
    model, scaler = get_model(X, y, scaler)

    predicted_price = predict_next(model, data, scaler)

    # ✅ FIXED ERROR HERE
    current_price = float(data['Close'].iloc[-1])

    # Improved signal logic
    diff = (predicted_price - current_price) / current_price

    if diff > 0.02:
        signal = "BUY"
    elif diff < -0.02:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "ticker": ticker,
        "current_price": current_price,
        "predicted_price": predicted_price,
        "signal": signal,
        "confidence": round(abs(diff), 4),
        "timestamp": datetime.now().isoformat()
    }

# =========================================
# FASTAPI APP
# =========================================
app = FastAPI()

@app.get("/")
def home():
    return {"message": "LSTM Stock API Running 🚀"}

@app.get("/predict")
def predict(ticker: str = "AAPL"):
    return run_pipeline(ticker)

# =========================================
# RUN SERVER
# =========================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
