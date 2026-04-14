import streamlit as st
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime

# =========================================
# CONFIG
# =========================================
API_URL = "http://localhost:8002/predict_multi"
# 👉 Change to your cloud API later:
# API_URL = "https://your-api.onrender.com/predict_multi"

st.set_page_config(page_title="AI Trading Dashboard", layout="wide")

st.title("📊 AI Trading Dashboard")
st.caption("LSTM-Based Stock Prediction System")

# =========================================
# SIDEBAR
# =========================================
st.sidebar.header("⚙️ Settings")

tickers = st.sidebar.text_input(
    "Enter Stocks (comma-separated)",
    "AAPL,TSLA,RELIANCE.NS"
)

refresh = st.sidebar.slider("Auto Refresh (seconds)", 0, 120, 0)

# =========================================
# FETCH DATA FUNCTION
# =========================================
def get_predictions(tickers):
    try:
        response = requests.get(f"{API_URL}?tickers={tickers}", timeout=30)
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

# =========================================
# MAIN BUTTON
# =========================================
if st.button("🚀 Get Predictions") or refresh > 0:

    data = get_predictions(tickers)

    if data:
        results = data.get("results", [])
        df = pd.DataFrame(results)

        # =========================================
        # TABLE VIEW
        # =========================================
        st.subheader("📈 Prediction Results")
        st.dataframe(df, use_container_width=True)

        # =========================================
        # SIGNAL SUMMARY
        # =========================================
        st.subheader("📢 Signals")

        for r in results:
            signal = r.get("signal")
            ticker = r.get("ticker")

            if signal == "BUY":
                st.success(f"🚀 BUY → {ticker}")
            elif signal == "SELL":
                st.error(f"📉 SELL → {ticker}")
            else:
                st.warning(f"⏸ HOLD → {ticker}")

        # =========================================
        # CHART SECTION
        # =========================================
        st.subheader("📊 Price Chart")

        selected_ticker = st.selectbox(
            "Select stock for chart",
            [r["ticker"] for r in results if "ticker" in r]
        )

        try:
            chart_data = yf.download(selected_ticker, period="1mo")
            st.line_chart(chart_data["Close"])
        except:
            st.warning("Chart data not available")

# =========================================
# AUTO REFRESH
# =========================================
if refresh > 0:
    import time
    time.sleep(refresh)
    st.experimental_rerun()
