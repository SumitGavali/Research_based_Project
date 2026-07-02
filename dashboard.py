# ============================================
# SIMPLE STREAMLIT DASHBOARD - FULLY FIXED
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests

# Page config
st.set_page_config(page_title="Volatility Dashboard", layout="wide")

st.title("📊 Volatility Forecasting Dashboard")
st.markdown("---")

# ============================================
# SIDEBAR
# ============================================
st.sidebar.header("Settings")

ticker = st.sidebar.selectbox(
    "Select Asset",
    ["SPY", "QQQ", "TLT", "GLD", "USO", "VIX", "DXY"]
)

days = st.sidebar.slider("Days to Display", 30, 730, 252)
target_vol = st.sidebar.slider("Target Volatility", 0.05, 0.30, 0.15, 0.01)

# ============================================
# LOAD DATA
# ============================================
@st.cache_data
def fetch_data(ticker, days):
    end = datetime.now()
    start = end - timedelta(days=days)
    df = yf.download(ticker, start=start, end=end, progress=False)
    return df

data = fetch_data(ticker, days)

if data is not None and not data.empty:
    # Calculate metrics
    close_prices = data['Close']
    returns = close_prices.pct_change()
    
    # Calculate rolling volatility
    vol_20 = returns.rolling(20).std() * np.sqrt(252)
    vol_60 = returns.rolling(60).std() * np.sqrt(252)
    
    # Get current values (safely)
    try:
        current_price = float(close_prices.iloc[-1])
    except:
        current_price = 0.0
    
    try:
        current_vol = float(vol_20.iloc[-1]) if not pd.isna(vol_20.iloc[-1]) else 0.0
    except:
        current_vol = 0.0
    
    try:
        avg_vol_20 = float(vol_20.mean()) if not pd.isna(vol_20.mean()) else 0.0
    except:
        avg_vol_20 = 0.0
    
    # ============================================
    # METRICS
    # ============================================
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(f"{ticker} Price", f"${current_price:.2f}")
    
    with col2:
        st.metric("Current Vol", f"{current_vol*100:.2f}%")
    
    with col3:
        st.metric("Avg Vol (20d)", f"{avg_vol_20*100:.2f}%")
    
    with col4:
        position = target_vol / current_vol if current_vol > 0 else 1.0
        st.metric("Position Size", f"{position:.2f}x")
    
    # ============================================
    # PRICE & VOLATILITY CHART
    # ============================================
    st.subheader("Price & Volatility")
    
    fig = go.Figure()
    
    # Price
    fig.add_trace(go.Scatter(
        x=data.index, y=close_prices,
        name='Price', line=dict(color='blue', width=2),
        yaxis='y2'
    ))
    
    # Volatility - convert to list
    vol_20_clean = vol_20.dropna()
    if len(vol_20_clean) > 0:
        fig.add_trace(go.Scatter(
            x=vol_20_clean.index.tolist(), 
            y=(vol_20_clean * 100).tolist(),
            name='20-day Vol', line=dict(color='orange', width=2)
        ))
    
    vol_60_clean = vol_60.dropna()
    if len(vol_60_clean) > 0:
        fig.add_trace(go.Scatter(
            x=vol_60_clean.index.tolist(),
            y=(vol_60_clean * 100).tolist(),
            name='60-day Vol', line=dict(color='green', width=2)
        ))
    
    fig.update_layout(
        title=f'{ticker} Price and Volatility',
        xaxis_title='Date',
        yaxis_title='Volatility (%)',
        yaxis2=dict(title='Price ($)', overlaying='y', side='right'),
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ============================================
    # VOLATILITY DISTRIBUTION - FIXED
    # ============================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Volatility Distribution")
        vol_data = vol_20.dropna() * 100
        if len(vol_data) > 0:
            # Convert to Python list using .tolist() on values
            vol_list = vol_data.values.tolist()
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=vol_list,
                nbinsx=30,
                marker_color='orange',
                opacity=0.7
            ))
            if current_vol > 0:
                fig.add_vline(x=current_vol * 100, line_dash="dash", line_color="red")
            fig.update_layout(
                xaxis_title='Volatility (%)',
                yaxis_title='Frequency',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No volatility data available")
    
    with col2:
        st.subheader("Return Distribution")
        returns_data = returns.dropna() * 100
        if len(returns_data) > 0:
            # Convert to Python list using .values.tolist()
            returns_list = returns_data.values.tolist()
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=returns_list,
                nbinsx=30,
                marker_color='blue',
                opacity=0.7
            ))
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            fig.update_layout(
                xaxis_title='Return (%)',
                yaxis_title='Frequency',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No return data available")
    
    # ============================================
    # FORECAST SECTION
    # ============================================
    st.subheader("🔮 Volatility Forecast")
    
    # Try to call the API
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json={"ticker": ticker, "target_vol": target_vol},
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Current Volatility",
                    f"{current_vol*100:.2f}%"
                )
            
            with col2:
                pred_vol = float(result['predicted_volatility']) * 100
                change = pred_vol - (current_vol * 100)
                st.metric(
                    "Forecasted Volatility",
                    f"{pred_vol:.2f}%",
                    delta=f"{change:+.2f}%"
                )
            
            with col3:
                st.metric(
                    "Position Size",
                    f"{float(result['position_size']):.2f}x"
                )
        else:
            st.warning("API not available. Running local prediction...")
            
            # Fallback: simple prediction
            pred_vol = current_vol * 1.1 if current_vol > 0 else 0.15
            position = target_vol / pred_vol if pred_vol > 0 else 1.0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Vol", f"{current_vol*100:.2f}%")
            with col2:
                st.metric("Forecast", f"{pred_vol*100:.2f}%", 
                         delta=f"{(pred_vol-current_vol)*100:.2f}%")
            with col3:
                st.metric("Position", f"{position:.2f}x")
    
    except Exception as e:
        st.warning("API not running. Using simple forecast...")
        pred_vol = current_vol * 1.1 if current_vol > 0 else 0.15
        position = target_vol / pred_vol if pred_vol > 0 else 1.0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Vol", f"{current_vol*100:.2f}%")
        with col2:
            st.metric("Forecast", f"{pred_vol*100:.2f}%", 
                     delta=f"{(pred_vol-current_vol)*100:.2f}%")
        with col3:
            st.metric("Position", f"{position:.2f}x")
    
    # ============================================
    # RECENT DATA TABLE
    # ============================================
    with st.expander("📊 Recent Data"):
        recent_data = data.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume']].round(2)
        st.dataframe(recent_data)

else:
    st.error(f"Failed to fetch data for {ticker}")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")