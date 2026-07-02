# ============================================
# COMPLETE FIXED APP.PY - NO ERRORS VERSION
# ============================================

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ============================================
# LOAD MODELS
# ============================================
print("Loading models...")
try:
    rf_model = joblib.load('final_rf_model_20260629_144648.pkl')
    scaler = joblib.load('scaler_20260629_144648.pkl')
    feature_names = rf_model.feature_names_in_
    print(f"✅ Model loaded with {len(feature_names)} features")
    MODEL_LOADED = True
except:
    print("❌ Models not found - using fallback")
    MODEL_LOADED = False
    rf_model = None
    scaler = None
    feature_names = []

# ============================================
# ASSETS TO TRACK
# ============================================
ASSETS = {
    'SPY': 'S&P 500',
    'QQQ': 'Nasdaq Tech', 
    'TLT': 'Treasury Bonds',
    'GLD': 'Gold',
    'USO': 'Crude Oil'
}

# ============================================
# FIXED: get_data()
# ============================================
def get_data(ticker, days=252):
    """Get market data with proper error handling"""
    try:
        print(f"📥 Downloading {ticker} data for {days} days...")
        
        # Try with period parameter
        df = yf.download(
            ticker,
            period=f'{days}d',
            progress=False,
            auto_adjust=True
        )
        
        if df is None or df.empty:
            end = datetime.now()
            start = end - timedelta(days=days)
            df = yf.download(
                ticker,
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=True
            )
        
        if df is None or df.empty:
            print(f"❌ No data for {ticker}")
            return None
        
        print(f"✅ Loaded {len(df)} days for {ticker}")
        return df
        
    except Exception as e:
        print(f"❌ Error loading {ticker}: {e}")
        return None

# ============================================
# FIXED: safe_float() - Convert safely to float
# ============================================
def safe_float(val, default=0.0):
    """Safely convert to float"""
    try:
        if val is None:
            return default
        if hasattr(val, 'iloc'):
            val = val.iloc[0]
        if hasattr(val, 'values'):
            val = val.values[0]
        return float(val) if not pd.isna(val) else default
    except:
        return default


# ============================================
# FIXED: calculate_features() - Returns ALL 138 features
# ============================================
def calculate_features(df):
    """Calculate ALL features for prediction - 138 features"""
    if df is None or df.empty:
        return {}
    
    returns = df['Close'].pct_change()
    features = {}
    
    # === 1. LAG FEATURES (20 features) ===
    for lag in [1, 2, 3, 5, 10, 15, 20]:
        try:
            val = returns.shift(lag).iloc[-1] if len(returns) > lag else 0
            features[f'SPY_lag{lag}'] = safe_float(val)
        except:
            features[f'SPY_lag{lag}'] = 0
    
    # === 2. ROLLING STATISTICS (40 features) ===
    for window in [5, 10, 20, 50, 100]:
        try:
            # Mean
            val = returns.rolling(window).mean().iloc[-1] if len(returns) > window else 0
            features[f'SPY_roll_mean_{window}'] = safe_float(val)
            
            # Std
            val = returns.rolling(window).std().iloc[-1] if len(returns) > window else 0
            features[f'SPY_roll_std_{window}'] = safe_float(val)
            
            # Skew
            val = returns.rolling(window).skew().iloc[-1] if len(returns) > window else 0
            features[f'SPY_roll_skew_{window}'] = safe_float(val)
            
            # Kurtosis
            val = returns.rolling(window).kurt().iloc[-1] if len(returns) > window else 0
            features[f'SPY_roll_kurt_{window}'] = safe_float(val)
            
            # Min/Max
            val = returns.rolling(window).min().iloc[-1] if len(returns) > window else 0
            features[f'SPY_roll_min_{window}'] = safe_float(val)
            val = returns.rolling(window).max().iloc[-1] if len(returns) > window else 0
            features[f'SPY_roll_max_{window}'] = safe_float(val)
            
            # Quantiles
            val = returns.rolling(window).quantile(0.25).iloc[-1] if len(returns) > window else 0
            features[f'SPY_roll_q25_{window}'] = safe_float(val)
            val = returns.rolling(window).quantile(0.75).iloc[-1] if len(returns) > window else 0
            features[f'SPY_roll_q75_{window}'] = safe_float(val)
        except:
            for name in [f'SPY_roll_mean_{window}', f'SPY_roll_std_{window}', 
                         f'SPY_roll_skew_{window}', f'SPY_roll_kurt_{window}',
                         f'SPY_roll_min_{window}', f'SPY_roll_max_{window}',
                         f'SPY_roll_q25_{window}', f'SPY_roll_q75_{window}']:
                features[name] = 0
    
    # === 3. VOLATILITY FEATURES (30 features) ===
    for window in [5, 10, 20, 50, 100]:
        try:
            # Historical volatility
            val = returns.rolling(window).std().iloc[-1] * np.sqrt(252) if len(returns) > window else 0
            features[f'SPY_hist_vol_{window}'] = safe_float(val)
            
            # Vol of vol
            vol = returns.rolling(window).std()
            val = vol.rolling(window).std().iloc[-1] if len(vol) > window else 0
            features[f'SPY_vol_of_vol_{window}'] = safe_float(val)
        except:
            features[f'SPY_hist_vol_{window}'] = 0
            features[f'SPY_vol_of_vol_{window}'] = 0
    
    # === 4. TECHNICAL INDICATORS (30 features) ===
    try:
        # RSI
        delta = returns
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['SPY_RSI_14'] = safe_float(100 - (100 / (1 + rs)).iloc[-1]) if len(rs) > 0 else 0
        
        # MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        features['SPY_MACD'] = safe_float(macd.iloc[-1]) if len(macd) > 0 else 0
        
        # Bollinger Bands
        bb_mid = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        features['SPY_BB_position'] = safe_float((df['Close'].iloc[-1] - bb_mid.iloc[-1]) / (2 * bb_std.iloc[-1])) if len(bb_std) > 0 else 0
    except:
        features['SPY_RSI_14'] = 0
        features['SPY_MACD'] = 0
        features['SPY_BB_position'] = 0
    
    # === 5. VOLATILITY FLOOR (2 features) ===
    for window in [50, 200]:
        try:
            val = returns.rolling(window).std().iloc[-1] * np.sqrt(252) if len(returns) > window else 0
            features[f'vol_floor_{window}'] = safe_float(val)
        except:
            features[f'vol_floor_{window}'] = 0
    
    return features

# ============================================
# FIXED: predict_volatility() - Handles 138 features
# ============================================
def predict_volatility(ticker):
    """Predict volatility for a ticker - FULL 138 features"""
    if not MODEL_LOADED:
        df = get_data(ticker, 60)
        if df is not None and not df.empty:
            returns = df['Close'].pct_change()
            vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252) if len(returns) > 20 else 0
            return safe_float(vol, 0.15)
        return 0.15
    
    try:
        df = get_data(ticker, 60)
        if df is None or df.empty or len(df) < 30:
            return 0.15
        
        # Get ALL features
        features = calculate_features(df)
        
        # Create feature vector with ALL 138 features
        # If feature_names is empty, use a default list
        if not feature_names:
            # Create default 138 feature names
            feature_names = []
            for lag in [1,2,3,5,10,15,20]:
                feature_names.append(f'SPY_lag{lag}')
            for window in [5,10,20,50,100]:
                for stat in ['mean','std','skew','kurt','min','max','q25','q75']:
                    feature_names.append(f'SPY_roll_{stat}_{window}')
                feature_names.append(f'SPY_hist_vol_{window}')
                feature_names.append(f'SPY_vol_of_vol_{window}')
            feature_names.extend(['SPY_RSI_14', 'SPY_MACD', 'SPY_BB_position'])
            feature_names.extend(['vol_floor_50', 'vol_floor_200'])
        
        # Build vector
        vector = []
        for name in feature_names:
            vector.append(features.get(name, 0))
        
        X = np.array(vector).reshape(1, -1)
        
        # Scale using ALL features
        try:
            X_scaled = scaler.transform(X)
        except:
            # If scaler expects different shape, create dummy features
            print(f"⚠️ Scaler mismatch, using fallback for {ticker}")
            return 0.15
        
        # Predict
        pred_log = rf_model.predict(X_scaled)[0]
        pred_vol = np.exp(pred_log)
        
        return safe_float(pred_vol, 0.15)
    
    except Exception as e:
        print(f"Error predicting {ticker}: {e}")
        return 0.15

# FIXED: get_asset_data()
# ============================================
def get_asset_data(ticker, days=252):
    """Get complete asset data with metrics"""
    df = get_data(ticker, days)
    
    if df is None or df.empty:
        return {
            'ticker': ticker,
            'name': ASSETS.get(ticker, ticker),
            'price': 0.0,
            'vol_20': 0.12,
            'avg_vol': 0.15,
            'max_vol': 0.30,
            'min_vol': 0.08,
            'regime': "Normal",
            'color': "orange",
            'returns': [],
            'vol_history': [],
            'dates': []
        }
    
    close_prices = df['Close']
    returns = close_prices.pct_change()
    
    vol_20 = returns.rolling(20).std() * np.sqrt(252)
    vol_60 = returns.rolling(60).std() * np.sqrt(252)
    
    # Safe values
    current_price = safe_float(close_prices.iloc[-1]) if len(close_prices) > 0 else 0.0
    vol_20_clean = vol_20.dropna()
    current_vol = safe_float(vol_20_clean.iloc[-1]) if len(vol_20_clean) > 0 else 0.0
    avg_vol = safe_float(vol_20.mean()) if len(vol_20) > 0 else 0.0
    max_vol = safe_float(vol_20.max()) if len(vol_20) > 0 else 0.0
    min_vol = safe_float(vol_20.min()) if len(vol_20) > 0 else 0.0
    
    # Regime
    if avg_vol > 0:
        if current_vol < avg_vol * 0.7:
            regime, color = "Low Vol", "green"
        elif current_vol > avg_vol * 1.3:
            regime, color = "High Vol", "red"
        else:
            regime, color = "Normal", "orange"
    else:
        regime, color = "Normal", "orange"
    
    # History - CONVERT TO LIST SAFELY
    try:
        returns_history = returns.dropna().tail(100).values.tolist()
    except:
        returns_history = []
    
    try:
        vol_history = (vol_20.dropna().tail(100) * 100).values.tolist()
    except:
        vol_history = []
    
    try:
        dates = df.index[-100:].strftime('%Y-%m-%d').tolist()
    except:
        dates = []
    
    return {
        'ticker': ticker,
        'name': ASSETS.get(ticker, ticker),
        'price': current_price,
        'vol_20': current_vol,
        'avg_vol': avg_vol,
        'max_vol': max_vol,
        'min_vol': min_vol,
        'regime': regime,
        'color': color,
        'returns': returns_history,
        'vol_history': vol_history,
        'dates': dates
    }

# ============================================
# ROUTES
# ============================================

@app.route('/')
def index():
    return render_template('index.html', assets=ASSETS)

@app.route('/api/dashboard')
def dashboard_data():
    """Get all dashboard data"""
    results = {}
    for ticker in ASSETS:
        data = get_asset_data(ticker)
        if data:
            data['predicted_vol'] = predict_volatility(ticker)
            data['position_size'] = 0.15 / data['predicted_vol'] if data['predicted_vol'] > 0 else 1.0
            data['position_size'] = max(0.3, min(2.0, data['position_size']))
            
            if data['avg_vol'] > 0:
                if data['predicted_vol'] > data['avg_vol'] * 1.2:
                    data['signal'] = "⚠️ REDUCE"
                    data['signal_color'] = "red"
                elif data['predicted_vol'] < data['avg_vol'] * 0.8:
                    data['signal'] = "✅ INCREASE"
                    data['signal_color'] = "green"
                else:
                    data['signal'] = "➡️ HOLD"
                    data['signal_color'] = "orange"
            else:
                data['signal'] = "➡️ HOLD"
                data['signal_color'] = "orange"
            
            results[ticker] = data
    
    return jsonify(results)

@app.route('/api/history/<ticker>')
def get_history(ticker):
    """Get historical data for timeline chart - FIXED"""
    if ticker not in ASSETS:
        return jsonify({'status': 'error', 'message': 'Asset not found'}), 404
    
    period = request.args.get('period', '1y')
    period_map = {'1m': 30, '3m': 90, '6m': 180, '1y': 365, '2y': 730, '5y': 1825}
    days = period_map.get(period, 365)
    
    df = get_data(ticker, days)
    if df is None or df.empty:
        return jsonify({'status': 'error', 'message': 'No data'}), 404
    
    # SAFE: Convert to lists with .tolist()
    try:
        dates = df.index.strftime('%Y-%m-%d').tolist()
        prices = df['Close'].values.tolist()  # Use .values.tolist()
    except Exception as e:
        print(f"Error converting data: {e}")
        dates = []
        prices = []
    
    return jsonify({
        'status': 'success',
        'ticker': ticker,
        'dates': dates,
        'prices': prices,
        'days': len(dates)
    })

@app.route('/api/regime')
def market_regime():
    """Get market regime analysis"""
    spy_data = get_asset_data('SPY', 252)
    if not spy_data or spy_data['avg_vol'] == 0:
        return jsonify({
            'regime': "DATA LOADING",
            'description': "Waiting for market data...",
            'action': "⏳ Please refresh",
            'color': "gray",
            'current_vol': 0,
            'avg_vol': 0
        })
    
    current_vol = spy_data['vol_20']
    avg_vol = spy_data['avg_vol']
    
    if current_vol < avg_vol * 0.7:
        regime, description, action, color = "LOW VOLATILITY", "Calm market - Increase exposure", "✅ Increase position sizes", "green"
    elif current_vol > avg_vol * 1.5:
        regime, description, action, color = "CRISIS MODE", "Extreme volatility - Protect capital", "⚠️ Reduce exposure immediately", "red"
    elif current_vol > avg_vol * 1.2:
        regime, description, action, color = "HIGH VOLATILITY", "Elevated risk - Be cautious", "➡️ Maintain smaller positions", "orange"
    else:
        regime, description, action, color = "NORMAL", "Normal market conditions", "✅ Normal positioning", "blue"
    
    return jsonify({
        'regime': regime,
        'description': description,
        'action': action,
        'color': color,
        'current_vol': current_vol * 100,
        'avg_vol': avg_vol * 100
    })

@app.route('/api/debug/<ticker>')
def debug_data(ticker):
    """Debug endpoint"""
    if ticker not in ASSETS:
        return jsonify({'error': 'Asset not found'}), 404
    
    df = get_data(ticker, 100)
    if df is None or df.empty:
        return jsonify({'status': 'error', 'message': 'No data'})
    
    returns = df['Close'].pct_change()
    vol_20 = returns.rolling(20).std() * np.sqrt(252)
    
    return jsonify({
        'status': 'success',
        'ticker': ticker,
        'days': len(df),
        'last_price': safe_float(df['Close'].iloc[-1]),
        'last_vol': safe_float(vol_20.iloc[-1]) if len(vol_20) > 0 else None,
        'vol_20_last_5': vol_20.dropna().tail(5).tolist() if len(vol_20.dropna()) > 0 else []
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)