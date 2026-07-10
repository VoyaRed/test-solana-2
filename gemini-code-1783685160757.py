import os
import time
import threading
import numpy as np
import pandas as pd
import pandas_ta as ta
import onnxruntime as ort
import ccxt
from fastapi import FastAPI

app = FastAPI()

# Global tracking for paper trading logs visible via web browser
trade_logs = []

# Load ONNX Engine
MODEL_PATH = "veto_engine.onnx"
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
label_name = session.get_outputs()[0].name
prob_name = session.get_outputs()[1].name

# Initialize Coinbase Exchange connection (Public data only for paper trading)
exchange = ccxt.coinbase({
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}  # Points to Perp/Futures markets
})
SYMBOL = 'SOL/USDC:USDC'  # Coinbase SOL Perpetual pair

def fetch_and_engineer_features():
    try:
        # Fetch 100 historical 15m candles to calculate indicators safely
        candles = exchange.fetch_ohlcv(SYMBOL, timeframe='15m', limit=100)
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # --- FEATURE ENGINEERING MATCHING YOUR MODEL (10 FEATURES) ---
        # 1 & 2. ADX metrics
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['currentADX'] = adx_df['ADX_14']
        df['prevADX'] = adx_df['ADX_14'].shift(1)
        
        # 3. RSI
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        # 4. Relative Volume (rvol) vs 20-period average
        df['rvol'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # 5. ATR Percentage
        atr = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atrPercentage'] = atr / df['close']
        
        # 6, 7, 8. Candle Mechanics
        df['bodySize'] = (df['close'] - df['open']).abs()
        df['lowerWick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['upperWick'] = df['high'] - df[['open', 'close']].max(axis=1)
        
        # 9 & 10. Strategic System Intent Flags (Mock placeholders - replace with your entry signal logic)
        df['directionIntent'] = 1.0  # 1 for Long, -1 for Short based on your core strategy
        df['isWhipsaw'] = 0.0        # Flag 1.0 if inside a choppy regime, else 0.0
        
        # Grab the most recently CLOSED candle (Index -2, since -1 is changing live)
        live_row = df.iloc[-2]
        
        # Check for NaNs due to indicator warmup periods
        if live_row.isnull().any():
            return None, "Indicators warming up..."
            
        feature_order = [
            'rvol', 'rsi', 'currentADX', 'prevADX', 'atrPercentage', 
            'bodySize', 'lowerWick', 'upperWick', 'directionIntent', 'isWhipsaw'
        ]
        
        input_vector = live_row[feature_order].values.astype(np.float32).reshape(1, 10)
        timestamp_str = pd.to_datetime(live_row['timestamp'], unit='ms').strftime('%Y-%m-%d %H:%M:%S')
        
        return input_vector, timestamp_str
        
    except Exception as e:
        return None, f"Data fetch error: {str(e)}"

def trading_loop():
    """Background loop that executes every 15 minutes at the candle boundary"""
    print("🚀 Paper Trading Core Engine Started...")
    while True:
        # Code execution syncs to the next 15-minute close interval
        current_time = time.time()
        time_to_next_candle = 900 - (current_time % 900)
        # Sleep until the exact second the current candle finishes
        time.sleep(time_to_next_candle + 5) # 5-second buffer for exchange latency
        
        features, meta = fetch_and_engineer_features()
        
        if features is None:
            log_msg = f"⚠️ [SKIPPED] {meta}"
            print(log_msg)
            trade_logs.append(log_msg)
            continue
            
        # Run inference
        pred_label, pred_prob = session.run([label_name, prob_name], {input_name: features})
        
        label = int(pred_label[0])
        prob_loss = float(pred_prob[0][0])
        prob_win = float(pred_prob[0][1])
        
        # Filter action based on model outcome
        # (Assuming 0 = Veto/Loss, 1 = Safe/Win setup)
        decision = "❌ VETO (Trade Blocked)" if label == 0 else "✅ ALLOWED (Simulating Entry)"
        
        log_entry = f"🕒 [{meta}] Win Prob: {prob_win:.2%} | Loss Prob: {prob_loss:.2%} | Action: {decision}"
        print(log_entry)
        trade_logs.append(log_entry)
        
        # Limit global log list size in memory
        if len(trade_logs) > 200:
            trade_logs.pop(0)

# Start trading loop execution in an independent background thread
threading.Thread(target=trading_loop, daemon=True).start()

@app.get("/")
def health_and_dashboard():
    """Web UI to prevent Render Free Tier from sleeping and monitor outputs"""
    return {
        "status": "online",
        "market": "SOL-PERP (Coinbase)",
        "model_loaded": True,
        "recent_paper_trades": trade_logs[::-1] # Show newest logs first
    }