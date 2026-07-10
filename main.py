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

# Global tracking arrays and metrics
trade_logs = []
performance = {
    "wins": 0,
    "losses": 0,
    "total_trades": 0,
    "win_rate": "0.00%"
}
active_trade = None  # Will hold {"entry_price": float, "direction": float, "timestamp": str}

# Load ONNX Engine
MODEL_PATH = "veto_engine.onnx"
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
label_name = session.get_outputs()[0].name
prob_name = session.get_outputs()[1].name

# Initialize Coinbase Exchange connection
exchange = ccxt.coinbase({
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})
SYMBOL = 'SOL/USDC:USDC'

def fetch_and_engineer_features():
    try:
        candles = exchange.fetch_ohlcv(SYMBOL, timeframe='15m', limit=100)
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # --- FEATURE ENGINEERING (10 FEATURES) ---
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['currentADX'] = adx_df['ADX_14']
        df['prevADX'] = adx_df['ADX_14'].shift(1)
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['rvol'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        atr = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atrPercentage'] = atr / df['close']
        
        df['bodySize'] = (df['close'] - df['open']).abs()
        df['lowerWick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['upperWick'] = df['high'] - df[['open', 'close']].max(axis=1)
        
        # Strategy Direction Flags
        df['directionIntent'] = 1.0  # 1.0 for Long, -1.0 for Short
        df['isWhipsaw'] = 0.0
        
        live_row = df.iloc[-2]
        
        if live_row.isnull().any():
            return None, "Indicators warming up...", None
            
        feature_order = [
            'rvol', 'rsi', 'currentADX', 'prevADX', 'atrPercentage', 
            'bodySize', 'lowerWick', 'upperWick', 'directionIntent', 'isWhipsaw'
        ]
        
        input_vector = live_row[feature_order].values.astype(np.float32).reshape(1, 10)
        timestamp_str = pd.to_datetime(live_row['timestamp'], unit='ms').strftime('%Y-%m-%d %H:%M:%S')
        current_close = float(live_row['close'])
        
        return input_vector, timestamp_str, current_close
        
    except Exception as e:
        return None, f"Data fetch error: {str(e)}", None

def trading_loop():
    global active_trade
    print("🚀 Upgraded Paper Trading Engine with Performance Ledger Tracking Started...")
    
    while True:
        # Sync loop execution to the next 15-minute close interval
        current_time = time.time()
        time_to_next_candle = 900 - (current_time % 900)
        time.sleep(time_to_next_candle + 5)
        
        features, meta, current_close = fetch_and_engineer_features()
        
        if features is None:
            print(f"⚠️ [SKIPPED] {meta}")
            continue
            
        # 1. SETTLE ANY ACTIVE HISTORICAL TRADE FIRST
        if active_trade is not None:
            entry_p = active_trade["entry_price"]
            direction = active_trade["direction"]
            
            # Determine outcome based on price delta and direction
            if direction == 1.0:  # Long Setup
                is_win = current_close > entry_p
            else:                 # Short Setup
                is_win = current_close < entry_p
                
            pnl_pct = ((current_close - entry_p) / entry_p) * direction * 100
            
            # Update running stats object
            performance["total_trades"] += 1
            if is_win:
                performance["wins"] += 1
                outcome_str = "🎉 WIN"
            else:
                performance["losses"] += 1
                outcome_str = "🛑 LOSS"
                
            # Calculate updated win percentage string
            win_val = (performance["wins"] / performance["total_trades"]) * 100
            performance["win_rate"] = f"{win_val:.2f}%"
            
            settlement_log = f"📊 [SETTLED] Trade from {active_trade['timestamp']} | Outcome: {outcome_str} ({pnl_pct:+.2f}%) | Entry: {entry_p} -> Exit: {current_close}"
            print(settlement_log)
            trade_logs.append(settlement_log)
            
            # Clear position memory to make room for new trade opportunities
            active_trade = None

        # 2. RUN INFERENCE FOR NEW TRADE OPPORTUNITIES
        pred_label, pred_prob = session.run([label_name, prob_name], {input_name: features})
        
        label = int(pred_label[0])
        prob_win = float(pred_prob[0][1])
        direction_intent = float(features[0][8]) # Extract directionIntent from vector
        
        if label == 1:
            # Veto engine approved the signal! Record paper trade entry details
            active_trade = {
                "entry_price": current_close,
                "direction": direction_intent,
                "timestamp": meta
            }
            decision = f"✅ ALLOWED (Simulating Entry @ {current_close})"
        else:
            decision = "❌ VETO (Trade Blocked)"
            
        log_entry = f"🕒 [{meta}] Win Prob: {prob_win:.2%} | Action: {decision}"
        print(log_entry)
        trade_logs.append(log_entry)
        
        if len(trade_logs) > 200:
            trade_logs.pop(0)

# Start background tracking execution loop
threading.Thread(target=trading_loop, daemon=True).start()

@app.api_route("/", methods=["GET", "HEAD"])
def health_and_dashboard():
    """Exposes current paper trading win rates and logs directly via the browser"""
    return {
        "status": "online",
        "market": "SOL-PERP (Coinbase)",
        "live_metrics": performance,
        "current_position": active_trade,
        "recent_activity_logs": trade_logs[::-1]
    }
