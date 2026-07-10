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

# --- HIGH-PERFORMANCE PRO-MODE THREAD-SAFE GLOBAL STATE ---
STATE = {
    "active_trade": None,  # Persists across multiple 15-minute candles until SL/TP is hit
    "performance": {
        "wins": 0,
        "losses": 0,
        "total_trades": 0,
        "win_rate": "0.00%",
        "gross_pnl_usdc": 0.0,  # CRITICAL: Calculated STRICTLY based on winning trades
        "net_pnl_usdc": 0.0     # Aggregate PNL minus trade transaction fees
    },
    "trade_logs": []
}

# --- SYSTEM CONSTANTS & CONFIGURATIONS (MIRRORED FROM BACKTESTER_2.JS) ---
RISK_SETTINGS = {
    "atrStopMultiplier": 2.0,     # Matches backtester_2.js
    "atrProfitMultiplier": 2.0,   # Matches backtester_2.js
    "breakevenMultiplier": 5.0,   # Triggers breakeven adjustment at 5x ATR
    "takerFeePerc": 0.0010,       # 0.10% Taker execution fee
    "makerFeePerc": 0.00095,      # 0.095% Maker limit exit fee
    "riskPct": 0.01               # 1% equity risk dynamic ceiling per trade
}

# --- ENGINE MODEL INITIALIZATION ---
MODEL_PATH = "veto_engine.onnx"

print(f"🤖 Initializing Inference Engine using file: {MODEL_PATH}")
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
label_name = session.get_outputs()[0].name
prob_name = session.get_outputs()[1].name

# Initialize Exchange Core Connection
exchange = ccxt.coinbase({
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})
SYMBOL = 'SOL/USDC'

def fetch_and_engineer_features():
    """Fetches full lookback frames from market data and runs feature extraction pipeline."""
    try:
        # Pull 100 candles to guarantee full mathematical warming limits for ADX and ATR
        candles = exchange.fetch_ohlcv(SYMBOL, timeframe='15m', limit=100)
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # --- TECHNICAL INDICATOR PIPELINE (10 TARGET FEATURES) ---
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['currentADX'] = adx_df['ADX_14']
        df['prevADX'] = adx_df['ADX_14'].shift(1)
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['rvol'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        atr = ta.atr(df['high'], df['low'], df['close'], length=14)
        # CRITICAL ALIGNMENT: Matches (rawATR / currentClose) * 100 from backtester_2.js
        df['atrPercentage'] = (atr / df['close']) * 100
        
        df['bodySize'] = (df['close'] - df['open']).abs()
        df['lowerWick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['upperWick'] = df['high'] - df[['open', 'close']].max(axis=1)
        
        # Strategy Execution Direction Vectors (1.0 = Long, -1.0 = Short)
        df['directionIntent'] = 1.0  
        df['isWhipsaw'] = 0.0
        
        # Isolate the most recently CLOSED candle frame to avoid live painting errors
        live_row = df.iloc[-2]
        
        if live_row.isnull().any():
            return None, "Indicators warming up...", None, None
            
        feature_order = [
            'rvol', 'rsi', 'currentADX', 'prevADX', 'atrPercentage', 
            'bodySize', 'lowerWick', 'upperWick', 'directionIntent', 'isWhipsaw'
        ]
        
        input_vector = live_row[feature_order].values.astype(np.float32).reshape(1, 10)
        timestamp_str = pd.to_datetime(live_row['timestamp'], unit='ms').strftime('%Y-%m-%d %H:%M:%S')
        
        pricing_data = {
            "close": float(live_row['close']),
            "high": float(live_row['high']),
            "low": float(live_row['low']),
            "atr": float(atr.iloc[-2])
        }
        
        return input_vector, timestamp_str, pricing_data, live_row.to_dict()
        
    except Exception as e:
        return None, f"Data fetch error: {str(e)}", None, None

def trading_loop():
    """Main execution engine processing trade cycles synchronized to 15m intervals."""
    print("🍰 UpsideDownCake 24/7 Production Engine Running Safely...")
    
    while True:
        # Strict chron-sync logic to execute directly at the turn of the 15-minute mark
        current_time = time.time()
        time_to_next_candle = 900 - (current_time % 900)
        time.sleep(time_to_next_candle + 3)  # 3-second buffer to guarantee candle close finalization
        
        # Fetch fresh dataset updates
        features, meta, pricing, raw_row = fetch_and_engineer_features()
        if features is None:
            print(f"⚠️ [SKIPPED BLOCK CYCLE] {meta}")
            continue
            
        # 1. EVALUATE TRANSITIONAL ACTIVE MULTI-CANDLE POSITION RULES (Matches backtester_2.js)
        if STATE["active_trade"] is not None:
            pos = STATE["active_trade"]
            trade_closed = False
            exit_price = 0.0
            
            # Extract underlying position sizing references
            breakeven_trigger = pos["atr"] * RISK_SETTINGS["breakevenMultiplier"]
            fee_buffer = pos["entry_price"] * RISK_SETTINGS["takerFeePerc"] * 2  # Cover in/out fees
            
            if pos["direction"] == 1.0:  # Long Trade Tracking Logic
                # Check for Breakeven Protection triggers
                if pricing["high"] >= (pos["entry_price"] + breakeven_trigger) and pos["sl"] < pos["entry_price"]:
                    pos["sl"] = pos["entry_price"] + fee_buffer
                    print(f"🛡️ [BREAKEVEN TRIGGERED] Long Stop Loss shifted up to protect capital floor.")
                
                # Check execution bracket violations
                if pricing["low"] <= pos["sl"]:
                    exit_price = pos["sl"]
                    trade_closed = True
                elif pricing["high"] >= pos["tp"]:
                    exit_price = pos["tp"]
                    trade_closed = True
            else:  # Short Trade Tracking Logic
                if pricing["low"] <= (pos["entry_price"] - breakeven_trigger) and pos["sl"] > pos["entry_price"]:
                    pos["sl"] = pos["entry_price"] - fee_buffer
                    print(f"🛡️ [BREAKEVEN TRIGGERED] Short Stop Loss shifted down to protect capital ceiling.")
                    
                if pricing["high"] >= pos["sl"]:
                    exit_price = pos["sl"]
                    trade_closed = True
                elif pricing["low"] <= pos["tp"]:
                    exit_price = pos["tp"]
                    trade_closed = True
            
            # Settlement Execution Blocks
            if trade_closed:
                entry_fee_cost = (pos["entry_price"] * pos["contract_size"]) * RISK_SETTINGS["takerFeePerc"]
                exit_fee_cost = (exit_price * pos["contract_size"]) * RISK_SETTINGS["makerFeePerc"]
                total_fees = entry_fee_cost + exit_fee_cost
                
                if pos["direction"] == 1.0:
                    gross_pnl = (exit_price - pos["entry_price"]) * pos["contract_size"]
                else:
                    gross_pnl = (pos["entry_price"] - exit_price) * pos["contract_size"]
                    
                net_pnl = gross_pnl - total_fees
                
                # State Update Logic Rules
                STATE["performance"]["total_trades"] += 1
                STATE["performance"]["net_pnl_usdc"] += net_pnl
                
                if net_pnl > 0:
                    STATE["performance"]["wins"] += 1
                    STATE["performance"]["gross_pnl_usdc"] += net_pnl  # CRITICAL: Gross PNL strictly sums winning trades
                    outcome_str = "🎉 WIN"
                else:
                    STATE["performance"]["losses"] += 1
                    outcome_str = "🛑 LOSS"
                    
                calc_wr = (STATE["performance"]["wins"] / STATE["performance"]["total_trades"]) * 100
                STATE["performance"]["win_rate"] = f"{calc_wr:.2f}%"
                
                settle_msg = f"📊 [SETTLED] Trade from {pos['timestamp']} Closed @ {exit_price} | {outcome_str} | Net PNL: {net_pnl:+.4f} USDC"
                print(settle_msg)
                STATE["trade_logs"].append(settle_msg)
                STATE["active_trade"] = None  # Clear allocation slot
                
        # 2. RUN INFERENCE BRAIN FOR NEW SETUP ENTRIES (IF UNHEDGED)
        if STATE["active_trade"] is None:
            pred_label, pred_prob = session.run([label_name, prob_name], {input_name: features})
            
            label = int(pred_label[0])
            prob_win = float(pred_prob[0][1])
            direction_intent = float(features[0][8])
            
            if label == 1:
                # Dynamic Sizing Engine Framework
                try:
                    balance_data = exchange.fetch_balance()
                    account_equity = float(balance_data['free']['USDC'] if 'USDC' in balance_data['free'] else 100.0)
                except Exception:
                    account_equity = 100.0  # Fallback allocation layer
                    
                stop_loss_distance = pricing["atr"] * RISK_SETTINGS["atrStopMultiplier"]
                
                if stop_loss_distance > 0:
                    target_dollar_risk = account_equity * RISK_SETTINGS["riskPct"]
                    calculated_size = target_dollar_risk / stop_loss_distance
                    # Safe rounding floor mapping to standard crypto lot steps
                    contract_size = round(calculated_size, 2) if calculated_size >= 0.01 else 0.25
                else:
                    contract_size = 0.25  # Standard default backstop
                    
                entry_p = pricing["close"]
                
                # Establish dynamic bracket targets based on calculated index directions
                if direction_intent == 1.0:
                    sl_target = entry_p - stop_loss_distance
                    tp_target = entry_p + (pricing["atr"] * RISK_SETTINGS["atrProfitMultiplier"])
                else:
                    sl_target = entry_p + stop_loss_distance
                    tp_target = entry_p - (pricing["atr"] * RISK_SETTINGS["atrProfitMultiplier"])
                    
                STATE["active_trade"] = {
                    "entry_price": entry_p,
                    "direction": direction_intent,
                    "timestamp": meta,
                    "contract_size": contract_size,
                    "sl": sl_target,
                    "tp": tp_target,
                    "atr": pricing["atr"]
                }
                decision_msg = f"✅ ALLOWED (Brain entry confirmation of {contract_size} SOL slots @ {entry_p})"
            else:
                decision_msg = "❌ VETO (Conditions blocked by XGBoost filter layer)"
                
            log_msg = f"🕒 [{meta}] Veto Engine Conviction Prob: {prob_win:.2%} | Action: {decision_msg}"
            print(log_msg)
            STATE["trade_logs"].append(log_msg)
            
        # Clean lookback queues to avoid running out of RAM over long deployment cycles
        if len(STATE["trade_logs"]) > 200:
            STATE["trade_logs"].pop(0)

# Run Background Worker Pipeline
threading.Thread(target=trading_loop, daemon=True).start()

@app.api_route("/", methods=["GET", "HEAD"])
def health_and_dashboard():
    """Serves real-time system tracking directly to client interface frames."""
    # System Price Offset Sync Correction: Subtract exactly 1.0 from data outputs
    adjusted_position = None
    if STATE["active_trade"] is not None:
        adjusted_position = STATE["active_trade"].copy()
        adjusted_position["entry_price"] = round(adjusted_position["entry_price"] - 1.0, 4)
        adjusted_position["sl"] = round(adjusted_position["sl"] - 1.0, 4)
        adjusted_position["tp"] = round(adjusted_position["tp"] - 1.0, 4)
        
    return {
        "status": "online",
        "market": "SOL-PERP (Coinbase-Pro-Context)",
        "live_metrics": {
            "wins": STATE["performance"]["wins"],
            "losses": STATE["performance"]["losses"],
            "total_trades": STATE["performance"]["total_trades"],
            "win_rate": STATE["performance"]["win_rate"],
            "gross_pnl_usdc": round(STATE["performance"]["gross_pnl_usdc"], 4),
            "net_pnl_usdc": round(STATE["performance"]["net_pnl_usdc"], 4)
        },
        "current_position": adjusted_position,
        "recent_activity_logs": STATE["trade_logs"][::-1]
    }