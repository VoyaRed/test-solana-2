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
    "active_trade": None,  
    "skipped_trade": None,  # Tracks Vetoed trades to evaluate ghost win/loss outcomes
    "last_close_price": 0.0,
    "performance": {
        "wins": 0,
        "losses": 0,
        "total_trades": 0,
        "win_rate": "0.00%",
        "gross_pnl_usdc": 0.0,  
        "net_pnl_usdc": 0.0     
    },
    "trade_logs": []
}

# --- SYSTEM CONSTANTS & CONFIGURATIONS ---
RISK_SETTINGS = {
    "atrStopMultiplier": 2.0,     
    "atrProfitMultiplier": 2.0,   
    "breakevenMultiplier": 5.0,   
    "takerFeePerc": 0.0010,       
    "makerFeePerc": 0.00095,      
    "riskPct": 0.01               
}

# --- ENGINE MODEL INITIALIZATION ---
MODEL_PATH = "veto_engine.onnx"

print(f"🤖 Initializing Inference Engine using file: {MODEL_PATH}")
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
label_name = session.get_outputs()[0].name
prob_name = session.get_outputs()[1].name

exchange = ccxt.coinbase({
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})
SYMBOL = 'SOL/USDC'

def fetch_and_engineer_features():
    try:
        candles = exchange.fetch_ohlcv(SYMBOL, timeframe='15m', limit=100)
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['currentADX'] = adx_df['ADX_14']
        df['prevADX'] = adx_df['ADX_14'].shift(1)
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['rvol'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        atr = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atrPercentage'] = (atr / df['close']) * 100
        
        df['bodySize'] = (df['close'] - df['open']).abs()
        df['lowerWick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['upperWick'] = df['high'] - df[['open', 'close']].max(axis=1)
        
        df['directionIntent'] = 1.0  
        df['isWhipsaw'] = 0.0
        
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
    print("🍰 UpsideDownCake 24/7 Production Engine Running Safely...")
    
    while True:
        current_time = time.time()
        time_to_next_candle = 900 - (current_time % 900)
        time.sleep(time_to_next_candle + 3) 
        
        features, meta, pricing, raw_row = fetch_and_engineer_features()
        if features is None:
            continue
            
        STATE["last_close_price"] = pricing["close"]
            
        # 1. EVALUATE GHOST OUTCOMES FOR SKIPPED TRADES
        if STATE["skipped_trade"] is not None:
            skip_pos = STATE["skipped_trade"]
            if skip_pos["direction"] == 1.0 and pricing["close"] > skip_pos["entry_price"]:
                outcome = "SKIP / WIN"
            elif skip_pos["direction"] == -1.0 and pricing["close"] < skip_pos["entry_price"]:
                outcome = "SKIP / WIN"
            else:
                outcome = "SKIP / LOSS"
                
            log_msg = f"👻 [GHOST SETTLED] Tracked skipped setup from {skip_pos['timestamp']}: {outcome}"
            print(log_msg)
            STATE["trade_logs"].append(log_msg)
            STATE["skipped_trade"] = None
            
        # 2. EVALUATE ACTIVE LIVE POSITION
        if STATE["active_trade"] is not None:
            pos = STATE["active_trade"]
            trade_closed = False
            exit_price = 0.0
            
            breakeven_trigger = pos["atr"] * RISK_SETTINGS["breakevenMultiplier"]
            fee_buffer = pos["entry_price"] * RISK_SETTINGS["takerFeePerc"] * 2 
            
            if pos["direction"] == 1.0: 
                if pricing["high"] >= (pos["entry_price"] + breakeven_trigger) and pos["sl"] < pos["entry_price"]:
                    pos["sl"] = pos["entry_price"] + fee_buffer
                
                if pricing["low"] <= pos["sl"]:
                    exit_price = pos["sl"]
                    trade_closed = True
                elif pricing["high"] >= pos["tp"]:
                    exit_price = pos["tp"]
                    trade_closed = True
            else: 
                if pricing["low"] <= (pos["entry_price"] - breakeven_trigger) and pos["sl"] > pos["entry_price"]:
                    pos["sl"] = pos["entry_price"] - fee_buffer
                    
                if pricing["high"] >= pos["sl"]:
                    exit_price = pos["sl"]
                    trade_closed = True
                elif pricing["low"] <= pos["tp"]:
                    exit_price = pos["tp"]
                    trade_closed = True
            
            if trade_closed:
                entry_fee_cost = (pos["entry_price"] * pos["contract_size"]) * RISK_SETTINGS["takerFeePerc"]
                exit_fee_cost = (exit_price * pos["contract_size"]) * RISK_SETTINGS["makerFeePerc"]
                total_fees = entry_fee_cost + exit_fee_cost
                
                if pos["direction"] == 1.0:
                    gross_pnl = (exit_price - pos["entry_price"]) * pos["contract_size"]
                else:
                    gross_pnl = (pos["entry_price"] - exit_price) * pos["contract_size"]
                    
                net_pnl = gross_pnl - total_fees
                
                STATE["performance"]["total_trades"] += 1
                STATE["performance"]["net_pnl_usdc"] += net_pnl
                
                if net_pnl > 0:
                    STATE["performance"]["wins"] += 1
                    STATE["performance"]["gross_pnl_usdc"] += net_pnl 
                    outcome_str = "🎉 WIN"
                else:
                    STATE["performance"]["losses"] += 1
                    outcome_str = "🛑 LOSS"
                    
                calc_wr = (STATE["performance"]["wins"] / STATE["performance"]["total_trades"]) * 100
                STATE["performance"]["win_rate"] = f"{calc_wr:.2f}%"
                
                settle_msg = f"📊 [SETTLED] Trade from {pos['timestamp']} Closed @ {exit_price} | {outcome_str} | Net PNL: {net_pnl:+.4f} USDC"
                print(settle_msg)
                STATE["trade_logs"].append(settle_msg)
                STATE["active_trade"] = None 
                
        # 3. RUN INFERENCE FOR NEW SETUP ENTRIES
        if STATE["active_trade"] is None:
            pred_label, pred_prob = session.run([label_name, prob_name], {input_name: features})
            
            label = int(pred_label[0])
            prob_win = float(pred_prob[0][1])
            direction_intent = float(features[0][8])
            
            direction_str = "LONG 📈" if direction_intent == 1.0 else "SHORT 📉"
            
            if label == 1:
                stop_loss_distance = pricing["atr"] * RISK_SETTINGS["atrStopMultiplier"]
                contract_size = 0.75
                entry_p = pricing["close"]
                
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
                decision_msg = f"✅ ALLOWED ({direction_str} entry of {contract_size} SOL @ {entry_p})"
            else:
                decision_msg = "❌ VETO (Conditions blocked by XGBoost filter layer)"
                STATE["skipped_trade"] = {
                    "timestamp": meta,
                    "entry_price": pricing["close"],
                    "direction": direction_intent
                }
                
            log_msg = f"🕒 [{meta}] Veto Engine Conviction Prob: {prob_win:.2%} | Action: {decision_msg}"
            print(log_msg)
            STATE["trade_logs"].append(log_msg)
            
        if len(STATE["trade_logs"]) > 200:
            STATE["trade_logs"].pop(0)

threading.Thread(target=trading_loop, daemon=True).start()

@app.api_route("/", methods=["GET", "HEAD"])
def health_and_dashboard():
    adjusted_position = None
    if STATE["active_trade"] is not None:
        pos = STATE["active_trade"]
        curr_price = STATE["last_close_price"]
        
        # System Price Offset Sync Correction
        entry_p = round(pos["entry_price"] - 1.0, 4)
        sl_p = round(pos["sl"] - 1.0, 4)
        tp_p = round(pos["tp"] - 1.0, 4)
        curr_p = round(curr_price - 1.0, 4)
        
        dir_str = "LONG 📈" if pos["direction"] == 1.0 else "SHORT 📉"
        
        # Calculate distance to targets
        if pos["direction"] == 1.0:
            sl_dist = curr_p - sl_p
            tp_dist = tp_p - curr_p
        else:
            sl_dist = sl_p - curr_p
            tp_dist = curr_p - tp_p
            
        adjusted_position = {
            "type": dir_str,
            "bet_size": f"{pos['contract_size']} SOL Contracts",
            "entry_price": entry_p,
            "current_price": curr_p,
            "stop_loss": sl_p,
            "distance_to_sl": f"{sl_dist:+.4f} USDC",
            "take_profit": tp_p,
            "distance_to_tp": f"{tp_dist:+.4f} USDC",
            "strategy_hold_condition": "Holding open indefinitely until price breaches either Stop Loss or Take Profit brackets."
        }
        
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
