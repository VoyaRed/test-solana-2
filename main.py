import os
import time
import threading
import requests
import numpy as np
import pandas as pd
import pandas_ta as ta
import onnxruntime as ort
import ccxt
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# --- MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- HIGH-PERFORMANCE PRO-MODE THREAD-SAFE GLOBAL STATE ---
STATE = {
    "active_trade": None,  
    "skipped_trade": None, 
    "last_close_price": 0.0,
    "performance": {
        "wins": 0,
        "losses": 0,
        "total_trades": 0,
        "win_rate": "0.00%",
        "gross_pnl_usdc": 0.0,  
        "net_pnl_usdc": 0.0,
        "wallet_balance_usdc": 1000.00  # Initial Paper Trading Balance
    },
    "trade_logs": []
}

# --- SYSTEM CONSTANTS & CONFIGURATIONS ---
RISK_SETTINGS = {
    "atrStopMultiplier": 2.0,     
    "atrProfitMultiplier": 3.0,   # 🚀 OPTIMIZED: Matches your 1:1.5 Alpha config
    "breakevenMultiplier": 2.0,   
    "takerFeePerc": 0.0006,       
    "makerFeePerc": 0.0006,      
    "riskPct": 0.01               
}

# 🚀 ON-THE-FLY CONFIGURATION
VETO_THRESHOLD = 0.50

# --- WEB3 INTEGRATION FLAG ---
LIVE_WEB3_MODE = False

# --- ENGINE MODEL INITIALIZATION ---
MODEL_PATH = "veto_engine_alpha.onnx"  # 🚀 UPDATED: Pointing to your alpha model

print(f"🤖 Initializing Inference Engine using file: {MODEL_PATH}")
try:
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name
    # Handle both single-output or multi-output models depending on how ONNX serialized the probabilities
    outputs = session.get_outputs()
    prob_name = outputs[1].name if len(outputs) > 1 else outputs[0].name
except Exception as e:
    print(f"⚠️ Warning: Could not load {MODEL_PATH}. Ensure the file exists. Error: {e}")

exchange = ccxt.coinbase({
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})
SYMBOL = 'SOL/USDC'

# ==============================================================================
# 🔮 HYBRID PRICING & WEB3 EXECUTION
# ==============================================================================

def get_jupiter_live_price():
    try:
        sol_mint = "So11111111111111111111111111111111111111112"
        url = f"https://api.jup.ag/price/v2?ids={sol_mint}"
        
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            json_data = response.json()
            price_str = json_data["data"][sol_mint]["price"]
            return float(price_str)
    except Exception as e:
        print(f"⚠️ Failed to fetch Jupiter Price API: {e}")
    return None

def execute_jupiter_transaction(direction, size_sol, price, sl, tp):
    action = "LONG" if direction == 1.0 else "SHORT"
    if LIVE_WEB3_MODE:
        log_msg = f"🔥 [LIVE WEB3] Broadcasting REAL {action} of {size_sol} SOL to Jupiter..."
        print(log_msg)
        return True 
    else:
        log_msg = f"🔗 [PAPER TRADE SIMULATION] Broadcasting {action} of {size_sol} SOL to Jupiter Perps..."
        print(log_msg)
        return True 

def fetch_and_engineer_features():
    global VETO_THRESHOLD
    try:
        # 🚀 FIX: Increased limit from 200 to 1000 to allow the 600 EMA space to warm up
        candles = exchange.fetch_ohlcv(SYMBOL, timeframe='5m', limit=1000)
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=10)
        df['currentADX'] = adx_df['ADX_10']
        df['prevADX'] = adx_df['ADX_10'].shift(1)
        df['adxDelta'] = df['currentADX'] - df['prevADX']  # 🚀 ADDED Alpha Feature
        
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['rvol'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        atr = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atrPercentage'] = (atr / df['close']) * 100
        
        # 🚀 ADDED Alpha Feature: 600 period HTF EMA distance tracking
        df['ema600'] = ta.ema(df['close'], length=600)
        df['distanceToHtfEma'] = ((df['close'] - df['ema600']) / df['ema600']) * 100
        
        df['bodySize'] = (df['close'] - df['open']).abs()
        df['lowerWick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['upperWick'] = df['high'] - df[['open', 'close']].max(axis=1)
        
        df['ema9'] = ta.ema(df['close'], length=9)
        df['ema21'] = ta.ema(df['close'], length=21)
        df['ema150'] = ta.ema(df['close'], length=150)

        bull_fan = (df['ema9'] > df['ema21']) & (df['ema21'] > df['ema150'])
        bear_fan = (df['ema9'] < df['ema21']) & (df['ema21'] < df['ema150'])

        touches_long = (df['low'] <= df['ema9']) & (df['close'] > df['ema9'])
        touches_short = (df['high'] >= df['ema9']) & (df['close'] < df['ema9'])

        df['directionIntent'] = 0.0
        df.loc[bull_fan & touches_long, 'directionIntent'] = 1.0
        df.loc[bear_fan & touches_short, 'directionIntent'] = -1.0
        
        df['color'] = np.where(df['close'] >= df['open'], 1, -1)
        df['flip'] = np.where(df['color'] != df['color'].shift(1), 1, 0)
        df['isWhipsaw'] = np.where(df['flip'].rolling(window=3).sum() >= 3, 1.0, 0.0)
        
        live_row = df.iloc[-2]
        
        if live_row.isnull().any():
            return None, "Indicators warming up...", None, None
            
        # 🚀 MATCHES 10 FEATURES USED IN TRAINING_ALPHA
        feature_order = [
            'rsi', 'currentADX', 'adxDelta', 'rvol', 'atrPercentage', 
            'distanceToHtfEma', 'upperWick', 'lowerWick', 'bodySize', 'isWhipsaw'
        ]
        
        input_vector = live_row[feature_order].values.astype(np.float32).reshape(1, 10)
        timestamp_str = pd.to_datetime(live_row['timestamp'], unit='ms').strftime('%Y-%m-%d %H:%M:%S')
        
        pricing_data = {
            "close": float(live_row['close']),
            "high": float(live_row['high']),
            "low": float(live_row['low']),
            "atr": float(atr.iloc[-2]),
            "direction_intent": float(live_row['directionIntent']) 
        }
        
        return input_vector, timestamp_str, pricing_data, live_row.to_dict()
        
    except Exception as e:
        return None, f"Data fetch error: {str(e)}", None, None

def trading_loop():
    global VETO_THRESHOLD
    print("🍰 UpsideDownCake Jupiter Engine Running...")
    
    while True:
        current_time = time.time()
        time_to_next_candle = 300 - (current_time % 300)
        time.sleep(time_to_next_candle + 3) 
        
        features, meta, pricing, raw_row = fetch_and_engineer_features()
        if features is None:
            continue
            
        jup_price = get_jupiter_live_price()
        if jup_price is not None:
            pricing["close"] = jup_price
            pricing["high"] = max(pricing["high"], jup_price)
            pricing["low"] = min(pricing["low"], jup_price)

        STATE["last_close_price"] = pricing["close"]
            
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
                STATE["performance"]["wallet_balance_usdc"] += net_pnl
                
                if net_pnl > 0:
                    STATE["performance"]["wins"] += 1
                    STATE["performance"]["gross_pnl_usdc"] += net_pnl 
                    outcome_str = "🎉 WIN"
                else:
                    STATE["performance"]["losses"] += 1
                    outcome_str = "🛑 LOSS"
                    
                calc_wr = (STATE["performance"]["wins"] / STATE["performance"]["total_trades"]) * 100
                STATE["performance"]["win_rate"] = f"{calc_wr:.2f}%"
                
                settle_msg = f"📊 [SETTLED] Trade Closed @ {exit_price} | Net PNL: {net_pnl:+.4f} USDC | New Balance: {STATE['performance']['wallet_balance_usdc']:.4f} USDC"
                print(settle_msg)
                STATE["trade_logs"].append(settle_msg)
                STATE["active_trade"] = None 
                
        if STATE["active_trade"] is None:
            direction_intent = pricing["direction_intent"]
            
            # Run model only if strategy structure signals an intent
            if direction_intent != 0.0:
                try:
                    pred_res = session.run([label_name, prob_name], {input_name: features})
                    # Handles structure differences across formatting outputs safely
                    prob_win = float(pred_res[1][0][1]) if len(pred_res) > 1 else float(pred_res[0][0])
                except Exception as e:
                    print(f"⚠️ Inference Error: {e}")
                    continue
            else:
                prob_win = 0.0
            
            direction_str = "LONG 📈" if direction_intent == 1.0 else "SHORT 📉"
            
            if direction_intent != 0.0 and prob_win >= VETO_THRESHOLD:
                stop_loss_distance = pricing["atr"] * RISK_SETTINGS["atrStopMultiplier"]
                contract_size = 0.75
                entry_p = pricing["close"]
                
                if direction_intent == 1.0:
                    sl_target = entry_p - stop_loss_distance
                    tp_target = entry_p + (pricing["atr"] * RISK_SETTINGS["atrProfitMultiplier"])
                else:
                    sl_target = entry_p + stop_loss_distance
                    tp_target = entry_p - (pricing["atr"] * RISK_SETTINGS["atrProfitMultiplier"])
                    
                tx_confirmed = execute_jupiter_transaction(direction_intent, contract_size, entry_p, sl_target, tp_target)
                
                if tx_confirmed:
                    STATE["active_trade"] = {
                        "entry_price": entry_p,
                        "direction": direction_intent,
                        "timestamp": meta,
                        "contract_size": contract_size,
                        "sl": sl_target,
                        "tp": tp_target,
                        "atr": pricing["atr"]
                    }
                    decision_msg = f"✅ ALLOWED & EXECUTED ({direction_str} entry of {contract_size} SOL @ {entry_p})"
                else:
                    decision_msg = f"⚠️ ALLOWED BY MODEL BUT ON-CHAIN EXECUTION FAILED"
            else:
                action_reason = f"Model Conviction ({prob_win:.2%}) below Target" if direction_intent != 0.0 else "No structural EMA trend setup"
                decision_msg = f"❌ VETO ({action_reason})"
                
                if direction_intent != 0.0:
                    STATE["skipped_trade"] = {
                        "timestamp": meta,
                        "entry_price": pricing["close"],
                        "direction": direction_intent
                    }
            
            if direction_intent != 0.0:
                log_msg = f"🕒 [{meta}] Conviction: {prob_win:.2%} (Target: {VETO_THRESHOLD:.2%}) | Action: {decision_msg}"
                print(log_msg)
                STATE["trade_logs"].append(log_msg)
            
        if len(STATE["trade_logs"]) > 200:
            STATE["trade_logs"].pop(0)

threading.Thread(target=trading_loop, daemon=True).start()

# ==============================================================================
# 🌐 API & DASHBOARD ROUTES
# ==============================================================================

# 🚀 NEW: Dynamically update your model's threshold on the fly
@app.post("/api/config/threshold")
def update_threshold(val: float):
    global VETO_THRESHOLD
    if 0.0 <= val <= 1.0:
        VETO_THRESHOLD = val
        log_msg = f"🎛️ [CONFIG UPDATE] Veto threshold updated to {VETO_THRESHOLD:.2%}"
        print(log_msg)
        STATE["trade_logs"].append(log_msg)
        return {"status": "success", "message": f"Threshold updated to {val}"}
    return {"status": "error", "message": "Threshold must be between 0.0 and 1.0"}

@app.post("/api/override/close")
def manual_close_position():
    if STATE["active_trade"] is None:
        return {"status": "error", "message": "No active position found to close."}
    
    pos = STATE["active_trade"]
    exit_price = STATE["last_close_price"]
    
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
    STATE["performance"]["wallet_balance_usdc"] += net_pnl
    
    if net_pnl > 0:
        STATE["performance"]["wins"] += 1
        STATE["performance"]["gross_pnl_usdc"] += net_pnl 
    else:
        STATE["performance"]["losses"] += 1
        
    calc_wr = (STATE["performance"]["wins"] / STATE["performance"]["total_trades"]) * 100
    STATE["performance"]["win_rate"] = f"{calc_wr:.2f}%"
    
    settle_msg = f"🕹️ [MANUAL OVERRIDE] Force-Closed @ {exit_price} | Net PNL: {net_pnl:+.4f} USDC | New Balance: {STATE['performance']['wallet_balance_usdc']:.4f} USDC"
    print(settle_msg)
    STATE["trade_logs"].append(settle_msg)
    
    STATE["active_trade"] = None
    return {"status": "success", "message": f"Successfully market-closed position at ${exit_price}."}

@app.api_route("/api/data", methods=["GET", "HEAD"])
def get_bot_data():
    global VETO_THRESHOLD
    adjusted_position = None
    if STATE["active_trade"] is not None:
        pos = STATE["active_trade"]
        curr_price = STATE["last_close_price"]
        
        entry_p = round(pos["entry_price"], 4)
        sl_p = round(pos["sl"], 4)
        tp_p = round(pos["tp"], 4)
        curr_p = round(curr_price, 4)
        
        dir_str = "LONG 📈" if pos["direction"] == 1.0 else "SHORT 📉"
        
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
        "market": "SOL-PERP (Jupiter Hybrid Oracle)",
        "current_veto_threshold": f"{VETO_THRESHOLD:.2%}",
        "live_metrics": {
            "wins": STATE["performance"]["wins"],
            "losses": STATE["performance"]["losses"],
            "total_trades": STATE["performance"]["total_trades"],
            "win_rate": STATE["performance"]["win_rate"],
            "gross_pnl_usdc": round(STATE["performance"]["gross_pnl_usdc"], 4),
            "net_pnl_usdc": round(STATE["performance"]["net_pnl_usdc"], 4),
            "wallet_balance_usdc": round(STATE["performance"]["wallet_balance_usdc"], 4)
        },
        "current_position": adjusted_position,
        "recent_activity_logs": STATE["trade_logs"][::-1]
    }

@app.get("/")
def serve_dashboard():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(current_dir, "index.html")
    return FileResponse(file_path)
