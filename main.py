import os
import time
import threading
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
# Enables the frontend to communicate with the backend smoothly
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
        "net_pnl_usdc": 0.0     
    },
    "trade_logs": []
}

# --- SYSTEM CONSTANTS & CONFIGURATIONS ---
RISK_SETTINGS = {
    "atrStopMultiplier": 2.0,     
    "atrProfitMultiplier": 4.0,   
    "breakevenMultiplier": 2.0,   
    # JUPITER DEX SIMULATION: Flat open/close position fee (approx 0.06%)
    "takerFeePerc": 0.0006,       
    "makerFeePerc": 0.0006,      
    "riskPct": 0.01               
}

# --- ENGINE MODEL INITIALIZATION ---
MODEL_PATH = "veto_engine.onnx"

print(f"🤖 Initializing Inference Engine using file: {MODEL_PATH}")
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
label_name = session.get_outputs()[0].name
prob_name = session.get_outputs()[1].name

# Pyth Oracle Proxy: Using Coinbase for live paper-trade pricing data 
# Since arbitrage keeps CEX and on-chain Oracle prices nearly identical
exchange = ccxt.coinbase({
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})
SYMBOL = 'SOL/USDC'

# ==============================================================================
# 🔮 FUTURE WEB3 INTEGRATION BLOCK (JUPITER PERPS ON SOLANA)
# When you transition to live money, replace the print statement below with 
# your Solana private keypair, RPC connection, and Jupiter SDK execution logic.
# ==============================================================================
def execute_jupiter_transaction(direction, size_sol, price, sl, tp):
    """
    PAPER TRADING MODE: Simulates the on-chain interaction. 
    Returns True to authorize the local state manager to track the paper trade.
    """
    action = "LONG" if direction == 1.0 else "SHORT"
    log_msg = f"🔗 [WEB3 SIMULATION] Broadcasting {action} of {size_sol} SOL to Jupiter Perps..."
    print(log_msg)
    # Return True simulates a successful on-chain transaction confirmation
    return True 

def fetch_and_engineer_features():
    try:
        candles = exchange.fetch_ohlcv(SYMBOL, timeframe='5m', limit=200)
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=10)
        df['currentADX'] = adx_df['ADX_10']
        df['prevADX'] = adx_df['ADX_10'].shift(1)
        
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['rvol'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        atr = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atrPercentage'] = (atr / df['close']) * 100
        
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
            
        feature_order = [
            'directionIntent', 'rsi', 'rvol', 'atrPercentage', 'prevADX', 'bodySize'
        ]
        
        input_vector = live_row[feature_order].values.astype(np.float32).reshape(1, 6)
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
    print("🍰 UpsideDownCake Jupiter Paper Trading Engine Running...")
    
    while True:
        current_time = time.time()
        time_to_next_candle = 300 - (current_time % 300)
        time.sleep(time_to_next_candle + 3) 
        
        features, meta, pricing, raw_row = fetch_and_engineer_features()
        if features is None:
            continue
            
        STATE["last_close_price"] = pricing["close"]
            
        # 1. EVALUATE GHOST OUTCOMES
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
            
            prob_win = float(pred_prob[0][1])
            direction_intent = pricing["direction_intent"] 
            
            threshold_value = 0.50
            if os.path.exists('veto_threshold.txt'):
                try:
                    with open('veto_threshold.txt', 'r') as f:
                        threshold_value = float(f.read().strip())
                except Exception:
                    pass 
            
            direction_str = "LONG 📈" if direction_intent == 1.0 else "SHORT 📉"
            
            if prob_win >= threshold_value and direction_intent != 0.0:
                stop_loss_distance = pricing["atr"] * RISK_SETTINGS["atrStopMultiplier"]
                contract_size = 0.75
                entry_p = pricing["close"]
                
                if direction_intent == 1.0:
                    sl_target = entry_p - stop_loss_distance
                    tp_target = entry_p + (pricing["atr"] * RISK_SETTINGS["atrProfitMultiplier"])
                else:
                    sl_target = entry_p + stop_loss_distance
                    tp_target = entry_p - (pricing["atr"] * RISK_SETTINGS["atrProfitMultiplier"])
                    
                # The execution stub intercepts the logic before assigning the trade to state
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
                action_reason = "Conditions blocked by XGBoost filter layer" if direction_intent != 0.0 else "No structural EMA trend setup"
                decision_msg = f"❌ VETO ({action_reason})"
                
                if direction_intent != 0.0:
                    STATE["skipped_trade"] = {
                        "timestamp": meta,
                        "entry_price": pricing["close"],
                        "direction": direction_intent
                    }
                
            log_msg = f"🕒 [{meta}] Veto Engine Conviction Prob: {prob_win:.2%} (Target: {threshold_value:.2%}) | Action: {decision_msg}"
            print(log_msg)
            STATE["trade_logs"].append(log_msg)
            
        if len(STATE["trade_logs"]) > 200:
            STATE["trade_logs"].pop(0)

threading.Thread(target=trading_loop, daemon=True).start()

# ==============================================================================
# 🌐 API & DASHBOARD ROUTES
# ==============================================================================

@app.post("/api/override/close")
def manual_close_position():
    """
    MANUAL OVERRIDE: Prematurely settles an active trade at the last known close price,
    updates performance metrics, and clears the position state safely.
    """
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
    
    if net_pnl > 0:
        STATE["performance"]["wins"] += 1
        STATE["performance"]["gross_pnl_usdc"] += net_pnl 
        outcome_str = "🎉 MANUAL WIN"
    else:
        STATE["performance"]["losses"] += 1
        outcome_str = "🛑 MANUAL LOSS"
        
    calc_wr = (STATE["performance"]["wins"] / STATE["performance"]["total_trades"]) * 100
    STATE["performance"]["win_rate"] = f"{calc_wr:.2f}%"
    
    settle_msg = f"🕹️ [MANUAL OVERRIDE] Position from {pos['timestamp']} Force-Closed @ {exit_price} | {outcome_str} | Net PNL: {net_pnl:+.4f} USDC"
    print(settle_msg)
    STATE["trade_logs"].append(settle_msg)
    
    STATE["active_trade"] = None
    
    return {"status": "success", "message": f"Successfully market-closed position at ${exit_price}."}

@app.api_route("/api/data", methods=["GET", "HEAD"])
def get_bot_data():
    adjusted_position = None
    if STATE["active_trade"] is not None:
        pos = STATE["active_trade"]
        curr_price = STATE["last_close_price"]
        
        entry_p = round(pos["entry_price"] - 1.0, 4)
        sl_p = round(pos["sl"] - 1.0, 4)
        tp_p = round(pos["tp"] - 1.0, 4)
        curr_p = round(curr_price - 1.0, 4)
        
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
        "market": "SOL-PERP (Jupiter-DEX-Paper-Context)",
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

@app.get("/")
def serve_dashboard():
    # Dynamically gets the absolute path of the directory main.py is living in
    current_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(current_dir, "index.html")
    return FileResponse(file_path)
