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
LIVE_WEB3_MODE = False

# 🚀 DISCORD WEBHOOK URLS
DISCORD_WEBHOOK_EXECUTIONS = "https://discord.com/api/webhooks/1526005962823962758/mlxDLG2JxPRV0VWkOoTcYcuIW76dX1cXuRLwWaw3wLNMvk80FxTzKRdgjnoTB9rl_vXH"
DISCORD_WEBHOOK_VETOES = "https://discord.com/api/webhooks/1526006069971783903/3oDB8tECDCs1DUp1Z6YlNdgsrkndFdHmSRpR_1KcFvJ4wdaTYTHuGDq-xwCb1NSqKiNt"
DISCORD_WEBHOOK_GHOSTS = "https://discord.com/api/webhooks/1526006633296166962/h7tDGT9JZEqcb2IdWtb2SW_IJHd75nqrSFg7th-IPl2LreN07jccaIoMX-lhKSl7wmbZ"

# --- ENGINE MODEL INITIALIZATION ---
MODEL_PATH = "veto_engine_alpha.onnx"  

print(f"🤖 Initializing Inference Engine using file: {MODEL_PATH}")
try:
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name
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
# 📢 DISCORD NOTIFICATION HELPER
# ==============================================================================

def _post_webhook(url, data):
    try:
        requests.post(url, json=data, timeout=5)
    except Exception as e:
        print(f"⚠️ Discord Webhook Error: {e}")

def send_discord_webhook(url, title, description, color, fields=None):
    if not url: return
    data = {
        "embeds": [{
            "title": title,
            "description": description,
            "color": color,
            "fields": fields or [],
            "footer": {"text": f"UpsideDownCake AI Engine • {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}"}
        }]
    }
    # Run in a background thread so it doesn't block your trading loop!
    threading.Thread(target=_post_webhook, args=(url, data), daemon=True).start()

# ==============================================================================
# 🔮 HYBRID PRICING & WEB3 EXECUTION
# ==============================================================================

def get_jupiter_live_price():
    try:
        sol_mint = "So11111111111111111111111111111111111111112"
        url = f"https://api.jup.ag/price/v3?ids={sol_mint}"
        
        headers = {
            "x-api-key": "YOUR_JUPITER_API_KEY", 
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json"
        }

        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            json_data = response.json()
            price_map = json_data.get("data", json_data)
            
            if sol_mint in price_map:
                token_data = price_map[sol_mint]
                price_str = token_data.get("usdPrice", token_data.get("price"))
                if price_str is not None:
                    return float(price_str)
                else:
                    print(f"⚠️ [DEBUG] Missing 'usdPrice' in Jupiter response: {token_data}")
            else:
                print(f"⚠️ [DEBUG] Missing expected V3 key in Jupiter response: {json_data}")
        else:
            print(f"⚠️ [DEBUG] Jupiter API returned Status Code: {response.status_code}")
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
        all_candles = []
        batch_limit = 300
        now = exchange.milliseconds()
        
        since = now - (1500 * 5 * 60 * 1000) 
        
        for _ in range(5):  
            batch = exchange.fetch_ohlcv(SYMBOL, timeframe='5m', since=since, limit=batch_limit)
            if not batch:
                break
            all_candles.extend(batch)
            since = batch[-1][0] + (5 * 60 * 1000)
            time.sleep(5)  
            
        if len(all_candles) == 0:
            return None, "No data fetched from exchange", None, None
            
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df.drop_duplicates(subset=['timestamp'], inplace=True)
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=10)
        df['currentADX'] = adx_df['ADX_10']
        df['prevADX'] = adx_df['ADX_10'].shift(1)
        df['adxDelta'] = df['currentADX'] - df['prevADX']  
        
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['rvol'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        atr = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atrPercentage'] = (atr / df['close']) * 100
        
        df['ema600'] = ta.ema(df['close'], length=600)
        df['distanceToHtfEma'] = ((df['close'] - df['ema600']) / df['ema600']) * 100
        
        df['bodySize'] = (df['close'] - df['open']).abs()
        df['lowerWick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['upperWick'] = df['high'] - df[['open', 'close']].max(axis=1)
        
        df['ema9'] = ta.ema(df['close'], length=9)
        df['ema21'] = ta.ema(df['close'], length=21)
        df['ema150'] = ta.ema(df['close'], length=150)

        buffer = df['ema9'] * 0.0005 
        
        bull_fan = (df['ema9'] > df['ema21']) & (df['ema21'] > df['ema150'])
        bear_fan = (df['ema9'] < df['ema21']) & (df['ema21'] < df['ema150'])

        touches_long = (df['low'] <= (df['ema9'] + buffer)) & (df['close'] > df['ema9'])
        touches_short = (df['high'] >= (df['ema9'] - buffer)) & (df['close'] < df['ema9'])

        df['directionIntent'] = 0.0
        df.loc[bull_fan & touches_long, 'directionIntent'] = 1.0
        df.loc[bear_fan & touches_short, 'directionIntent'] = -1.0
        
        df['color'] = np.where(df['close'] >= df['open'], 1, -1)
        df['flip'] = np.where(df['color'] != df['color'].shift(1), 1, 0)
        df['isWhipsaw'] = np.where(df['flip'].rolling(window=3).sum() >= 3, 1.0, 0.0)
        
        now_ms = time.time() * 1000
        last_candle_time = df.iloc[-1]['timestamp']
        
        if (now_ms - last_candle_time) < 300000:
            live_row = df.iloc[-2]
        else:
            live_row = df.iloc[-1]
            
        if live_row.isnull().any():
            return None, "Indicators warming up...", None, None
            
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
            "atr": float(atr.iloc[-1] if (now_ms - last_candle_time) >= 300000 else atr.iloc[-2]),
            "direction_intent": float(live_row['directionIntent']) 
        }
        
        return input_vector, timestamp_str, pricing_data, live_row.to_dict()
        
    except Exception as e:
        return None, f"Data fetch error: {str(e)}", None, None

def trading_loop():
    global VETO_THRESHOLD
    print("🍰 UpsideDownCake Jupiter Engine Running...")
    
    first_run = True 

    while True:
        current_time = time.time()
        time_to_next_candle = 300 - (current_time % 300)
        
        if first_run:
            print("\n🚀 [DEBUG] First execution! Bypassing the 5-minute sync delay to test pipeline immediately...")
            first_run = False
        else:
            print(f"\n⏳ [DEBUG] Sleeping for {time_to_next_candle:.0f} seconds until next 5m candle close...")
            time.sleep(time_to_next_candle + 15) 
        
        print("🔄 [DEBUG] Fetching exchange data and engineering 1500 candles...")
        features, meta, pricing, raw_row = fetch_and_engineer_features()
        
        if features is None:
            print(f"⚠️ [DEBUG] Data fetch failed or warming up: {meta}")
            time.sleep(10) 
            continue
            
        print(f"✅ [DEBUG] Data loaded! Timestamp: {meta} | Exchange Close: ${pricing['close']:.4f}")

        jup_price = get_jupiter_live_price()
        if jup_price is not None:
            print(f"🦎 [DEBUG] Jupiter Web3 Price fetched: ${jup_price:.4f}")
            pricing["close"] = jup_price
            pricing["high"] = max(pricing["high"], jup_price)
            pricing["low"] = min(pricing["low"], jup_price)
        else:
            print("⚠️ [DEBUG] Could not fetch Jupiter price, falling back to Exchange data.")

        STATE["last_close_price"] = pricing["close"]
            
        # --- GHOST SETTLEMENT LOGIC ---
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
            
            # 📢 WEBHOOK: Ghost Trade Resolution
            color = 0x00FF00 if "WIN" in outcome else 0xFF0000
            dir_str = "LONG 📈" if skip_pos["direction"] == 1.0 else "SHORT 📉"
            fields = [
                {"name": "Direction", "value": dir_str, "inline": True},
                {"name": "Theoretical Entry", "value": f"${skip_pos['entry_price']:.4f}", "inline": True},
                {"name": "Resolution Price", "value": f"${pricing['close']:.4f}", "inline": True},
                {"name": "Ghost Outcome", "value": outcome, "inline": False}
            ]
            send_discord_webhook(DISCORD_WEBHOOK_GHOSTS, "👻 Ghost Trade Resolved", log_msg, color, fields)

            STATE["skipped_trade"] = None
            
        # --- ACTIVE TRADE MANAGEMENT ---
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
                
                # 📢 WEBHOOK: Live Trade Settlement
                color = 0x00FF00 if net_pnl > 0 else 0xFF0000
                dir_str = "LONG 📈" if pos["direction"] == 1.0 else "SHORT 📉"
                fields = [
                    {"name": "Direction", "value": dir_str, "inline": True},
                    {"name": "Entry Price", "value": f"${pos['entry_price']:.4f}", "inline": True},
                    {"name": "Exit Price", "value": f"${exit_price:.4f}", "inline": True},
                    {"name": "Net PNL", "value": f"{net_pnl:+.4f} USDC", "inline": True},
                    {"name": "New Balance", "value": f"{STATE['performance']['wallet_balance_usdc']:.4f} USDC", "inline": True},
                    {"name": "Overall Win Rate", "value": STATE["performance"]["win_rate"], "inline": True}
                ]
                send_discord_webhook(DISCORD_WEBHOOK_EXECUTIONS, f"📊 Trade Settled: {outcome_str}", settle_msg, color, fields)

                STATE["active_trade"] = None 
                
        # --- NEW ENTRY EVALUATION LOGIC ---
        if STATE["active_trade"] is None:
            direction_intent = pricing["direction_intent"]
            print(f"🧭 [DEBUG] EMA Trend Direction Intent: {direction_intent}")
            
            try:
                pred_res = session.run([label_name, prob_name], {input_name: features})
                prob_win = float(pred_res[1][0][1]) if len(pred_res) > 1 else float(pred_res[0][0])
                print(f"🧠 [DEBUG] ONNX Engine Prediction -> Win Probability: {prob_win:.2%}")
            except Exception as e:
                print(f"⚠️ [DEBUG] Inference Error: {e}")
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
                    
                    # 📢 WEBHOOK: New Trade Executed
                    color = 0x00FF00 if direction_intent == 1.0 else 0xFF0000
                    fields = [
                        {"name": "Direction", "value": direction_str, "inline": True},
                        {"name": "Entry Price", "value": f"${entry_p:.4f}", "inline": True},
                        {"name": "Position Size", "value": f"{contract_size} SOL", "inline": True},
                        {"name": "Stop Loss", "value": f"${sl_target:.4f}", "inline": True},
                        {"name": "Take Profit", "value": f"${tp_target:.4f}", "inline": True},
                        {"name": "Conviction", "value": f"{prob_win:.2%}", "inline": True},
                        {"name": "Current ATR", "value": f"${pricing['atr']:.4f}", "inline": True}
                    ]
                    title = "🟢 LONG Executed" if direction_intent == 1.0 else "🔴 SHORT Executed"
                    send_discord_webhook(DISCORD_WEBHOOK_EXECUTIONS, title, decision_msg, color, fields)

                else:
                    decision_msg = f"⚠️ ALLOWED BY MODEL BUT ON-CHAIN EXECUTION FAILED"
            else:
                action_reason = f"Model Conviction ({prob_win:.2%}) below Target" if direction_intent != 0.0 else "No structural EMA trend setup (Intent is 0)"
                decision_msg = f"❌ VETO ({action_reason})"
                
                if direction_intent != 0.0:
                    STATE["skipped_trade"] = {
                        "timestamp": meta,
                        "entry_price": pricing["close"],
                        "direction": direction_intent
                    }
                    
                    # 📢 WEBHOOK: Trade Vetoed by Model
                    color = 0xFFA500 # Orange
                    fields = [
                        {"name": "Direction", "value": direction_str, "inline": True},
                        {"name": "Current Price", "value": f"${pricing['close']:.4f}", "inline": True},
                        {"name": "Conviction", "value": f"{prob_win:.2%} (Needs {VETO_THRESHOLD:.2%})", "inline": True},
                        {"name": "Current ATR", "value": f"${pricing['atr']:.4f}", "inline": True},
                        {"name": "Action", "value": "Tracked as Ghost Setup 👻", "inline": True}
                    ]
                    send_discord_webhook(DISCORD_WEBHOOK_VETOES, "⚠️ Trade Vetoed", decision_msg, color, fields)
            
            log_msg = f"🕒 [{meta}] Conviction: {prob_win:.2%} (Target: {VETO_THRESHOLD:.2%}) | Action: {decision_msg}"
            print(log_msg)
            STATE["trade_logs"].append(log_msg)
            
        if len(STATE["trade_logs"]) > 200:
            STATE["trade_logs"].pop(0)

threading.Thread(target=trading_loop, daemon=True).start()

# ==============================================================================
# 🌐 API & DASHBOARD ROUTES
# ==============================================================================

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
    
    # 📢 WEBHOOK: Manual Override Closure
    color = 0x00FF00 if net_pnl > 0 else 0xFF0000
    dir_str = "LONG 📈" if pos["direction"] == 1.0 else "SHORT 📉"
    fields = [
        {"name": "Direction", "value": dir_str, "inline": True},
        {"name": "Entry Price", "value": f"${pos['entry_price']:.4f}", "inline": True},
        {"name": "Exit Price (Force)", "value": f"${exit_price:.4f}", "inline": True},
        {"name": "Net PNL", "value": f"{net_pnl:+.4f} USDC", "inline": True},
        {"name": "New Balance", "value": f"{STATE['performance']['wallet_balance_usdc']:.4f} USDC", "inline": True}
    ]
    send_discord_webhook(DISCORD_WEBHOOK_EXECUTIONS, "🕹️ Manual Force Close", settle_msg, color, fields)
    
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

@app.head("/")
@app.get("/")
def serve_dashboard():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(current_dir, "index.html")
    return FileResponse(file_path)
