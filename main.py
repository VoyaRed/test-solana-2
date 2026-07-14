import os
import time
import threading
import requests
import numpy as np
import pandas as pd
import pandas_ta as ta
import onnxruntime as ort
import ccxt
from fastapi import FastAPI, Path
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# --- MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- INDEPENDENT GLOBAL STATES ---
def create_default_state(initial_balance):
    return {
        "active_trade": None,  
        "skipped_trade": None, 
        "last_close_price": 0.0,
        "performance": {
            "trade_id_counter": 0,  
            "wins": 0, "losses": 0, "consecutive_losses": 0,
            "total_trades": 0, "win_rate": "0.00%",
            "gross_pnl_usdc": 0.0, "net_pnl_usdc": 0.0,
            "wallet_balance_usdc": initial_balance  
        },
        "trade_logs": []
    }

STATE_SOL = create_default_state(1000.00)
STATE_CB = create_default_state(1000.00)

# --- INDEPENDENT CONFIGURATIONS ---
RISK_SETTINGS = {
    "atrStopMultiplier": 2.0,     
    "atrProfitMultiplier": 3.0,   
    "takerFeePerc": 0.0006,       
    "makerFeePerc": 0.0006,      
    "riskPct": 0.02              
}

CONFIG_SOL = { "veto": 0.50, "leverage": 10.0, "margin": 1.0, "live_mode": False }
CONFIG_CB = { "veto": 0.50, "leverage": 10.0, "margin": 0.05, "live_mode": False } # Using 0.05 BTC as default margin mock

NEPTUNE_WALLET_PRIVATE_KEY = os.getenv("NEPTUNE_WALLET_PRIVATE_KEY")
DISCORD_WEBHOOK_EXECUTIONS = os.getenv("DISCORD_WEBHOOK_EXECUTIONS", "") # Replaced static URLs for safety
DISCORD_WEBHOOK_VETOES = os.getenv("DISCORD_WEBHOOK_VETOES", "")
DISCORD_WEBHOOK_GHOSTS = os.getenv("DISCORD_WEBHOOK_GHOSTS", "")

# --- MODEL INITIALIZATION ---
MODEL_SOL = "veto_engine_alpha.onnx"  
MODEL_CB = "veto_engine_cb.onnx"

def load_onnx_session(path, name):
    print(f"🤖 initializing {name} engine using file: {path}")
    try:
        session = ort.InferenceSession(path)
        input_name = session.get_inputs()[0].name
        label_name = session.get_outputs()[0].name
        outputs = session.get_outputs()
        prob_name = outputs[1].name if len(outputs) > 1 else outputs[0].name
        return session, input_name, label_name, prob_name
    except Exception as e:
        print(f"⚠️ warning: could not load {path}. error: {e}")
        return None, None, None, None

session_sol, in_sol, lbl_sol, prob_sol = load_onnx_session(MODEL_SOL, "solana perps")
session_cb, in_cb, lbl_cb, prob_cb = load_onnx_session(MODEL_CB, "coinbase perps")

# Two separate CCXT instances to avoid rate limit/thread collisions
exchange_sol = ccxt.coinbase({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
exchange_cb = ccxt.coinbase({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})

SYMBOL_SOL = 'SOL/USDC'
SYMBOL_CB = 'BTC/USDC' # Using BTC for the Coinbase bot distinction

# ==============================================================================
# 📢 DISCORD NOTIFICATION HELPER
# ==============================================================================
def _post_webhook(url, data):
    try:
        requests.post(url, json=data, timeout=5)
    except Exception as e:
        print(f"⚠️ discord webhook error: {e}")

def send_discord_webhook(url, title, description, color, fields=None):
    if not url: return
    data = {
        "embeds": [{
            "title": title, "description": description, "color": color,
            "fields": fields or [],
            "footer": {"text": f"system engine • {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}"}
        }]
    }
    threading.Thread(target=_post_webhook, args=(url, data), daemon=True).start()

# ==============================================================================
# 🔮 PRICING & MOCK EXECUTION LOGIC
# ==============================================================================
def get_jupiter_live_price():
    # Placeholder for Jupiter API fetching
    return None

def execute_transaction(network, direction, size, price, sl, tp, leverage, margin, is_live):
    action = "long" if direction == 1.0 else "short"
    if is_live:
        print(f"🔥 [{network} live web3] broadcasting {action} of {size} units")
        return True 
    else:
        print(f"🔗 [{network} paper] emulating {action} of {size} units...")
        return True 

def fetch_and_engineer_features(exchange, symbol):
    try:
        all_candles = []
        now = exchange.milliseconds()
        since = now - (1500 * 5 * 60 * 1000) 
        
        for _ in range(5):  
            batch = exchange.fetch_ohlcv(symbol, timeframe='5m', since=since, limit=300)
            if not batch: break
            all_candles.extend(batch)
            since = batch[-1][0] + (5 * 60 * 1000)
            time.sleep(1)  
            
        if len(all_candles) == 0:
            return None, "no data", None, None
            
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
        
        df['ema9'], df['ema21'], df['ema150'] = ta.ema(df['close'], length=9), ta.ema(df['close'], length=21), ta.ema(df['close'], length=150)

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
        live_row = df.iloc[-2] if (now_ms - last_candle_time) < 300000 else df.iloc[-1]
            
        if live_row.isnull().any():
            return None, "warming up...", None, None
            
        feature_order = ['rsi', 'currentADX', 'adxDelta', 'rvol', 'atrPercentage', 'distanceToHtfEma', 'upperWick', 'lowerWick', 'bodySize', 'isWhipsaw']
        input_vector = live_row[feature_order].values.astype(np.float32).reshape(1, 10)
        timestamp_str = pd.to_datetime(live_row['timestamp'], unit='ms').strftime('%Y-%m-%d %H:%M:%S')
        
        pricing_data = {
            "close": float(live_row['close']), "high": float(live_row['high']), "low": float(live_row['low']),
            "atr": float(atr.iloc[-1] if (now_ms - last_candle_time) >= 300000 else atr.iloc[-2]),
            "direction_intent": float(live_row['directionIntent']) 
        }
        return input_vector, timestamp_str, pricing_data, live_row.to_dict()
    except Exception as e:
        return None, f"data error: {e}", None, None

# ==============================================================================
# 🤖 UNIVERSAL TRADING ENGINE LOOP
# ==============================================================================
def engine_loop(bot_name, symbol, exchange, session, in_name, lbl_name, prob_name, state, config):
    print(f"🌊 {bot_name} engine running for {symbol}...")
    first_run = True 

    while True:
        time_to_next_candle = 300 - (time.time() % 300)
        if first_run: first_run = False
        else: time.sleep(time_to_next_candle + 15) 
        
        features, meta, pricing, raw_row = fetch_and_engineer_features(exchange, symbol)
        if features is None:
            time.sleep(10) 
            continue
            
        if bot_name == "SOL":
            jup_price = get_jupiter_live_price()
            if jup_price:
                pricing["close"] = jup_price
                pricing["high"], pricing["low"] = max(pricing["high"], jup_price), min(pricing["low"], jup_price)

        state["last_close_price"] = pricing["close"]
            
        # Ghost Log Settlement
        if state["skipped_trade"] is not None:
            skip_pos = state["skipped_trade"]
            if (skip_pos["direction"] == 1.0 and pricing["close"] > skip_pos["entry_price"]) or \
               (skip_pos["direction"] == -1.0 and pricing["close"] < skip_pos["entry_price"]):
                outcome = "skip / win"
            else: outcome = "skip / loss"
                
            log_msg = f"👻 [ghost] {bot_name} resolved from {skip_pos['timestamp']}: {outcome}"
            state["trade_logs"].append(log_msg)
            state["skipped_trade"] = None
            
        # Active Trade Management
        if state["active_trade"] is not None:
            pos = state["active_trade"]
            trade_closed, exit_price = False, 0.0
            
            if pos["direction"] == 1.0: 
                if pricing["low"] <= pos["sl"]: exit_price, trade_closed = pos["sl"], True
                elif pricing["high"] >= pos["tp"]: exit_price, trade_closed = pos["tp"], True
            else: 
                if pricing["high"] >= pos["sl"]: exit_price, trade_closed = pos["sl"], True
                elif pricing["low"] <= pos["tp"]: exit_price, trade_closed = pos["tp"], True
            
            if trade_closed:
                fees = ((pos["entry_price"] * pos["contract_size"]) * RISK_SETTINGS["takerFeePerc"]) + \
                       ((exit_price * pos["contract_size"]) * RISK_SETTINGS["makerFeePerc"])
                
                gross_pnl = ((exit_price - pos["entry_price"]) if pos["direction"] == 1.0 else (pos["entry_price"] - exit_price)) * pos["contract_size"]
                net_pnl = gross_pnl - fees
                
                state["performance"]["total_trades"] += 1
                state["performance"]["net_pnl_usdc"] += net_pnl
                state["performance"]["wallet_balance_usdc"] += net_pnl
                
                if net_pnl > 0:
                    state["performance"]["wins"] += 1
                    state["performance"]["gross_pnl_usdc"] += net_pnl 
                    state["performance"]["consecutive_losses"] = 0
                else:
                    state["performance"]["losses"] += 1
                    state["performance"]["consecutive_losses"] += 1
                    
                calc_wr = (state["performance"]["wins"] / state["performance"]["total_trades"]) * 100
                state["performance"]["win_rate"] = f"{calc_wr:.2f}%"
                
                settle_msg = f"📊 [{bot_name} settled] trade {pos['trade_id']} closed @ {exit_price} | pnl: {net_pnl:+.4f} usdc | balance: {state['performance']['wallet_balance_usdc']:.4f} usdc"
                state["trade_logs"].append(settle_msg)
                state["active_trade"] = None 
                
        # New Entry Evaluation
        if state["active_trade"] is None:
            direction_intent = pricing["direction_intent"]
            try:
                pred_res = session.run([lbl_name, prob_name], {in_name: features}) if session else [[0], [[0, 0]]]
                prob_win = float(pred_res[1][0][1]) if len(pred_res) > 1 else float(pred_res[0][0])
            except:
                prob_win = 0.0
            
            if direction_intent != 0.0 and prob_win >= config["veto"]:
                stop_dist = pricing["atr"] * RISK_SETTINGS["atrStopMultiplier"]
                entry_p = pricing["close"]
                
                applied_risk = RISK_SETTINGS["riskPct"] / 2.0 if state["performance"]["consecutive_losses"] >= 3 else RISK_SETTINGS["riskPct"]
                margin_usdc = state["performance"]["wallet_balance_usdc"] * applied_risk
                dynamic_margin_asset = margin_usdc / entry_p
                contract_size = dynamic_margin_asset * config["leverage"]
                
                config["margin"] = dynamic_margin_asset 
                
                sl_target = entry_p - stop_dist if direction_intent == 1.0 else entry_p + stop_dist
                tp_target = entry_p + (pricing["atr"] * RISK_SETTINGS["atrProfitMultiplier"]) if direction_intent == 1.0 else entry_p - (pricing["atr"] * RISK_SETTINGS["atrProfitMultiplier"])
                    
                tx_confirmed = execute_transaction(bot_name, direction_intent, contract_size, entry_p, sl_target, tp_target, config["leverage"], config["margin"], config["live_mode"])
                
                if tx_confirmed:
                    state["performance"]["trade_id_counter"] += 1
                    state["active_trade"] = {
                        "trade_id": state["performance"]["trade_id_counter"],
                        "entry_price": entry_p, "direction": direction_intent,
                        "timestamp": meta, "contract_size": contract_size,
                        "leverage": config["leverage"], "margin_sol": config["margin"],
                        "sl": sl_target, "tp": tp_target, "atr": pricing["atr"]
                    }
                    decision_msg = f"✅ executed {bot_name} ({contract_size:.4f} units @ {config['leverage']}x)"
                else: decision_msg = f"⚠️ {bot_name} execution failed"
            else:
                reason = f"conviction ({prob_win:.2%})" if direction_intent != 0.0 else "no trend"
                decision_msg = f"❌ veto ({reason})"
                if direction_intent != 0.0:
                    state["skipped_trade"] = {"timestamp": meta, "entry_price": pricing["close"], "direction": direction_intent}
            
            state["trade_logs"].append(f"🕒 [{meta}] {bot_name} action: {decision_msg}")
            
        if len(state["trade_logs"]) > 200: state["trade_logs"].pop(0)

# Spin up independent threads
threading.Thread(target=engine_loop, args=("SOL", SYMBOL_SOL, exchange_sol, session_sol, in_sol, lbl_sol, prob_sol, STATE_SOL, CONFIG_SOL), daemon=True).start()
threading.Thread(target=engine_loop, args=("CB", SYMBOL_CB, exchange_cb, session_cb, in_cb, lbl_cb, prob_cb, STATE_CB, CONFIG_CB), daemon=True).start()

# ==============================================================================
# 🌐 DYNAMIC API ROUTES (SERVES BOTH BOTS)
# ==============================================================================

def get_bot_context(bot_type: str):
    bt = bot_type.lower()
    if bt == "sol": return STATE_SOL, CONFIG_SOL
    elif bt == "cb": return STATE_CB, CONFIG_CB
    return None, None

class TradingModePayload(BaseModel):
    mode: str

@app.post("/api/{bot_type}/settings/mode")
def update_trading_mode(bot_type: str, payload: TradingModePayload):
    state, config = get_bot_context(bot_type)
    if not state: return {"status": "error"}
    
    if payload.mode.upper() == "LIVE":
        config["live_mode"] = True
        state["trade_logs"].append(f"🔥 [{bot_type.upper()}] ui changed environment to live.")
    elif payload.mode.upper() == "PAPER":
        config["live_mode"] = False
        state["trade_logs"].append(f"⚙️ [{bot_type.upper()}] ui changed environment to paper.")
        
    return {"status": "success"}

@app.post("/api/{bot_type}/config/threshold")
def update_threshold(bot_type: str, val: float):
    state, config = get_bot_context(bot_type)
    if state and 0.0 <= val <= 1.0:
        config["veto"] = val
        state["trade_logs"].append(f"🎛️ [{bot_type.upper()}] veto threshold updated to {val:.2%}")
        return {"status": "success"}
    return {"status": "error"}

@app.post("/api/{bot_type}/config/leverage")
def update_leverage(bot_type: str, val: float):
    state, config = get_bot_context(bot_type)
    if state and val >= 1.0:
        config["leverage"] = val
        state["trade_logs"].append(f"🎛️ [{bot_type.upper()}] leverage target updated to {val}x")
        return {"status": "success"}
    return {"status": "error"}

@app.post("/api/{bot_type}/override/close")
def manual_close_position(bot_type: str):
    state, config = get_bot_context(bot_type)
    if not state or state["active_trade"] is None:
        return {"status": "error", "message": "no active position found."}
    
    pos = state["active_trade"]
    exit_price = state["last_close_price"]
    
    fees = ((pos["entry_price"] * pos["contract_size"]) * RISK_SETTINGS["takerFeePerc"]) + ((exit_price * pos["contract_size"]) * RISK_SETTINGS["makerFeePerc"])
    gross_pnl = ((exit_price - pos["entry_price"]) if pos["direction"] == 1.0 else (pos["entry_price"] - exit_price)) * pos["contract_size"]
    net_pnl = gross_pnl - fees
    
    state["performance"]["total_trades"] += 1
    state["performance"]["net_pnl_usdc"] += net_pnl
    state["performance"]["wallet_balance_usdc"] += net_pnl
    
    if net_pnl > 0:
        state["performance"]["wins"] += 1
        state["performance"]["gross_pnl_usdc"] += net_pnl 
        state["performance"]["consecutive_losses"] = 0
    else:
        state["performance"]["losses"] += 1
        state["performance"]["consecutive_losses"] += 1
        
    state["performance"]["win_rate"] = f"{(state['performance']['wins'] / state['performance']['total_trades']) * 100:.2f}%"
    state["trade_logs"].append(f"🕹️ [{bot_type.upper()} override] closed @ {exit_price} | pnl: {net_pnl:+.4f} usdc")
    state["active_trade"] = None
    return {"status": "success"}

@app.api_route("/api/{bot_type}/data", methods=["GET", "HEAD"])
def get_bot_data(bot_type: str):
    state, config = get_bot_context(bot_type)
    if not state: return {"status": "error"}

    adjusted_position = None
    if state["active_trade"] is not None:
        pos = state["active_trade"]
        curr_p = round(state["last_close_price"], 4)
        dir_str = "long 📈" if pos["direction"] == 1.0 else "short 📉"
        
        sl_dist = (pos["sl"] - curr_p) if pos["direction"] == 1.0 else (curr_p - pos["sl"])
        tp_dist = (pos["tp"] - curr_p) if pos["direction"] == 1.0 else (curr_p - pos["tp"])
            
        adjusted_position = {
            "type": dir_str, "bet_size": f"{pos['contract_size']:.4f} units",
            "margin_locked": f"{pos.get('margin_sol', config['margin']):.4f} units",
            "leverage_multiplier": f"{pos.get('leverage', config['leverage'])}x",
            "entry_price": round(pos["entry_price"], 4), "current_price": curr_p,
            "stop_loss": round(pos["sl"], 4), "distance_to_sl": f"{sl_dist:+.4f} usdc",
            "take_profit": round(pos["tp"], 4), "distance_to_tp": f"{tp_dist:+.4f} usdc"
        }
        
    market_name = "project neptune® | solana perps bot" if bot_type.lower() == "sol" else "project neptune® | link perps bot"

    return {
        "status": "online",
        "market": market_name,
        "trading_mode": "LIVE" if config["live_mode"] else "PAPER",
        "current_price": state["last_close_price"],  
        "config_metrics": {
            "current_veto_threshold": f"{config['veto']:.2%}",
            "active_leverage_setting": f"{config['leverage']}x",
            "active_margin_setting": f"{config['margin']:.4f} units"
        },
        "live_metrics": {
            "wins": state["performance"]["wins"], "losses": state["performance"]["losses"],
            "total_trades": state["performance"]["total_trades"], "win_rate": state["performance"]["win_rate"],
            "gross_pnl_usdc": round(state["performance"]["gross_pnl_usdc"], 4),
            "net_pnl_usdc": round(state["performance"]["net_pnl_usdc"], 4),
            "wallet_balance_usdc": round(state["performance"]["wallet_balance_usdc"], 4)
        },
        "current_position": adjusted_position,
        "recent_activity_logs": state["trade_logs"][::-1]
    }

@app.head("/")
@app.get("/")
def serve_dashboard():
    # Replaced FileResponse to prevent FileNotFoundError during platform health checks
    return {"status": "online", "message": "API is running"}
