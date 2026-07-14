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
STATE_LINK = create_default_state(1000.00)

# --- INDEPENDENT BOT CONFIGURATIONS & WEBHOOKS ---

# 1. SOLANA BOT CONFIGURATION (5-Minute)
CONFIG_SOL = { 
    "timeframe": "5m",
    "timeframe_sec": 300,
    "veto": 0.50, 
    "leverage": 10.0, 
    "margin": 1.0, 
    "live_mode": False,
    "risk": {
        "atrStopMultiplier": 2.0,     
        "atrProfitMultiplier": 3.0,   
        "takerFeePerc": 0.0006,       
        "makerFeePerc": 0.0006,      
        "riskPct": 0.02              
    },
    "trade": {
        "timeStopCandles": 9999 
    },
    "webhooks": {
        "execution": "https://discord.com/api/webhooks/1526386725079744763/26M6X4lrbzLDD1y1UYFskeujLPOYjR7H5ToPisyD0kWKChMb_2SwYYxMEk2WLMyZgWCi",
        "veto":      "https://discord.com/api/webhooks/1526386989925011726/DvasexYCeu-NWDiFE4tuHxglgUJbgZKk6LDvtrMNH0qANcxhqhpf1KW5a5E0J7ORoMDn",
        "settle":    "https://discord.com/api/webhooks/1526387397833785435/Jx43isvgxVYSGagONyMBrYc7QzylbArfz9OlyNUCdwncZntSrpF7e-lOKpRuPcjx2qmf",
        "ghost":     "https://discord.com/api/webhooks/1526387315683885058/M6gBr41Rvq1SfX_n535SB8yh5HyyZ_AHGYSRLT_CcITXYe2R6lI0KGuY13cEF9dZiqJs"
    }
}

# 2. LINK BOT CONFIGURATION (15-Minute)
CONFIG_LINK = { 
    "timeframe": "15m",
    "timeframe_sec": 900,
    "veto": 0.55,           
    "leverage": 5.0,        
    "margin": 0.05, 
    "live_mode": False,
    "trade": {
        "contractSize": 1.0,          
        "amountContracts": 14.0,       
        "timeStopCandles": 24  
    },
    "risk": {
        "atrStopMultiplier": 1.0,     
        "atrProfitMultiplier": 2.5,   
        "slippagePerc": 0.0002,       
        "takerFeePerc": 0.0006,       
        "makerFeePerc": 0.0003,
        "riskPct": 0.02
    },
    "webhooks": {
        "execution": "https://discord.com/api/webhooks/1526387660971577355/0BJ1hzZCrH00ZYSiIKPlmTU-khceP0cAIfz5s7tbyzerEd4mXX4sDmZ76RixJ_KFB06U",
        "veto":      "https://discord.com/api/webhooks/1526388299168616559/cLXIyKoYpzyc7rZdB2z61uG7xEO_opIvOHknUEuTXLUH2NwzFmz6HYTe50Atioybb8Hz",
        "settle":    "https://discord.com/api/webhooks/1526388137197178880/hsBeBTYWqFjuoTrbfolYH4eEXCcdXAdGIRZPFW11UmRHNZOsu8ijOUhvdqP4OhiaY0Oa",
        "ghost":     "https://discord.com/api/webhooks/1526388304000585869/XMgRT1F1rrqoM-YO2MMmOQgIDxs5M7LOLCifVgRQsDksrpZVSotF27dZ-Utmxqaj3gcq"
    }
}

NEPTUNE_WALLET_PRIVATE_KEY = os.getenv("NEPTUNE_WALLET_PRIVATE_KEY")

# --- MODEL INITIALIZATION ---
MODEL_SOL = "veto_engine_alpha.onnx"  
MODEL_LINK = "veto_engine_alpha_15m_timebound.onnx" 

def load_onnx_session(path, name):
    print(f"🤖 initializing {name} engine using file: {path}")
    if not os.path.exists(path):
        print(f"⚠️ warning: {path} not found. Ensure the model file is in the root directory.")
        return None, None, None, None
    try:
        session = ort.InferenceSession(path)
        input_name = session.get_inputs()[0].name
        label_name = session.get_outputs()[0].name
        outputs = session.get_outputs()
        prob_name = outputs[1].name if len(outputs) > 1 else outputs[0].name
        return session, input_name, label_name, prob_name
    except Exception as e:
        print(f"⚠️ error loading {path}: {e}")
        return None, None, None, None

session_sol, in_sol, lbl_sol, prob_sol = load_onnx_session(MODEL_SOL, "SOLANA PERPS")
session_link, in_link, lbl_link, prob_link = load_onnx_session(MODEL_LINK, "LINK PERPS")

exchange_sol = ccxt.coinbase({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
exchange_link = ccxt.coinbase({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})

SYMBOL_SOL = 'SOL/USDC'
SYMBOL_LINK = 'LINK/USDC' 

# ==============================================================================
# 📢 DISCORD NOTIFICATION HELPER
# ==============================================================================
def _post_webhook(url, data):
    try:
        requests.post(url, json=data, timeout=5)
    except Exception as e:
        pass

def send_discord_webhook(url, title, description, color, fields=None):
    if not url or "YOUR_" in url: return
    
    data = {
        "embeds": [{
            "title": title, "description": description, "color": color,
            "fields": fields or [],
            "footer": {"text": f"project neptune® • {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}"}
        }]
    }
    threading.Thread(target=_post_webhook, args=(url, data), daemon=True).start()

# ==============================================================================
# 🔮 PRICING & MOCK EXECUTION LOGIC
# ==============================================================================
def get_jupiter_live_price():
    try:
        sol_mint = "So11111111111111111111111111111111111111112"
        url = f"https://api.jup.ag/price/v3?ids={sol_mint}"
        headers = {"Accept": "application/json"}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            json_data = response.json()
            price_map = json_data.get("data", json_data)
            if sol_mint in price_map:
                price_str = price_map[sol_mint].get("usdPrice", price_map[sol_mint].get("price"))
                if price_str: return float(price_str)
    except Exception as e:
        pass
    return None

def execute_transaction(network, direction, size, price, sl, tp, leverage, margin, is_live):
    action = "long" if direction == 1.0 else "short"
    if is_live:
        print(f"🔥 [{network} live web3] broadcasting {action} of {size} units")
        return True 
    else:
        print(f"🔗 [{network} paper] emulating {action} of {size} units...")
        return True 

# ==============================================================================
# 🧠 AI FEATURE ENGINEERING PIPELINES
# ==============================================================================

# Pipeline A: Solana 5-Minute Strategy
def engineer_features_sol(df, config):
    try:
        adx_len = 10
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=adx_len)
        df['currentADX'] = adx_df[f'ADX_{adx_len}']
        df['prevADX'] = adx_df[f'ADX_{adx_len}'].shift(1)
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
        
        df['color'] = np.where(df['close'] >= df['open'], 1, -1)
        df['flip'] = np.where(df['color'] != df['color'].shift(1), 1, 0)
        df['isWhipsaw'] = np.where(df['flip'].rolling(window=3).sum() >= 3, 1.0, 0.0)

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

        now_ms = time.time() * 1000
        last_candle_time = df.iloc[-1]['timestamp']
        live_row = df.iloc[-2] if (now_ms - last_candle_time) < (config["timeframe_sec"] * 1000) else df.iloc[-1]
            
        if live_row.isnull().any(): return None, "warming up...", None
            
        feature_order = ['rsi', 'currentADX', 'adxDelta', 'rvol', 'atrPercentage', 'distanceToHtfEma', 'upperWick', 'lowerWick', 'bodySize', 'isWhipsaw']
        input_vector = live_row[feature_order].values.astype(np.float32).reshape(1, 10)
        
        pricing_data = {
            "close": float(live_row['close']), "high": float(live_row['high']), "low": float(live_row['low']),
            "atr": float(atr.iloc[-1] if (now_ms - last_candle_time) >= (config["timeframe_sec"] * 1000) else atr.iloc[-2]),
            "direction_intent": float(live_row['directionIntent']) 
        }
        timestamp_str = pd.to_datetime(live_row['timestamp'], unit='ms').strftime('%Y-%m-%d %H:%M:%S')
        return input_vector, timestamp_str, pricing_data
    except Exception as e:
        return None, f"solana engineering error: {e}", None

# Pipeline B: LINK 15-Minute Strategy
def engineer_features_link(df, config):
    try:
        df['rsi_15m'] = ta.rsi(df['close'], length=14)
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=10)
        df['currentADX'] = adx_df['ADX_10']
        
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df_1h = df.resample('1h', on='datetime').agg({'close': 'last'}).dropna()
        df_1h['rsi_1h'] = ta.rsi(df_1h['close'], length=14)
        df['rsi_1h'] = df['datetime'].map(df_1h['rsi_1h']).fillna(method='ffill')

        vol_sma20 = df['volume'].rolling(window=20).mean()
        df['rvol'] = df['volume'] / vol_sma20
        candle_range = (df['high'] - df['low']).replace(0, 0.00001)
        signed_delta = ((df['close'] - df['open']) / candle_range) * df['volume']
        df['vol_delta_ratio'] = signed_delta / vol_sma20.replace(0, 1)

        df['tr'] = ta.true_range(df['high'], df['low'], df['close'])
        tr_sum14 = df['tr'].rolling(14).mean()
        tr_sum100 = df['tr'].rolling(100).mean().replace(0, 1)
        df['atr_ratio'] = tr_sum14 / tr_sum100
        raw_atr = df['tr'].rolling(14).mean()

        df['htfEma'] = ta.ema(df['close'], length=400)
        price_std_dev = df['close'].rolling(50).std()
        df['macro_ema_zscore'] = (df['close'] - df['htfEma']) / price_std_dev.replace(0, 1)

        lowest_recent = df['low'].shift(1).rolling(21).min()
        highest_recent = df['high'].shift(1).rolling(21).max()
        sweep_long = ((df['low'] < lowest_recent) & (df['close'] > df['open'])).astype(int)
        sweep_short = ((df['high'] > highest_recent) & (df['close'] < df['open'])).astype(int)
        df['liquidity_sweep'] = np.maximum(sweep_long, sweep_short)

        body_size = (df['close'] - df['open']).abs()
        df['body_to_range_ratio'] = body_size / candle_range

        df['color'] = np.where(df['close'] >= df['open'], 1, -1)
        df['flip'] = np.where(df['color'] != df['color'].shift(1), 1, 0)
        df['isWhipsaw'] = np.where(df['flip'].rolling(window=3).sum() >= 3, 1.0, 0.0)

        df['ema9'] = ta.ema(df['close'], length=9)
        df['ema21'] = ta.ema(df['close'], length=21)
        df['ema100'] = ta.ema(df['close'], length=100)
        
        bull_fan = (df['ema9'] > df['ema21']) & (df['ema21'] > df['ema100'])
        bear_fan = (df['ema9'] < df['ema21']) & (df['ema21'] < df['ema100'])
        touches_long = (df['low'] <= df['ema9']) & (df['close'] > df['ema9'])
        touches_short = (df['high'] >= df['ema9']) & (df['close'] < df['ema9'])

        df['directionIntent'] = 0.0
        df.loc[bull_fan & touches_long, 'directionIntent'] = 1.0
        df.loc[bear_fan & touches_short, 'directionIntent'] = -1.0

        now_ms = time.time() * 1000
        last_candle_time = df.iloc[-1]['timestamp']
        live_row = df.iloc[-2] if (now_ms - last_candle_time) < (config["timeframe_sec"] * 1000) else df.iloc[-1]

        if live_row.isnull().any(): return None, "warming up...", None

        feature_order = [
            'rsi_15m', 'rsi_1h', 'currentADX', 'rvol', 'vol_delta_ratio', 
            'atr_ratio', 'macro_ema_zscore', 'liquidity_sweep', 'body_to_range_ratio', 'isWhipsaw'
        ]
        input_vector = live_row[feature_order].values.astype(np.float32).reshape(1, 10)
        
        pricing_data = {
            "close": float(live_row['close']), "high": float(live_row['high']), "low": float(live_row['low']),
            "atr": float(raw_atr.iloc[-1] if (now_ms - last_candle_time) >= (config["timeframe_sec"] * 1000) else raw_atr.iloc[-2]),
            "direction_intent": float(live_row['directionIntent']) 
        }
        timestamp_str = pd.to_datetime(live_row['timestamp'], unit='ms').strftime('%Y-%m-%d %H:%M:%S')
        return input_vector, timestamp_str, pricing_data
    except Exception as e:
        return None, f"link engineering error: {e}", None

def get_market_data_and_features(bot_name, exchange, symbol, config):
    try:
        all_candles = []
        since = exchange.milliseconds() - (1500 * config["timeframe_sec"] * 1000) 
        
        for _ in range(5):  
            batch = exchange.fetch_ohlcv(symbol, timeframe=config["timeframe"], since=since, limit=300)
            if not batch: break
            all_candles.extend(batch)
            since = batch[-1][0] + (config["timeframe_sec"] * 1000)
            time.sleep(1)  
            
        if len(all_candles) == 0: return None, "no data", None
            
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df.drop_duplicates(subset=['timestamp'], inplace=True)
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        if bot_name == "SOL": return engineer_features_sol(df, config)
        elif bot_name == "LINK": return engineer_features_link(df, config)
        
    except Exception as e:
        return None, f"fetch error: {e}", None

# ==============================================================================
# 🤖 UNIVERSAL TRADING ENGINE LOOP
# ==============================================================================
def engine_loop(bot_name, symbol, exchange, session, in_name, lbl_name, prob_name, state, config):
    print(f"🌊 {bot_name} engine running for {symbol} on {config['timeframe']}...")
    first_run = True 

    while True:
        tf_sec = config["timeframe_sec"]
        time_to_next_candle = tf_sec - (time.time() % tf_sec)
        
        if first_run: first_run = False
        else: time.sleep(time_to_next_candle + 10) 
        
        features, meta, pricing = get_market_data_and_features(bot_name, exchange, symbol, config)
        if features is None:
            time.sleep(10) 
            continue
            
        if bot_name == "SOL":
            jup_price = get_jupiter_live_price()
            if jup_price:
                pricing["close"] = jup_price
                pricing["high"] = max(pricing["high"], jup_price)
                pricing["low"] = min(pricing["low"], jup_price)

        state["last_close_price"] = pricing["close"]
            
        # 1. GHOST LOG SETTLEMENT
        if state["skipped_trade"] is not None:
            skip_pos = state["skipped_trade"]
            if (skip_pos["direction"] == 1.0 and pricing["close"] > skip_pos["entry_price"]) or \
               (skip_pos["direction"] == -1.0 and pricing["close"] < skip_pos["entry_price"]):
                outcome = "skip / win"
            else: outcome = "skip / loss"
                
            log_msg = f"👻 [ghost] {bot_name} resolved from {skip_pos['timestamp']}: {outcome}"
            state["trade_logs"].append(log_msg)
            
            color = 0x00e5ff if "win" in outcome else 0xFF0000
            dir_str = "LONG 📈" if skip_pos["direction"] == 1.0 else "SHORT 📉"
            discord_fields = [
                {"name": "Direction", "value": dir_str, "inline": True},
                {"name": "Theoretical Entry", "value": f"${skip_pos['entry_price']:.4f}", "inline": True},
                {"name": "Resolution Price", "value": f"${pricing['close']:.4f}", "inline": True},
                {"name": "Ghost Outcome", "value": outcome, "inline": False}
            ]
            send_discord_webhook(config["webhooks"]["ghost"], f"👻 {bot_name} Ghost Resolved", log_msg, color, discord_fields)
            state["skipped_trade"] = None
            
        # 2. ACTIVE TRADE MANAGEMENT
        if state["active_trade"] is not None:
            pos = state["active_trade"]
            trade_closed, exit_price, reason = False, 0.0, ""
            
            if pos["direction"] == 1.0: 
                if pricing["low"] <= pos["sl"]: exit_price, trade_closed, reason = pos["sl"], True, "Stop Loss"
                elif pricing["high"] >= pos["tp"]: exit_price, trade_closed, reason = pos["tp"], True, "Take Profit"
            else: 
                if pricing["high"] >= pos["sl"]: exit_price, trade_closed, reason = pos["sl"], True, "Stop Loss"
                elif pricing["low"] <= pos["tp"]: exit_price, trade_closed, reason = pos["tp"], True, "Take Profit"
            
            if not trade_closed:
                current_time_ms = time.time() * 1000
                time_in_trade_ms = current_time_ms - pos["entry_timestamp_ms"]
                max_time_ms = config["trade"]["timeStopCandles"] * config["timeframe_sec"] * 1000
                if time_in_trade_ms >= max_time_ms:
                    exit_price = pricing["close"]
                    trade_closed = True
                    reason = "Vertical Time-Stop Expiration"
            
            if trade_closed:
                fees = ((pos["entry_price"] * pos["contract_size"]) * config["risk"]["takerFeePerc"]) + \
                       ((exit_price * pos["contract_size"]) * config["risk"]["makerFeePerc"])
                
                gross_pnl = ((exit_price - pos["entry_price"]) if pos["direction"] == 1.0 else (pos["entry_price"] - exit_price)) * pos["contract_size"]
                net_pnl = gross_pnl - fees
                
                state["performance"]["total_trades"] += 1
                state["performance"]["net_pnl_usdc"] += net_pnl
                state["performance"]["wallet_balance_usdc"] += net_pnl
                
                if net_pnl > 0:
                    state["performance"]["wins"] += 1
                    state["performance"]["gross_pnl_usdc"] += net_pnl 
                    state["performance"]["consecutive_losses"] = 0
                    outcome_str = "🎉 Win"
                else:
                    state["performance"]["losses"] += 1
                    state["performance"]["consecutive_losses"] += 1
                    outcome_str = "🛑 Loss"
                    
                calc_wr = (state["performance"]["wins"] / state["performance"]["total_trades"]) * 100
                state["performance"]["win_rate"] = f"{calc_wr:.2f}%"
                
                settle_msg = f"📊 [{bot_name} Settled] trade {pos['trade_id']} closed via {reason} @ {exit_price} | pnl: {net_pnl:+.4f} usdc | balance: {state['performance']['wallet_balance_usdc']:.4f} usdc"
                state["trade_logs"].append(settle_msg)
                
                color = 0x00e5ff if net_pnl > 0 else 0xFF0000
                dir_str = "LONG 📈" if pos["direction"] == 1.0 else "SHORT 📉"
                discord_fields = [
                    {"name": "Direction", "value": dir_str, "inline": True},
                    {"name": "Entry Price", "value": f"${pos['entry_price']:.4f}", "inline": True},
                    {"name": "Exit Price", "value": f"${exit_price:.4f}", "inline": True},
                    {"name": "Net PnL", "value": f"{net_pnl:+.4f} USDC", "inline": True},
                    {"name": "Margin Used", "value": f"{pos.get('margin_base', config['margin']):.4f} units @ {pos.get('leverage', config['leverage'])}x", "inline": True},
                    {"name": "New Balance", "value": f"{state['performance']['wallet_balance_usdc']:.4f} USDC", "inline": True},
                    {"name": "Overall Win Rate", "value": state["performance"]["win_rate"], "inline": False}
                ]
                title = f"Trade {pos['trade_id']} | 📊 {bot_name} Settled: {outcome_str}"
                send_discord_webhook(config["webhooks"]["settle"], title, settle_msg, color, discord_fields)
                state["active_trade"] = None 
                
        # 3. NEW ENTRY EVALUATION
        if state["active_trade"] is None:
            direction_intent = pricing["direction_intent"]
            try:
                pred_res = session.run([lbl_name, prob_name], {in_name: features}) if session else [[0], [[0, 0]]]
                prob_win = float(pred_res[1][0][1]) if len(pred_res) > 1 else float(pred_res[0][0])
            except Exception as e:
                prob_win = 0.0
            
            direction_str = "LONG 📈" if direction_intent == 1.0 else "SHORT 📉"
            
            if direction_intent != 0.0 and prob_win >= config["veto"]:
                stop_dist = pricing["atr"] * config["risk"]["atrStopMultiplier"]
                entry_p = pricing["close"]
                
                applied_risk = config["risk"]["riskPct"] / 2.0 if state["performance"]["consecutive_losses"] >= 3 else config["risk"]["riskPct"]
                margin_usdc = state["performance"]["wallet_balance_usdc"] * applied_risk
                dynamic_margin_asset = margin_usdc / entry_p
                contract_size = dynamic_margin_asset * config["leverage"]
                
                config["margin"] = dynamic_margin_asset 
                
                sl_target = entry_p - stop_dist if direction_intent == 1.0 else entry_p + stop_dist
                tp_target = entry_p + (pricing["atr"] * config["risk"]["atrProfitMultiplier"]) if direction_intent == 1.0 else entry_p - (pricing["atr"] * config["risk"]["atrProfitMultiplier"])
                    
                tx_confirmed = execute_transaction(bot_name, direction_intent, contract_size, entry_p, sl_target, tp_target, config["leverage"], config["margin"], config["live_mode"])
                
                if tx_confirmed:
                    state["performance"]["trade_id_counter"] += 1
                    t_id = state["performance"]["trade_id_counter"]
                    state["active_trade"] = {
                        "trade_id": t_id,
                        "entry_price": entry_p, "direction": direction_intent,
                        "timestamp": meta, "entry_timestamp_ms": time.time() * 1000,
                        "contract_size": contract_size,
                        "leverage": config["leverage"], "margin_base": config["margin"],
                        "sl": sl_target, "tp": tp_target, "atr": pricing["atr"]
                    }
                    decision_msg = f"✅ Allowed & executed ({contract_size:.4f} units @ {config['leverage']}x)"
                    
                    color = 0x00e5ff if direction_intent == 1.0 else 0xFF0000
                    discord_fields = [
                        {"name": "Direction", "value": direction_str, "inline": True},
                        {"name": "Entry Price", "value": f"${entry_p:.4f}", "inline": True},
                        {"name": "Locked Margin", "value": f"{config['margin']:.4f} units", "inline": True},
                        {"name": "Leverage Applied", "value": f"{config['leverage']}x", "inline": True},
                        {"name": "Total Order Size", "value": f"{contract_size:.4f} units", "inline": True},
                        {"name": "Stop Loss", "value": f"${sl_target:.4f}", "inline": True},
                        {"name": "Take Profit", "value": f"${tp_target:.4f}", "inline": True}
                    ]
                    title = f"Trade {t_id} | 🟢 {bot_name} Long Executed" if direction_intent == 1.0 else f"Trade {t_id} | 🔴 {bot_name} Short Executed"
                    send_discord_webhook(config["webhooks"]["execution"], title, decision_msg, color, discord_fields)
                    
                else: 
                    decision_msg = f"⚠️ {bot_name} execution failed on chain/exchange"
            else:
                reason = f"Conviction ({prob_win:.2%})" if direction_intent != 0.0 else "No structural EMA setup"
                decision_msg = f"❌ Veto ({reason})"
                
                if direction_intent != 0.0:
                    state["skipped_trade"] = {"timestamp": meta, "entry_price": pricing["close"], "direction": direction_intent}
                    
                    color = 0xFFA500
                    discord_fields = [
                        {"name": "Direction", "value": direction_str, "inline": True},
                        {"name": "Current Price", "value": f"${pricing['close']:.4f}", "inline": True},
                        {"name": "Conviction", "value": f"{prob_win:.2%} (needs {config['veto']:.2%})", "inline": True}
                    ]
                    send_discord_webhook(config["webhooks"]["veto"], f"🛡️ {bot_name} Trade Vetoed", decision_msg, color, discord_fields)
            
            state["trade_logs"].append(f"🕒 [{meta}] {bot_name} Action: {decision_msg}")
            
        if len(state["trade_logs"]) > 200: state["trade_logs"].pop(0)

threading.Thread(target=engine_loop, args=("SOL", SYMBOL_SOL, exchange_sol, session_sol, in_sol, lbl_sol, prob_sol, STATE_SOL, CONFIG_SOL), daemon=True).start()
threading.Thread(target=engine_loop, args=("LINK", SYMBOL_LINK, exchange_link, session_link, in_link, lbl_link, prob_link, STATE_LINK, CONFIG_LINK), daemon=True).start()

# ==============================================================================
# 🌐 DYNAMIC API ROUTES
# ==============================================================================

def get_bot_context(bot_type: str):
    bt = bot_type.lower()
    if bt == "sol": return STATE_SOL, CONFIG_SOL
    elif bt == "link": return STATE_LINK, CONFIG_LINK
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
    
    fees = ((pos["entry_price"] * pos["contract_size"]) * config["risk"]["takerFeePerc"]) + ((exit_price * pos["contract_size"]) * config["risk"]["makerFeePerc"])
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
            "margin_locked": f"{pos.get('margin_base', config['margin']):.4f} units",
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
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"status": "online", "message": "API is running (index.html not found)"}
