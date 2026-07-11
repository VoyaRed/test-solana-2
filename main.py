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
    
    # Replicate structural fee rules from your trading loop
    entry_fee_cost = (pos["entry_price"] * pos["contract_size"]) * RISK_SETTINGS["takerFeePerc"]
    exit_fee_cost = (exit_price * pos["contract_size"]) * RISK_SETTINGS["makerFeePerc"]
    total_fees = entry_fee_cost + exit_fee_cost
    
    if pos["direction"] == 1.0:
        gross_pnl = (exit_price - pos["entry_price"]) * pos["contract_size"]
    else:
        gross_pnl = (pos["entry_price"] - exit_price) * pos["contract_size"]
        
    net_pnl = gross_pnl - total_fees
    
    # Record and update performance state
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
    
    # Log execution to brain activity ledger
    settle_msg = f"🕹️ [MANUAL OVERRIDE] Position from {pos['timestamp']} Force-Closed @ {exit_price} | {outcome_str} | Net PNL: {net_pnl:+.4f} USDC"
    print(settle_msg)
    STATE["trade_logs"].append(settle_msg)
    
    # Evict position from engine memory
    STATE["active_trade"] = None
    
    return {"status": "success", "message": f"Successfully market-closed position at ${exit_price}."}
