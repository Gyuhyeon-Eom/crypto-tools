#!/usr/bin/env python3
"""
Crypto Monitor - 1회 실행 후 종료 (cron/launchd로 주기 호출)
포지션 상태를 체크하고 Slack으로 전송
"""

import json
import os
import sys
import urllib.request
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(line_buffering=True)

KST = timezone(timedelta(hours=9))
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitor_state.json")

# === 포지션 설정 ===
POSITIONS = [
    {
        "symbol": "TONUSDT",
        "name": "TON",
        "direction": "LONG",
        "entry": 1.4068,
        "sl": 1.355,
        "tp1": 1.464,
        "tp2": 1.47,
    },
]


def get_price(symbol):
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = json.loads(urllib.request.urlopen(req, timeout=10).read())
        return float(data["price"])
    except Exception as e:
        print(f"Price error {symbol}: {e}", flush=True)
        return None


def make_bar(current, low, high, width=20):
    if high == low:
        return "=" * width
    ratio = (current - low) / (high - low)
    ratio = max(0, min(1, ratio))
    filled = int(ratio * width)
    return "█" * filled + "·" * (width - filled)


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def main():
    state = load_state()
    btc_ref = state.get("btc_ref")
    
    btc = get_price("BTCUSDT")
    if btc is None:
        print("BTC price fetch failed", flush=True)
        return
    
    if btc_ref is None:
        btc_ref = btc
        state["btc_ref"] = btc_ref
    
    now = datetime.now(KST).strftime("%H:%M")
    btc_chg = (btc / btc_ref - 1) * 100
    
    lines = [f"[내 포지션 {now}]"]
    lines.append(f" BTC ${btc:,.0f} ({btc_chg:+.1f}%)")
    
    is_alert = False
    
    for pos in POSITIONS:
        price = get_price(pos["symbol"])
        if price is None:
            lines.append(f" {pos['name']}: 조회 실패")
            continue
        
        entry = pos["entry"]
        sl = pos["sl"]
        tp1 = pos["tp1"]
        pnl = (price / entry - 1) * 100
        if pos["direction"] == "SHORT":
            pnl = -pnl
        
        bar = make_bar(price, sl, tp1)
        
        sl_dist = abs(price - sl) / price * 100
        tp_dist = abs(tp1 - price) / price * 100
        
        warning = ""
        if sl_dist < 1.5:
            warning = " ⚠️ SL 근접!"
            is_alert = True
        elif tp_dist < 1.5:
            warning = " 🎯 TP 근접!"
            is_alert = True
        
        if price <= sl and pos["direction"] == "LONG":
            warning = " 🚨 SL 도달!"
            is_alert = True
        elif price >= tp1 and pos["direction"] == "LONG":
            warning = " ✅ TP1 도달!"
            is_alert = True
        
        lines.append(f" {pos['name']} ${price:.4f} ({pnl:+.2f}%){warning}")
        lines.append(f" [${sl}]|{bar}|[${tp1}]")
    
    header = "--- ALERT" if is_alert else "---"
    msg = f"```{header} {now} ---\n" + "\n".join(lines) + "```"
    
    print(msg, flush=True)
    
    save_state(state)


if __name__ == "__main__":
    main()
