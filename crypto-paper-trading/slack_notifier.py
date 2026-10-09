#!/usr/bin/env python3
"""
v4 봇 알림 감시 - v4_alerts.jsonl을 감시하고 OPEN/CLOSE 이벤트를 stdout으로 출력
openclaw이 이걸 실행하지 않음 - 봇 자체에서 알림 파일에 기록하면
메인 세션의 heartbeat이나 주기적 체크에서 읽어서 Slack 전송
"""

import json
import os
import time
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
ALERT_FILE = os.path.join(WORKSPACE, "v4_alerts.jsonl")
SENT_FILE = os.path.join(WORKSPACE, "v4_alerts_sent.txt")


def get_sent_count():
    if os.path.exists(SENT_FILE):
        with open(SENT_FILE, 'r') as f:
            return int(f.read().strip())
    return 0


def set_sent_count(n):
    with open(SENT_FILE, 'w') as f:
        f.write(str(n))


def get_new_alerts():
    if not os.path.exists(ALERT_FILE):
        return []
    
    sent = get_sent_count()
    alerts = []
    with open(ALERT_FILE, 'r') as f:
        for i, line in enumerate(f):
            if i >= sent:
                try:
                    alerts.append(json.loads(line.strip()))
                except:
                    pass
    
    if alerts:
        set_sent_count(sent + len(alerts))
    
    return alerts


def format_alert(alert):
    t = alert.get('type', '')
    ts = alert.get('timestamp', '')[:16]
    
    if t == 'OPEN':
        side = alert.get('side', '')
        sym = alert.get('symbol', '')
        ep = alert.get('entry_price', 0)
        size = alert.get('size_usdt', 0)
        lev = alert.get('leverage', 10)
        sl = alert.get('sl', 0)
        tp = alert.get('tp', 0)
        reasons = ', '.join(alert.get('reasons', []))
        accum = alert.get('accum_score', 0)
        accum_tag = f" | 매집점수 {accum}" if accum > 0 else ""
        
        return (f"[OPEN] {side} {sym} @ {ep}\n"
                f"투자: {size:.0f} USDT x{lev}{accum_tag}\n"
                f"SL: {sl:.6g} | TP: {tp:.6g}\n"
                f"근거: {reasons}")
    
    elif t == 'CLOSE':
        result = alert.get('result', '')
        side = alert.get('side', '')
        sym = alert.get('symbol', '')
        ep = alert.get('entry_price', 0)
        xp = alert.get('exit_price', 0)
        pnl = alert.get('net_pnl', 0)
        pnl_pct = alert.get('pnl_pct', 0)
        reason = alert.get('close_reason', '')
        total = alert.get('total_trades', 0)
        wr = alert.get('win_rate', 0)
        cum = alert.get('cumulative_pnl', 0)
        ret = alert.get('portfolio_return', 0)
        
        return (f"[CLOSE {result}] {side} {sym}\n"
                f"{ep} -> {xp} | {pnl:+.1f} USDT ({pnl_pct:+.1f}%)\n"
                f"사유: {reason}\n"
                f"총 {total}건 승률 {wr:.0f}% | 누적 {cum:+.0f} | 수익률 {ret:+.1f}%")
    
    elif t == 'BTC_TREND':
        return f"[BTC 추세] {alert.get('from','')} -> {alert.get('to','')} (score {alert.get('score',0)})"
    
    return None


if __name__ == "__main__":
    alerts = get_new_alerts()
    trade_alerts = [a for a in alerts if a.get('type') in ('OPEN', 'CLOSE')]
    
    for alert in trade_alerts:
        msg = format_alert(alert)
        if msg:
            print(msg)
            print("---")
