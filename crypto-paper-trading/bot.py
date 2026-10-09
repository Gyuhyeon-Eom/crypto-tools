#!/usr/bin/env python3
"""
코인 선물 페이퍼 트레이딩 봇 v3
- 종료: 2026-04-06 22:00 KST
- 매시간 정시 리포트 -> report_queue.jsonl (감시 스크립트가 Slack 전송)
- 매매 이벤트 -> alerts.jsonl
"""

import json
import os
import time
import math
import urllib.request
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(WORKSPACE, "state.json")
ALERT_FILE = os.path.join(WORKSPACE, "alerts.jsonl")
REPORT_FILE = os.path.join(WORKSPACE, "report_queue.jsonl")
SUMMARY_FILE = os.path.join(WORKSPACE, "summary.txt")

# === 설정 ===
USDT_KRW_RATE = 1390
LEVERAGE = 10
MAX_POSITIONS = 3
POSITION_SIZE_PCT = 0.25
STOP_LOSS_PCT = -2.0
TAKE_PROFIT_PCT = 3.5
TAKER_FEE = 0.0005
SCAN_INTERVAL = 300  # 5분

# 종료 시간: 2026-04-06 22:00 KST
END_TIME = datetime(2026, 4, 6, 21, 55, tzinfo=KST)

TARGET_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "BNBUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "NEARUSDT", "APTUSDT", "SUIUSDT", "ARBUSDT",
    "OPUSDT", "INJUSDT", "TIAUSDT", "SEIUSDT", "JUPUSDT",
    "WIFUSDT", "ONDOUSDT", "ENAUSDT",
    "FETUSDT", "RENDERUSDT", "ARUSDT", "STXUSDT", "RUNEUSDT",
    "ZECUSDT", "STOUSDT", "DRIFTUSDT", "ZETAUSDT", "ONGUSDT",
    "1000PEPEUSDT", "1000FLOKIUSDT",
]

def api_get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except:
        return None

def write_alert(alert_data):
    alert_data['timestamp'] = datetime.now(KST).isoformat()
    with open(ALERT_FILE, 'a') as f:
        f.write(json.dumps(alert_data, ensure_ascii=False) + '\n')

def write_report(report_text):
    """매시간 리포트를 큐 파일에 기록"""
    entry = {
        'timestamp': datetime.now(KST).isoformat(),
        'report': report_text,
        'sent': False,
    }
    with open(REPORT_FILE, 'a') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def get_klines(symbol, interval="5m", limit=50):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = api_get(url)
    if not data:
        return None
    return [{'open': float(d[1]), 'high': float(d[2]), 'low': float(d[3]),
             'close': float(d[4]), 'volume': float(d[5]), 'quote_volume': float(d[7])} for d in data]

def calc_ema(prices, period):
    if len(prices) < period:
        return None
    m = 2 / (period + 1)
    ema = sum(prices[:period]) / period
    for p in prices[period:]:
        ema = (p - ema) * m + ema
    return ema

def calc_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50
    gains, losses = [], []
    for i in range(1, len(prices)):
        d = prices[i] - prices[i-1]
        gains.append(max(d, 0)); losses.append(max(-d, 0))
    ag = sum(gains[-period:]) / period
    al = sum(losses[-period:]) / period
    if al == 0: return 100
    return 100 - (100 / (1 + ag/al))

def calc_volume_spike(candles, lookback=20):
    if len(candles) < lookback + 1:
        return 1
    avg = sum(c['quote_volume'] for c in candles[-(lookback+1):-1]) / lookback
    return candles[-1]['quote_volume'] / avg if avg > 0 else 1

def calc_bb(prices, period=20, mult=2):
    if len(prices) < period:
        return None, None, None
    r = prices[-period:]
    sma = sum(r) / period
    std = math.sqrt(sum((p-sma)**2 for p in r) / period)
    return sma, sma + mult*std, sma - mult*std

def analyze_signal(symbol):
    candles = get_klines(symbol, "5m", 50)
    if not candles or len(candles) < 30:
        return None
    closes = [c['close'] for c in candles]
    cp = closes[-1]
    ema9 = calc_ema(closes, 9); ema21 = calc_ema(closes, 21)
    if not ema9 or not ema21: return None
    rsi = calc_rsi(closes, 14)
    vol = calc_volume_spike(candles)
    _, bb_up, bb_lo = calc_bb(closes)

    # LONG
    lc, lr = 0, []
    if ema9 > ema21: lc += 1; lr.append("EMA9>21 상승추세")
    if rsi < 35: lc += 2; lr.append(f"RSI {rsi:.0f} 과매도")
    elif rsi < 45: lc += 1; lr.append(f"RSI {rsi:.0f} 저점권")
    if vol > 2.0: lc += 1; lr.append(f"거래량 {vol:.1f}x 급증")
    if closes[-1] > closes[-2] > closes[-3]: lc += 1; lr.append("3봉 연속상승")
    if cp > ema21 and abs(cp - ema9)/cp < 0.003: lc += 1; lr.append("EMA9 눌림목")
    if bb_lo and cp <= bb_lo * 1.005: lc += 1; lr.append("BB 하단 터치")

    # SHORT
    sc, sr = 0, []
    if ema9 < ema21: sc += 1; sr.append("EMA9<21 하락추세")
    if rsi > 70: sc += 2; sr.append(f"RSI {rsi:.0f} 과매수")
    elif rsi > 60: sc += 1; sr.append(f"RSI {rsi:.0f} 고점권")
    if vol > 2.0: sc += 1; sr.append(f"거래량 {vol:.1f}x 급증")
    if closes[-1] < closes[-2] < closes[-3]: sc += 1; sr.append("3봉 연속하락")
    if bb_up and cp >= bb_up * 0.995: sc += 1; sr.append("BB 상단 터치")

    if lc >= 3 and lc > sc:
        return {'symbol': symbol, 'signal': 'LONG', 'confidence': lc, 'price': cp,
                'rsi': rsi, 'ema9': ema9, 'ema21': ema21, 'vol_spike': vol, 'reasons': lr}
    if sc >= 3 and sc > lc:
        return {'symbol': symbol, 'signal': 'SHORT', 'confidence': sc, 'price': cp,
                'rsi': rsi, 'ema9': ema9, 'ema21': ema21, 'vol_spike': vol, 'reasons': sr}
    return None

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return None

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

def get_price(symbol):
    d = api_get(f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}")
    return float(d['price']) if d else None

def calc_pnl_pct(side, entry, current):
    if side == "LONG":
        return (current - entry) / entry * 100
    return (entry - current) / entry * 100

def open_position(state, sig):
    sym = sig['symbol']
    for p in state['positions']:
        if p['symbol'] == sym: return False
    size = state['balance'] * POSITION_SIZE_PCT
    if size < 10: return False

    ep = sig['price']; side = sig['signal']
    if side == "LONG":
        sl = ep * (1 + STOP_LOSS_PCT/100); tp = ep * (1 + TAKE_PROFIT_PCT/100)
    else:
        sl = ep * (1 - STOP_LOSS_PCT/100); tp = ep * (1 - TAKE_PROFIT_PCT/100)
    
    pv = size * LEVERAGE; fee = pv * TAKER_FEE
    pos = {'symbol': sym, 'side': side, 'entry_price': ep, 'size_usdt': size,
           'position_value': pv, 'leverage': LEVERAGE,
           'entry_time': datetime.now(KST).isoformat(),
           'sl': sl, 'tp': tp, 'entry_fee': fee,
           'reasons': sig['reasons'], 'rsi': sig['rsi'], 'confidence': sig['confidence']}
    
    state['balance'] -= size + fee
    state['total_fees'] += fee
    state['positions'].append(pos)
    
    print(f"  >>> OPEN {side} {sym} @ {ep} | {size:.1f} USDT x{LEVERAGE}")
    
    write_alert({
        'type': 'OPEN', 'side': side, 'symbol': sym.replace('USDT','/USDT'),
        'entry_price': ep, 'size_usdt': round(size,2), 'leverage': LEVERAGE,
        'sl': round(sl,6), 'tp': round(tp,6),
        'reasons': sig['reasons'], 'rsi': round(sig['rsi'],1),
        'confidence': sig['confidence'],
        'balance': round(state['balance'],2),
        'open_positions': len(state['positions']),
    })
    return True

def close_position(state, idx, current_price, close_reason):
    pos = state['positions'][idx]
    pnl_pct = calc_pnl_pct(pos['side'], pos['entry_price'], current_price)
    pnl_usdt = pos['size_usdt'] * LEVERAGE * pnl_pct / 100
    exit_fee = pos['position_value'] * TAKER_FEE
    net_pnl = pnl_usdt - exit_fee
    
    state['balance'] += pos['size_usdt'] + net_pnl
    state['total_fees'] += exit_fee
    
    entry_dt = datetime.fromisoformat(pos['entry_time'])
    dur_min = (datetime.now(KST) - entry_dt).total_seconds() / 60
    
    trade = {'symbol': pos['symbol'], 'side': pos['side'],
             'entry_price': pos['entry_price'], 'exit_price': current_price,
             'pnl_pct': round(pnl_pct,2), 'net_pnl': round(net_pnl,2),
             'close_reason': close_reason, 'duration_min': round(dur_min,1),
             'close_time': datetime.now(KST).isoformat(),
             'reasons_entry': pos.get('reasons',[])}
    state['closed_trades'].append(trade)
    
    wins = [t for t in state['closed_trades'] if t['net_pnl'] > 0]
    total = len(state['closed_trades'])
    wr = len(wins)/total*100 if total else 0
    cum = sum(t['net_pnl'] for t in state['closed_trades'])
    
    print(f"  <<< CLOSE {pos['side']} {pos['symbol']} | {close_reason} | {net_pnl:+.2f} USDT")
    
    write_alert({
        'type': 'CLOSE', 'side': pos['side'],
        'symbol': pos['symbol'].replace('USDT','/USDT'),
        'entry_price': pos['entry_price'], 'exit_price': current_price,
        'pnl_pct': round(pnl_pct,2), 'net_pnl': round(net_pnl,2),
        'close_reason': close_reason, 'duration_min': round(dur_min,1),
        'reasons_entry': pos.get('reasons',[]),
        'total_trades': total, 'win_rate': round(wr,1),
        'cumulative_pnl': round(cum,2),
        'balance': round(state['balance'],2),
    })
    state['positions'].pop(idx)

def check_positions(state):
    to_close = []
    for i, pos in enumerate(state['positions']):
        cp = get_price(pos['symbol'])
        if not cp: continue
        pnl = calc_pnl_pct(pos['side'], pos['entry_price'], cp)
        
        reason = None
        if pos['side'] == "LONG":
            if cp <= pos['sl']: reason = "STOP_LOSS"
            elif cp >= pos['tp']: reason = "TAKE_PROFIT"
        else:
            if cp >= pos['sl']: reason = "STOP_LOSS"
            elif cp <= pos['tp']: reason = "TAKE_PROFIT"
        if pnl * LEVERAGE <= -90: reason = "LIQUIDATION"
        
        if reason:
            to_close.append((i, cp, reason))
        else:
            print(f"  ... HOLD {pos['side']} {pos['symbol']} {pnl:+.2f}% ({pos['size_usdt']*LEVERAGE*pnl/100:+.1f} USDT)")
    
    for i, cp, reason in sorted(to_close, key=lambda x: x[0], reverse=True):
        close_position(state, i, cp, reason)

def generate_hourly_report(state):
    """매시간 리포트 생성"""
    now = datetime.now(KST)
    bal = state['balance']
    
    # 미실현 평가
    unrealized = 0
    pos_lines = []
    for pos in state['positions']:
        cp = get_price(pos['symbol'])
        if cp:
            pnl = calc_pnl_pct(pos['side'], pos['entry_price'], cp)
            pnl_usdt = pos['size_usdt'] * LEVERAGE * pnl / 100
            unrealized += pnl_usdt
            emoji = "+" if pnl > 0 else ""
            pos_lines.append(f"  {pos['side']} {pos['symbol'].replace('USDT','')} | 진입 {pos['entry_price']} -> 현재 {cp} | {emoji}{pnl:.2f}% ({emoji}{pnl_usdt:.1f} USDT)")
    
    total_val = bal + sum(p['size_usdt'] for p in state['positions']) + unrealized
    total_pnl = total_val - state['initial_balance']
    total_pnl_pct = total_pnl / state['initial_balance'] * 100
    
    wins = [t for t in state['closed_trades'] if t['net_pnl'] > 0]
    losses = [t for t in state['closed_trades'] if t['net_pnl'] <= 0]
    total_trades = len(state['closed_trades'])
    wr = len(wins)/total_trades*100 if total_trades else 0
    
    # 최근 거래 (마지막 5건)
    recent = state['closed_trades'][-5:]
    trade_lines = []
    for t in recent:
        e = "W" if t['net_pnl'] > 0 else "L"
        trade_lines.append(f"  [{e}] {t['side']} {t['symbol'].replace('USDT','')} | {t['net_pnl']:+.1f} USDT ({t['pnl_pct']:+.1f}%) | {t['close_reason']}")
    
    report = f"""[{now.strftime('%m/%d %H:%M')} 정시 리포트]
총 평가금: {total_val:.0f} USDT ({total_val*USDT_KRW_RATE:,.0f}원)
손익: {total_pnl:+.1f} USDT ({total_pnl_pct:+.2f}%) | {total_pnl*USDT_KRW_RATE:+,.0f}원
거래: {total_trades}건 (승{len(wins)}/패{len(losses)}) 승률 {wr:.0f}%
수수료: {state['total_fees']:.1f} USDT

보유 포지션 ({len(state['positions'])}개):
{chr(10).join(pos_lines) if pos_lines else '  (없음)'}"""
    
    if trade_lines:
        report += f"\n\n최근 거래:\n{chr(10).join(trade_lines)}"
    
    return report

def main():
    print(f"봇 시작: {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S KST')}")
    print(f"종료 예정: {END_TIME.strftime('%Y-%m-%d %H:%M KST')}")
    
    state = load_state()
    if not state:
        print("ERROR: 기존 state.json 없음")
        return
    
    save_state(state)
    
    last_report_hour = -1
    
    while datetime.now(KST) < END_TIME:
        state['scan_count'] += 1
        now = datetime.now(KST)
        print(f"\n--- 스캔 #{state['scan_count']} | {now.strftime('%m/%d %H:%M:%S')} | 잔고 {state['balance']:.1f} | 포지션 {len(state['positions'])} ---")
        
        # 포지션 체크
        if state['positions']:
            check_positions(state)
        
        # 새 시그널
        if len(state['positions']) < MAX_POSITIONS:
            signals = []
            for sym in TARGET_SYMBOLS:
                sig = analyze_signal(sym)
                if sig: signals.append(sig)
                time.sleep(0.12)
            signals.sort(key=lambda x: x['confidence'], reverse=True)
            for sig in signals:
                if len(state['positions']) >= MAX_POSITIONS: break
                if sig['confidence'] >= 3:
                    open_position(state, sig)
        
        save_state(state)
        
        # 매시간 정시 리포트 (정시 기준 +-3분 이내)
        current_hour = now.hour
        if current_hour != last_report_hour and now.minute < 6:
            report = generate_hourly_report(state)
            write_report(report)
            print(f"\n[HOURLY REPORT]\n{report}")
            last_report_hour = current_hour
        
        time.sleep(SCAN_INTERVAL)
    
    # === 종료 ===
    print("\n=== 세션 종료, 전체 포지션 청산 ===")
    for i in range(len(state['positions'])-1, -1, -1):
        cp = get_price(state['positions'][i]['symbol'])
        if cp: close_position(state, i, cp, "SESSION_END")
    
    save_state(state)
    final = generate_hourly_report(state)
    write_report("[최종 리포트]\n" + final)
    write_alert({'type': 'FINAL', 'report': final})
    print(final)
    
    with open(os.path.join(WORKSPACE, "DONE"), 'w') as f:
        f.write(datetime.now(KST).isoformat())

if __name__ == "__main__":
    main()
