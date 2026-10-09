#!/usr/bin/env python3
"""
공격적 코인 선물 페이퍼 트레이딩 봇
- 300만원 시작, 일 20% 목표
- 레버리지 20x, 스캘핑
- 손절 -1% / 익절 +2% (손익비 1:2)
- 1분 스캔
- 종료: 2026-04-06 22:00 KST
"""

import json
import os
import time
import math
import urllib.request
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(WORKSPACE, "agg_state.json")
ALERT_FILE = os.path.join(WORKSPACE, "agg_alerts.jsonl")
LOG_FILE = os.path.join(WORKSPACE, "agg_bot.log")

# === 설정 ===
INITIAL_KRW = 3_000_000
USDT_KRW = 1390
INITIAL_USDT = round(INITIAL_KRW / USDT_KRW, 2)  # ~2158 USDT
LEVERAGE = 20
MAX_POSITIONS = 2  # 집중 투자
POSITION_SIZE_PCT = 0.40  # 잔고의 40%
STOP_LOSS_PCT = -1.0  # -1% 타이트 손절
TAKE_PROFIT_PCT = 2.0  # +2% 익절 (손익비 1:2)
TAKER_FEE = 0.0005
SCAN_INTERVAL = 60  # 1분

END_TIME = datetime(2026, 4, 6, 21, 55, tzinfo=KST)

# 변동성 큰 종목 위주
TARGET_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "BNBUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT",
    "NEARUSDT", "APTUSDT", "SUIUSDT", "ARBUSDT",
    "OPUSDT", "INJUSDT", "SEIUSDT", "JUPUSDT",
    "WIFUSDT", "ONDOUSDT", "ENAUSDT",
    "FETUSDT", "RENDERUSDT", "RUNEUSDT",
    "ZECUSDT", "STOUSDT", "DRIFTUSDT", "ZETAUSDT",
    "1000PEPEUSDT", "1000FLOKIUSDT",
]

def api_get(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8) as r:
            return json.loads(r.read())
    except:
        return None

def write_alert(data):
    data['timestamp'] = datetime.now(KST).isoformat()
    with open(ALERT_FILE, 'a') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

def get_klines(symbol, interval="1m", limit=60):
    data = api_get(f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}")
    if not data: return None
    return [{'open':float(d[1]),'high':float(d[2]),'low':float(d[3]),
             'close':float(d[4]),'volume':float(d[5]),'qv':float(d[7])} for d in data]

def get_klines_5m(symbol, limit=30):
    data = api_get(f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=5m&limit={limit}")
    if not data: return None
    return [{'open':float(d[1]),'high':float(d[2]),'low':float(d[3]),
             'close':float(d[4]),'volume':float(d[5]),'qv':float(d[7])} for d in data]

def ema(prices, p):
    if len(prices) < p: return None
    m = 2/(p+1); e = sum(prices[:p])/p
    for x in prices[p:]: e = (x-e)*m+e
    return e

def rsi(prices, p=14):
    if len(prices) < p+1: return 50
    g,l = [],[]
    for i in range(1,len(prices)):
        d = prices[i]-prices[i-1]; g.append(max(d,0)); l.append(max(-d,0))
    ag = sum(g[-p:])/p; al = sum(l[-p:])/p
    if al==0: return 100
    return 100-(100/(1+ag/al))

def vol_spike(candles, lb=20):
    if len(candles) < lb+1: return 1
    avg = sum(c['qv'] for c in candles[-(lb+1):-1])/lb
    return candles[-1]['qv']/avg if avg>0 else 1

def bb(prices, p=20, m=2):
    if len(prices)<p: return None,None,None
    r=prices[-p:]; s=sum(r)/p; std=math.sqrt(sum((x-s)**2 for x in r)/p)
    return s, s+m*std, s-m*std

def vwap(candles, periods=20):
    """VWAP 계산"""
    if len(candles) < periods: return None
    recent = candles[-periods:]
    total_pv = sum((c['high']+c['low']+c['close'])/3 * c['volume'] for c in recent)
    total_vol = sum(c['volume'] for c in recent)
    return total_pv / total_vol if total_vol > 0 else None

def momentum_score(candles_1m, candles_5m):
    """모멘텀 스코어 (-10 ~ +10)"""
    if not candles_1m or not candles_5m: return 0
    
    score = 0
    c1 = [c['close'] for c in candles_1m]
    c5 = [c['close'] for c in candles_5m]
    cp = c1[-1]
    
    # 1분봉 EMA 크로스
    e5 = ema(c1, 5); e13 = ema(c1, 13)
    if e5 and e13:
        if e5 > e13: score += 2
        else: score -= 2
    
    # 5분봉 EMA 크로스
    e9 = ema(c5, 9); e21 = ema(c5, 21)
    if e9 and e21:
        if e9 > e21: score += 2
        else: score -= 2
    
    # RSI (1분봉)
    r1 = rsi(c1, 10)
    if r1 < 25: score += 3  # 극단 과매도 -> 반등 기대
    elif r1 < 40: score += 1
    elif r1 > 75: score -= 3  # 극단 과매수 -> 하락 기대
    elif r1 > 60: score -= 1
    
    # 거래량 급증
    vs = vol_spike(candles_1m, 15)
    if vs > 3: score += 2 if score > 0 else -2  # 추세 방향 강화
    elif vs > 2: score += 1 if score > 0 else -1
    
    # VWAP 위치
    vw = vwap(candles_1m, 20)
    if vw:
        if cp > vw * 1.001: score += 1  # VWAP 위 = 강세
        elif cp < vw * 0.999: score -= 1  # VWAP 아래 = 약세
    
    # 연속 캔들 방향
    if len(c1) >= 4:
        if c1[-1]>c1[-2]>c1[-3]>c1[-4]: score += 2
        elif c1[-1]<c1[-2]<c1[-3]<c1[-4]: score -= 2
    
    # BB 터치
    _, bu, bl = bb(c1)
    if bl and cp <= bl*1.002: score += 2  # 하단 터치 반등
    if bu and cp >= bu*0.998: score -= 2  # 상단 터치 하락
    
    return max(-10, min(10, score))

def analyze(symbol):
    """종목 분석 -> 시그널"""
    c1m = get_klines(symbol, "1m", 60)
    c5m = get_klines_5m(symbol, 30)
    if not c1m or len(c1m) < 30: return None
    
    ms = momentum_score(c1m, c5m)
    cp = c1m[-1]['close']
    closes = [c['close'] for c in c1m]
    r = rsi(closes, 10)
    vs = vol_spike(c1m, 15)
    
    e5 = ema(closes, 5); e13 = ema(closes, 13)
    
    reasons = []
    
    if ms >= 5:
        if e5 and e13 and e5 > e13: reasons.append("1m EMA5>13")
        if r < 40: reasons.append(f"RSI {r:.0f} 과매도 반등")
        if vs > 2: reasons.append(f"거래량 {vs:.1f}x")
        if len(closes)>=3 and closes[-1]>closes[-2]>closes[-3]: reasons.append("연속상승")
        _, _, bl = bb(closes)
        if bl and cp <= bl*1.005: reasons.append("BB하단 반등")
        vw = vwap(c1m)
        if vw and cp > vw: reasons.append("VWAP 위")
        
        return {'symbol':symbol,'signal':'LONG','score':ms,'price':cp,
                'rsi':r,'vol_spike':vs,'reasons':reasons}
    
    elif ms <= -5:
        if e5 and e13 and e5 < e13: reasons.append("1m EMA5<13")
        if r > 60: reasons.append(f"RSI {r:.0f} 과매수 하락")
        if vs > 2: reasons.append(f"거래량 {vs:.1f}x")
        if len(closes)>=3 and closes[-1]<closes[-2]<closes[-3]: reasons.append("연속하락")
        _, bu, _ = bb(closes)
        if bu and cp >= bu*0.995: reasons.append("BB상단 저항")
        vw = vwap(c1m)
        if vw and cp < vw: reasons.append("VWAP 아래")
        
        return {'symbol':symbol,'signal':'SHORT','score':abs(ms),'price':cp,
                'rsi':r,'vol_spike':vs,'reasons':reasons}
    
    return None

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE,'r') as f: return json.load(f)
    return {
        'balance': INITIAL_USDT,
        'initial_balance': INITIAL_USDT,
        'positions': [],
        'closed_trades': [],
        'total_fees': 0,
        'scan_count': 0,
        'start_time': datetime.now(KST).isoformat(),
        'daily_start_balance': INITIAL_USDT,
        'daily_pnl': 0,
        'today': datetime.now(KST).strftime('%Y-%m-%d'),
    }

def save_state(s):
    with open(STATE_FILE,'w') as f: json.dump(s, f, indent=2, ensure_ascii=False)

def get_price(sym):
    d = api_get(f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={sym}")
    return float(d['price']) if d else None

def pnl_pct(side, entry, current):
    return ((current-entry)/entry*100) if side=="LONG" else ((entry-current)/entry*100)

def open_pos(state, sig):
    sym = sig['symbol']
    for p in state['positions']:
        if p['symbol']==sym: return False
    
    size = state['balance'] * POSITION_SIZE_PCT
    if size < 5: return False
    
    ep = sig['price']; side = sig['signal']
    if side=="LONG":
        sl = ep*(1+STOP_LOSS_PCT/100); tp = ep*(1+TAKE_PROFIT_PCT/100)
    else:
        sl = ep*(1-STOP_LOSS_PCT/100); tp = ep*(1-TAKE_PROFIT_PCT/100)
    
    pv = size*LEVERAGE; fee = pv*TAKER_FEE
    pos = {'symbol':sym,'side':side,'entry_price':ep,'size_usdt':size,
           'position_value':pv,'leverage':LEVERAGE,
           'entry_time':datetime.now(KST).isoformat(),
           'sl':sl,'tp':tp,'entry_fee':fee,
           'reasons':sig['reasons'],'score':sig['score'],'rsi':sig['rsi']}
    
    state['balance'] -= size+fee
    state['total_fees'] += fee
    state['positions'].append(pos)
    
    msg = f"OPEN {side} {sym.replace('USDT','')} @ {ep} | {size:.0f} USDT x{LEVERAGE} | SL {sl:.6g} TP {tp:.6g} | 근거: {', '.join(sig['reasons'])}"
    print(f"  >>> {msg}")
    write_alert({'type':'OPEN','side':side,'symbol':sym.replace('USDT','/USDT'),
                 'entry_price':ep,'size_usdt':round(size,1),'leverage':LEVERAGE,
                 'sl':round(sl,6),'tp':round(tp,6),
                 'reasons':sig['reasons'],'score':sig['score'],'rsi':round(sig['rsi'],1),
                 'balance':round(state['balance'],1),'positions':len(state['positions'])})
    return True

def close_pos(state, idx, cp, reason):
    pos = state['positions'][idx]
    pp = pnl_pct(pos['side'], pos['entry_price'], cp)
    pu = pos['size_usdt']*LEVERAGE*pp/100
    ef = pos['position_value']*TAKER_FEE
    net = pu - ef
    
    state['balance'] += pos['size_usdt'] + net
    state['total_fees'] += ef
    state['daily_pnl'] += net
    
    dur = (datetime.now(KST)-datetime.fromisoformat(pos['entry_time'])).total_seconds()/60
    
    trade = {'symbol':pos['symbol'],'side':pos['side'],
             'entry_price':pos['entry_price'],'exit_price':cp,
             'pnl_pct':round(pp,2),'net_pnl':round(net,2),
             'close_reason':reason,'duration_min':round(dur,1),
             'close_time':datetime.now(KST).isoformat(),
             'reasons':pos.get('reasons',[])}
    state['closed_trades'].append(trade)
    
    wins = [t for t in state['closed_trades'] if t['net_pnl']>0]
    tt = len(state['closed_trades'])
    wr = len(wins)/tt*100 if tt else 0
    cum = sum(t['net_pnl'] for t in state['closed_trades'])
    total_val = state['balance'] + sum(p['size_usdt'] for p in state['positions'] if p!=pos)
    total_ret = (total_val - state['initial_balance'])/state['initial_balance']*100
    
    emoji = "W" if net>0 else "L"
    msg = f"CLOSE [{emoji}] {pos['side']} {pos['symbol'].replace('USDT','')} @ {cp} | {net:+.1f} USDT ({pp:+.2f}%) | {reason} | {dur:.0f}분 | 총{tt}건 승률{wr:.0f}% | 누적{cum:+.0f} | 총수익률{total_ret:+.1f}%"
    print(f"  <<< {msg}")
    write_alert({'type':'CLOSE','result':emoji,'side':pos['side'],
                 'symbol':pos['symbol'].replace('USDT','/USDT'),
                 'entry_price':pos['entry_price'],'exit_price':cp,
                 'pnl_pct':round(pp,2),'net_pnl':round(net,2),
                 'close_reason':reason,'duration_min':round(dur,1),
                 'reasons':pos.get('reasons',[]),
                 'total_trades':tt,'win_rate':round(wr,1),
                 'cumulative_pnl':round(cum,1),
                 'portfolio_return':round(total_ret,1),
                 'balance':round(state['balance'],1)})
    state['positions'].pop(idx)

def check_positions(state):
    to_close = []
    for i, pos in enumerate(state['positions']):
        cp = get_price(pos['symbol'])
        if not cp: continue
        pp = pnl_pct(pos['side'], pos['entry_price'], cp)
        
        reason = None
        if pos['side']=="LONG":
            if cp<=pos['sl']: reason="STOP_LOSS"
            elif cp>=pos['tp']: reason="TAKE_PROFIT"
        else:
            if cp>=pos['sl']: reason="STOP_LOSS"
            elif cp<=pos['tp']: reason="TAKE_PROFIT"
        if pp*LEVERAGE<=-95: reason="LIQUIDATION"
        
        # 트레일링: 수익 1.5% 넘었다가 0.5%로 내려오면 보호 익절
        lev_pnl = pp * LEVERAGE / 100  # 실제 포트 수익률
        if pp >= 1.5 and pp < 1.0:
            reason = "TRAILING_PROTECT"
        
        if reason:
            to_close.append((i,cp,reason))
        else:
            pu = pos['size_usdt']*LEVERAGE*pp/100
            print(f"  ... {pos['side']} {pos['symbol'].replace('USDT','')} {pp:+.2f}% ({pu:+.0f} USDT)")
    
    for i,cp,r in sorted(to_close, key=lambda x:x[0], reverse=True):
        close_pos(state, i, cp, r)

def hourly_report(state):
    bal = state['balance']
    ur = 0
    lines = []
    for p in state['positions']:
        cp = get_price(p['symbol'])
        if not cp: continue
        pp = pnl_pct(p['side'], p['entry_price'], cp)
        pu = p['size_usdt']*LEVERAGE*pp/100; ur += pu
        lines.append(f"  {p['side']} {p['symbol'].replace('USDT','')} | {pp:+.2f}% ({pu:+.0f})")
    
    tv = bal + sum(p['size_usdt'] for p in state['positions']) + ur
    tp = tv - state['initial_balance']
    tpp = tp/state['initial_balance']*100
    wins = [t for t in state['closed_trades'] if t['net_pnl']>0]
    tt = len(state['closed_trades'])
    wr = len(wins)/tt*100 if tt else 0
    
    now = datetime.now(KST)
    report = f"[AGG {now.strftime('%m/%d %H:%M')}] {tv:.0f} USDT ({tpp:+.1f}%) | 거래{tt}건 승률{wr:.0f}%"
    if lines: report += "\n" + "\n".join(lines)
    
    write_alert({'type':'HOURLY','report':report,'portfolio':round(tv,1),
                 'return_pct':round(tpp,1),'trades':tt,'win_rate':round(wr,1)})
    return report

def main():
    now = datetime.now(KST)
    print(f"[AGG BOT] 시작: {now.strftime('%Y-%m-%d %H:%M:%S KST')}")
    print(f"자금: {INITIAL_USDT:.0f} USDT (300만원) | x{LEVERAGE} | SL {STOP_LOSS_PCT}% TP {TAKE_PROFIT_PCT}%")
    print(f"종료: {END_TIME.strftime('%Y-%m-%d %H:%M KST')}")
    
    state = load_state()
    save_state(state)
    
    write_alert({'type':'START','balance':state['initial_balance'],
                 'leverage':LEVERAGE,'sl':STOP_LOSS_PCT,'tp':TAKE_PROFIT_PCT})
    
    last_report_hour = -1
    
    while datetime.now(KST) < END_TIME:
        state['scan_count'] += 1
        now = datetime.now(KST)
        
        # 날짜 바뀌면 daily 리셋
        today = now.strftime('%Y-%m-%d')
        if today != state.get('today',''):
            state['daily_start_balance'] = state['balance'] + sum(p['size_usdt'] for p in state['positions'])
            state['daily_pnl'] = 0
            state['today'] = today
        
        bal = state['balance']
        n_pos = len(state['positions'])
        
        if state['scan_count'] % 5 == 0:  # 5분마다 로그
            print(f"\n--- AGG #{state['scan_count']} | {now.strftime('%m/%d %H:%M:%S')} | 잔고 {bal:.0f} | 포지션 {n_pos} ---")
        
        # 포지션 체크
        if state['positions']:
            check_positions(state)
        
        # 새 진입
        if len(state['positions']) < MAX_POSITIONS:
            signals = []
            for sym in TARGET_SYMBOLS:
                sig = analyze(sym)
                if sig: signals.append(sig)
                time.sleep(0.08)
            
            signals.sort(key=lambda x: x['score'], reverse=True)
            
            for sig in signals:
                if len(state['positions']) >= MAX_POSITIONS: break
                if sig['score'] >= 5:
                    open_pos(state, sig)
        
        save_state(state)
        
        # 매시간 리포트
        ch = now.hour
        if ch != last_report_hour and now.minute < 3:
            r = hourly_report(state)
            print(f"\n{r}")
            last_report_hour = ch
        
        time.sleep(SCAN_INTERVAL)
    
    # === 종료 ===
    print("\n=== AGG BOT 종료 ===")
    for i in range(len(state['positions'])-1,-1,-1):
        cp = get_price(state['positions'][i]['symbol'])
        if cp: close_pos(state, i, cp, "SESSION_END")
    
    save_state(state)
    final = hourly_report(state)
    write_alert({'type':'FINAL','report':final})
    print(final)
    
    with open(os.path.join(WORKSPACE,"AGG_DONE"),'w') as f:
        f.write(datetime.now(KST).isoformat())

if __name__ == "__main__":
    main()
