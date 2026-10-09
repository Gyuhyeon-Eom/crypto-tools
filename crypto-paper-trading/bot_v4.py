#!/usr/bin/env python3
"""
코인 선물 페이퍼 트레이딩 봇 v4
- 10x 레버리지, 매집 패턴 감지, 과열 종목 필터
- 멀티타임프레임 추세 (1h/15m/5m)
- BTC/ETH/SOL + 메이저 알트 집중
- 종료 없음 (계속 매매)
"""

import json
import os
import time
import math
import signal
import sys
import urllib.request
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(WORKSPACE, "v4_state.json")
ALERT_FILE = os.path.join(WORKSPACE, "v4_alerts.jsonl")
REPORT_FILE = os.path.join(WORKSPACE, "v4_report_queue.jsonl")

# === 설정 ===
USDT_KRW_RATE = 1390
INITIAL_USDT = 5755  # 합산 복구
LEVERAGE = 10
MAX_POSITIONS = 3
POSITION_SIZE_PCT = 0.30    # 잔고 30% (10x면 실 리스크 ~4.5%/건)
STOP_LOSS_PCT = -1.5        # SL -1.5%
TAKE_PROFIT_PCT = 3.0       # TP +3% (손익비 1:2)
TRAILING_ACTIVATE = 2.0     # +2%부터 트레일링
TRAILING_CALLBACK = 0.8     # 고점 대비 0.8% 되돌림시 익절
TAKER_FEE = 0.0005
SCAN_INTERVAL = 180          # 3분

# 종료 없음 - 계속 매매
END_TIME = None

# 메이저만 - 유동성 높고 BTC 추세 따라가는 종목
TARGET_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "NEARUSDT", "APTUSDT", "SUIUSDT",
    "ARBUSDT", "OPUSDT",
]

# 24h 상승률 이 이상이면 "이미 너무 오른 종목" -> 롱 진입 안 함
OVERHEATED_LONG_PCT = 10.0   # 24h +10% 이상이면 롱 스킵
OVERHEATED_SHORT_PCT = -10.0  # 24h -10% 이상 하락이면 숏 스킵 (반등 가능)


# ==================== API ====================

def api_get(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except:
        return None


def get_price(symbol):
    d = api_get(f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}")
    return float(d['price']) if d else None


def get_24h_change(symbol):
    """24시간 변동률 조회"""
    d = api_get(f"https://fapi.binance.com/fapi/v1/ticker/24hr?symbol={symbol}")
    if d:
        return float(d.get('priceChangePercent', 0))
    return 0


def get_klines(symbol, interval="5m", limit=50):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = api_get(url)
    if not data:
        return None
    return [{
        'open': float(d[1]), 'high': float(d[2]), 'low': float(d[3]),
        'close': float(d[4]), 'volume': float(d[5]), 'quote_volume': float(d[7]),
        'timestamp': int(d[0])
    } for d in data]


# ==================== 지표 ====================

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
        d = prices[i] - prices[i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    ag = sum(gains[-period:]) / period
    al = sum(losses[-period:]) / period
    if al == 0:
        return 100
    return 100 - (100 / (1 + ag / al))


def calc_volume_spike(candles, lookback=20):
    if len(candles) < lookback + 1:
        return 1
    avg = sum(c['quote_volume'] for c in candles[-(lookback + 1):-1]) / lookback
    return candles[-1]['quote_volume'] / avg if avg > 0 else 1


def calc_bb(prices, period=20, mult=2):
    if len(prices) < period:
        return None, None, None
    r = prices[-period:]
    sma = sum(r) / period
    std = math.sqrt(sum((p - sma) ** 2 for p in r) / period)
    return sma, sma + mult * std, sma - mult * std


def calc_atr(candles, period=14):
    if len(candles) < period + 1:
        return None
    trs = []
    for i in range(1, len(candles)):
        h, l, pc = candles[i]['high'], candles[i]['low'], candles[i - 1]['close']
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    return sum(trs[-period:]) / period


def calc_obv(candles):
    """On-Balance Volume - 매집/분산 판단"""
    if len(candles) < 2:
        return []
    obv = [0]
    for i in range(1, len(candles)):
        if candles[i]['close'] > candles[i-1]['close']:
            obv.append(obv[-1] + candles[i]['volume'])
        elif candles[i]['close'] < candles[i-1]['close']:
            obv.append(obv[-1] - candles[i]['volume'])
        else:
            obv.append(obv[-1])
    return obv


# ==================== 매집 패턴 감지 ====================

def detect_accumulation(candles_5m, candles_1h):
    """
    세력 매집 패턴 감지 (어제 배운 잡알트 세력 매집량 매매법 적용)
    
    매집 신호:
    1. 가격은 횡보/소폭 하락인데 거래량 증가 (가격 안 내리고 버팀)
    2. OBV 상승 추세 (가격 대비 거래량이 매수 쪽으로 쏠림)
    3. 하단 꼬리 긴 캔들 반복 (아래서 매수 받아먹음)
    4. 볼린저 밴드 수축 (변동성 축소 -> 곧 터질 준비)
    
    Returns: score (0~10), reasons
    """
    if not candles_5m or len(candles_5m) < 30:
        return 0, []
    
    score = 0
    reasons = []
    closes = [c['close'] for c in candles_5m]
    
    # 1. 가격 횡보 + 거래량 증가 패턴
    # 최근 10봉 vs 이전 20봉 가격 변동률
    recent_range = (max(closes[-10:]) - min(closes[-10:])) / closes[-10] * 100
    prev_range = (max(closes[-30:-10]) - min(closes[-30:-10])) / closes[-30] * 100 if len(closes) >= 30 else recent_range
    
    recent_vol = sum(c['quote_volume'] for c in candles_5m[-10:]) / 10
    prev_vol = sum(c['quote_volume'] for c in candles_5m[-30:-10]) / 20 if len(candles_5m) >= 30 else recent_vol
    
    # 가격 변동 줄었는데 거래량 늘었으면 -> 매집
    if recent_range < prev_range * 0.7 and recent_vol > prev_vol * 1.3:
        score += 3
        reasons.append(f"횡보+거래량증가 (범위 {recent_range:.1f}% vs {prev_range:.1f}%, 볼륨 {recent_vol/prev_vol:.1f}x)")
    elif recent_vol > prev_vol * 1.5:
        score += 1
        reasons.append(f"거래량 증가 {recent_vol/prev_vol:.1f}x")
    
    # 2. OBV 상승 추세
    obv = calc_obv(candles_5m)
    if len(obv) >= 20:
        obv_recent = obv[-1]
        obv_mid = obv[-10]
        obv_old = obv[-20]
        if obv_recent > obv_mid > obv_old:
            score += 2
            reasons.append("OBV 상승추세 (매집)")
        elif obv_recent > obv_old:
            score += 1
            reasons.append("OBV 순매수")
    
    # 3. 하단 꼬리 긴 캔들 (아래서 받아먹는 세력)
    tail_count = 0
    for c in candles_5m[-10:]:
        body = abs(c['close'] - c['open'])
        lower_tail = min(c['open'], c['close']) - c['low']
        if body > 0 and lower_tail > body * 1.5:
            tail_count += 1
    if tail_count >= 3:
        score += 2
        reasons.append(f"하단꼬리 {tail_count}회 (매수세)")
    elif tail_count >= 2:
        score += 1
        reasons.append(f"하단꼬리 {tail_count}회")
    
    # 4. 볼린저 밴드 수축 (스퀴즈 -> 폭발 임박)
    sma, bb_up, bb_lo = calc_bb(closes)
    if sma and bb_up and bb_lo:
        bb_width = (bb_up - bb_lo) / sma * 100
        if bb_width < 1.5:  # 매우 좁은 밴드
            score += 2
            reasons.append(f"BB 스퀴즈 ({bb_width:.1f}%)")
        elif bb_width < 2.5:
            score += 1
            reasons.append(f"BB 수축 ({bb_width:.1f}%)")
    
    # 5. 1시간봉에서도 매집 확인 (더 큰 그림)
    if candles_1h and len(candles_1h) >= 10:
        obv_1h = calc_obv(candles_1h)
        if len(obv_1h) >= 5:
            if obv_1h[-1] > obv_1h[-5]:
                score += 1
                reasons.append("1h OBV 강세")
    
    return score, reasons


# ==================== 멀티타임프레임 추세 ====================

def get_trend_mtf(symbol):
    """
    멀티타임프레임 추세 분석 (1h + 15m + 5m)
    Returns: 'STRONG_BULL', 'BULL', 'NEUTRAL', 'BEAR', 'STRONG_BEAR'
    + trend_score (-6 ~ +6)
    """
    frames = [
        ("1h", 50, 3),   # 1시간봉, weight 3
        ("15m", 50, 2),  # 15분봉, weight 2
        ("5m", 50, 1),   # 5분봉, weight 1
    ]
    
    total_score = 0
    details = []
    
    for interval, limit, weight in frames:
        candles = get_klines(symbol, interval, limit)
        if not candles or len(candles) < 25:
            continue
        
        closes = [c['close'] for c in candles]
        e9 = calc_ema(closes, 9)
        e21 = calc_ema(closes, 21)
        r = calc_rsi(closes, 14)
        
        if e9 and e21:
            if e9 > e21:
                total_score += weight
                details.append(f"{interval}:BULL")
            else:
                total_score -= weight
                details.append(f"{interval}:BEAR")
        time.sleep(0.1)
    
    if total_score >= 5:
        return "STRONG_BULL", total_score, details
    elif total_score >= 2:
        return "BULL", total_score, details
    elif total_score <= -5:
        return "STRONG_BEAR", total_score, details
    elif total_score <= -2:
        return "BEAR", total_score, details
    return "NEUTRAL", total_score, details


def get_btc_trend():
    """BTC 멀티타임프레임 추세"""
    trend, score, details = get_trend_mtf("BTCUSDT")
    return trend, score


# ==================== 시그널 분석 ====================

def analyze_signal(symbol, btc_trend, btc_score):
    """시그널 분석 - 매집 패턴 + 과열 필터 + MTF 추세"""
    
    # 24h 변동률 체크 (과열 필터)
    change_24h = get_24h_change(symbol)
    
    candles_5m = get_klines(symbol, "5m", 50)
    candles_1h = get_klines(symbol, "1h", 30)
    
    if not candles_5m or len(candles_5m) < 30:
        return None
    
    closes = [c['close'] for c in candles_5m]
    cp = closes[-1]
    
    # 기본 지표
    ema9 = calc_ema(closes, 9)
    ema21 = calc_ema(closes, 21)
    if not ema9 or not ema21:
        return None
    
    r = calc_rsi(closes, 14)
    vol = calc_volume_spike(candles_5m)
    _, bb_up, bb_lo = calc_bb(closes)
    
    # 매집 패턴 감지
    accum_score, accum_reasons = detect_accumulation(candles_5m, candles_1h)
    
    # 종목별 MTF 추세
    sym_trend, sym_score, sym_details = get_trend_mtf(symbol)
    
    # === LONG 시그널 ===
    long_score = 0
    long_reasons = []
    
    # BTC 필터
    if btc_trend in ("BEAR", "STRONG_BEAR"):
        long_score = -99  # BTC 하락 중 롱 차단
    # 과열 필터: 이미 24h +10% 이상 오른 종목은 롱 위험
    elif change_24h > OVERHEATED_LONG_PCT:
        long_score = -99
    else:
        # MTF 추세
        if sym_trend == "STRONG_BULL":
            long_score += 3
            long_reasons.append(f"MTF 강한 상승 ({'/'.join(sym_details)})")
        elif sym_trend == "BULL":
            long_score += 2
            long_reasons.append(f"MTF 상승 ({'/'.join(sym_details)})")
        
        # 매집 패턴 (핵심!)
        if accum_score >= 5:
            long_score += 3
            long_reasons.extend(accum_reasons[:2])
        elif accum_score >= 3:
            long_score += 2
            long_reasons.extend(accum_reasons[:1])
        
        # EMA 추세
        if ema9 > ema21:
            long_score += 1
            long_reasons.append("EMA9>21")
        
        # RSI
        if r < 30:
            long_score += 2
            long_reasons.append(f"RSI {r:.0f} 극과매도")
        elif r < 40:
            long_score += 1
            long_reasons.append(f"RSI {r:.0f} 과매도")
        
        # 거래량
        if vol > 2.5:
            long_score += 1
            long_reasons.append(f"거래량 {vol:.1f}x")
        
        # BB 하단
        if bb_lo and cp <= bb_lo * 1.003:
            long_score += 1
            long_reasons.append("BB 하단")
        
        # BTC 보너스
        if btc_trend == "STRONG_BULL":
            long_score += 1
            long_reasons.append("BTC 강세")
    
    # === SHORT 시그널 ===
    short_score = 0
    short_reasons = []
    
    if btc_trend in ("BULL", "STRONG_BULL"):
        short_score = -99  # BTC 상승 중 숏 차단
    elif change_24h < OVERHEATED_SHORT_PCT:
        short_score = -99  # 이미 -10% 이상 빠진 종목 숏 위험 (반등)
    else:
        if sym_trend == "STRONG_BEAR":
            short_score += 3
            short_reasons.append(f"MTF 강한 하락 ({'/'.join(sym_details)})")
        elif sym_trend == "BEAR":
            short_score += 2
            short_reasons.append(f"MTF 하락 ({'/'.join(sym_details)})")
        
        if ema9 < ema21:
            short_score += 1
            short_reasons.append("EMA9<21")
        if r > 70:
            short_score += 2
            short_reasons.append(f"RSI {r:.0f} 극과매수")
        elif r > 60:
            short_score += 1
            short_reasons.append(f"RSI {r:.0f} 과매수")
        if vol > 2.5:
            short_score += 1
            short_reasons.append(f"거래량 {vol:.1f}x")
        if bb_up and cp >= bb_up * 0.997:
            short_score += 1
            short_reasons.append("BB 상단")
        if btc_trend == "STRONG_BEAR":
            short_score += 1
            short_reasons.append("BTC 약세")
    
    # 최소 5점 (매집 감지로 더 확실한 자리만)
    MIN_SCORE = 5
    
    result = None
    if long_score >= MIN_SCORE and long_score > short_score:
        result = {
            'symbol': symbol, 'signal': 'LONG', 'confidence': long_score,
            'price': cp, 'rsi': r, 'ema9': ema9, 'ema21': ema21,
            'vol_spike': vol, 'reasons': long_reasons,
            'accum_score': accum_score, 'change_24h': change_24h,
            'sym_trend': sym_trend,
        }
    elif short_score >= MIN_SCORE and short_score > long_score:
        result = {
            'symbol': symbol, 'signal': 'SHORT', 'confidence': short_score,
            'price': cp, 'rsi': r, 'ema9': ema9, 'ema21': ema21,
            'vol_spike': vol, 'reasons': short_reasons,
            'accum_score': 0, 'change_24h': change_24h,
            'sym_trend': sym_trend,
        }
    
    return result


# ==================== 상태 관리 ====================

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return None


def init_state():
    return {
        'version': 'v4',
        'balance': INITIAL_USDT,
        'initial_balance': INITIAL_USDT,
        'positions': [],
        'closed_trades': [],
        'total_fees': 0,
        'scan_count': 0,
        'start_time': datetime.now(KST).isoformat(),
    }


def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


# ==================== 로깅 ====================

def write_alert(data):
    data['timestamp'] = datetime.now(KST).isoformat()
    with open(ALERT_FILE, 'a') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')


def write_report(report_text):
    entry = {
        'timestamp': datetime.now(KST).isoformat(),
        'report': report_text,
        'sent': False,
    }
    with open(REPORT_FILE, 'a') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


# ==================== 트레이딩 ====================

def calc_pnl_pct(side, entry, current):
    if side == "LONG":
        return (current - entry) / entry * 100
    return (entry - current) / entry * 100


def open_position(state, sig):
    sym = sig['symbol']
    for p in state['positions']:
        if p['symbol'] == sym:
            return False
    
    size = state['balance'] * POSITION_SIZE_PCT
    if size < 10:
        return False
    
    ep = sig['price']
    side = sig['signal']
    
    if side == "LONG":
        sl = ep * (1 + STOP_LOSS_PCT / 100)
        tp = ep * (1 + TAKE_PROFIT_PCT / 100)
    else:
        sl = ep * (1 - STOP_LOSS_PCT / 100)
        tp = ep * (1 - TAKE_PROFIT_PCT / 100)
    
    pv = size * LEVERAGE
    fee = pv * TAKER_FEE
    
    pos = {
        'symbol': sym, 'side': side, 'entry_price': ep,
        'size_usdt': size, 'position_value': pv, 'leverage': LEVERAGE,
        'entry_time': datetime.now(KST).isoformat(),
        'sl': sl, 'tp': tp, 'entry_fee': fee,
        'reasons': sig['reasons'], 'rsi': sig['rsi'],
        'confidence': sig['confidence'],
        'accum_score': sig.get('accum_score', 0),
        'change_24h': sig.get('change_24h', 0),
        'max_pnl_pct': 0,
    }
    
    state['balance'] -= size + fee
    state['total_fees'] += fee
    state['positions'].append(pos)
    
    accum_tag = f" | 매집점수 {sig.get('accum_score', 0)}" if sig.get('accum_score', 0) > 0 else ""
    print(f"  >>> OPEN {side} {sym} @ {ep} | {size:.1f} USDT x{LEVERAGE}{accum_tag}")
    print(f"      24h변동: {sig.get('change_24h', 0):+.1f}% | 추세: {sig.get('sym_trend', '?')}")
    print(f"      근거: {', '.join(sig['reasons'])}")
    
    write_alert({
        'type': 'OPEN', 'side': side,
        'symbol': sym.replace('USDT', '/USDT'),
        'entry_price': ep, 'size_usdt': round(size, 2),
        'leverage': LEVERAGE, 'sl': round(sl, 6), 'tp': round(tp, 6),
        'reasons': sig['reasons'], 'rsi': round(sig['rsi'], 1),
        'confidence': sig['confidence'],
        'accum_score': sig.get('accum_score', 0),
        'change_24h': round(sig.get('change_24h', 0), 1),
        'sym_trend': sig.get('sym_trend', ''),
        'balance': round(state['balance'], 2),
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
    
    trade = {
        'symbol': pos['symbol'], 'side': pos['side'],
        'entry_price': pos['entry_price'], 'exit_price': current_price,
        'pnl_pct': round(pnl_pct, 2), 'net_pnl': round(net_pnl, 2),
        'close_reason': close_reason, 'duration_min': round(dur_min, 1),
        'close_time': datetime.now(KST).isoformat(),
        'reasons_entry': pos.get('reasons', []),
        'max_pnl_pct': round(pos.get('max_pnl_pct', 0), 2),
        'accum_score': pos.get('accum_score', 0),
    }
    state['closed_trades'].append(trade)
    
    wins = [t for t in state['closed_trades'] if t['net_pnl'] > 0]
    total = len(state['closed_trades'])
    wr = len(wins) / total * 100 if total else 0
    cum = sum(t['net_pnl'] for t in state['closed_trades'])
    total_val = state['balance'] + sum(
        p['size_usdt'] for j, p in enumerate(state['positions']) if j != idx
    )
    total_ret = (total_val - state['initial_balance']) / state['initial_balance'] * 100
    
    result = "W" if net_pnl > 0 else "L"
    print(f"  <<< CLOSE [{result}] {pos['side']} {pos['symbol']} | {close_reason}")
    print(f"      {net_pnl:+.2f} USDT ({pnl_pct:+.2f}%) | {dur_min:.0f}분 | 총{total}건 승률{wr:.0f}%")
    
    write_alert({
        'type': 'CLOSE', 'result': result, 'side': pos['side'],
        'symbol': pos['symbol'].replace('USDT', '/USDT'),
        'entry_price': pos['entry_price'], 'exit_price': current_price,
        'pnl_pct': round(pnl_pct, 2), 'net_pnl': round(net_pnl, 2),
        'close_reason': close_reason, 'duration_min': round(dur_min, 1),
        'reasons_entry': pos.get('reasons', []),
        'total_trades': total, 'win_rate': round(wr, 1),
        'cumulative_pnl': round(cum, 2),
        'portfolio_return': round(total_ret, 1),
        'balance': round(state['balance'], 2),
    })
    state['positions'].pop(idx)


def check_positions(state):
    to_close = []
    
    for i, pos in enumerate(state['positions']):
        cp = get_price(pos['symbol'])
        if not cp:
            continue
        
        pnl = calc_pnl_pct(pos['side'], pos['entry_price'], cp)
        
        if pnl > pos.get('max_pnl_pct', 0):
            pos['max_pnl_pct'] = pnl
        
        reason = None
        
        # 기본 SL/TP
        if pos['side'] == "LONG":
            if cp <= pos['sl']:
                reason = "STOP_LOSS"
            elif cp >= pos['tp']:
                reason = "TAKE_PROFIT"
        else:
            if cp >= pos['sl']:
                reason = "STOP_LOSS"
            elif cp <= pos['tp']:
                reason = "TAKE_PROFIT"
        
        # 트레일링 스탑
        if not reason and pos.get('max_pnl_pct', 0) >= TRAILING_ACTIVATE:
            callback = pos['max_pnl_pct'] - pnl
            if callback >= TRAILING_CALLBACK:
                reason = f"TRAILING (peak {pos['max_pnl_pct']:.1f}%)"
        
        # 청산 방지
        if pnl * LEVERAGE <= -90:
            reason = "LIQUIDATION"
        
        if reason:
            to_close.append((i, cp, reason))
        else:
            pu = pos['size_usdt'] * LEVERAGE * pnl / 100
            trail = f" [peak {pos['max_pnl_pct']:.1f}%]" if pos.get('max_pnl_pct', 0) > 0.5 else ""
            print(f"  ... {pos['side']} {pos['symbol'].replace('USDT','')} {pnl:+.2f}% ({pu:+.1f} USDT){trail}")
    
    for i, cp, reason in sorted(to_close, key=lambda x: x[0], reverse=True):
        close_position(state, i, cp, reason)


# ==================== 리포트 ====================

def generate_report(state, prefix=""):
    now = datetime.now(KST)
    bal = state['balance']
    
    unrealized = 0
    pos_lines = []
    for pos in state['positions']:
        cp = get_price(pos['symbol'])
        if cp:
            pnl = calc_pnl_pct(pos['side'], pos['entry_price'], cp)
            pnl_usdt = pos['size_usdt'] * LEVERAGE * pnl / 100
            unrealized += pnl_usdt
            sym = pos['symbol'].replace('USDT', '')
            trail = f" [peak {pos.get('max_pnl_pct', 0):.1f}%]" if pos.get('max_pnl_pct', 0) > 0.5 else ""
            accum = f" (매집{pos.get('accum_score', 0)})" if pos.get('accum_score', 0) > 0 else ""
            pos_lines.append(f"  {pos['side']} {sym} @ {pos['entry_price']} -> {cp} | {pnl:+.2f}% ({pnl_usdt:+.1f}){trail}{accum}")
    
    total_val = bal + sum(p['size_usdt'] for p in state['positions']) + unrealized
    total_pnl = total_val - state['initial_balance']
    total_pnl_pct = total_pnl / state['initial_balance'] * 100
    
    wins = [t for t in state['closed_trades'] if t['net_pnl'] > 0]
    losses = [t for t in state['closed_trades'] if t['net_pnl'] <= 0]
    total_trades = len(state['closed_trades'])
    wr = len(wins) / total_trades * 100 if total_trades else 0
    
    recent = state['closed_trades'][-5:]
    trade_lines = []
    for t in recent:
        e = "W" if t['net_pnl'] > 0 else "L"
        trade_lines.append(f"  [{e}] {t['side']} {t['symbol'].replace('USDT','')} | {t['net_pnl']:+.1f} ({t['pnl_pct']:+.1f}%) | {t['close_reason']}")
    
    report = f"""{prefix}[v4 {now.strftime('%m/%d %H:%M')}] 10x | BTC필터 | 매집감지
총 평가: {total_val:.0f} USDT ({total_val * USDT_KRW_RATE:,.0f}원)
손익: {total_pnl:+.1f} USDT ({total_pnl_pct:+.2f}%)
거래: {total_trades}건 (W{len(wins)}/L{len(losses)}) 승률 {wr:.0f}%
수수료: {state['total_fees']:.1f} USDT

포지션 ({len(state['positions'])}):
{chr(10).join(pos_lines) if pos_lines else '  (없음)'}"""
    
    if trade_lines:
        report += f"\n\n최근:\n{chr(10).join(trade_lines)}"
    
    return report


# ==================== 메인 ====================

def main():
    now = datetime.now(KST)
    print(f"=== v4 페이퍼 트레이딩 ===")
    print(f"시간: {now.strftime('%Y-%m-%d %H:%M:%S KST')}")
    print(f"자금: {INITIAL_USDT} USDT | 10x | SL {STOP_LOSS_PCT}% / TP {TAKE_PROFIT_PCT}%")
    print(f"트레일링: +{TRAILING_ACTIVATE}% 활성, {TRAILING_CALLBACK}% 콜백")
    print(f"매집 감지: ON | 과열 필터: 24h>{OVERHEATED_LONG_PCT}% 롱차단")
    print(f"MTF 추세: 1h/15m/5m | BTC 필터: ON")
    print(f"종목: {len(TARGET_SYMBOLS)}개 메이저")
    print(f"종료: 없음 (계속 매매)\n")
    
    # 시그널 핸들러 - kill 시 포지션 정리 후 종료
    def graceful_exit(signum, frame):
        print(f"\n=== 시그널 {signum} 수신, 포지션 정리 중... ===")
        for i in range(len(state['positions']) - 1, -1, -1):
            cp = get_price(state['positions'][i]['symbol'])
            if cp:
                close_position(state, i, cp, "MANUAL_STOP")
        save_state(state)
        final = generate_report(state, "[수동종료] ")
        write_report(final)
        write_alert({'type': 'MANUAL_STOP', 'report': final})
        print(f"\n{final}")
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, graceful_exit)
    signal.signal(signal.SIGINT, graceful_exit)
    
    state = load_state()
    if not state or state.get('version') != 'v4':
        state = init_state()
    
    save_state(state)
    
    write_alert({
        'type': 'START', 'version': 'v4',
        'balance': state['initial_balance'],
        'leverage': LEVERAGE,
        'features': ['btc_filter', 'accumulation_detect', 'overheat_filter', 'mtf_trend', 'trailing_stop'],
    })
    
    last_report_hour = -1
    last_btc_trend = "NEUTRAL"
    
    while True:
        state['scan_count'] += 1
        now = datetime.now(KST)
        
        # BTC 추세
        btc_trend, btc_score = get_btc_trend()
        if btc_trend != last_btc_trend:
            print(f"\n  ** BTC: {last_btc_trend} -> {btc_trend} (score {btc_score})")
            write_alert({'type': 'BTC_TREND', 'from': last_btc_trend, 'to': btc_trend, 'score': btc_score})
            last_btc_trend = btc_trend
        
        print(f"\n--- v4 #{state['scan_count']} | {now.strftime('%H:%M:%S')} | "
              f"잔고 {state['balance']:.0f} | 포지션 {len(state['positions'])} | BTC:{btc_trend}({btc_score}) ---")
        
        # 포지션 체크
        if state['positions']:
            check_positions(state)
        
        # 새 시그널
        if len(state['positions']) < MAX_POSITIONS:
            signals = []
            for sym in TARGET_SYMBOLS:
                sig = analyze_signal(sym, btc_trend, btc_score)
                if sig:
                    signals.append(sig)
                time.sleep(0.15)
            
            # 매집 점수 > 신뢰도 순 정렬
            signals.sort(key=lambda x: (x.get('accum_score', 0), x['confidence']), reverse=True)
            
            for sig in signals:
                if len(state['positions']) >= MAX_POSITIONS:
                    break
                open_position(state, sig)
        
        save_state(state)
        
        # 매시간 리포트
        ch = now.hour
        if ch != last_report_hour and now.minute < 6:
            report = generate_report(state)
            write_report(report)
            print(f"\n{report}")
            last_report_hour = ch
        
        time.sleep(SCAN_INTERVAL)
    
    # while True이므로 여기 도달 안 함 (graceful_exit으로 종료)


if __name__ == "__main__":
    main()
