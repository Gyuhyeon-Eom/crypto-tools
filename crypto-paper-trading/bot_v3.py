#!/usr/bin/env python3
"""
코인 선물 페이퍼 트레이딩 봇 v3
- 교훈 반영 전면 재설계
- 5x 레버리지, BTC 추세 필터, 트레일링 스탑
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
STATE_FILE = os.path.join(WORKSPACE, "v3_state.json")
ALERT_FILE = os.path.join(WORKSPACE, "v3_alerts.jsonl")
REPORT_FILE = os.path.join(WORKSPACE, "v3_report_queue.jsonl")

# === 설정 ===
USDT_KRW_RATE = 1390

# 보수적 + 공격적 합산 시드 복구
INITIAL_USDT_CONSERVATIVE = 3597  # 500만원
INITIAL_USDT_AGGRESSIVE = 2158   # 300만원
INITIAL_USDT = INITIAL_USDT_CONSERVATIVE + INITIAL_USDT_AGGRESSIVE  # 5755 USDT (~800만원)

LEVERAGE = 5           # v1(10x), v2(20x) -> v3(5x) 대폭 축소
MAX_POSITIONS = 3
POSITION_SIZE_PCT = 0.25  # 잔고의 25% -> 리스크 ~1.9%/건
STOP_LOSS_PCT = -1.5      # v2(-1%) 너무 타이트, v1(-2%) 적당 -> -1.5% 절충
TAKE_PROFIT_PCT = 3.0     # 손익비 1:2
TRAILING_ACTIVATE = 2.0   # 수익 2% 넘으면 트레일링 시작
TRAILING_CALLBACK = 1.0   # 고점 대비 1% 되돌림 시 익절
TAKER_FEE = 0.0005
SCAN_INTERVAL = 180        # 3분 (1분=노이즈, 5분=느림 -> 절충)

# 종료 시간
END_TIME = datetime(2026, 4, 6, 21, 55, tzinfo=KST)

# 유동성 높은 대형/중형만 (소형 알트 STO/DRIFT/ZETA/ONG 제거)
TARGET_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "BNBUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "NEARUSDT", "APTUSDT", "SUIUSDT", "ARBUSDT",
    "OPUSDT", "INJUSDT", "FETUSDT", "RENDERUSDT",
    "1000PEPEUSDT",
]


# ==================== API ====================

def api_get(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return None


def get_price(symbol):
    d = api_get(f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}")
    return float(d['price']) if d else None


def get_klines(symbol, interval="5m", limit=50):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = api_get(url)
    if not data:
        return None
    return [{
        'open': float(d[1]), 'high': float(d[2]), 'low': float(d[3]),
        'close': float(d[4]), 'volume': float(d[5]), 'quote_volume': float(d[7])
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
    """Average True Range - 변동성 측정"""
    if len(candles) < period + 1:
        return None
    trs = []
    for i in range(1, len(candles)):
        h, l, pc = candles[i]['high'], candles[i]['low'], candles[i - 1]['close']
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    return sum(trs[-period:]) / period


def calc_macd(prices, fast=12, slow=26, signal=9):
    """MACD 계산"""
    ema_fast = calc_ema(prices, fast)
    ema_slow = calc_ema(prices, slow)
    if not ema_fast or not ema_slow:
        return None, None, None
    macd_line = ema_fast - ema_slow
    # signal line은 간략 계산 (최근 MACD 값 기반)
    if len(prices) < slow + signal:
        return macd_line, None, None
    # 최근 signal 구간의 MACD 근사
    macd_vals = []
    for i in range(signal + 1):
        sub = prices[:len(prices) - i] if i > 0 else prices
        ef = calc_ema(sub, fast)
        es = calc_ema(sub, slow)
        if ef and es:
            macd_vals.insert(0, ef - es)
    if len(macd_vals) >= signal:
        sig_line = sum(macd_vals[-signal:]) / signal
        histogram = macd_line - sig_line
        return macd_line, sig_line, histogram
    return macd_line, None, None


# ==================== BTC 추세 필터 ====================

def get_btc_trend():
    """
    BTC 추세 판단: 5분봉 + 15분봉 EMA 크로스
    Returns: 'BULLISH', 'BEARISH', 'NEUTRAL'
    """
    # 5분봉
    c5m = get_klines("BTCUSDT", "5m", 50)
    # 15분봉
    c15m = get_klines("BTCUSDT", "15m", 50)

    if not c5m or not c15m:
        return "NEUTRAL"

    closes_5m = [c['close'] for c in c5m]
    closes_15m = [c['close'] for c in c15m]

    # 5분봉 EMA9/21
    ema9_5m = calc_ema(closes_5m, 9)
    ema21_5m = calc_ema(closes_5m, 21)

    # 15분봉 EMA9/21
    ema9_15m = calc_ema(closes_15m, 9)
    ema21_15m = calc_ema(closes_15m, 21)

    if not all([ema9_5m, ema21_5m, ema9_15m, ema21_15m]):
        return "NEUTRAL"

    bull_score = 0
    bear_score = 0

    # 5분봉 추세
    if ema9_5m > ema21_5m:
        bull_score += 1
    else:
        bear_score += 1

    # 15분봉 추세 (더 높은 가중치)
    if ema9_15m > ema21_15m:
        bull_score += 2
    else:
        bear_score += 2

    # RSI 보조
    rsi_15m = calc_rsi(closes_15m, 14)
    if rsi_15m > 55:
        bull_score += 1
    elif rsi_15m < 45:
        bear_score += 1

    if bull_score >= 3:
        return "BULLISH"
    elif bear_score >= 3:
        return "BEARISH"
    return "NEUTRAL"


# ==================== 시그널 분석 ====================

def analyze_signal(symbol, btc_trend):
    """
    5분봉 기반 시그널 분석
    BTC 추세 필터: 하락장에서 롱 X, 상승장에서 숏 X
    """
    candles = get_klines(symbol, "5m", 50)
    if not candles or len(candles) < 30:
        return None

    closes = [c['close'] for c in candles]
    cp = closes[-1]

    # 기본 지표
    ema9 = calc_ema(closes, 9)
    ema21 = calc_ema(closes, 21)
    if not ema9 or not ema21:
        return None

    r = calc_rsi(closes, 14)
    vol = calc_volume_spike(candles)
    _, bb_up, bb_lo = calc_bb(closes)
    atr = calc_atr(candles)
    macd_line, macd_sig, macd_hist = calc_macd(closes)

    # === LONG 시그널 ===
    long_score = 0
    long_reasons = []

    # BTC 필터: BTC 하락 중이면 롱 진입 차단
    if btc_trend == "BEARISH":
        long_score = -99  # 절대 진입 불가
    else:
        if ema9 > ema21:
            long_score += 2
            long_reasons.append("EMA9>21 상승추세")
        if r < 30:
            long_score += 2
            long_reasons.append(f"RSI {r:.0f} 극과매도")
        elif r < 40:
            long_score += 1
            long_reasons.append(f"RSI {r:.0f} 과매도")
        if vol > 2.5:
            long_score += 1
            long_reasons.append(f"거래량 {vol:.1f}x 급증")
        if bb_lo and cp <= bb_lo * 1.003:
            long_score += 1
            long_reasons.append("BB 하단 터치")
        if macd_hist and macd_hist > 0 and macd_line and macd_line > 0:
            long_score += 1
            long_reasons.append("MACD 강세")
        # EMA 눌림목 (가격이 EMA21 위에서 EMA9 근처까지 내려옴)
        if cp > ema21 and abs(cp - ema9) / cp < 0.002:
            long_score += 1
            long_reasons.append("EMA9 눌림목")
        if btc_trend == "BULLISH":
            long_score += 1
            long_reasons.append("BTC 상승추세")

    # === SHORT 시그널 ===
    short_score = 0
    short_reasons = []

    # BTC 필터: BTC 상승 중이면 숏 진입 차단
    if btc_trend == "BULLISH":
        short_score = -99
    else:
        if ema9 < ema21:
            short_score += 2
            short_reasons.append("EMA9<21 하락추세")
        if r > 70:
            short_score += 2
            short_reasons.append(f"RSI {r:.0f} 극과매수")
        elif r > 60:
            short_score += 1
            short_reasons.append(f"RSI {r:.0f} 과매수")
        if vol > 2.5:
            short_score += 1
            short_reasons.append(f"거래량 {vol:.1f}x 급증")
        if bb_up and cp >= bb_up * 0.997:
            short_score += 1
            short_reasons.append("BB 상단 터치")
        if macd_hist and macd_hist < 0 and macd_line and macd_line < 0:
            short_score += 1
            short_reasons.append("MACD 약세")
        if btc_trend == "BEARISH":
            short_score += 1
            short_reasons.append("BTC 하락추세")

    # 최소 4점 이상 + 상대방보다 높아야 진입 (v1은 3점이라 너무 느슨했음)
    MIN_SCORE = 4

    if long_score >= MIN_SCORE and long_score > short_score:
        return {
            'symbol': symbol, 'signal': 'LONG', 'confidence': long_score,
            'price': cp, 'rsi': r, 'ema9': ema9, 'ema21': ema21,
            'vol_spike': vol, 'reasons': long_reasons, 'atr': atr
        }
    if short_score >= MIN_SCORE and short_score > long_score:
        return {
            'symbol': symbol, 'signal': 'SHORT', 'confidence': short_score,
            'price': cp, 'rsi': r, 'ema9': ema9, 'ema21': ema21,
            'vol_spike': vol, 'reasons': short_reasons, 'atr': atr
        }
    return None


# ==================== 상태 관리 ====================

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return None


def init_state():
    return {
        'version': 'v3',
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
    # 같은 심볼 중복 진입 방지
    for p in state['positions']:
        if p['symbol'] == sym:
            return False

    size = state['balance'] * POSITION_SIZE_PCT
    if size < 10:
        return False

    ep = sig['price']
    side = sig['signal']

    # SL/TP 계산
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
        'max_pnl_pct': 0,  # 트레일링 스탑용 최대 수익률 추적
    }

    state['balance'] -= size + fee
    state['total_fees'] += fee
    state['positions'].append(pos)

    # 실제 리스크 계산
    risk_pct = abs(STOP_LOSS_PCT) * LEVERAGE * size / (state['balance'] + size) 
    
    print(f"  >>> OPEN {side} {sym} @ {ep} | {size:.1f} USDT x{LEVERAGE} | 리스크 {risk_pct:.1f}%")
    print(f"      근거: {', '.join(sig['reasons'])}")

    write_alert({
        'type': 'OPEN', 'side': side,
        'symbol': sym.replace('USDT', '/USDT'),
        'entry_price': ep, 'size_usdt': round(size, 2),
        'leverage': LEVERAGE, 'sl': round(sl, 6), 'tp': round(tp, 6),
        'reasons': sig['reasons'], 'rsi': round(sig['rsi'], 1),
        'confidence': sig['confidence'],
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
    print(f"  <<< CLOSE [{result}] {pos['side']} {pos['symbol']} | {close_reason} | {net_pnl:+.2f} USDT ({pnl_pct:+.2f}%)")
    print(f"      총 {total}건 승률 {wr:.0f}% | 누적 {cum:+.1f} USDT | 수익률 {total_ret:+.1f}%")

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
    """포지션 체크: SL/TP + 트레일링 스탑"""
    to_close = []

    for i, pos in enumerate(state['positions']):
        cp = get_price(pos['symbol'])
        if not cp:
            continue

        pnl = calc_pnl_pct(pos['side'], pos['entry_price'], cp)

        # 최대 수익률 업데이트 (트레일링용)
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

        # 트레일링 스탑: 수익 2% 넘었다가 고점 대비 1% 되돌림
        if not reason and pos.get('max_pnl_pct', 0) >= TRAILING_ACTIVATE:
            callback = pos['max_pnl_pct'] - pnl
            if callback >= TRAILING_CALLBACK:
                reason = f"TRAILING_STOP (고점 {pos['max_pnl_pct']:.1f}% -> 현재 {pnl:.1f}%)"

        # 청산 방지
        if pnl * LEVERAGE <= -90:
            reason = "LIQUIDATION"

        if reason:
            to_close.append((i, cp, reason))
        else:
            pu = pos['size_usdt'] * LEVERAGE * pnl / 100
            trail_info = f" [최대 {pos.get('max_pnl_pct', 0):.1f}%]" if pos.get('max_pnl_pct', 0) > 0.5 else ""
            print(f"  ... HOLD {pos['side']} {pos['symbol']} {pnl:+.2f}% ({pu:+.1f} USDT){trail_info}")

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
            sym_short = pos['symbol'].replace('USDT', '')
            trail = f" [peak {pos.get('max_pnl_pct', 0):.1f}%]" if pos.get('max_pnl_pct', 0) > 0.5 else ""
            pos_lines.append(f"  {pos['side']} {sym_short} | {pos['entry_price']} -> {cp} | {pnl:+.2f}% ({pnl_usdt:+.1f} USDT){trail}")

    total_val = bal + sum(p['size_usdt'] for p in state['positions']) + unrealized
    total_pnl = total_val - state['initial_balance']
    total_pnl_pct = total_pnl / state['initial_balance'] * 100

    wins = [t for t in state['closed_trades'] if t['net_pnl'] > 0]
    losses = [t for t in state['closed_trades'] if t['net_pnl'] <= 0]
    total_trades = len(state['closed_trades'])
    wr = len(wins) / total_trades * 100 if total_trades else 0

    # 최근 5건
    recent = state['closed_trades'][-5:]
    trade_lines = []
    for t in recent:
        e = "W" if t['net_pnl'] > 0 else "L"
        trade_lines.append(f"  [{e}] {t['side']} {t['symbol'].replace('USDT', '')} | {t['net_pnl']:+.1f} USDT ({t['pnl_pct']:+.1f}%) | {t['close_reason']}")

    report = f"""{prefix}[v3 {now.strftime('%m/%d %H:%M')}] 5x 레버리지 | BTC 필터
총 평가금: {total_val:.0f} USDT ({total_val * USDT_KRW_RATE:,.0f}원)
손익: {total_pnl:+.1f} USDT ({total_pnl_pct:+.2f}%) | {total_pnl * USDT_KRW_RATE:+,.0f}원
거래: {total_trades}건 (승{len(wins)}/패{len(losses)}) 승률 {wr:.0f}%
수수료: {state['total_fees']:.1f} USDT

보유 포지션 ({len(state['positions'])}개):
{chr(10).join(pos_lines) if pos_lines else '  (없음)'}"""

    if trade_lines:
        report += f"\n\n최근 거래:\n{chr(10).join(trade_lines)}"

    return report


# ==================== 메인 루프 ====================

def main():
    now = datetime.now(KST)
    print(f"=== 페이퍼 트레이딩 v3 시작 ===")
    print(f"시간: {now.strftime('%Y-%m-%d %H:%M:%S KST')}")
    print(f"자금: {INITIAL_USDT} USDT (~{INITIAL_USDT * USDT_KRW_RATE:,.0f}원)")
    print(f"레버리지: {LEVERAGE}x | SL: {STOP_LOSS_PCT}% | TP: {TAKE_PROFIT_PCT}%")
    print(f"트레일링: {TRAILING_ACTIVATE}% 활성화, {TRAILING_CALLBACK}% 콜백")
    print(f"BTC 추세 필터: ON")
    print(f"종료: {END_TIME.strftime('%Y-%m-%d %H:%M KST')}")
    print(f"종목: {len(TARGET_SYMBOLS)}개 (대형/중형만)\n")

    state = load_state()
    if not state or state.get('version') != 'v3':
        print("새 v3 상태 초기화")
        state = init_state()

    save_state(state)

    write_alert({
        'type': 'START', 'version': 'v3',
        'balance': state['initial_balance'],
        'leverage': LEVERAGE,
        'sl': STOP_LOSS_PCT, 'tp': TAKE_PROFIT_PCT,
        'trailing': f'{TRAILING_ACTIVATE}%/{TRAILING_CALLBACK}%',
        'btc_filter': True,
        'symbols': len(TARGET_SYMBOLS),
    })

    last_report_hour = -1
    last_btc_trend = "NEUTRAL"

    while datetime.now(KST) < END_TIME:
        state['scan_count'] += 1
        now = datetime.now(KST)

        # BTC 추세 업데이트 (매 스캔)
        btc_trend = get_btc_trend()
        if btc_trend != last_btc_trend:
            print(f"\n  ** BTC 추세 변경: {last_btc_trend} -> {btc_trend}")
            write_alert({'type': 'BTC_TREND', 'from': last_btc_trend, 'to': btc_trend})
            last_btc_trend = btc_trend

        print(f"\n--- v3 스캔 #{state['scan_count']} | {now.strftime('%m/%d %H:%M:%S')} | "
              f"잔고 {state['balance']:.1f} | 포지션 {len(state['positions'])} | BTC: {btc_trend} ---")

        # 포지션 체크 (SL/TP/트레일링)
        if state['positions']:
            check_positions(state)

        # 새 시그널 탐색
        if len(state['positions']) < MAX_POSITIONS:
            signals = []
            for sym in TARGET_SYMBOLS:
                if sym == "BTCUSDT":
                    # BTC는 추세 필터 대상이므로 별도 판단
                    pass
                sig = analyze_signal(sym, btc_trend)
                if sig:
                    signals.append(sig)
                time.sleep(0.12)

            signals.sort(key=lambda x: x['confidence'], reverse=True)

            for sig in signals:
                if len(state['positions']) >= MAX_POSITIONS:
                    break
                if sig['confidence'] >= 4:
                    open_position(state, sig)

        save_state(state)

        # 매시간 정시 리포트
        current_hour = now.hour
        if current_hour != last_report_hour and now.minute < 6:
            report = generate_report(state)
            write_report(report)
            print(f"\n[HOURLY REPORT]\n{report}")
            last_report_hour = current_hour

        time.sleep(SCAN_INTERVAL)

    # === 종료 ===
    print("\n=== v3 세션 종료, 전체 포지션 청산 ===")
    for i in range(len(state['positions']) - 1, -1, -1):
        cp = get_price(state['positions'][i]['symbol'])
        if cp:
            close_position(state, i, cp, "SESSION_END")

    save_state(state)
    final = generate_report(state, prefix="[최종] ")
    write_report(final)
    write_alert({'type': 'FINAL', 'report': final})
    print(f"\n{final}")

    with open(os.path.join(WORKSPACE, "V3_DONE"), 'w') as f:
        f.write(datetime.now(KST).isoformat())


if __name__ == "__main__":
    main()
