#!/usr/bin/env python3
"""
v4 스윙 스캐너 - 빠른 재스캔
RSI100/상폐코인 필터 추가, top 15 정상 종목만
"""

import json
import math
import time
import urllib.request
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))
MIN_QUOTE_VOLUME_24H = 50_000_000

# 바이낸스 상폐 예정/비정상 종목 블랙리스트
BLACKLIST = {
    'ALPACAUSDT', 'BNXUSDT', 'ALPHAUSDT', 'PORT3USDT', 'VIDTUSDT',
    'SXPUSDT', 'AGIXUSDT', 'LINAUSDT', 'MEMEFIUSDT', 'LEVERUSDT',
    'NEIROETHUSDT', 'FTMUSDT', 'WAVESUSDT', 'OMNIUSDT',
    'BADGERUSDT', 'NKNUSDT', 'MDTUSDT',
}


def api_get(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except:
        return None


def get_all_futures_tickers():
    data = api_get("https://fapi.binance.com/fapi/v1/ticker/24hr")
    if not data: return []
    results = []
    for d in data:
        sym = d['symbol']
        if not sym.endswith('USDT') or sym in BLACKLIST:
            continue
        vol = float(d.get('quoteVolume', 0))
        change = float(d.get('priceChangePercent', 0))
        price = float(d.get('lastPrice', 0))
        # 24h 변동 > 30% 이면 비정상
        if vol >= MIN_QUOTE_VOLUME_24H and price > 0 and abs(change) < 30:
            results.append({'symbol': sym, 'price': price, 'change_24h': change, 'volume_24h': vol})
    results.sort(key=lambda x: x['volume_24h'], reverse=True)
    return results


def get_klines(symbol, interval, limit=100):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = api_get(url)
    if not data: return None
    return [{'open': float(d[1]), 'high': float(d[2]), 'low': float(d[3]),
             'close': float(d[4]), 'volume': float(d[5]), 'quote_volume': float(d[7]),
             'timestamp': int(d[0])} for d in data]


def calc_ema(prices, period):
    if len(prices) < period: return None
    m = 2 / (period + 1)
    ema = sum(prices[:period]) / period
    for p in prices[period:]:
        ema = (p - ema) * m + ema
    return ema


def calc_rsi(prices, period=14):
    if len(prices) < period + 1: return 50
    gains, losses = [], []
    for i in range(1, len(prices)):
        d = prices[i] - prices[i-1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    ag = sum(gains[-period:]) / period
    al = sum(losses[-period:]) / period
    if al == 0: return 100
    return 100 - (100 / (1 + ag / al))


def calc_bb(prices, period=20, mult=2):
    if len(prices) < period: return None, None, None
    r = prices[-period:]
    sma = sum(r) / period
    std = math.sqrt(sum((p - sma)**2 for p in r) / period)
    return sma, sma + mult * std, sma - mult * std


def calc_macd(prices, fast=12, slow=26, signal=9):
    if len(prices) < slow + signal: return None, None, None
    m_fast = 2 / (fast + 1)
    m_slow = 2 / (slow + 1)
    ema_f = sum(prices[:fast]) / fast
    ema_s = sum(prices[:slow]) / slow
    macd_line = []
    for i, p in enumerate(prices):
        if i > 0:
            if i >= fast:
                ema_f = (p - ema_f) * m_fast + ema_f
            else:
                ema_f = sum(prices[:i+1]) / (i+1)
        if i >= slow - 1:
            ema_s = (p - ema_s) * m_slow + ema_s
            macd_line.append(ema_f - ema_s)
    if len(macd_line) < signal: return None, None, None
    m_sig = 2 / (signal + 1)
    sig_ema = sum(macd_line[:signal]) / signal
    for v in macd_line[signal:]:
        sig_ema = (v - sig_ema) * m_sig + sig_ema
    return macd_line[-1], sig_ema, macd_line[-1] - sig_ema


def calc_obv(candles):
    if len(candles) < 2: return []
    obv = [0]
    for i in range(1, len(candles)):
        if candles[i]['close'] > candles[i-1]['close']:
            obv.append(obv[-1] + candles[i]['volume'])
        elif candles[i]['close'] < candles[i-1]['close']:
            obv.append(obv[-1] - candles[i]['volume'])
        else:
            obv.append(obv[-1])
    return obv


def calc_atr(candles, period=14):
    if len(candles) < period + 1: return None
    trs = []
    for i in range(1, len(candles)):
        h, l, pc = candles[i]['high'], candles[i]['low'], candles[i-1]['close']
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs[-period:]) / period


def analyze_swing(symbol, price, change_24h, vol_24h):
    candles_4h = get_klines(symbol, "4h", 100)
    if not candles_4h or len(candles_4h) < 50: return None
    time.sleep(0.05)
    candles_1h = get_klines(symbol, "1h", 100)
    if not candles_1h or len(candles_1h) < 50: return None
    time.sleep(0.05)
    candles_1d = get_klines(symbol, "1d", 60)
    time.sleep(0.05)

    closes_4h = [c['close'] for c in candles_4h]
    closes_1h = [c['close'] for c in candles_1h]
    closes_1d = [c['close'] for c in candles_1d] if candles_1d else []
    cp = closes_4h[-1]

    ema9_4h = calc_ema(closes_4h, 9)
    ema21_4h = calc_ema(closes_4h, 21)
    ema50_4h = calc_ema(closes_4h, 50)
    rsi_4h = calc_rsi(closes_4h, 14)
    macd_4h, macd_sig_4h, macd_hist_4h = calc_macd(closes_4h)
    sma_4h, bb_up_4h, bb_lo_4h = calc_bb(closes_4h)
    atr_4h = calc_atr(candles_4h)
    obv_4h = calc_obv(candles_4h)

    ema9_1h = calc_ema(closes_1h, 9)
    ema21_1h = calc_ema(closes_1h, 21)
    rsi_1h = calc_rsi(closes_1h, 14)
    macd_1h, _, macd_hist_1h = calc_macd(closes_1h)

    ema20_1d = calc_ema(closes_1d, 20) if len(closes_1d) >= 20 else None
    rsi_1d = calc_rsi(closes_1d, 14) if len(closes_1d) >= 15 else 50

    if not ema9_4h or not ema21_4h: return None

    # RSI 비정상 필터
    if rsi_4h >= 95 or rsi_4h <= 5: return None

    bb_width_4h = None
    if sma_4h and bb_up_4h and bb_lo_4h:
        bb_width_4h = (bb_up_4h - bb_lo_4h) / sma_4h * 100

    obv_trend = "NEUTRAL"
    if len(obv_4h) >= 20:
        if obv_4h[-1] > obv_4h[-10] > obv_4h[-20]: obv_trend = "ACCUMULATION"
        elif obv_4h[-1] < obv_4h[-10] < obv_4h[-20]: obv_trend = "DISTRIBUTION"
        elif obv_4h[-1] > obv_4h[-20]: obv_trend = "NET_BUY"
        elif obv_4h[-1] < obv_4h[-20]: obv_trend = "NET_SELL"

    # 지지/저항
    highs = [c['high'] for c in candles_4h[-50:]]
    lows = [c['low'] for c in candles_4h[-50:]]
    supports = sorted([l for l in [min(lows[-20:]), min(lows)] if l < cp], reverse=True)[:2]
    resistances = sorted([h for h in [max(highs[-20:]), max(highs)] if h > cp])[:2]

    long_score = 0; long_reasons = []
    short_score = 0; short_reasons = []

    # 4H EMA
    if ema9_4h > ema21_4h:
        long_score += 2; long_reasons.append("4h EMA9>21")
    else:
        short_score += 2; short_reasons.append("4h EMA9<21")

    if ema50_4h:
        if cp > ema50_4h: long_score += 1; long_reasons.append("4h >EMA50")
        else: short_score += 1; short_reasons.append("4h <EMA50")

    # 4H MACD
    if macd_hist_4h is not None:
        if macd_hist_4h > 0: long_score += 1; long_reasons.append("4h MACD+")
        else: short_score += 1; short_reasons.append("4h MACD-")

    # 4H RSI
    if rsi_4h < 30: long_score += 3; long_reasons.append(f"4h RSI{rsi_4h:.0f} 극과매도")
    elif rsi_4h < 40: long_score += 2; long_reasons.append(f"4h RSI{rsi_4h:.0f}")
    elif rsi_4h < 45: long_score += 1; long_reasons.append(f"4h RSI{rsi_4h:.0f}")
    elif rsi_4h > 70: short_score += 3; short_reasons.append(f"4h RSI{rsi_4h:.0f} 극과매수")
    elif rsi_4h > 60: short_score += 2; short_reasons.append(f"4h RSI{rsi_4h:.0f} 과매수")
    elif rsi_4h > 55: short_score += 1; short_reasons.append(f"4h RSI{rsi_4h:.0f}")

    # BB
    if bb_lo_4h and cp <= bb_lo_4h * 1.005:
        long_score += 2; long_reasons.append("4h BB하단")
    elif bb_up_4h and cp >= bb_up_4h * 0.995:
        short_score += 2; short_reasons.append("4h BB상단")

    # OBV
    if obv_trend == "ACCUMULATION": long_score += 2; long_reasons.append("OBV매집")
    elif obv_trend == "DISTRIBUTION": short_score += 2; short_reasons.append("OBV분산")
    elif obv_trend == "NET_BUY": long_score += 1; long_reasons.append("OBV순매수")
    elif obv_trend == "NET_SELL": short_score += 1; short_reasons.append("OBV순매도")

    # 1H
    if ema9_1h and ema21_1h:
        if ema9_1h > ema21_1h: long_score += 1; long_reasons.append("1h EMA상승")
        else: short_score += 1; short_reasons.append("1h EMA하락")

    if rsi_1h < 35: long_score += 1; long_reasons.append(f"1h RSI{rsi_1h:.0f}")
    elif rsi_1h > 65: short_score += 1; short_reasons.append(f"1h RSI{rsi_1h:.0f}")

    if macd_hist_1h is not None:
        if macd_hist_1h > 0: long_score += 1
        else: short_score += 1

    # 일봉
    if ema20_1d:
        if cp > ema20_1d: long_score += 1; long_reasons.append("일봉>EMA20")
        else: short_score += 1; short_reasons.append("일봉<EMA20")

    if rsi_1d < 35: long_score += 1; long_reasons.append(f"1d RSI{rsi_1d:.0f}")
    elif rsi_1d > 65: short_score += 1; short_reasons.append(f"1d RSI{rsi_1d:.0f}")

    # 지지/저항 근접
    entry_quality = "AVERAGE"
    if supports:
        dist = (cp - supports[0]) / cp * 100
        if dist < 2.0: long_score += 2; long_reasons.append(f"지지근접({dist:.1f}%)"); entry_quality = "GOOD"
    if resistances:
        dist = (resistances[0] - cp) / cp * 100
        if dist < 2.0: short_score += 2; short_reasons.append(f"저항근접({dist:.1f}%)"); entry_quality = "GOOD"

    best_side = None; best_score = 0; best_reasons = []
    if long_score >= 6 and long_score > short_score + 2:
        best_side = "LONG"; best_score = long_score; best_reasons = long_reasons
    elif short_score >= 6 and short_score > long_score + 2:
        best_side = "SHORT"; best_score = short_score; best_reasons = short_reasons
    if not best_side: return None

    # SL/TP ATR 기반
    if atr_4h and atr_4h > 0:
        if best_side == "LONG":
            sl = cp - atr_4h * 2; tp1 = cp + atr_4h * 3; tp2 = cp + atr_4h * 5
        else:
            sl = cp + atr_4h * 2; tp1 = cp - atr_4h * 3; tp2 = cp - atr_4h * 5
    else:
        factor = 0.03
        if best_side == "LONG":
            sl = cp * (1 - factor); tp1 = cp * (1 + factor * 1.5); tp2 = cp * (1 + factor * 2.5)
        else:
            sl = cp * (1 + factor); tp1 = cp * (1 - factor * 1.5); tp2 = cp * (1 - factor * 2.5)

    sl_pct = abs(cp - sl) / cp * 100
    tp1_pct = abs(tp1 - cp) / cp * 100
    tp2_pct = abs(tp2 - cp) / cp * 100

    return {
        'symbol': symbol, 'price': cp, 'side': best_side, 'score': best_score,
        'long_score': long_score, 'short_score': short_score,
        'rsi_4h': round(rsi_4h, 1), 'rsi_1h': round(rsi_1h, 1), 'rsi_1d': round(rsi_1d, 1),
        'bb_width_4h': round(bb_width_4h, 2) if bb_width_4h else None,
        'obv_trend': obv_trend,
        'macd_hist_4h': round(macd_hist_4h, 6) if macd_hist_4h else None,
        'change_24h': round(change_24h, 1), 'vol_24h_m': round(vol_24h / 1e6, 1),
        'supports': [round(s, 6) for s in supports[:2]],
        'resistances': [round(r, 6) for r in resistances[:2]],
        'entry_quality': entry_quality,
        'sl': round(sl, 6), 'tp1': round(tp1, 6), 'tp2': round(tp2, 6),
        'sl_pct': round(sl_pct, 1), 'tp1_pct': round(tp1_pct, 1), 'tp2_pct': round(tp2_pct, 1),
        'atr_4h': round(atr_4h, 6) if atr_4h else None,
        'reasons': best_reasons,
    }


def main():
    now = datetime.now(KST)
    print(f"{'='*65}")
    print(f"  v4 스윙 스캐너 (정상종목) | {now.strftime('%Y-%m-%d %H:%M:%S KST')}")
    print(f"  프레임: 1D + 4H + 1H | 볼륨 >= $50M | 24h변동 <30%")
    print(f"{'='*65}")

    tickers = get_all_futures_tickers()
    print(f"\n  필터 통과: {len(tickers)}개")

    # BTC
    btc_4h = get_klines("BTCUSDT", "4h", 100)
    if btc_4h:
        c = [x['close'] for x in btc_4h]
        e9 = calc_ema(c, 9); e21 = calc_ema(c, 21)
        r = calc_rsi(c, 14)
        _, _, mh = calc_macd(c)
        print(f"  BTC 4H: {'BULL' if e9>e21 else 'BEAR'} | RSI {r:.1f} | MACD hist {mh:.1f}")
        print(f"  EMA9={e9:.0f} EMA21={e21:.0f} 현재={c[-1]:.0f}")

    print(f"\n  스캔 중...")
    signals = []
    for i, t in enumerate(tickers):
        result = analyze_swing(t['symbol'], t['price'], t['change_24h'], t['volume_24h'])
        if result:
            signals.append(result)
        if (i+1) % 20 == 0:
            print(f"    {i+1}/{len(tickers)}")
        time.sleep(0.05)

    signals.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n{'='*65}")
    print(f"  결과: {len(signals)}개 스윙 시그널")
    print(f"{'='*65}\n")

    for rank, s in enumerate(signals[:10], 1):
        sym = s['symbol'].replace('USDT', '')
        bb = f"{s['bb_width_4h']}%" if s['bb_width_4h'] else "-"
        print(f"#{rank} {sym} {s['side']} | 점수 {s['score']} (L:{s['long_score']} S:{s['short_score']})")
        print(f"   가격: {s['price']} | RSI: 4h={s['rsi_4h']} 1h={s['rsi_1h']} 1d={s['rsi_1d']}")
        print(f"   OBV: {s['obv_trend']} | BB폭: {bb} | MACD hist: {s['macd_hist_4h']}")
        print(f"   24h: {s['change_24h']:+.1f}% | 볼륨: ${s['vol_24h_m']:.0f}M")
        print(f"   지지: {s['supports']} | 저항: {s['resistances']}")
        print(f"   근거: {', '.join(s['reasons'])}")
        print(f"   진입: {s['price']} ({s['entry_quality']}) | SL: {s['sl']} (-{s['sl_pct']}%)")
        print(f"   TP1: {s['tp1']} (+{s['tp1_pct']}%) | TP2: {s['tp2']} (+{s['tp2_pct']}%)")
        if s['atr_4h']: print(f"   ATR(4h): {s['atr_4h']}")
        print()

    with open('/Users/eomgyuhyeon/.openclaw/workspace/crypto-paper-trading/v4_swing_result2.json', 'w') as f:
        json.dump({'scan_time': now.isoformat(), 'signals': signals[:15]}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
