#!/usr/bin/env python3
"""
v4 로직 기반 실시간 스캐너 - Bybit 선물 버전
"""

import json
import math
import time
import urllib.request
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))
MIN_QUOTE_VOLUME_24H = 20_000_000  # Bybit은 Binance보다 볼륨 작으므로 $20M
OVERHEATED_LONG_PCT = 10.0
OVERHEATED_SHORT_PCT = -10.0


def api_get(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return None


def get_all_futures_tickers():
    """Bybit USDT 무기한 선물 전 종목"""
    data = api_get("https://api.bybit.com/v5/market/tickers?category=linear")
    if not data or data.get('retCode') != 0:
        return []
    results = []
    for d in data['result']['list']:
        sym = d['symbol']
        if not sym.endswith('USDT'):
            continue
        vol = float(d.get('turnover24h', 0))
        change = float(d.get('price24hPcnt', 0)) * 100
        price = float(d.get('lastPrice', 0))
        if vol >= MIN_QUOTE_VOLUME_24H and price > 0:
            results.append({
                'symbol': sym,
                'price': price,
                'change_24h': change,
                'volume_24h': vol,
            })
    results.sort(key=lambda x: x['volume_24h'], reverse=True)
    return results


def get_klines(symbol, interval="5", limit=50):
    """Bybit kline. interval: 1,3,5,15,30,60,120,240,360,720,D,W,M"""
    url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={interval}&limit={limit}"
    data = api_get(url)
    if not data or data.get('retCode') != 0 or not data['result']['list']:
        return None
    # Bybit returns newest first, reverse for chronological
    raw = data['result']['list'][::-1]
    return [{
        'open': float(d[1]), 'high': float(d[2]), 'low': float(d[3]),
        'close': float(d[4]), 'volume': float(d[5]),
        'quote_volume': float(d[6]) if len(d) > 6 else float(d[5]) * float(d[4]),
        'timestamp': int(d[0])
    } for d in raw]


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


def calc_obv(candles):
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


def detect_accumulation(candles_5m, candles_1h):
    if not candles_5m or len(candles_5m) < 30:
        return 0, []
    score = 0
    reasons = []
    closes = [c['close'] for c in candles_5m]

    recent_range = (max(closes[-10:]) - min(closes[-10:])) / closes[-10] * 100
    prev_range = (max(closes[-30:-10]) - min(closes[-30:-10])) / closes[-30] * 100 if len(closes) >= 30 else recent_range
    recent_vol = sum(c['quote_volume'] for c in candles_5m[-10:]) / 10
    prev_vol = sum(c['quote_volume'] for c in candles_5m[-30:-10]) / 20 if len(candles_5m) >= 30 else recent_vol

    if recent_range < prev_range * 0.7 and recent_vol > prev_vol * 1.3:
        score += 3
        reasons.append(f"횡보+볼륨증가({recent_vol/prev_vol:.1f}x)")
    elif recent_vol > prev_vol * 1.5:
        score += 1
        reasons.append(f"볼륨증가 {recent_vol/prev_vol:.1f}x")

    obv = calc_obv(candles_5m)
    if len(obv) >= 20:
        if obv[-1] > obv[-10] > obv[-20]:
            score += 2
            reasons.append("OBV상승(매집)")
        elif obv[-1] > obv[-20]:
            score += 1
            reasons.append("OBV순매수")

    tail_count = 0
    for c in candles_5m[-10:]:
        body = abs(c['close'] - c['open'])
        lower_tail = min(c['open'], c['close']) - c['low']
        if body > 0 and lower_tail > body * 1.5:
            tail_count += 1
    if tail_count >= 3:
        score += 2
        reasons.append(f"하단꼬리{tail_count}회")
    elif tail_count >= 2:
        score += 1
        reasons.append(f"하단꼬리{tail_count}회")

    sma, bb_up, bb_lo = calc_bb(closes)
    if sma and bb_up and bb_lo:
        bb_width = (bb_up - bb_lo) / sma * 100
        if bb_width < 1.5:
            score += 2
            reasons.append(f"BB스퀴즈({bb_width:.1f}%)")
        elif bb_width < 2.5:
            score += 1
            reasons.append(f"BB수축({bb_width:.1f}%)")

    if candles_1h and len(candles_1h) >= 10:
        obv_1h = calc_obv(candles_1h)
        if len(obv_1h) >= 5:
            if obv_1h[-1] > obv_1h[-5]:
                score += 1
                reasons.append("1h OBV강세")

    return score, reasons


def get_trend_mtf(symbol):
    # Bybit intervals: 60=1h, 15=15m, 5=5m
    frames = [("60", 50, 3), ("15", 50, 2), ("5", 50, 1)]
    total_score = 0
    details = []
    for interval, limit, weight in frames:
        candles = get_klines(symbol, interval, limit)
        if not candles or len(candles) < 25:
            continue
        closes = [c['close'] for c in candles]
        e9 = calc_ema(closes, 9)
        e21 = calc_ema(closes, 21)
        if e9 and e21:
            label = {"60": "1h", "15": "15m", "5": "5m"}.get(interval, interval)
            if e9 > e21:
                total_score += weight
                details.append(f"{label}:UP")
            else:
                total_score -= weight
                details.append(f"{label}:DN")
        time.sleep(0.05)

    if total_score >= 5: return "STRONG_BULL", total_score, details
    elif total_score >= 2: return "BULL", total_score, details
    elif total_score <= -5: return "STRONG_BEAR", total_score, details
    elif total_score <= -2: return "BEAR", total_score, details
    return "NEUTRAL", total_score, details


def analyze_full(symbol, price, change_24h, vol_24h, btc_trend, btc_score):
    candles_5m = get_klines(symbol, "5", 50)
    candles_1h = get_klines(symbol, "60", 30)
    if not candles_5m or len(candles_5m) < 30:
        return None

    closes = [c['close'] for c in candles_5m]
    cp = closes[-1]
    ema9 = calc_ema(closes, 9)
    ema21 = calc_ema(closes, 21)
    if not ema9 or not ema21:
        return None

    r = calc_rsi(closes, 14)
    vol = calc_volume_spike(candles_5m)
    _, bb_up, bb_lo = calc_bb(closes)
    accum_score, accum_reasons = detect_accumulation(candles_5m, candles_1h)
    sym_trend, sym_score, sym_details = get_trend_mtf(symbol)

    # LONG
    long_score = 0
    long_reasons = []
    if btc_trend in ("BEAR", "STRONG_BEAR"):
        long_score = -99
    elif change_24h > OVERHEATED_LONG_PCT:
        long_score = -99
    else:
        if sym_trend == "STRONG_BULL":
            long_score += 3; long_reasons.append(f"MTF강세({'/'.join(sym_details)})")
        elif sym_trend == "BULL":
            long_score += 2; long_reasons.append(f"MTF상승({'/'.join(sym_details)})")
        if accum_score >= 5:
            long_score += 3; long_reasons.extend(accum_reasons[:2])
        elif accum_score >= 3:
            long_score += 2; long_reasons.extend(accum_reasons[:1])
        if ema9 > ema21:
            long_score += 1; long_reasons.append("EMA9>21")
        if r < 30:
            long_score += 2; long_reasons.append(f"RSI{r:.0f}")
        elif r < 40:
            long_score += 1; long_reasons.append(f"RSI{r:.0f}")
        if vol > 2.5:
            long_score += 1; long_reasons.append(f"볼륨{vol:.1f}x")
        if bb_lo and cp <= bb_lo * 1.003:
            long_score += 1; long_reasons.append("BB하단")
        if btc_trend == "STRONG_BULL":
            long_score += 1; long_reasons.append("BTC강세")

    # SHORT
    short_score = 0
    short_reasons = []
    if btc_trend in ("BULL", "STRONG_BULL"):
        short_score = -99
    elif change_24h < OVERHEATED_SHORT_PCT:
        short_score = -99
    else:
        if sym_trend == "STRONG_BEAR":
            short_score += 3; short_reasons.append(f"MTF약세({'/'.join(sym_details)})")
        elif sym_trend == "BEAR":
            short_score += 2; short_reasons.append(f"MTF하락({'/'.join(sym_details)})")
        if ema9 < ema21:
            short_score += 1; short_reasons.append("EMA9<21")
        if r > 70:
            short_score += 2; short_reasons.append(f"RSI{r:.0f}")
        elif r > 60:
            short_score += 1; short_reasons.append(f"RSI{r:.0f}")
        if vol > 2.5:
            short_score += 1; short_reasons.append(f"볼륨{vol:.1f}x")
        if bb_up and cp >= bb_up * 0.997:
            short_score += 1; short_reasons.append("BB상단")
        if btc_trend == "STRONG_BEAR":
            short_score += 1; short_reasons.append("BTC약세")

    best_side = None
    best_score = 0
    best_reasons = []

    if long_score >= 4 and long_score > short_score:
        best_side = "LONG"
        best_score = long_score
        best_reasons = long_reasons
    elif short_score >= 4 and short_score > long_score:
        best_side = "SHORT"
        best_score = short_score
        best_reasons = short_reasons

    if not best_side:
        return None

    return {
        'symbol': symbol,
        'price': cp,
        'side': best_side,
        'score': best_score,
        'rsi': round(r, 1),
        'ema9': ema9,
        'ema21': ema21,
        'vol_spike': round(vol, 1),
        'accum_score': accum_score,
        'accum_reasons': accum_reasons,
        'change_24h': round(change_24h, 1),
        'vol_24h_m': round(vol_24h / 1e6, 1),
        'sym_trend': sym_trend,
        'sym_score': sym_score,
        'reasons': best_reasons,
    }


def main():
    now = datetime.now(KST)
    print(f"{'='*60}")
    print(f"  v4 Bybit 전종목 스캐너 | {now.strftime('%Y-%m-%d %H:%M:%S KST')}")
    print(f"  볼륨 필터: 24h >= ${MIN_QUOTE_VOLUME_24H/1e6:.0f}M")
    print(f"{'='*60}")

    print("\n[1] Bybit USDT 무기한 선물 전 종목 조회 중...")
    tickers = get_all_futures_tickers()
    print(f"    볼륨 필터 통과: {len(tickers)}개")

    print("\n[2] BTC 멀티타임프레임 추세 분석...")
    btc_trend, btc_score, btc_details = get_trend_mtf("BTCUSDT")
    print(f"    BTC: {btc_trend} (score {btc_score}) | {' / '.join(btc_details)}")

    print(f"\n[3] {len(tickers)}개 종목 시그널 스캔 중...")
    signals = []
    for i, t in enumerate(tickers):
        sym = t['symbol']
        if sym == 'BTCUSDT':
            pass
        result = analyze_full(sym, t['price'], t['change_24h'], t['volume_24h'], btc_trend, btc_score)
        if result:
            signals.append(result)
        if (i + 1) % 20 == 0:
            print(f"    ... {i+1}/{len(tickers)} 완료")
        time.sleep(0.12)

    signals.sort(key=lambda x: (x['accum_score'], x['score']), reverse=True)

    print(f"\n{'='*60}")
    print(f"  스캔 결과: {len(signals)}개 시그널 발견")
    print(f"  BTC 추세: {btc_trend} (score {btc_score})")
    print(f"{'='*60}")

    if not signals:
        print("\n  시그널 없음 - 진입 대기")
    else:
        print(f"\n{'─'*60}")
        print(f"  {'순위':>3} | {'종목':<12} | {'포지션':<6} | {'점수':>4} | {'매집':>4} | {'RSI':>5} | {'24h':>7} | {'볼륨(M$)':>8} | 근거")
        print(f"{'─'*60}")
        for rank, s in enumerate(signals[:20], 1):
            sym_name = s['symbol'].replace('USDT', '')
            reasons_str = ', '.join(s['reasons'][:3])
            print(f"  {rank:>3} | {sym_name:<12} | {s['side']:<6} | {s['score']:>4} | {s['accum_score']:>4} | {s['rsi']:>5} | {s['change_24h']:>+6.1f}% | {s['vol_24h_m']:>7.0f} | {reasons_str}")

        print(f"\n{'='*60}")
        print("  TOP 5 상세 분석")
        print(f"{'='*60}")
        for rank, s in enumerate(signals[:5], 1):
            sym_name = s['symbol'].replace('USDT', '')
            print(f"\n  #{rank} {sym_name} {s['side']} (점수 {s['score']}, 매집 {s['accum_score']})")
            print(f"     가격: {s['price']} | RSI: {s['rsi']} | 추세: {s['sym_trend']}({s['sym_score']})")
            print(f"     24h: {s['change_24h']:+.1f}% | 볼륨: ${s['vol_24h_m']:.0f}M | 볼스파이크: {s['vol_spike']}x")
            print(f"     근거: {', '.join(s['reasons'])}")
            if s['accum_reasons']:
                print(f"     매집: {', '.join(s['accum_reasons'])}")
            ep = s['price']
            if s['side'] == 'LONG':
                sl = ep * (1 - 1.5 / 100)
                tp = ep * (1 + 3.0 / 100)
            else:
                sl = ep * (1 + 1.5 / 100)
                tp = ep * (1 - 3.0 / 100)
            print(f"     10x 기준 -> 진입: {ep} | SL: {sl:.6g} (-1.5%) | TP: {tp:.6g} (+3%)")

    output = {
        'scan_time': now.isoformat(),
        'btc_trend': btc_trend,
        'btc_score': btc_score,
        'total_scanned': len(tickers),
        'signals_found': len(signals),
        'top_signals': signals[:20],
    }
    with open('/Users/eomgyuhyeon/.openclaw/workspace/crypto-paper-trading/v4_scan_result_bybit.json', 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: v4_scan_result_bybit.json")


if __name__ == "__main__":
    main()
