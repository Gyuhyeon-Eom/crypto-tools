#!/usr/bin/env python3
"""
v5 Bybit 중기 스캐너 (6-10h 홀드)
=================================
v4 대비 추가:
1. 알트 상대강도 (vs BTC) - BTC보다 강한 알트 감지
2. 펀딩비 - 롱/숏 쏠림 반전 감지
3. 오픈인터레스트 변화 - 포지션 과열 체크
4. 도미넌스 이탈 모드 - BTC 약세 + 알트 강세 시 롱 허용
5. 4h 가중치 강화 (중기 맞춤)
"""

import json
import math
import time
import urllib.request
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))
MIN_QUOTE_VOLUME_24H = 20_000_000
OVERHEATED_LONG_PCT = 15.0
OVERHEATED_SHORT_PCT = -15.0


def api_get(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except:
        return None


# ─── 데이터 수집 ───

def get_all_futures_tickers():
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
        funding = float(d.get('fundingRate', 0)) * 100  # %로 변환
        oi = float(d.get('openInterest', 0))
        if vol >= MIN_QUOTE_VOLUME_24H and price > 0:
            results.append({
                'symbol': sym,
                'price': price,
                'change_24h': change,
                'volume_24h': vol,
                'funding_rate': funding,
                'open_interest': oi,
            })
    results.sort(key=lambda x: x['volume_24h'], reverse=True)
    return results


def get_klines(symbol, interval, limit=50):
    url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={interval}&limit={limit}"
    data = api_get(url)
    if not data or data.get('retCode') != 0 or not data['result']['list']:
        return None
    raw = data['result']['list'][::-1]
    return [{
        'open': float(d[1]), 'high': float(d[2]), 'low': float(d[3]),
        'close': float(d[4]), 'volume': float(d[5]),
        'quote_volume': float(d[6]) if len(d) > 6 else float(d[5]) * float(d[4]),
        'timestamp': int(d[0])
    } for d in raw]


def get_funding_history(symbol, limit=3):
    """최근 펀딩비 히스토리 (추세 판단용)"""
    url = f"https://api.bybit.com/v5/market/funding/history?category=linear&symbol={symbol}&limit={limit}"
    data = api_get(url)
    if not data or data.get('retCode') != 0 or not data['result']['list']:
        return []
    return [float(d['fundingRate']) * 100 for d in data['result']['list']]


def get_btc_dominance_trend():
    """BTC 도미넌스 추세 추정: BTC vs 전체 시장 변동 비교"""
    # CoinGecko global
    data = api_get("https://api.coingecko.com/api/v3/global")
    if not data:
        return None, None
    d = data['data']
    btc_dom = d['market_cap_percentage']['btc']
    mcap_change = d['market_cap_change_percentage_24h_usd']
    return btc_dom, mcap_change


def get_relative_strength(tickers, btc_change):
    """각 종목의 BTC 대비 상대강도 계산"""
    rs = {}
    for t in tickers:
        sym = t['symbol']
        if sym == 'BTCUSDT':
            rs[sym] = 0
        else:
            rs[sym] = t['change_24h'] - btc_change
    return rs


# ─── 지표 계산 ───

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


def calc_macd(prices, fast=12, slow=26, signal=9):
    """MACD - 중기 추세 확인용"""
    ema_fast = calc_ema(prices, fast)
    ema_slow = calc_ema(prices, slow)
    if not ema_fast or not ema_slow:
        return None, None, None
    macd_line = ema_fast - ema_slow
    # 간단 시그널: 최근 값만
    return macd_line, ema_fast, ema_slow


def calc_support_resistance(candles_4h):
    if not candles_4h or len(candles_4h) < 10:
        return None, None
    highs = [c['high'] for c in candles_4h[-20:]]
    lows = [c['low'] for c in candles_4h[-20:]]
    return min(lows), max(highs)


# ─── 패턴 감지 ───

def detect_accumulation(candles_1h, candles_4h):
    if not candles_1h or len(candles_1h) < 20:
        return 0, []
    score = 0
    reasons = []
    closes = [c['close'] for c in candles_1h]

    recent = 6
    prev = 14
    if len(closes) >= recent + prev:
        recent_range = (max(closes[-recent:]) - min(closes[-recent:])) / closes[-recent] * 100
        prev_range = (max(closes[-(recent+prev):-recent]) - min(closes[-(recent+prev):-recent])) / closes[-(recent+prev)] * 100
        recent_vol = sum(c['quote_volume'] for c in candles_1h[-recent:]) / recent
        prev_vol = sum(c['quote_volume'] for c in candles_1h[-(recent+prev):-recent]) / prev

        if prev_vol > 0:
            if recent_range < prev_range * 0.7 and recent_vol > prev_vol * 1.2:
                score += 3
                reasons.append(f"횡보+볼륨증가({recent_vol/prev_vol:.1f}x)")
            elif recent_vol > prev_vol * 1.5:
                score += 1
                reasons.append(f"볼륨증가 {recent_vol/prev_vol:.1f}x")

    obv = calc_obv(candles_1h)
    if len(obv) >= 12:
        if obv[-1] > obv[-6] > obv[-12]:
            score += 2
            reasons.append("1h OBV상승(매집)")
        elif obv[-1] > obv[-12]:
            score += 1
            reasons.append("1h OBV순매수")

    tail_count = 0
    for c in candles_1h[-6:]:
        body = abs(c['close'] - c['open'])
        lower_tail = min(c['open'], c['close']) - c['low']
        if body > 0 and lower_tail > body * 1.5:
            tail_count += 1
    if tail_count >= 3:
        score += 2
        reasons.append(f"1h 하단꼬리{tail_count}회")
    elif tail_count >= 2:
        score += 1
        reasons.append(f"1h 하단꼬리{tail_count}회")

    sma, bb_up, bb_lo = calc_bb(closes)
    if sma and bb_up and bb_lo:
        bb_width = (bb_up - bb_lo) / sma * 100
        if bb_width < 2.0:
            score += 2
            reasons.append(f"1h BB스퀴즈({bb_width:.1f}%)")
        elif bb_width < 3.5:
            score += 1
            reasons.append(f"1h BB수축({bb_width:.1f}%)")

    if candles_4h and len(candles_4h) >= 6:
        obv_4h = calc_obv(candles_4h)
        if len(obv_4h) >= 3:
            if obv_4h[-1] > obv_4h[-3]:
                score += 1
                reasons.append("4h OBV강세")

    return score, reasons


def detect_distribution(candles_1h, candles_4h):
    """분배(매도 집중) 패턴 감지 - SHORT용"""
    if not candles_1h or len(candles_1h) < 20:
        return 0, []
    score = 0
    reasons = []
    closes = [c['close'] for c in candles_1h]

    # 상단 꼬리 (매도 압력)
    upper_tail_count = 0
    for c in candles_1h[-6:]:
        body = abs(c['close'] - c['open'])
        upper_tail = c['high'] - max(c['open'], c['close'])
        if body > 0 and upper_tail > body * 1.5:
            upper_tail_count += 1
    if upper_tail_count >= 3:
        score += 2
        reasons.append(f"1h 상단꼬리{upper_tail_count}회")
    elif upper_tail_count >= 2:
        score += 1
        reasons.append(f"1h 상단꼬리{upper_tail_count}회")

    # OBV 하락 (매도 분배)
    obv = calc_obv(candles_1h)
    if len(obv) >= 12:
        if obv[-1] < obv[-6] < obv[-12]:
            score += 2
            reasons.append("1h OBV하락(분배)")
        elif obv[-1] < obv[-12]:
            score += 1
            reasons.append("1h OBV순매도")

    # 4h OBV
    if candles_4h and len(candles_4h) >= 6:
        obv_4h = calc_obv(candles_4h)
        if len(obv_4h) >= 3:
            if obv_4h[-1] < obv_4h[-3]:
                score += 1
                reasons.append("4h OBV약세")

    return score, reasons


# ─── MTF 추세 ───

def get_trend_mtf(symbol):
    """4h(가중4) / 1h(가중2) / 15m(가중1) - 중기 맞춤 가중치"""
    frames = [("240", 50, 4), ("60", 50, 2), ("15", 50, 1)]
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
            label = {"240": "4h", "60": "1h", "15": "15m"}.get(interval, interval)
            if e9 > e21:
                total_score += weight
                details.append(f"{label}:UP")
            else:
                total_score -= weight
                details.append(f"{label}:DN")
        time.sleep(0.05)

    if total_score >= 6: return "STRONG_BULL", total_score, details
    elif total_score >= 3: return "BULL", total_score, details
    elif total_score <= -6: return "STRONG_BEAR", total_score, details
    elif total_score <= -3: return "BEAR", total_score, details
    return "NEUTRAL", total_score, details


# ─── 펀딩비 분석 ───

def analyze_funding(current_funding, funding_history):
    """
    펀딩비 분석:
    - 양수 높으면 롱 과열 → 숏 유리
    - 음수 높으면 숏 과열 → 롱 유리
    - 추세 변화도 감지
    """
    score = 0  # 양수=숏유리, 음수=롱유리
    reason = None

    if current_funding > 0.03:  # 0.03% 이상 = 롱 과열
        score = 2
        reason = f"펀딩{current_funding:.3f}%(롱과열)"
    elif current_funding > 0.015:
        score = 1
        reason = f"펀딩{current_funding:.3f}%(롱우세)"
    elif current_funding < -0.03:  # 숏 과열
        score = -2
        reason = f"펀딩{current_funding:.3f}%(숏과열)"
    elif current_funding < -0.015:
        score = -1
        reason = f"펀딩{current_funding:.3f}%(숏우세)"

    # 펀딩비 추세 (연속 상승/하락)
    if funding_history and len(funding_history) >= 3:
        if all(funding_history[i] > funding_history[i+1] for i in range(len(funding_history)-1)):
            if score > 0:
                score += 1
                reason = (reason or "") + "+연속상승"
        elif all(funding_history[i] < funding_history[i+1] for i in range(len(funding_history)-1)):
            if score < 0:
                score -= 1
                reason = (reason or "") + "+연속하락"

    return score, reason


# ─── 종합 분석 ───

def analyze_full(symbol, ticker_data, btc_trend, btc_score, btc_change, rel_strength,
                 dominance_falling, dominance_val):
    price = ticker_data['price']
    change_24h = ticker_data['change_24h']
    vol_24h = ticker_data['volume_24h']
    funding_rate = ticker_data['funding_rate']

    candles_1h = get_klines(symbol, "60", 30)
    candles_4h = get_klines(symbol, "240", 30)
    if not candles_1h or len(candles_1h) < 20:
        return None

    closes_1h = [c['close'] for c in candles_1h]
    cp = closes_1h[-1]

    ema9 = calc_ema(closes_1h, 9)
    ema21 = calc_ema(closes_1h, 21)
    if not ema9 or not ema21:
        return None

    r = calc_rsi(closes_1h, 14)
    vol = calc_volume_spike(candles_1h)
    _, bb_up, bb_lo = calc_bb(closes_1h)
    accum_score, accum_reasons = detect_accumulation(candles_1h, candles_4h)
    dist_score, dist_reasons = detect_distribution(candles_1h, candles_4h)
    sym_trend, sym_score, sym_details = get_trend_mtf(symbol)
    support, resistance = calc_support_resistance(candles_4h)

    # MACD (1h)
    macd_val, _, _ = calc_macd(closes_1h)

    # 펀딩비
    funding_hist = get_funding_history(symbol, 3)
    funding_score, funding_reason = analyze_funding(funding_rate, funding_hist)

    # 상대강도
    rs = rel_strength.get(symbol, 0)

    # ─── LONG 점수 ───
    long_score = 0
    long_reasons = []

    # 도미넌스 이탈 모드: BTC 약세여도 알트가 강하면 롱 허용
    btc_blocks_long = False
    alt_dominance_mode = False

    if btc_trend == "STRONG_BEAR":
        # 도미넌스 이탈 체크: BTC 약세인데 이 알트가 BTC보다 강하면 허용
        if rs > 1.5 and dominance_falling:
            alt_dominance_mode = True
            long_reasons.append(f"도미넌스이탈(RS:{rs:+.1f}%)")
        else:
            btc_blocks_long = True
    elif btc_trend == "BEAR":
        if rs > 1.0:
            alt_dominance_mode = True
            long_reasons.append(f"BTC대비강세(RS:{rs:+.1f}%)")

    if change_24h > OVERHEATED_LONG_PCT:
        btc_blocks_long = True

    if not btc_blocks_long:
        # MTF 추세
        if sym_trend == "STRONG_BULL":
            long_score += 3; long_reasons.append(f"MTF강세({'/'.join(sym_details)})")
        elif sym_trend == "BULL":
            long_score += 2; long_reasons.append(f"MTF상승({'/'.join(sym_details)})")

        # 매집 패턴
        if accum_score >= 5:
            long_score += 3; long_reasons.extend(accum_reasons[:2])
        elif accum_score >= 3:
            long_score += 2; long_reasons.extend(accum_reasons[:1])

        # EMA
        if ema9 > ema21:
            long_score += 1; long_reasons.append("1h EMA9>21")

        # RSI
        if r < 30:
            long_score += 2; long_reasons.append(f"1h RSI{r:.0f}")
        elif r < 40:
            long_score += 1; long_reasons.append(f"1h RSI{r:.0f}")

        # 볼륨 스파이크
        if vol > 2.0:
            long_score += 1; long_reasons.append(f"볼륨{vol:.1f}x")

        # BB 하단
        if bb_lo and cp <= bb_lo * 1.005:
            long_score += 1; long_reasons.append("1h BB하단")

        # 4h 지지선
        if support and cp <= support * 1.01:
            long_score += 1; long_reasons.append("4h 지지선")

        # 펀딩비 (숏 과열 = 롱 유리)
        if funding_score <= -1:
            long_score += abs(funding_score)
            if funding_reason:
                long_reasons.append(funding_reason)

        # 상대강도 보너스
        if rs > 2.0:
            long_score += 2; long_reasons.append(f"BTC대비강세({rs:+.1f}%)")
        elif rs > 1.0:
            long_score += 1; long_reasons.append(f"BTC대비선방({rs:+.1f}%)")

        # MACD
        if macd_val and macd_val > 0:
            long_score += 1; long_reasons.append("MACD양전환")

        # BTC 강세 보너스
        if btc_trend == "STRONG_BULL":
            long_score += 1; long_reasons.append("BTC강세")

        # 도미넌스 이탈 보너스
        if alt_dominance_mode:
            long_score += 1

    # ─── SHORT 점수 ───
    short_score = 0
    short_reasons = []

    btc_blocks_short = False
    if btc_trend == "STRONG_BULL":
        btc_blocks_short = True
    if change_24h < OVERHEATED_SHORT_PCT:
        btc_blocks_short = True

    if not btc_blocks_short:
        if sym_trend == "STRONG_BEAR":
            short_score += 3; short_reasons.append(f"MTF약세({'/'.join(sym_details)})")
        elif sym_trend == "BEAR":
            short_score += 2; short_reasons.append(f"MTF하락({'/'.join(sym_details)})")

        # 분배 패턴
        if dist_score >= 3:
            short_score += 2; short_reasons.extend(dist_reasons[:2])
        elif dist_score >= 2:
            short_score += 1; short_reasons.extend(dist_reasons[:1])

        if ema9 < ema21:
            short_score += 1; short_reasons.append("1h EMA9<21")

        if r > 70:
            short_score += 2; short_reasons.append(f"1h RSI{r:.0f}")
        elif r > 60:
            short_score += 1; short_reasons.append(f"1h RSI{r:.0f}")

        if vol > 2.0:
            short_score += 1; short_reasons.append(f"볼륨{vol:.1f}x")

        if bb_up and cp >= bb_up * 0.995:
            short_score += 1; short_reasons.append("1h BB상단")

        if resistance and cp >= resistance * 0.99:
            short_score += 1; short_reasons.append("4h 저항선")

        # 펀딩비 (롱 과열 = 숏 유리)
        if funding_score >= 1:
            short_score += funding_score
            if funding_reason:
                short_reasons.append(funding_reason)

        # 상대약세
        if rs < -2.0:
            short_score += 2; short_reasons.append(f"BTC대비약세({rs:+.1f}%)")
        elif rs < -1.0:
            short_score += 1; short_reasons.append(f"BTC대비부진({rs:+.1f}%)")

        # MACD
        if macd_val and macd_val < 0:
            short_score += 1; short_reasons.append("MACD음전환")

        if btc_trend == "STRONG_BEAR":
            short_score += 1; short_reasons.append("BTC약세")

    # ─── 최종 판정 ───
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
        'rsi_1h': round(r, 1),
        'ema9': ema9,
        'ema21': ema21,
        'vol_spike': round(vol, 1),
        'accum_score': accum_score,
        'accum_reasons': accum_reasons,
        'dist_score': dist_score,
        'dist_reasons': dist_reasons,
        'change_24h': round(change_24h, 1),
        'vol_24h_m': round(vol_24h / 1e6, 1),
        'sym_trend': sym_trend,
        'sym_score': sym_score,
        'reasons': best_reasons,
        'support': support,
        'resistance': resistance,
        'funding_rate': round(funding_rate, 4),
        'funding_score': funding_score,
        'rel_strength': round(rs, 2),
        'macd': round(macd_val, 6) if macd_val else None,
        'alt_dom_mode': alt_dominance_mode if best_side == 'LONG' else False,
    }


def main():
    now = datetime.now(KST)
    print(f"{'='*70}")
    print(f"  v5 Bybit 중기 스캐너 (6-10h) | {now.strftime('%Y-%m-%d %H:%M:%S KST')}")
    print(f"  MTF: 4h(x4)/1h(x2)/15m(x1) | 펀딩비+상대강도+도미넌스")
    print(f"{'='*70}")

    # 1) 전 종목 조회
    print("\n[1] Bybit USDT 무기한 선물 전 종목 조회...")
    tickers = get_all_futures_tickers()
    print(f"    볼륨 필터 통과: {len(tickers)}개")

    # BTC 변동
    btc_change = 0
    for t in tickers:
        if t['symbol'] == 'BTCUSDT':
            btc_change = t['change_24h']
            break
    print(f"    BTC 24h: {btc_change:+.2f}%")

    # 2) BTC 도미넌스
    print("\n[2] BTC 도미넌스 확인...")
    btc_dom, mcap_change = get_btc_dominance_trend()
    dominance_falling = False
    if btc_dom:
        # BTC가 시장보다 더 빠지면 도미넌스 하락 추정
        dominance_falling = btc_change < (mcap_change if mcap_change else 0)
        dom_status = "하락추정" if dominance_falling else "유지/상승"
        print(f"    BTC 도미넌스: {btc_dom:.2f}% | 시총변동: {mcap_change:.2f}% | 도미넌스: {dom_status}")
    else:
        print("    도미넌스 데이터 없음 (API 제한)")

    # 3) BTC MTF
    print("\n[3] BTC 멀티타임프레임 추세 (4h/1h/15m)...")
    btc_trend, btc_score, btc_details = get_trend_mtf("BTCUSDT")
    print(f"    BTC: {btc_trend} (score {btc_score}) | {' / '.join(btc_details)}")

    # 4) 상대강도
    rel_strength = get_relative_strength(tickers, btc_change)

    # 5) 전 종목 스캔
    print(f"\n[4] {len(tickers)}개 종목 시그널 스캔 중...")
    signals = []
    ticker_map = {t['symbol']: t for t in tickers}
    for i, t in enumerate(tickers):
        sym = t['symbol']
        result = analyze_full(
            sym, t, btc_trend, btc_score, btc_change,
            rel_strength, dominance_falling, btc_dom
        )
        if result:
            signals.append(result)
        if (i + 1) % 10 == 0:
            print(f"    ... {i+1}/{len(tickers)} 완료")
        time.sleep(0.15)

    # LONG/SHORT 분리 후 각각 정렬
    longs = [s for s in signals if s['side'] == 'LONG']
    shorts = [s for s in signals if s['side'] == 'SHORT']
    longs.sort(key=lambda x: (x['score'], x['accum_score'], x['rel_strength']), reverse=True)
    shorts.sort(key=lambda x: (x['score'], x['dist_score'], -x['rel_strength']), reverse=True)

    # 합치기 (점수 순)
    all_signals = sorted(signals, key=lambda x: x['score'], reverse=True)

    print(f"\n{'='*70}")
    print(f"  스캔 결과: {len(signals)}개 시그널 (LONG {len(longs)} / SHORT {len(shorts)})")
    print(f"  BTC: {btc_trend} (score {btc_score}) | 도미넌스: {btc_dom:.1f}% {'(하락중)' if dominance_falling else ''}")
    print(f"{'='*70}")

    if not signals:
        print("\n  시그널 없음 - 진입 대기")
    else:
        # LONG 시그널
        if longs:
            print(f"\n  === LONG 시그널 ({len(longs)}개) ===")
            print(f"{'─'*80}")
            print(f"  {'순위':>3} | {'종목':<10} | {'점수':>4} | {'매집':>4} | {'1hRSI':>5} | {'RS':>6} | {'펀딩':>7} | {'24h':>7} | {'볼륨M$':>6} | 근거")
            print(f"{'─'*80}")
            for rank, s in enumerate(longs[:10], 1):
                sym_name = s['symbol'].replace('USDT', '')
                reasons_str = ', '.join(s['reasons'][:3])
                dom_tag = " [DOM]" if s.get('alt_dom_mode') else ""
                print(f"  {rank:>3} | {sym_name:<10} | {s['score']:>4} | {s['accum_score']:>4} | {s['rsi_1h']:>5} | {s['rel_strength']:>+5.1f}% | {s['funding_rate']:>+6.3f}% | {s['change_24h']:>+6.1f}% | {s['vol_24h_m']:>5.0f} | {reasons_str}{dom_tag}")

        # SHORT 시그널
        if shorts:
            print(f"\n  === SHORT 시그널 ({len(shorts)}개) ===")
            print(f"{'─'*80}")
            print(f"  {'순위':>3} | {'종목':<10} | {'점수':>4} | {'분배':>4} | {'1hRSI':>5} | {'RS':>6} | {'펀딩':>7} | {'24h':>7} | {'볼륨M$':>6} | 근거")
            print(f"{'─'*80}")
            for rank, s in enumerate(shorts[:10], 1):
                sym_name = s['symbol'].replace('USDT', '')
                reasons_str = ', '.join(s['reasons'][:3])
                print(f"  {rank:>3} | {sym_name:<10} | {s['score']:>4} | {s['dist_score']:>4} | {s['rsi_1h']:>5} | {s['rel_strength']:>+5.1f}% | {s['funding_rate']:>+6.3f}% | {s['change_24h']:>+6.1f}% | {s['vol_24h_m']:>5.0f} | {reasons_str}")

        # TOP 5 상세
        print(f"\n{'='*70}")
        print("  TOP 5 상세 분석 (점수 순)")
        print(f"{'='*70}")
        for rank, s in enumerate(all_signals[:5], 1):
            sym_name = s['symbol'].replace('USDT', '')
            dom_tag = " [도미넌스이탈]" if s.get('alt_dom_mode') else ""
            print(f"\n  #{rank} {sym_name} {s['side']}{dom_tag} (시그널 {s['score']}, 매집 {s['accum_score']}, 분배 {s['dist_score']})")
            print(f"     가격: {s['price']} | 1h RSI: {s['rsi_1h']} | 추세: {s['sym_trend']}({s['sym_score']})")
            print(f"     24h: {s['change_24h']:+.1f}% | 볼륨: ${s['vol_24h_m']:.0f}M | 볼스파이크: {s['vol_spike']}x")
            print(f"     펀딩비: {s['funding_rate']:+.4f}% | 상대강도: {s['rel_strength']:+.2f}% vs BTC")
            if s.get('macd') is not None:
                print(f"     MACD: {s['macd']:.6f}")
            if s['support'] and s['resistance']:
                print(f"     4h 지지: {s['support']:.6g} | 4h 저항: {s['resistance']:.6g}")
            print(f"     근거: {', '.join(s['reasons'])}")
            if s['accum_reasons']:
                print(f"     매집: {', '.join(s['accum_reasons'])}")
            if s['dist_reasons']:
                print(f"     분배: {', '.join(s['dist_reasons'])}")

            ep = s['price']
            if s['side'] == 'LONG':
                sl = ep * (1 - 2.0 / 100)
                tp = ep * (1 + 5.0 / 100)
                print(f"     10x -> 진입: {ep} | SL: {sl:.6g} (-2%) | TP: {tp:.6g} (+5%)")
            else:
                sl = ep * (1 + 2.0 / 100)
                tp = ep * (1 - 5.0 / 100)
                print(f"     10x -> 진입: {ep} | SL: {sl:.6g} (-2%) | TP: {tp:.6g} (+5%)")

    output = {
        'scan_time': now.isoformat(),
        'version': 'v5',
        'btc_trend': btc_trend,
        'btc_score': btc_score,
        'btc_change_24h': btc_change,
        'btc_dominance': btc_dom,
        'dominance_falling': dominance_falling,
        'total_scanned': len(tickers),
        'signals_found': len(signals),
        'long_count': len(longs),
        'short_count': len(shorts),
        'top_longs': longs[:10],
        'top_shorts': shorts[:10],
    }
    with open('/Users/eomgyuhyeon/.openclaw/workspace/crypto-paper-trading/v5_scan_result.json', 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n결과 저장: v5_scan_result.json")


if __name__ == "__main__":
    main()
