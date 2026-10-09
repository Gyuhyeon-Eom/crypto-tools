#!/usr/bin/env python3
"""
급등(100%+) 전조 스캐너 v2 - 소형캡 집중
==========================================
시총 큰 코인(TRX, ADA, BCH 등) 100% 급등? 안 한다.
100-200% 급등은 소형/마이크로캡에서 발생.

필터: OI $500K ~ $30M (소형~중소형만)
"""

import json
import math
import time
import urllib.request
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))

# 소형캡 필터
MIN_VOLUME_24H = 1_000_000      # 최소 $1M (너무 작으면 유동성 위험)
MAX_OI_VALUE = 30_000_000       # OI $30M 이하 (소형)
MIN_OI_VALUE = 300_000          # OI $300K 이상 (최소 유동성)


def api_get(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except:
        return None


def get_small_cap_tickers():
    data = api_get("https://api.bybit.com/v5/market/tickers?category=linear")
    if not data or data.get('retCode') != 0:
        return []
    results = []
    for d in data['result']['list']:
        sym = d['symbol']
        if not sym.endswith('USDT'):
            continue
        vol = float(d.get('turnover24h', 0))
        price = float(d.get('lastPrice', 0))
        oi = float(d.get('openInterest', 0))
        oi_val = oi * price
        chg = float(d.get('price24hPcnt', 0)) * 100
        funding = float(d.get('fundingRate', 0)) * 100

        # 스테이블코인 제외
        EXCLUDE = {'USDCUSDT','USDTUSDT','DAIUSDT','TUSDUSDT','BUSDUSDT','FDUSDUSDT','EURUSDT','PAXGUSDT','XAUTUSDT'}
        if sym in EXCLUDE:
            continue

        if (vol >= MIN_VOLUME_24H and price > 0
            and MIN_OI_VALUE <= oi_val <= MAX_OI_VALUE):
            results.append({
                'symbol': sym,
                'price': price,
                'change_24h': chg,
                'volume_24h': vol,
                'funding_rate': funding,
                'oi_value': oi_val,
                'oi_raw': oi,
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


def get_funding_history(symbol, limit=8):
    url = f"https://api.bybit.com/v5/market/funding/history?category=linear&symbol={symbol}&limit={limit}"
    data = api_get(url)
    if not data or data.get('retCode') != 0 or not data['result']['list']:
        return []
    return [float(d['fundingRate']) * 100 for d in data['result']['list']]


def analyze_pump_signals(symbol, ticker):
    score = 0
    signals = []
    details = {}

    price = ticker['price']
    vol_24h = ticker['volume_24h']
    funding = ticker['funding_rate']
    oi_value = ticker['oi_value']

    # 일봉
    candles_1d = get_klines(symbol, "D", 30)
    if not candles_1d or len(candles_1d) < 10:
        return None
    closes_1d = [c['close'] for c in candles_1d]
    vols_1d = [c['quote_volume'] for c in candles_1d]

    # 4h봉
    candles_4h = get_klines(symbol, "240", 50)
    if not candles_4h or len(candles_4h) < 15:
        return None
    closes_4h = [c['close'] for c in candles_4h]
    vols_4h = [c['quote_volume'] for c in candles_4h]

    # 1h봉
    candles_1h = get_klines(symbol, "60", 48)

    # ── 1) 볼륨 선행 폭발 ──
    if len(vols_1d) >= 8:
        avg_vol_7d = sum(vols_1d[-8:-1]) / 7
        today_vol = vols_1d[-1]
        price_chg_3d = abs(closes_1d[-1] - closes_1d[-4]) / closes_1d[-4] * 100 if len(closes_1d) >= 4 else 99

        if avg_vol_7d > 0:
            vol_ratio = today_vol / avg_vol_7d
            details['vol_ratio_1d'] = round(vol_ratio, 1)
            if vol_ratio >= 3.0 and price_chg_3d < 5:
                score += 5; signals.append(f"볼륨폭발({vol_ratio:.1f}x)+가격정체")
            elif vol_ratio >= 2.0 and price_chg_3d < 5:
                score += 3; signals.append(f"볼륨증가({vol_ratio:.1f}x)+가격정체")
            elif vol_ratio >= 1.5 and price_chg_3d < 3:
                score += 1; signals.append(f"볼륨소폭증가({vol_ratio:.1f}x)")

    if len(vols_4h) >= 15:
        avg_vol_4h = sum(vols_4h[-13:-1]) / 12
        recent_vol_4h = sum(vols_4h[-3:]) / 3
        if avg_vol_4h > 0:
            vol_ratio_4h = recent_vol_4h / avg_vol_4h
            if vol_ratio_4h >= 3.0:
                score += 3; signals.append(f"4h볼륨폭발({vol_ratio_4h:.1f}x)")
            elif vol_ratio_4h >= 2.0:
                score += 2; signals.append(f"4h볼륨급증({vol_ratio_4h:.1f}x)")

    # ── 2) BB 극단적 수축 ──
    sma, bb_up, bb_lo = calc_bb(closes_1d)
    if sma and bb_up and bb_lo and sma > 0:
        bb_width = (bb_up - bb_lo) / sma * 100
        details['bb_width_1d'] = round(bb_width, 2)
        if bb_width < 5:
            score += 4; signals.append(f"1d BB극단수축({bb_width:.1f}%)")
        elif bb_width < 8:
            score += 2; signals.append(f"1d BB수축({bb_width:.1f}%)")
        elif bb_width < 12:
            score += 1; signals.append(f"1d BB좁음({bb_width:.1f}%)")

    sma4, bb_up4, bb_lo4 = calc_bb(closes_4h)
    if sma4 and bb_up4 and bb_lo4 and sma4 > 0:
        bb_width_4h = (bb_up4 - bb_lo4) / sma4 * 100
        details['bb_width_4h'] = round(bb_width_4h, 2)
        if bb_width_4h < 3:
            score += 3; signals.append(f"4h BB극단수축({bb_width_4h:.1f}%)")
        elif bb_width_4h < 5:
            score += 1; signals.append(f"4h BB수축({bb_width_4h:.1f}%)")

    # ── 3) OBV 다이버전스 ──
    obv_1d = calc_obv(candles_1d)
    if len(obv_1d) >= 10:
        price_chg_10d = (closes_1d[-1] - closes_1d[-10]) / closes_1d[-10] * 100
        obv_trend = obv_1d[-1] - obv_1d[-10]
        if price_chg_10d <= 0 and obv_trend > 0:
            score += 4; signals.append(f"OBV다이버전스(가격{price_chg_10d:+.1f}%/OBV상승)")
        elif price_chg_10d < 3 and obv_trend > 0:
            score += 2; signals.append("OBV순매집")

    obv_4h = calc_obv(candles_4h)
    if len(obv_4h) >= 20:
        price_chg_4h = (closes_4h[-1] - closes_4h[-20]) / closes_4h[-20] * 100
        if price_chg_4h <= 0 and obv_4h[-1] > obv_4h[-20]:
            score += 2; signals.append("4h OBV다이버전스")

    # ── 4) 거래량/OI 비율 ──
    if oi_value > 0:
        vol_oi_ratio = vol_24h / oi_value
        details['vol_oi_ratio'] = round(vol_oi_ratio, 2)
        if vol_oi_ratio >= 5.0:
            score += 4; signals.append(f"거래량/OI극단({vol_oi_ratio:.1f}x)")
        elif vol_oi_ratio >= 3.0:
            score += 2; signals.append(f"거래량/OI높음({vol_oi_ratio:.1f}x)")
        elif vol_oi_ratio >= 2.0:
            score += 1; signals.append(f"거래량/OI관심({vol_oi_ratio:.1f}x)")

    # ── 5) 펀딩비 + RSI ──
    rsi_1d = calc_rsi(closes_1d)
    rsi_4h = calc_rsi(closes_4h)
    details['rsi_1d'] = round(rsi_1d, 1)
    details['rsi_4h'] = round(rsi_4h, 1)

    funding_hist = get_funding_history(symbol, 8)
    neg_count = sum(1 for f in funding_hist if f < 0)

    if funding < -0.05 and rsi_4h < 35:
        score += 5; signals.append(f"숏스퀴즈세팅(펀딩{funding:.3f}%+RSI{rsi_4h:.0f})")
    elif funding < -0.03:
        score += 3; signals.append(f"강한숏과열(펀딩{funding:.3f}%)")
    elif funding < -0.01:
        score += 2; signals.append(f"펀딩마이너스({funding:.3f}%)")
    if neg_count >= 6:
        score += 2; signals.append(f"펀딩연속마이너스({neg_count}/8)")
    elif neg_count >= 4:
        score += 1; signals.append(f"펀딩마이너스빈발({neg_count}/8)")

    # ── 6) 대폭락 후 바닥 ──
    if len(candles_1d) >= 20:
        high_30d = max(c['high'] for c in candles_1d)
        drawdown = (price - high_30d) / high_30d * 100
        recent_range = (max(closes_1d[-5:]) - min(closes_1d[-5:])) / min(closes_1d[-5:]) * 100 if len(closes_1d) >= 5 else 99
        details['drawdown_30d'] = round(drawdown, 1)

        if drawdown < -60 and recent_range < 15:
            score += 4; signals.append(f"대폭락후바닥({drawdown:.0f}%)")
        elif drawdown < -40 and recent_range < 12:
            score += 3; signals.append(f"폭락후횡보({drawdown:.0f}%)")
        elif drawdown < -25 and recent_range < 10:
            score += 1; signals.append(f"하락후안정({drawdown:.0f}%)")

    # ── 7) 하단 꼬리 반복 ──
    if candles_1d:
        tail_count = 0
        for c in candles_1d[-7:]:
            body = abs(c['close'] - c['open'])
            lower_tail = min(c['open'], c['close']) - c['low']
            if body > 0 and lower_tail > body * 2:
                tail_count += 1
        if tail_count >= 4:
            score += 3; signals.append(f"강한매수방어({tail_count}/7일)")
        elif tail_count >= 3:
            score += 2; signals.append(f"매수방어({tail_count}/7일)")
        elif tail_count >= 2:
            score += 1; signals.append(f"하단꼬리({tail_count}/7일)")

    # ── 8) EMA 수렴/골든크로스 ──
    e9 = calc_ema(closes_1d, 9)
    e21 = calc_ema(closes_1d, 21)
    if e9 and e21 and e21 > 0:
        ema_gap = abs(e9 - e21) / e21 * 100
        if ema_gap < 0.5:
            score += 2; signals.append(f"EMA수렴({ema_gap:.2f}%)")
        elif ema_gap < 1.5:
            score += 1; signals.append(f"EMA근접({ema_gap:.2f}%)")
        if e9 > e21 and ema_gap < 2:
            score += 1; signals.append("골든크로스")

    # ── 9) 1h 단기 매집 ──
    if candles_1h and len(candles_1h) >= 20:
        obv_1h = calc_obv(candles_1h)
        if len(obv_1h) >= 12:
            if obv_1h[-1] > obv_1h[-6] > obv_1h[-12]:
                score += 2; signals.append("1h OBV계단식매집")

    if score < 6:
        return None

    return {
        'symbol': symbol,
        'price': price,
        'score': score,
        'signals': signals,
        'change_24h': round(ticker['change_24h'], 1),
        'vol_24h_m': round(vol_24h / 1e6, 1),
        'oi_m': round(oi_value / 1e6, 1),
        'funding_rate': round(funding, 4),
        'rsi_1d': details.get('rsi_1d', 50),
        'rsi_4h': details.get('rsi_4h', 50),
        'bb_width_1d': details.get('bb_width_1d'),
        'bb_width_4h': details.get('bb_width_4h'),
        'vol_ratio_1d': details.get('vol_ratio_1d'),
        'drawdown_30d': details.get('drawdown_30d'),
        'vol_oi_ratio': details.get('vol_oi_ratio'),
    }


def main():
    now = datetime.now(KST)
    print(f"{'='*70}")
    print(f"  PUMP SCANNER v2 - 소형캡 급등 전조 감지")
    print(f"  {now.strftime('%Y-%m-%d %H:%M:%S KST')}")
    print(f"  필터: OI ${MIN_OI_VALUE/1e3:.0f}K~${MAX_OI_VALUE/1e6:.0f}M | 볼륨 >=${MIN_VOLUME_24H/1e6:.0f}M")
    print(f"{'='*70}")

    print("\n[1] Bybit 소형캡 종목 조회...")
    tickers = get_small_cap_tickers()
    print(f"    {len(tickers)}개 소형캡 종목")

    print(f"\n[2] {len(tickers)}개 종목 급등 전조 스캔 중...")
    results = []
    for i, t in enumerate(tickers):
        result = analyze_pump_signals(t['symbol'], t)
        if result:
            results.append(result)
        if (i + 1) % 20 == 0:
            print(f"    ... {i+1}/{len(tickers)} 완료 ({len(results)}개 감지)")
        time.sleep(0.15)

    results.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n{'='*70}")
    print(f"  결과: {len(results)}개 급등 전조 감지 (소형캡)")
    print(f"{'='*70}")

    if not results:
        print("\n  전조 패턴 없음")
    else:
        print(f"\n{'─'*85}")
        print(f"  {'#':>2} | {'종목':<10} | {'점수':>3} | {'1dRSI':>5} | {'4hRSI':>5} | {'BB폭':>6} | {'볼비':>5} | {'OI_M$':>5} | {'펀딩':>7} | {'24h':>7} | 전조 시그널")
        print(f"{'─'*85}")
        for rank, s in enumerate(results[:20], 1):
            sym = s['symbol'].replace('USDT', '')
            bb = f"{s['bb_width_1d']:.1f}%" if s['bb_width_1d'] else "N/A"
            vr = f"{s['vol_ratio_1d']:.1f}x" if s['vol_ratio_1d'] else "N/A"
            sigs = ', '.join(s['signals'][:2])
            print(f"  {rank:>2} | {sym:<10} | {s['score']:>3} | {s['rsi_1d']:>5.0f} | {s['rsi_4h']:>5.0f} | {bb:>6} | {vr:>5} | {s['oi_m']:>4.1f} | {s['funding_rate']:>+6.3f}% | {s['change_24h']:>+6.1f}% | {sigs}")

        print(f"\n{'='*70}")
        print("  TOP 10 상세")
        print(f"{'='*70}")
        for rank, s in enumerate(results[:10], 1):
            sym = s['symbol'].replace('USDT', '')
            print(f"\n  #{rank} {sym} (전조점수: {s['score']})")
            print(f"     가격: {s['price']} | 24h: {s['change_24h']:+.1f}% | 볼륨: ${s['vol_24h_m']:.1f}M | OI: ${s['oi_m']:.1f}M")
            print(f"     RSI: 1d={s['rsi_1d']:.0f} 4h={s['rsi_4h']:.0f} | 펀딩: {s['funding_rate']:+.4f}%")
            if s.get('bb_width_1d'):
                print(f"     BB폭: 1d={s['bb_width_1d']:.2f}%" + (f" 4h={s['bb_width_4h']:.2f}%" if s.get('bb_width_4h') else ""))
            if s.get('drawdown_30d'):
                print(f"     30일 고점 대비: {s['drawdown_30d']:.1f}%")
            if s.get('vol_oi_ratio'):
                print(f"     거래량/OI: {s['vol_oi_ratio']:.2f}x")
            print(f"     전조:")
            for sig in s['signals']:
                print(f"       - {sig}")

    output = {
        'scan_time': now.isoformat(),
        'total_scanned': len(tickers),
        'detected': len(results),
        'results': results[:20],
    }
    with open('/Users/eomgyuhyeon/.openclaw/workspace/crypto-paper-trading/pump_scan_v2_result.json', 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n결과 저장: pump_scan_v2_result.json")


if __name__ == "__main__":
    main()
