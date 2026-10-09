#!/usr/bin/env python3
"""
알트코인 급등(100%+) 전조 패턴 스캐너
=====================================
급등 전조 공통 패턴 (데이터 기반):

1. 볼륨 선행 폭발 (Volume Precedes Price)
   - 가격은 아직 안 움직이는데 거래량이 2-5배 급증
   - 스마트머니/고래가 조용히 매집하는 신호
   - 최근 3-7일 평균 대비 거래량 급증 감지

2. 장기 횡보 후 BB 극단적 수축 (Squeeze)
   - 볼린저밴드 폭이 역대급으로 좁아짐
   - 에너지 축적 → 한 방향으로 폭발
   - BB width < 3% (일봉 기준)

3. OBV 다이버전스 (은밀한 매집)
   - 가격은 횡보/하락인데 OBV는 상승
   - 누군가 물량을 조용히 모으는 중

4. 저시총 + 고거래량 비율
   - 시총 대비 거래량이 비정상적으로 높음
   - 관심도가 시총에 비해 과도 = 뭔가 일어나고 있음

5. 고래 지갑/OI 급증
   - 오픈인터레스트 갑자기 늘어남
   - 큰 손들이 포지션 잡기 시작

6. 펀딩비 마이너스 + 가격 바닥
   - 숏 과열 상태에서 반전 시 숏스퀴즈로 폭등
   - 펀딩비 연속 마이너스 + RSI 과매도

7. 이전 고점 대비 80-90% 하락 후 바닥 다지기
   - ATH 대비 크게 빠진 후 더 이상 안 빠지는 구간
   - 매도 세력 소진 신호
"""

import json
import math
import time
import urllib.request
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))

# 더 넓게 스캔 (소형 포함 - 급등은 소형에서 많이 발생)
MIN_QUOTE_VOLUME_24H = 5_000_000  # $5M까지 낮춤


def api_get(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except:
        return None


def get_all_tickers():
    """Bybit USDT 선물 전종목"""
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
        funding = float(d.get('fundingRate', 0)) * 100
        oi = float(d.get('openInterest', 0))
        oi_value = oi * price  # OI를 달러 가치로
        if vol >= MIN_QUOTE_VOLUME_24H and price > 0:
            results.append({
                'symbol': sym,
                'price': price,
                'change_24h': change,
                'volume_24h': vol,
                'funding_rate': funding,
                'oi_value': oi_value,
                'oi_raw': oi,
            })
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
    """급등 전조 패턴 종합 분석"""
    score = 0
    signals = []
    details = {}

    price = ticker['price']
    vol_24h = ticker['volume_24h']
    funding = ticker['funding_rate']
    oi_value = ticker['oi_value']

    # ─── 1) 일봉 데이터 ───
    candles_1d = get_klines(symbol, "D", 30)
    if not candles_1d or len(candles_1d) < 14:
        return None

    closes_1d = [c['close'] for c in candles_1d]
    vols_1d = [c['quote_volume'] for c in candles_1d]

    # ─── 2) 4시간봉 데이터 ───
    candles_4h = get_klines(symbol, "240", 50)
    if not candles_4h or len(candles_4h) < 20:
        return None

    closes_4h = [c['close'] for c in candles_4h]
    vols_4h = [c['quote_volume'] for c in candles_4h]

    # ─── 3) 1시간봉 데이터 ───
    candles_1h = get_klines(symbol, "60", 48)

    # ============================================================
    # 패턴 1: 볼륨 선행 폭발 (Volume Precedes Price)
    # 가격 변동 < 3%인데 거래량이 평균의 2배 이상
    # ============================================================
    if len(vols_1d) >= 10:
        avg_vol_7d = sum(vols_1d[-8:-1]) / 7  # 최근 7일 평균 (오늘 제외)
        today_vol = vols_1d[-1]
        price_chg_3d = abs(closes_1d[-1] - closes_1d[-4]) / closes_1d[-4] * 100 if len(closes_1d) >= 4 else 99

        if avg_vol_7d > 0:
            vol_ratio = today_vol / avg_vol_7d
            details['vol_ratio_1d'] = round(vol_ratio, 1)

            if vol_ratio >= 3.0 and price_chg_3d < 5:
                score += 4
                signals.append(f"볼륨폭발({vol_ratio:.1f}x)+가격정체({price_chg_3d:.1f}%)")
            elif vol_ratio >= 2.0 and price_chg_3d < 5:
                score += 3
                signals.append(f"볼륨증가({vol_ratio:.1f}x)+가격정체({price_chg_3d:.1f}%)")
            elif vol_ratio >= 1.5 and price_chg_3d < 3:
                score += 1
                signals.append(f"볼륨소폭증가({vol_ratio:.1f}x)")

    # 4h 볼륨도 체크
    if len(vols_4h) >= 20:
        avg_vol_4h = sum(vols_4h[-13:-1]) / 12
        recent_vol_4h = sum(vols_4h[-3:]) / 3
        if avg_vol_4h > 0:
            vol_ratio_4h = recent_vol_4h / avg_vol_4h
            if vol_ratio_4h >= 2.5:
                score += 2
                signals.append(f"4h볼륨급증({vol_ratio_4h:.1f}x)")
            elif vol_ratio_4h >= 1.8:
                score += 1
                signals.append(f"4h볼륨증가({vol_ratio_4h:.1f}x)")

    # ============================================================
    # 패턴 2: BB 극단적 수축 (Squeeze)
    # ============================================================
    sma, bb_up, bb_lo = calc_bb(closes_1d)
    if sma and bb_up and bb_lo and sma > 0:
        bb_width = (bb_up - bb_lo) / sma * 100
        details['bb_width_1d'] = round(bb_width, 2)

        if bb_width < 3:
            score += 4
            signals.append(f"1d BB극단수축({bb_width:.1f}%)")
        elif bb_width < 5:
            score += 2
            signals.append(f"1d BB수축({bb_width:.1f}%)")
        elif bb_width < 8:
            score += 1
            signals.append(f"1d BB좁음({bb_width:.1f}%)")

    sma4, bb_up4, bb_lo4 = calc_bb(closes_4h)
    if sma4 and bb_up4 and bb_lo4 and sma4 > 0:
        bb_width_4h = (bb_up4 - bb_lo4) / sma4 * 100
        details['bb_width_4h'] = round(bb_width_4h, 2)
        if bb_width_4h < 2:
            score += 3
            signals.append(f"4h BB극단수축({bb_width_4h:.1f}%)")
        elif bb_width_4h < 4:
            score += 1
            signals.append(f"4h BB수축({bb_width_4h:.1f}%)")

    # ============================================================
    # 패턴 3: OBV 다이버전스 (은밀한 매집)
    # 가격 횡보/하락인데 OBV 상승
    # ============================================================
    obv_1d = calc_obv(candles_1d)
    if len(obv_1d) >= 10:
        price_chg_10d = (closes_1d[-1] - closes_1d[-10]) / closes_1d[-10] * 100
        obv_trend = obv_1d[-1] - obv_1d[-10]

        if price_chg_10d <= 0 and obv_trend > 0:
            score += 3
            signals.append(f"OBV다이버전스(가격{price_chg_10d:+.1f}%/OBV상승)")
        elif price_chg_10d < 3 and obv_trend > 0:
            score += 2
            signals.append(f"OBV순매집(가격정체+OBV상승)")

    # 4h OBV
    obv_4h = calc_obv(candles_4h)
    if len(obv_4h) >= 20:
        price_chg_4h = (closes_4h[-1] - closes_4h[-20]) / closes_4h[-20] * 100
        if price_chg_4h <= 0 and obv_4h[-1] > obv_4h[-20]:
            score += 2
            signals.append("4h OBV다이버전스")

    # ============================================================
    # 패턴 4: 거래량/시총 비율 (과도한 관심)
    # 24h 거래량이 추정시총의 30%+ = 비정상적 관심
    # ============================================================
    # OI를 시총 프록시로 사용 (선물이라 정확한 시총 없음)
    if oi_value > 0:
        vol_oi_ratio = vol_24h / oi_value
        details['vol_oi_ratio'] = round(vol_oi_ratio, 2)
        if vol_oi_ratio >= 5.0:
            score += 3
            signals.append(f"거래량/OI비율 극단({vol_oi_ratio:.1f}x)")
        elif vol_oi_ratio >= 2.0:
            score += 1
            signals.append(f"거래량/OI비율 높음({vol_oi_ratio:.1f}x)")

    # ============================================================
    # 패턴 5: 펀딩비 마이너스 + RSI 바닥 (숏스퀴즈 세팅)
    # ============================================================
    rsi_1d = calc_rsi(closes_1d)
    rsi_4h = calc_rsi(closes_4h)
    details['rsi_1d'] = round(rsi_1d, 1)
    details['rsi_4h'] = round(rsi_4h, 1)

    funding_hist = get_funding_history(symbol, 8)
    neg_funding_count = sum(1 for f in funding_hist if f < 0)

    if funding < -0.01 and rsi_4h < 35:
        score += 4
        signals.append(f"숏스퀴즈세팅(펀딩{funding:.3f}%+RSI{rsi_4h:.0f})")
    elif funding < -0.01:
        score += 2
        signals.append(f"펀딩마이너스({funding:.3f}%)")
    elif neg_funding_count >= 5:
        score += 2
        signals.append(f"펀딩연속마이너스({neg_funding_count}/8회)")

    # ============================================================
    # 패턴 6: ATH 대비 대폭 하락 후 바닥 다지기
    # 최근 30일 고점 대비 50%+ 하락 후 횡보
    # ============================================================
    if len(closes_1d) >= 30:
        high_30d = max(c['high'] for c in candles_1d)
        low_recent = min(closes_1d[-5:])
        drawdown = (price - high_30d) / high_30d * 100

        # 최근 5일 변동폭
        recent_range = (max(closes_1d[-5:]) - min(closes_1d[-5:])) / min(closes_1d[-5:]) * 100

        details['drawdown_30d'] = round(drawdown, 1)

        if drawdown < -50 and recent_range < 10:
            score += 3
            signals.append(f"대폭락후바닥({drawdown:.0f}%,최근변동{recent_range:.0f}%)")
        elif drawdown < -30 and recent_range < 8:
            score += 2
            signals.append(f"하락후횡보({drawdown:.0f}%,변동{recent_range:.0f}%)")

    # ============================================================
    # 패턴 7: 하단 꼬리 반복 (매수세 방어)
    # ============================================================
    if candles_1d:
        tail_count = 0
        for c in candles_1d[-7:]:
            body = abs(c['close'] - c['open'])
            lower_tail = min(c['open'], c['close']) - c['low']
            if body > 0 and lower_tail > body * 2:
                tail_count += 1
        if tail_count >= 4:
            score += 3
            signals.append(f"1d 하단꼬리반복({tail_count}/7일)")
        elif tail_count >= 3:
            score += 2
            signals.append(f"1d 하단꼬리({tail_count}/7일)")
        elif tail_count >= 2:
            score += 1
            signals.append(f"1d 하단꼬리({tail_count}/7일)")

    # ============================================================
    # 패턴 8: EMA 수렴 (방향 잡기 직전)
    # ============================================================
    e9 = calc_ema(closes_1d, 9)
    e21 = calc_ema(closes_1d, 21)
    if e9 and e21 and e21 > 0:
        ema_gap = abs(e9 - e21) / e21 * 100
        if ema_gap < 0.5:
            score += 2
            signals.append(f"EMA9/21수렴({ema_gap:.2f}%)")
        elif ema_gap < 1.0:
            score += 1
            signals.append(f"EMA근접({ema_gap:.2f}%)")

        # 골든크로스 임박
        if e9 > e21 and ema_gap < 1.0:
            score += 1
            signals.append("EMA골든크로스")

    # ============================================================
    # 보너스: 1h 단기 매집 확인
    # ============================================================
    if candles_1h and len(candles_1h) >= 20:
        obv_1h = calc_obv(candles_1h)
        if len(obv_1h) >= 12:
            if obv_1h[-1] > obv_1h[-6] > obv_1h[-12]:
                score += 1
                signals.append("1h OBV계단식상승")

    if score < 5:
        return None

    return {
        'symbol': symbol,
        'price': price,
        'score': score,
        'signals': signals,
        'change_24h': round(ticker['change_24h'], 1),
        'vol_24h_m': round(vol_24h / 1e6, 1),
        'funding_rate': round(funding, 4),
        'rsi_1d': details.get('rsi_1d', 50),
        'rsi_4h': details.get('rsi_4h', 50),
        'bb_width_1d': details.get('bb_width_1d'),
        'bb_width_4h': details.get('bb_width_4h'),
        'vol_ratio_1d': details.get('vol_ratio_1d'),
        'drawdown_30d': details.get('drawdown_30d'),
        'vol_oi_ratio': details.get('vol_oi_ratio'),
        'neg_funding_count': neg_funding_count,
    }


def main():
    now = datetime.now(KST)
    print(f"{'='*70}")
    print(f"  PUMP SCANNER - 급등 전조 패턴 감지")
    print(f"  {now.strftime('%Y-%m-%d %H:%M:%S KST')}")
    print(f"{'='*70}")
    print()
    print("  감지 패턴:")
    print("  1. 볼륨 선행 폭발 (가격 정체 + 거래량 급증)")
    print("  2. BB 극단적 수축 (에너지 축적)")
    print("  3. OBV 다이버전스 (은밀한 매집)")
    print("  4. 거래량/OI 비율 이상 (과도한 관심)")
    print("  5. 펀딩비 마이너스 + RSI 바닥 (숏스퀴즈)")
    print("  6. 대폭락 후 바닥 다지기")
    print("  7. 하단 꼬리 반복 (매수세 방어)")
    print("  8. EMA 수렴 (방향 잡기 직전)")
    print()

    print("[1] Bybit 전종목 조회...")
    tickers = get_all_tickers()
    print(f"    {len(tickers)}개 종목")

    print(f"\n[2] {len(tickers)}개 종목 급등 전조 스캔 중...")
    results = []
    for i, t in enumerate(tickers):
        result = analyze_pump_signals(t['symbol'], t)
        if result:
            results.append(result)
        if (i + 1) % 10 == 0:
            print(f"    ... {i+1}/{len(tickers)} 완료 ({len(results)}개 감지)")
        time.sleep(0.15)

    results.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n{'='*70}")
    print(f"  결과: {len(results)}개 급등 전조 감지")
    print(f"{'='*70}")

    if not results:
        print("\n  전조 패턴 감지된 종목 없음")
    else:
        print(f"\n{'─'*80}")
        print(f"  {'순위':>3} | {'종목':<10} | {'점수':>4} | {'1dRSI':>5} | {'4hRSI':>5} | {'BB폭':>6} | {'볼비':>5} | {'펀딩':>7} | {'24h':>7} | 전조 시그널")
        print(f"{'─'*80}")
        for rank, s in enumerate(results[:25], 1):
            sym_name = s['symbol'].replace('USDT', '')
            bb = f"{s['bb_width_1d']:.1f}%" if s['bb_width_1d'] else "N/A"
            vr = f"{s['vol_ratio_1d']:.1f}x" if s['vol_ratio_1d'] else "N/A"
            sigs = ', '.join(s['signals'][:2])
            print(f"  {rank:>3} | {sym_name:<10} | {s['score']:>4} | {s['rsi_1d']:>5.0f} | {s['rsi_4h']:>5.0f} | {bb:>6} | {vr:>5} | {s['funding_rate']:>+6.3f}% | {s['change_24h']:>+6.1f}% | {sigs}")

        print(f"\n{'='*70}")
        print("  TOP 10 상세 분석")
        print(f"{'='*70}")
        for rank, s in enumerate(results[:10], 1):
            sym_name = s['symbol'].replace('USDT', '')
            print(f"\n  #{rank} {sym_name} (전조점수: {s['score']})")
            print(f"     가격: {s['price']} | 24h: {s['change_24h']:+.1f}% | 볼륨: ${s['vol_24h_m']:.0f}M")
            print(f"     RSI: 1d={s['rsi_1d']:.0f} 4h={s['rsi_4h']:.0f} | 펀딩: {s['funding_rate']:+.4f}%")
            if s['bb_width_1d']:
                print(f"     BB폭: 1d={s['bb_width_1d']:.2f}%" + (f" 4h={s['bb_width_4h']:.2f}%" if s.get('bb_width_4h') else ""))
            if s.get('drawdown_30d'):
                print(f"     30일 고점 대비: {s['drawdown_30d']:.1f}%")
            if s.get('vol_oi_ratio'):
                print(f"     거래량/OI: {s['vol_oi_ratio']:.2f}x")
            print(f"     전조 시그널:")
            for sig in s['signals']:
                print(f"       - {sig}")

    # 저장
    output = {
        'scan_time': now.isoformat(),
        'total_scanned': len(tickers),
        'detected': len(results),
        'results': results[:25],
    }
    with open('/Users/eomgyuhyeon/.openclaw/workspace/crypto-paper-trading/pump_scan_result.json', 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n결과 저장: pump_scan_result.json")


if __name__ == "__main__":
    main()
