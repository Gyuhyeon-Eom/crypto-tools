#!/usr/bin/env python3
"""
Stage 1 — 급등 후보 선정 (일봉/4h 기반)
========================================
기존 pump_scanner_v2를 기반으로 하되, 더 넓은 풀에서 후보를 뽑는다.
이 스크립트는 30분~1시간마다 실행. 결과를 stage1_watchlist.json에 저장.
Stage 2가 이 파일을 읽어서 실시간 거래량 감시.

개선점:
- Bybit spot + linear 모두 스캔
- 더 세밀한 스코어링 (OBV 가속도, VWAP 이탈, 거래량 분포 비대칭)
- "급등 가능 후보" 기준: score >= 8
"""

import json
import math
import time
import urllib.request
import os
from datetime import datetime, timezone, timedelta
from collections import defaultdict

KST = timezone(timedelta(hours=9))
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
WATCHLIST_FILE = os.path.join(WORKSPACE, "stage1_watchlist.json")

# 필터 기준 — 소형~중형캡
MIN_VOLUME_24H = 500_000       # $500K 이상 (너무 작으면 유동성 위험)
MAX_VOLUME_24H = 500_000_000   # $500M 이하 (대형 제외)


def api_get(url, retries=2):
    for i in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except Exception as e:
            if i < retries:
                time.sleep(1)
            else:
                return None


def get_all_tickers():
    """Bybit linear (선물) + spot 모두에서 USDT 페어 수집"""
    EXCLUDE = {
        'USDCUSDT','USDTUSDT','DAIUSDT','TUSDUSDT','BUSDUSDT',
        'FDUSDUSDT','EURUSDT','PAXGUSDT','XAUTUSDT','BTCUSDT',
        'ETHUSDT','BNBUSDT','SOLUSDT','XRPUSDT',  # 대형 제외
    }
    
    results = {}
    
    # Linear (선물) — OI, 펀딩 정보 있음
    data = api_get("https://api.bybit.com/v5/market/tickers?category=linear")
    if data and data.get('retCode') == 0:
        for d in data['result']['list']:
            sym = d['symbol']
            if not sym.endswith('USDT') or sym in EXCLUDE:
                continue
            vol = float(d.get('turnover24h', 0))
            price = float(d.get('lastPrice', 0))
            if price <= 0 or not (MIN_VOLUME_24H <= vol <= MAX_VOLUME_24H):
                continue
            oi = float(d.get('openInterest', 0))
            results[sym] = {
                'symbol': sym,
                'price': price,
                'change_24h': float(d.get('price24hPcnt', 0)) * 100,
                'volume_24h': vol,
                'funding_rate': float(d.get('fundingRate', 0)) * 100,
                'oi_value': oi * price,
                'oi_raw': oi,
                'has_futures': True,
            }
    
    # Spot도 확인 (선물에 없는 코인)
    data = api_get("https://api.bybit.com/v5/market/tickers?category=spot")
    if data and data.get('retCode') == 0:
        for d in data['result']['list']:
            sym = d['symbol']
            if not sym.endswith('USDT') or sym in EXCLUDE or sym in results:
                continue
            vol = float(d.get('turnover24h', 0))
            price = float(d.get('lastPrice', 0))
            if price <= 0 or not (MIN_VOLUME_24H <= vol <= MAX_VOLUME_24H):
                continue
            results[sym] = {
                'symbol': sym,
                'price': price,
                'change_24h': float(d.get('price24hPcnt', 0)) * 100,
                'volume_24h': vol,
                'funding_rate': 0,
                'oi_value': 0,
                'oi_raw': 0,
                'has_futures': False,
            }
    
    return list(results.values())


def get_klines(symbol, interval, limit=50, category="linear"):
    url = f"https://api.bybit.com/v5/market/kline?category={category}&symbol={symbol}&interval={interval}&limit={limit}"
    data = api_get(url)
    if not data or data.get('retCode') != 0 or not data['result']['list']:
        return None
    raw = data['result']['list'][::-1]  # oldest first
    return [{
        'open': float(d[1]), 'high': float(d[2]), 'low': float(d[3]),
        'close': float(d[4]), 'volume': float(d[5]),
        'quote_volume': float(d[6]) if len(d) > 6 else float(d[5]) * float(d[4]),
        'timestamp': int(d[0])
    } for d in raw]


# ═══════ Technical Indicators ═══════

def ema(prices, period):
    if len(prices) < period:
        return None
    m = 2 / (period + 1)
    e = sum(prices[:period]) / period
    for p in prices[period:]:
        e = (p - e) * m + e
    return e


def ema_series(prices, period):
    if len(prices) < period:
        return []
    m = 2 / (period + 1)
    e = sum(prices[:period]) / period
    result = [e]
    for p in prices[period:]:
        e = (p - e) * m + e
        result.append(e)
    return result


def rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50
    gains, losses = [], []
    for i in range(1, len(prices)):
        d = prices[i] - prices[i-1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    ag = sum(gains[-period:]) / period
    al = sum(losses[-period:]) / period
    if al == 0:
        return 100
    return 100 - (100 / (1 + ag / al))


def bollinger(prices, period=20, mult=2):
    if len(prices) < period:
        return None, None, None
    r = prices[-period:]
    sma = sum(r) / period
    std = math.sqrt(sum((p - sma)**2 for p in r) / period)
    return sma, sma + mult * std, sma - mult * std


def obv(candles):
    if len(candles) < 2:
        return []
    o = [0]
    for i in range(1, len(candles)):
        if candles[i]['close'] > candles[i-1]['close']:
            o.append(o[-1] + candles[i]['volume'])
        elif candles[i]['close'] < candles[i-1]['close']:
            o.append(o[-1] - candles[i]['volume'])
        else:
            o.append(o[-1])
    return o


def vwap(candles, lookback=20):
    """VWAP 계산 (최근 lookback 캔들)"""
    if len(candles) < lookback:
        return None
    recent = candles[-lookback:]
    cum_pv = sum(c['close'] * c['volume'] for c in recent)
    cum_v = sum(c['volume'] for c in recent)
    if cum_v == 0:
        return None
    return cum_pv / cum_v


def atr(candles, period=14):
    """ATR (Average True Range)"""
    if len(candles) < period + 1:
        return None
    trs = []
    for i in range(1, len(candles)):
        h = candles[i]['high']
        l = candles[i]['low']
        pc = candles[i-1]['close']
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    return sum(trs[-period:]) / period


def volume_profile_skew(candles, lookback=20):
    """
    거래량 프로파일 비대칭도.
    양봉 거래량 vs 음봉 거래량 비율 → 매집 감지.
    """
    if len(candles) < lookback:
        return 0
    recent = candles[-lookback:]
    up_vol = sum(c['volume'] for c in recent if c['close'] >= c['open'])
    down_vol = sum(c['volume'] for c in recent if c['close'] < c['open'])
    total = up_vol + down_vol
    if total == 0:
        return 0
    return (up_vol - down_vol) / total  # -1 ~ +1


def mfi(candles, period=14):
    """Money Flow Index"""
    if len(candles) < period + 1:
        return 50
    pos_flow = 0
    neg_flow = 0
    for i in range(-period, 0):
        tp = (candles[i]['high'] + candles[i]['low'] + candles[i]['close']) / 3
        tp_prev = (candles[i-1]['high'] + candles[i-1]['low'] + candles[i-1]['close']) / 3
        mf = tp * candles[i]['volume']
        if tp > tp_prev:
            pos_flow += mf
        else:
            neg_flow += mf
    if neg_flow == 0:
        return 100
    ratio = pos_flow / neg_flow
    return 100 - (100 / (1 + ratio))


def chaikin_money_flow(candles, period=20):
    """CMF — Chaikin Money Flow"""
    if len(candles) < period:
        return 0
    recent = candles[-period:]
    cmf_sum = 0
    vol_sum = 0
    for c in recent:
        hl = c['high'] - c['low']
        if hl == 0:
            continue
        clv = ((c['close'] - c['low']) - (c['high'] - c['close'])) / hl
        cmf_sum += clv * c['volume']
        vol_sum += c['volume']
    if vol_sum == 0:
        return 0
    return cmf_sum / vol_sum


def get_funding_history(symbol, limit=8):
    url = f"https://api.bybit.com/v5/market/funding/history?category=linear&symbol={symbol}&limit={limit}"
    data = api_get(url)
    if not data or data.get('retCode') != 0 or not data['result']['list']:
        return []
    return [float(d['fundingRate']) * 100 for d in data['result']['list']]


# ═══════ Main Analysis ═══════

def analyze_candidate(ticker):
    """
    종합 전조 분석. score >= 8이면 watchlist에 추가.
    
    시그널 카테고리:
    A. 거래량 이상 (볼륨 선행, OBV 다이버전스, CMF)
    B. 변동성 수축 (BB squeeze, ATR 수축)
    C. 가격 구조 (바닥 다지기, 하단꼬리, EMA수렴)
    D. 매집 흔적 (vol profile skew, MFI, 대량 양봉)
    E. 선물 시그널 (펀딩비, OI)
    """
    sym = ticker['symbol']
    price = ticker['price']
    cat = "linear" if ticker['has_futures'] else "spot"
    
    score = 0
    signals = []
    details = {}
    
    # 캔들 데이터 수집
    c1d = get_klines(sym, "D", 30, cat)
    if not c1d or len(c1d) < 10:
        return None
    
    c4h = get_klines(sym, "240", 50, cat)
    if not c4h or len(c4h) < 15:
        return None
    
    c1h = get_klines(sym, "60", 48, cat)
    
    closes_1d = [c['close'] for c in c1d]
    closes_4h = [c['close'] for c in c4h]
    vols_1d = [c['quote_volume'] for c in c1d]
    vols_4h = [c['quote_volume'] for c in c4h]
    
    # ═══ A. 거래량 이상 ═══
    
    # A1. 일봉 볼륨 선행 폭발 (가격은 안 움직이는데 거래량 터짐)
    if len(vols_1d) >= 8:
        avg_v7 = sum(vols_1d[-8:-1]) / 7
        today_v = vols_1d[-1]
        price_chg_3d = abs(closes_1d[-1] - closes_1d[-4]) / closes_1d[-4] * 100 if len(closes_1d) >= 4 else 99
        
        if avg_v7 > 0:
            vr = today_v / avg_v7
            details['vol_ratio_1d'] = round(vr, 2)
            if vr >= 4.0 and price_chg_3d < 5:
                score += 6; signals.append(f"A1:볼륨폭발({vr:.1f}x)+가격정체")
            elif vr >= 2.5 and price_chg_3d < 5:
                score += 4; signals.append(f"A1:볼륨급증({vr:.1f}x)+가격정체")
            elif vr >= 1.8 and price_chg_3d < 3:
                score += 2; signals.append(f"A1:볼륨증가({vr:.1f}x)")
    
    # A2. 4h 볼륨 선행
    if len(vols_4h) >= 15:
        avg_v4h = sum(vols_4h[-13:-1]) / 12
        recent_v4h = sum(vols_4h[-3:]) / 3
        if avg_v4h > 0:
            vr4 = recent_v4h / avg_v4h
            details['vol_ratio_4h'] = round(vr4, 2)
            if vr4 >= 4.0:
                score += 4; signals.append(f"A2:4h볼륨폭발({vr4:.1f}x)")
            elif vr4 >= 2.5:
                score += 3; signals.append(f"A2:4h볼륨급증({vr4:.1f}x)")
            elif vr4 >= 1.8:
                score += 1; signals.append(f"A2:4h볼륨증가({vr4:.1f}x)")
    
    # A3. OBV 다이버전스 (가격↓ OBV↑ = 숨은 매집)
    obv_1d = obv(c1d)
    if len(obv_1d) >= 10:
        p_chg = (closes_1d[-1] - closes_1d[-10]) / closes_1d[-10] * 100
        obv_chg = obv_1d[-1] - obv_1d[-10]
        if p_chg <= -5 and obv_chg > 0:
            score += 5; signals.append(f"A3:강한OBV다이버전스(가격{p_chg:+.1f}%/OBV↑)")
        elif p_chg <= 0 and obv_chg > 0:
            score += 3; signals.append(f"A3:OBV다이버전스(가격{p_chg:+.1f}%/OBV↑)")
        elif p_chg < 3 and obv_chg > 0:
            score += 1; signals.append("A3:OBV순매집")
    
    # A4. OBV 가속도 (최근 5일 OBV 상승 속도 > 이전 5일)
    if len(obv_1d) >= 11:
        obv_accel_recent = obv_1d[-1] - obv_1d[-5]
        obv_accel_prev = obv_1d[-6] - obv_1d[-10]
        if obv_accel_prev != 0:
            accel_ratio = obv_accel_recent / abs(obv_accel_prev) if obv_accel_prev != 0 else 0
            if obv_accel_recent > 0 and accel_ratio > 2:
                score += 2; signals.append(f"A4:OBV가속({accel_ratio:.1f}x)")
    
    # A5. CMF (Chaikin Money Flow) — 자금 유입
    cmf_val = chaikin_money_flow(c4h, 20)
    details['cmf_4h'] = round(cmf_val, 3)
    if cmf_val > 0.15:
        score += 3; signals.append(f"A5:강한자금유입(CMF={cmf_val:.2f})")
    elif cmf_val > 0.08:
        score += 2; signals.append(f"A5:자금유입(CMF={cmf_val:.2f})")
    
    # ═══ B. 변동성 수축 ═══
    
    # B1. BB Squeeze (일봉)
    sma, bb_up, bb_lo = bollinger(closes_1d)
    if sma and bb_up and bb_lo and sma > 0:
        bb_w = (bb_up - bb_lo) / sma * 100
        details['bb_width_1d'] = round(bb_w, 2)
        if bb_w < 4:
            score += 5; signals.append(f"B1:극단BB수축({bb_w:.1f}%)")
        elif bb_w < 7:
            score += 3; signals.append(f"B1:BB수축({bb_w:.1f}%)")
        elif bb_w < 10:
            score += 1; signals.append(f"B1:BB좁음({bb_w:.1f}%)")
    
    # B2. BB Squeeze (4h)
    sma4, bb_up4, bb_lo4 = bollinger(closes_4h)
    if sma4 and bb_up4 and bb_lo4 and sma4 > 0:
        bb_w4 = (bb_up4 - bb_lo4) / sma4 * 100
        details['bb_width_4h'] = round(bb_w4, 2)
        if bb_w4 < 2.5:
            score += 3; signals.append(f"B2:4h극단BB수축({bb_w4:.1f}%)")
        elif bb_w4 < 4:
            score += 1; signals.append(f"B2:4hBB수축({bb_w4:.1f}%)")
    
    # B3. ATR 수축 (변동성 줄어듦 → 곧 터진다)
    atr_val = atr(c1d, 14)
    if atr_val and price > 0:
        atr_pct = atr_val / price * 100
        details['atr_pct_1d'] = round(atr_pct, 2)
        if atr_pct < 2:
            score += 2; signals.append(f"B3:ATR극단수축({atr_pct:.1f}%)")
        elif atr_pct < 4:
            score += 1; signals.append(f"B3:ATR수축({atr_pct:.1f}%)")
    
    # ═══ C. 가격 구조 ═══
    
    # C1. 대폭락 후 바닥 횡보
    if len(c1d) >= 20:
        high_30d = max(c['high'] for c in c1d)
        drawdown = (price - high_30d) / high_30d * 100
        recent_range = (max(closes_1d[-5:]) - min(closes_1d[-5:])) / min(closes_1d[-5:]) * 100 if len(closes_1d) >= 5 else 99
        details['drawdown_30d'] = round(drawdown, 1)
        details['recent_range_5d'] = round(recent_range, 1)
        
        if drawdown < -60 and recent_range < 15:
            score += 5; signals.append(f"C1:대폭락후바닥({drawdown:.0f}%,범위{recent_range:.0f}%)")
        elif drawdown < -40 and recent_range < 12:
            score += 3; signals.append(f"C1:폭락후횡보({drawdown:.0f}%)")
        elif drawdown < -25 and recent_range < 10:
            score += 1; signals.append(f"C1:하락후안정({drawdown:.0f}%)")
    
    # C2. 하단 꼬리 반복 (매수 방어)
    tail_count = 0
    for c in c1d[-7:]:
        body = abs(c['close'] - c['open'])
        lower_tail = min(c['open'], c['close']) - c['low']
        if body > 0 and lower_tail > body * 2:
            tail_count += 1
    if tail_count >= 4:
        score += 4; signals.append(f"C2:강한매수방어({tail_count}/7)")
    elif tail_count >= 3:
        score += 2; signals.append(f"C2:매수방어({tail_count}/7)")
    elif tail_count >= 2:
        score += 1; signals.append(f"C2:하단꼬리({tail_count}/7)")
    
    # C3. EMA 수렴 / 골든크로스
    e9 = ema(closes_1d, 9)
    e21 = ema(closes_1d, 21)
    if e9 and e21 and e21 > 0:
        gap = abs(e9 - e21) / e21 * 100
        details['ema_gap'] = round(gap, 2)
        if gap < 0.5:
            score += 3; signals.append(f"C3:EMA극단수렴({gap:.2f}%)")
        elif gap < 1.2:
            score += 1; signals.append(f"C3:EMA수렴({gap:.2f}%)")
        if e9 > e21 and gap < 2:
            score += 1; signals.append("C3:골든크로스")
    
    # C4. 가격이 VWAP 근처 or 아래 (매집 기회)
    vwap_val = vwap(c4h, 20)
    if vwap_val and price > 0:
        vwap_dist = (price - vwap_val) / vwap_val * 100
        details['vwap_dist_4h'] = round(vwap_dist, 2)
        if -5 < vwap_dist < -1:
            score += 2; signals.append(f"C4:VWAP하방({vwap_dist:+.1f}%)")
        elif -1 <= vwap_dist <= 1:
            score += 1; signals.append(f"C4:VWAP근접({vwap_dist:+.1f}%)")
    
    # ═══ D. 매집 흔적 ═══
    
    # D1. 거래량 프로파일 비대칭 (양봉 거래량 > 음봉 거래량)
    skew_4h = volume_profile_skew(c4h, 20)
    details['vol_skew_4h'] = round(skew_4h, 3)
    if skew_4h > 0.3:
        score += 3; signals.append(f"D1:강한매집비대칭({skew_4h:.2f})")
    elif skew_4h > 0.15:
        score += 2; signals.append(f"D1:매집비대칭({skew_4h:.2f})")
    
    # D2. MFI (자금흐름지수) — RSI의 거래량 가중 버전
    mfi_val = mfi(c4h, 14)
    details['mfi_4h'] = round(mfi_val, 1)
    if mfi_val < 20:
        score += 3; signals.append(f"D2:MFI과매도({mfi_val:.0f})")
    elif mfi_val < 30:
        score += 1; signals.append(f"D2:MFI저점({mfi_val:.0f})")
    
    # D3. "조용한 대량 양봉" — 큰 거래량 양봉이 나왔는데 가격 변화 적음
    if len(c4h) >= 10:
        for c in c4h[-6:]:
            if c['close'] > c['open']:  # 양봉
                pct_change = (c['close'] - c['open']) / c['open'] * 100
                avg_vol = sum(x['volume'] for x in c4h[-10:]) / 10
                if c['volume'] > avg_vol * 3 and pct_change < 2:
                    score += 2; signals.append("D3:조용한대량양봉")
                    break
    
    # D4. 1h OBV 계단식 매집
    if c1h and len(c1h) >= 20:
        obv_1h = obv(c1h)
        if len(obv_1h) >= 12:
            if obv_1h[-1] > obv_1h[-6] > obv_1h[-12]:
                score += 2; signals.append("D4:1hOBV계단매집")
    
    # ═══ E. 선물 시그널 ═══
    
    if ticker['has_futures']:
        funding = ticker['funding_rate']
        oi_val = ticker['oi_value']
        
        # E1. 펀딩비 + RSI 복합
        rsi_4h = rsi(closes_4h)
        rsi_1d = rsi(closes_1d)
        details['rsi_4h'] = round(rsi_4h, 1)
        details['rsi_1d'] = round(rsi_1d, 1)
        
        if funding < -0.05 and rsi_4h < 35:
            score += 6; signals.append(f"E1:숏스퀴즈세팅(FR{funding:.3f}%+RSI{rsi_4h:.0f})")
        elif funding < -0.03 and rsi_4h < 40:
            score += 4; signals.append(f"E1:숏과열(FR{funding:.3f}%+RSI{rsi_4h:.0f})")
        elif funding < -0.01:
            score += 2; signals.append(f"E1:펀딩마이너스({funding:.3f}%)")
        
        # E2. 펀딩비 연속 마이너스
        fh = get_funding_history(sym, 8)
        neg_cnt = sum(1 for f in fh if f < 0)
        if neg_cnt >= 6:
            score += 3; signals.append(f"E2:펀딩연속마이너스({neg_cnt}/8)")
        elif neg_cnt >= 4:
            score += 1; signals.append(f"E2:펀딩마이너스빈발({neg_cnt}/8)")
        
        # E3. 거래량/OI 비율 급등
        if oi_val > 0:
            voi = ticker['volume_24h'] / oi_val
            details['vol_oi_ratio'] = round(voi, 2)
            if voi >= 6:
                score += 4; signals.append(f"E3:Vol/OI극단({voi:.1f}x)")
            elif voi >= 3.5:
                score += 2; signals.append(f"E3:Vol/OI높음({voi:.1f}x)")
            elif voi >= 2:
                score += 1; signals.append(f"E3:Vol/OI관심({voi:.1f}x)")
    else:
        # 스팟 전용 — 기본 RSI만
        rsi_4h = rsi(closes_4h)
        rsi_1d = rsi(closes_1d)
        details['rsi_4h'] = round(rsi_4h, 1)
        details['rsi_1d'] = round(rsi_1d, 1)
        if rsi_4h < 25:
            score += 3; signals.append(f"E1:RSI극단과매도({rsi_4h:.0f})")
        elif rsi_4h < 35:
            score += 1; signals.append(f"E1:RSI과매도({rsi_4h:.0f})")
    
    # ═══ 최소 컷 ═══
    if score < 8:
        return None
    
    # 이미 24h에 +20% 이상 오른 건 제외 (이미 터진 것)
    if ticker['change_24h'] > 20:
        return None
    
    return {
        'symbol': sym,
        'price': price,
        'score': score,
        'signals': signals,
        'change_24h': round(ticker['change_24h'], 1),
        'vol_24h_m': round(ticker['volume_24h'] / 1e6, 2),
        'oi_m': round(ticker['oi_value'] / 1e6, 2) if ticker['oi_value'] else 0,
        'funding_rate': round(ticker['funding_rate'], 4),
        'has_futures': ticker['has_futures'],
        'details': details,
    }


def main():
    now = datetime.now(KST)
    print(f"{'='*70}")
    print(f"  STAGE 1 — 급등 후보 선정")
    print(f"  {now.strftime('%Y-%m-%d %H:%M:%S KST')}")
    print(f"{'='*70}")
    
    print("\n[1] Bybit 전체 종목 수집 (linear + spot)...")
    tickers = get_all_tickers()
    print(f"    {len(tickers)}개 종목 (소형~중형캡)")
    
    print(f"\n[2] {len(tickers)}개 종목 전조 분석 중...")
    results = []
    errors = 0
    for i, t in enumerate(tickers):
        try:
            result = analyze_candidate(t)
            if result:
                results.append(result)
        except Exception as e:
            errors += 1
        if (i + 1) % 30 == 0:
            print(f"    ... {i+1}/{len(tickers)} ({len(results)}개 감지, {errors}개 에러)")
        time.sleep(0.12)  # rate limit
    
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # 결과 출력
    print(f"\n{'='*70}")
    print(f"  결과: {len(results)}개 급등 후보 (score >= 8)")
    print(f"{'='*70}")
    
    for rank, r in enumerate(results[:30], 1):
        sym = r['symbol'].replace('USDT', '')
        sigs = ' | '.join(r['signals'][:3])
        fut = "F" if r['has_futures'] else "S"
        print(f"  #{rank:>2} [{fut}] {sym:<12} score={r['score']:>2} | 24h={r['change_24h']:>+6.1f}% | vol=${r['vol_24h_m']:.1f}M | {sigs}")
    
    # watchlist 저장
    watchlist = {
        'updated': now.isoformat(),
        'total_scanned': len(tickers),
        'candidates': len(results),
        'watchlist': results[:50],  # 상위 50개
    }
    with open(WATCHLIST_FILE, 'w') as f:
        json.dump(watchlist, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  watchlist 저장: {WATCHLIST_FILE} ({len(results[:50])}개)")
    
    return results


if __name__ == "__main__":
    main()
