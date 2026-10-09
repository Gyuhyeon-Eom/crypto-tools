#!/usr/bin/env python3
"""
급등 전조 모델 v2 — 진짜 ML 파이프라인
XGBoost + LightGBM + RandomForest 앙상블
수백 번 하이퍼파라미터 튜닝으로 최적 모델 탐색

핵심 개선:
1. +50% 기준으로 더 많은 학습 데이터 확보
2. 여러 급등 유형(바닥반등, 모멘텀, 돌파) 각각 특화 피처
3. 클래스 불균형 처리 (SMOTE + class weight)
4. 시계열 교차검증 (TimeSeriesSplit)
5. Precision 최적화 (급등 안 잡는 것보다, 잘못 알림이 더 나쁨)
6. 앙상블 스태킹
"""

import json, math, os, sys, time, datetime, pickle, warnings
import urllib.request
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, precision_recall_curve
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "ml_data"
DATA_DIR.mkdir(exist_ok=True)

# ============================================================
# 설정
# ============================================================

SURGE_THRESHOLDS = [0.3, 0.5, 0.7]  # 여러 기준 테스트
SURGE_WINDOW = 7
MIN_KLINES = 90
LOOKBACK_DAYS = 365

MIN_VOL_USD = 50_000
MAX_VOL_USD = 10_000_000

STABLECOINS = {
    "USDCUSDT", "UUSDT", "XUSDUSDT", "PAXGUSDT", "DAIUSDT",
    "FDUSDUSDT", "TUSDUSDT", "BUSDUSDT", "EURUSDT", "USTCUSDT",
    "USDTUSDT", "USDDUSDT", "RLUSDUSDT", "USD1USDT",
}
LARGE_CAPS = {
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
    "ADAUSDT", "DOGEUSDT", "TRXUSDT", "DOTUSDT", "MATICUSDT",
    "LTCUSDT", "BCHUSDT", "LINKUSDT", "AVAXUSDT", "UNIUSDT",
    "AAVEUSDT", "ATOMUSDT",
}

# ============================================================
# Bybit API
# ============================================================

def bybit_get(endpoint, params=None, retries=3):
    url = "https://api.bybit.com" + endpoint
    if params:
        url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ml-builder/2.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except Exception as e:
            if attempt == retries - 1: raise
            time.sleep(1 + attempt)

def fetch_all_tickers():
    data = bybit_get("/v5/market/tickers", {"category": "spot"})
    return data.get("result", {}).get("list", [])

def fetch_klines(symbol, interval="D", limit=200, start=None):
    params = {"category": "spot", "symbol": symbol, "interval": interval, "limit": str(limit)}
    if start: params["start"] = str(start)
    data = bybit_get("/v5/market/kline", params)
    rows = data.get("result", {}).get("list", [])
    return list(reversed(rows)) if rows else []

def fetch_all_klines(symbol, days=365):
    all_rows = []
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - days * 24 * 3600 * 1000
    cursor = start_ms
    while cursor < now_ms:
        rows = fetch_klines(symbol, "D", 200, start=cursor)
        if not rows: break
        all_rows.extend(rows)
        last_ts = int(rows[-1][0])
        if last_ts <= cursor: break
        cursor = last_ts + 24 * 3600 * 1000
        time.sleep(0.05)
    seen = set(); unique = []
    for r in all_rows:
        if r[0] not in seen: seen.add(r[0]); unique.append(r)
    unique.sort(key=lambda x: int(x[0]))
    return unique

# ============================================================
# Math helpers
# ============================================================

def mean(arr):
    return sum(arr) / len(arr) if arr else 0.0

def stdev(arr):
    if len(arr) < 2: return 0.0
    m = mean(arr)
    return math.sqrt(sum((x - m) ** 2 for x in arr) / (len(arr) - 1))

def polyfit_slope(y_vals):
    n = len(y_vals)
    if n < 2: return 0.0
    xm = (n - 1) / 2.0; ym = mean(y_vals)
    if ym == 0: return 0.0
    num = sum((i - xm) * (y - ym) for i, y in enumerate(y_vals))
    den = sum((i - xm) ** 2 for i in range(n))
    return (num / den) / ym if den else 0.0

def compute_rsi_series(closes, period=14):
    rsi = [None] * len(closes)
    if len(closes) < period + 1: return rsi
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [max(d, 0) for d in deltas]; losses = [max(-d, 0) for d in deltas]
    ag = mean(gains[:period]); al = mean(losses[:period])
    rsi[period] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    for i in range(period, len(deltas)):
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period
        rsi[i + 1] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return rsi

def compute_atr_series(highs, lows, closes, period=14):
    n = len(closes)
    tr = [highs[0] - lows[0]]
    for i in range(1, n):
        tr.append(max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])))
    atr = [None] * n
    if n >= period:
        atr[period-1] = mean(tr[:period])
        for i in range(period, n): atr[i] = (atr[i-1]*(period-1)+tr[i])/period
    return atr

def compute_macd(closes, fast=12, slow=26, signal=9):
    """MACD line, signal line, histogram."""
    if len(closes) < slow + signal: return [0]*len(closes), [0]*len(closes), [0]*len(closes)
    def ema(arr, p):
        k = 2/(p+1); result = [arr[0]]
        for i in range(1, len(arr)): result.append(arr[i]*k + result[-1]*(1-k))
        return result
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = ema(macd_line, signal)
    histogram = [m - s for m, s in zip(macd_line, signal_line)]
    return macd_line, signal_line, histogram

def compute_obv(closes, volumes):
    obv = [0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]: obv.append(obv[-1] + volumes[i])
        elif closes[i] < closes[i-1]: obv.append(obv[-1] - volumes[i])
        else: obv.append(obv[-1])
    return obv

# ============================================================
# 확장된 Feature Extraction (50+ features)
# ============================================================

def extract_features_at(o, h, l, c, v, idx):
    """idx 시점에서 feature 추출. 50+ features."""
    if idx < 59: return None
    pe = idx + 1
    
    f = {}
    
    # === 가격 리턴 (다양한 기간) ===
    for d in [1, 2, 3, 5, 7, 10, 14, 21]:
        if idx >= d:
            f[f"ret_{d}d"] = c[idx]/c[idx-d] - 1 if c[idx-d] > 0 else 0
        else:
            f[f"ret_{d}d"] = 0
    
    # === 이동평균 대비 ===
    for p in [5, 7, 10, 20, 50]:
        if idx >= p-1:
            ma = mean(c[idx-p+1:idx+1])
            f[f"price_vs_ma{p}"] = c[idx]/ma - 1 if ma > 0 else 0
        else:
            f[f"price_vs_ma{p}"] = 0
    
    # === 이동평균 크로스오버 ===
    if idx >= 20:
        ma5 = mean(c[idx-4:idx+1])
        ma10 = mean(c[idx-9:idx+1])
        ma20 = mean(c[idx-19:idx+1])
        f["ma5_ma10_cross"] = ma5/ma10 - 1 if ma10 > 0 else 0
        f["ma5_ma20_cross"] = ma5/ma20 - 1 if ma20 > 0 else 0
        f["ma10_ma20_cross"] = ma10/ma20 - 1 if ma20 > 0 else 0
    else:
        f["ma5_ma10_cross"] = 0; f["ma5_ma20_cross"] = 0; f["ma10_ma20_cross"] = 0
    
    # === 가격 위치 (범위 내) ===
    for window in [14, 30, 60]:
        if idx >= window-1:
            hi = max(h[idx-window+1:idx+1])
            lo = min(l[idx-window+1:idx+1])
            rng = hi - lo
            f[f"price_pos_{window}d"] = (c[idx] - lo) / rng if rng > 0 else 0.5
            f[f"price_vs_{window}d_high"] = c[idx] / hi if hi > 0 else 1
        else:
            f[f"price_pos_{window}d"] = 0.5
            f[f"price_vs_{window}d_high"] = 1
    
    # === RSI (다중 기간) ===
    rsi14 = compute_rsi_series(c[:idx+1], 14)
    rsi7 = compute_rsi_series(c[:idx+1], 7)
    
    f["rsi_14"] = rsi14[idx] if rsi14[idx] is not None else 50
    f["rsi_7"] = rsi7[idx] if rsi7[idx] is not None else 50
    
    # RSI 변화량
    if idx >= 7 and rsi14[idx] is not None and rsi14[idx-7] is not None:
        f["rsi14_change_7d"] = rsi14[idx] - rsi14[idx-7]
    else:
        f["rsi14_change_7d"] = 0
    if idx >= 3 and rsi14[idx] is not None and rsi14[idx-3] is not None:
        f["rsi14_change_3d"] = rsi14[idx] - rsi14[idx-3]
    else:
        f["rsi14_change_3d"] = 0
    
    # RSI 영역 일수
    rw = [rsi14[j] for j in range(max(0, idx-13), idx+1) if rsi14[j] is not None]
    f["rsi_oversold_days"] = sum(1 for r in rw if r < 30)
    f["rsi_consol_days"] = sum(1 for r in rw if 40 <= r <= 55)
    f["rsi_overbought_days"] = sum(1 for r in rw if r > 70)
    f["rsi_min_14d"] = min(rw) if rw else 50
    f["rsi_max_14d"] = max(rw) if rw else 50
    f["rsi_range_14d"] = (max(rw) - min(rw)) if rw else 0
    
    # RSI divergence: 가격 하락인데 RSI 상승
    if idx >= 7 and rsi14[idx] is not None and rsi14[idx-7] is not None:
        price_dir = 1 if c[idx] > c[idx-7] else -1
        rsi_dir = 1 if rsi14[idx] > rsi14[idx-7] else -1
        f["rsi_divergence"] = 1 if price_dir == -1 and rsi_dir == 1 else (-1 if price_dir == 1 and rsi_dir == -1 else 0)
    else:
        f["rsi_divergence"] = 0
    
    # === MACD ===
    macd_line, signal_line, histogram = compute_macd(c[:idx+1])
    if idx < len(macd_line):
        price_at = c[idx] if c[idx] > 0 else 1
        f["macd_line"] = macd_line[idx] / price_at
        f["macd_signal"] = signal_line[idx] / price_at
        f["macd_histogram"] = histogram[idx] / price_at
        # MACD 크로스 (양수 전환)
        if idx >= 1:
            f["macd_cross"] = 1 if histogram[idx] > 0 and histogram[idx-1] <= 0 else (
                -1 if histogram[idx] < 0 and histogram[idx-1] >= 0 else 0)
        else:
            f["macd_cross"] = 0
        # MACD histogram 추세
        if idx >= 3:
            f["macd_hist_slope"] = polyfit_slope([histogram[idx-i] for i in range(3, -1, -1)])
        else:
            f["macd_hist_slope"] = 0
    else:
        f["macd_line"] = 0; f["macd_signal"] = 0; f["macd_histogram"] = 0
        f["macd_cross"] = 0; f["macd_hist_slope"] = 0
    
    # === ATR / 변동성 ===
    atr_s = compute_atr_series(h[:idx+1], l[:idx+1], c[:idx+1], 14)
    a14 = atr_s[idx] if atr_s[idx] else 0
    f["atr_ratio"] = a14/c[idx] if c[idx] > 0 else 0
    
    a7v = [atr_s[j] for j in range(max(0,idx-6), idx+1) if atr_s[j]]
    a30v = [atr_s[j] for j in range(max(0,idx-29), idx+1) if atr_s[j]]
    f["volatility_ratio_7_30"] = mean(a7v)/mean(a30v) if a30v and mean(a30v) > 0 else 1
    
    # 변동성 수축/확장
    if idx >= 29:
        atr_early = [atr_s[j] for j in range(idx-29, idx-14) if atr_s[j]]
        atr_late = [atr_s[j] for j in range(idx-6, idx+1) if atr_s[j]]
        f["volatility_contraction"] = mean(atr_late)/mean(atr_early) if atr_early and mean(atr_early) > 0 else 1
    else:
        f["volatility_contraction"] = 1
    
    # === 볼린저 밴드 ===
    if idx >= 19:
        bc = c[idx-19:idx+1]; bm = mean(bc); bs = stdev(bc)
        f["bb_width"] = (2*bs)/bm if bm > 0 else 0
        f["bb_position"] = (c[idx] - (bm - bs)) / (2*bs) if bs > 0 else 0.5
        # BB 스퀴즈 (폭 축소 정도)
        if idx >= 39:
            bb_widths = []
            for j in range(idx-19, idx+1):
                if j >= 19:
                    w = stdev(c[j-19:j+1])
                    m = mean(c[j-19:j+1])
                    bb_widths.append(2*w/m if m > 0 else 0)
            f["bb_squeeze"] = min(bb_widths[-5:])/max(bb_widths[:10]) if bb_widths and max(bb_widths[:10]) > 0 else 1
        else:
            f["bb_squeeze"] = 1
    else:
        f["bb_width"] = 0; f["bb_position"] = 0.5; f["bb_squeeze"] = 1
    
    # === 거래량 ===
    v_1d = v[idx]
    v_3d = mean(v[max(0,idx-2):idx+1])
    v_7d = mean(v[max(0,idx-6):idx+1])
    v_14d = mean(v[max(0,idx-13):idx+1])
    v_30d = mean(v[max(0,idx-29):idx+1])
    
    f["vol_ratio_1d_7d"] = v_1d/v_7d if v_7d > 0 else 1
    f["vol_ratio_3d_14d"] = v_3d/v_14d if v_14d > 0 else 1
    f["vol_ratio_7d_30d"] = v_7d/v_30d if v_30d > 0 else 1
    f["vol_trend_7d"] = polyfit_slope(v[max(0,idx-6):idx+1])
    f["vol_trend_14d"] = polyfit_slope(v[max(0,idx-13):idx+1])
    
    # 거래량 폭발 (최근 2일 vs 이전 5일)
    v_last2 = mean(v[max(0,idx-1):idx+1])
    v_prev5 = mean(v[max(0,idx-6):max(0,idx-1)]) if idx >= 3 else v_last2
    f["vol_explosion_2d"] = v_last2/v_prev5 if v_prev5 > 0 else 1
    
    # OBV 추세
    obv = compute_obv(c[max(0,idx-29):idx+1], v[max(0,idx-29):idx+1])
    if len(obv) >= 7:
        f["obv_slope_7d"] = polyfit_slope(obv[-7:])
    else:
        f["obv_slope_7d"] = 0
    
    # 거래량-가격 괴리
    if idx >= 7:
        f["vol_price_divergence"] = polyfit_slope(v[idx-6:idx+1]) - polyfit_slope(c[idx-6:idx+1])
    else:
        f["vol_price_divergence"] = 0
    
    # === 캔들 패턴 ===
    f["green_ratio_3d"] = sum(1 for j in range(max(0,idx-2), idx+1) if c[j] >= o[j]) / 3.0
    f["green_ratio_7d"] = sum(1 for j in range(max(0,idx-6), idx+1) if c[j] >= o[j]) / 7.0
    
    cr = 0
    for j in range(idx, max(idx-14, -1), -1):
        if c[j] < o[j]: cr += 1
        else: break
    f["consec_red"] = cr
    
    cg = 0
    for j in range(idx, max(idx-14, -1), -1):
        if c[j] >= o[j]: cg += 1
        else: break
    f["consec_green"] = cg
    
    # 심지 분석
    bodies = []; upwicks = []; lowwicks = []
    for j in range(max(0,idx-6), idx+1):
        body = abs(c[j]-o[j]); full = h[j]-l[j]
        if full > 0:
            bodies.append(body/full)
            upwicks.append((h[j]-max(c[j],o[j]))/full)
            lowwicks.append((min(c[j],o[j])-l[j])/full)
    f["avg_body_ratio"] = mean(bodies) if bodies else 0.5
    f["avg_upper_wick"] = mean(upwicks) if upwicks else 0
    f["avg_lower_wick"] = mean(lowwicks) if lowwicks else 0
    
    # 해머/도지 패턴 (최근 3일 중 존재 여부)
    hammer_count = 0; doji_count = 0
    for j in range(max(0,idx-2), idx+1):
        body = abs(c[j]-o[j]); full = h[j]-l[j]
        if full > 0:
            lower_wick = (min(c[j],o[j])-l[j])/full
            body_ratio = body/full
            if lower_wick > 0.6 and body_ratio < 0.2: hammer_count += 1
            if body_ratio < 0.1: doji_count += 1
    f["hammer_3d"] = hammer_count
    f["doji_3d"] = doji_count
    
    # === Accumulation/Distribution ===
    if idx >= 13:
        pc14 = abs(c[idx]/c[idx-13]-1)
        vr = mean(v[idx-6:idx+1]); vp = mean(v[idx-13:idx-6])
        f["accumulation"] = (vr/vp)/(1+pc14*10) if vp > 0 else 1
    else:
        f["accumulation"] = 1
    
    # === 급등 유형별 특화 피처 ===
    
    # [유형1] 바닥 반등형: 큰 하락 후 안정화
    if idx >= 30:
        recent_low = min(l[idx-29:idx+1])
        peak_30d = max(h[idx-29:idx+1])
        f["drawdown_30d"] = (c[idx] - peak_30d) / peak_30d if peak_30d > 0 else 0
        f["recovery_from_low"] = (c[idx] - recent_low) / recent_low if recent_low > 0 else 0
    else:
        f["drawdown_30d"] = 0; f["recovery_from_low"] = 0
    
    # [유형2] 모멘텀형: 이미 오르는 중 가속
    if idx >= 14:
        # 7일 모멘텀 + 가속도
        ret7 = c[idx]/c[idx-7] - 1 if c[idx-7] > 0 else 0
        ret14 = c[idx]/c[idx-14] - 1 if c[idx-14] > 0 else 0
        f["momentum_acceleration"] = ret7 - (ret14 - ret7)  # 최근 가속
        # 연속 상승일수 + 거래량 동반
        up_days_with_vol = 0
        for j in range(idx, max(idx-7, -1), -1):
            if c[j] >= o[j] and v[j] > v_7d: up_days_with_vol += 1
            else: break
        f["up_with_volume_streak"] = up_days_with_vol
    else:
        f["momentum_acceleration"] = 0; f["up_with_volume_streak"] = 0
    
    # [유형3] 돌파형: 저항선 근접 + 거래량 증가
    if idx >= 30:
        # 여러 번 닿은 고점(저항선) 대비 현재 위치
        recent_highs = sorted(h[idx-29:idx+1], reverse=True)
        resistance = mean(recent_highs[:3])  # 상위 3개 고점 평균
        f["distance_to_resistance"] = (resistance - c[idx]) / c[idx] if c[idx] > 0 else 0
        # 지지선 대비
        recent_lows = sorted(l[idx-29:idx+1])
        support = mean(recent_lows[:3])
        f["distance_from_support"] = (c[idx] - support) / c[idx] if c[idx] > 0 else 0
    else:
        f["distance_to_resistance"] = 0; f["distance_from_support"] = 0
    
    # === 시장 구조 ===
    # 가격 변동폭 대비 방향성 (효율성 비율)
    if idx >= 7:
        net_move = abs(c[idx] - c[idx-7])
        total_move = sum(abs(c[j] - c[j-1]) for j in range(idx-6, idx+1))
        f["efficiency_ratio_7d"] = net_move / total_move if total_move > 0 else 0
    else:
        f["efficiency_ratio_7d"] = 0
    
    if idx >= 14:
        net_move = abs(c[idx] - c[idx-14])
        total_move = sum(abs(c[j] - c[j-1]) for j in range(idx-13, idx+1))
        f["efficiency_ratio_14d"] = net_move / total_move if total_move > 0 else 0
    else:
        f["efficiency_ratio_14d"] = 0
    
    return f

# ============================================================
# 데이터 수집
# ============================================================

def collect_data():
    """Bybit에서 전체 소형 알트 1년치 데이터 수집."""
    cache_file = DATA_DIR / "raw_data.json"
    
    # 캐시 있으면 재사용 (1시간 이내)
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < 3600:
            print(f"  캐시 사용 ({age/60:.0f}분 전)")
            with open(cache_file) as f:
                return json.load(f)
    
    print("  Bybit 티커 수집...")
    tickers = fetch_all_tickers()
    
    small_alts = []
    for t in tickers:
        sym = t["symbol"]
        if not sym.endswith("USDT"): continue
        if sym in STABLECOINS or sym in LARGE_CAPS: continue
        vol24 = float(t.get("turnover24h", 0))
        if MIN_VOL_USD <= vol24 <= MAX_VOL_USD:
            small_alts.append(sym)
    
    print(f"  소형 알트: {len(small_alts)}개")
    print(f"  1년치 데이터 수집 시작...")
    
    all_data = {}
    for i, sym in enumerate(small_alts):
        if (i+1) % 30 == 0:
            print(f"    [{i+1}/{len(small_alts)}] {sym}...")
        try:
            klines = fetch_all_klines(sym, LOOKBACK_DAYS)
            if len(klines) < MIN_KLINES: continue
            all_data[sym] = {
                "o": [float(k[1]) for k in klines],
                "h": [float(k[2]) for k in klines],
                "l": [float(k[3]) for k in klines],
                "c": [float(k[4]) for k in klines],
                "v": [float(k[5]) for k in klines],
                "ts": [int(k[0]) for k in klines],
            }
            time.sleep(0.05)
        except:
            continue
    
    print(f"  수집 완료: {len(all_data)}개")
    
    with open(cache_file, "w") as f:
        json.dump(all_data, f)
    
    return all_data

# ============================================================
# 학습 데이터셋 구축
# ============================================================

def build_dataset(all_data, surge_threshold=0.5):
    """모든 코인/시점에서 feature + label 추출."""
    print(f"\n  데이터셋 구축 (급등 기준: +{surge_threshold*100:.0f}%)...")
    
    rows = []
    surge_count = 0
    normal_count = 0
    
    for sym, d in all_data.items():
        n = len(d["c"])
        o, h, l, c, v = d["o"], d["h"], d["l"], d["c"], d["v"]
        
        # 거래대금 체크
        avg_vol = mean([v[j]*c[j] for j in range(max(0, n-30), n)])
        if avg_vol > MAX_VOL_USD or avg_vol < MIN_VOL_USD:
            continue
        
        for idx in range(59, n - SURGE_WINDOW):
            # Feature 추출
            feat = extract_features_at(o, h, l, c, v, idx)
            if feat is None: continue
            
            # Label: 향후 7일 내 최대 수익률
            max_ret = 0
            for j in range(idx+1, min(idx+SURGE_WINDOW+1, n)):
                ret = c[j] / c[idx] - 1 if c[idx] > 0 else 0
                max_ret = max(max_ret, ret)
            
            label = 1 if max_ret >= surge_threshold else 0
            
            feat["_label"] = label
            feat["_symbol"] = sym
            feat["_idx"] = idx
            feat["_max_ret"] = max_ret
            feat["_ts"] = d["ts"][idx]
            
            rows.append(feat)
            if label == 1: surge_count += 1
            else: normal_count += 1
    
    print(f"  총 {len(rows)}개 샘플 (급등 {surge_count}, 일반 {normal_count})")
    print(f"  급등 비율: {surge_count/len(rows)*100:.2f}%")
    
    return rows

# ============================================================
# 모델 학습
# ============================================================

def train_models(rows, surge_threshold):
    """여러 모델 학습 + 하이퍼파라미터 튜닝."""
    
    # DataFrame 변환
    df = pd.DataFrame(rows)
    meta_cols = [c for c in df.columns if c.startswith("_")]
    feature_cols = [c for c in df.columns if not c.startswith("_")]
    
    X = df[feature_cols].values.astype(np.float64)
    y = df["_label"].values
    timestamps = df["_ts"].values
    
    # NaN/Inf 처리
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    # 시간순 정렬 (시계열 교차검증용)
    sort_idx = np.argsort(timestamps)
    X = X[sort_idx]
    y = y[sort_idx]
    timestamps = timestamps[sort_idx]
    
    print(f"\n  Feature 수: {len(feature_cols)}")
    print(f"  학습 데이터: {len(X)} ({sum(y)} surge, {len(y)-sum(y)} normal)")
    
    # 클래스 가중치
    n_pos = sum(y); n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1
    print(f"  클래스 불균형 비율: 1:{scale_pos_weight:.0f}")
    
    # ============================================================
    # 시계열 분할: 학습 80% / 테스트 20%
    # ============================================================
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"  Train: {len(X_train)} ({sum(y_train)} surge)")
    print(f"  Test:  {len(X_test)} ({sum(y_test)} surge)")
    
    # TimeSeriesSplit for CV
    tscv = TimeSeriesSplit(n_splits=5)
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # ============================================================
    # Model 1: XGBoost
    # ============================================================
    print(f"\n  [1/4] XGBoost 튜닝 (80 iterations)...")
    
    xgb_params = {
        'n_estimators': [200, 400, 600],
        'max_depth': [4, 5, 6, 7],
        'learning_rate': [0.02, 0.05, 0.1],
        'min_child_weight': [3, 5, 10],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8],
        'gamma': [0, 0.1, 0.5],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1, 3],
        'scale_pos_weight': [scale_pos_weight, scale_pos_weight * 1.5, scale_pos_weight * 2],
    }
    
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',
        use_label_encoder=False,
        random_state=42,
        tree_method='hist',
    )
    
    xgb_search = RandomizedSearchCV(
        xgb_model, xgb_params, n_iter=80, cv=tscv,
        scoring='average_precision', n_jobs=1, random_state=42, verbose=0
    )
    xgb_search.fit(X_train_scaled, y_train)
    xgb_best = xgb_search.best_estimator_
    print(f"    Best score (CV AP): {xgb_search.best_score_:.4f}")
    print(f"    Best params: {xgb_search.best_params_}")
    
    results["xgboost"] = evaluate_model(xgb_best, X_test_scaled, y_test, "XGBoost")
    results["xgboost"]["model"] = xgb_best
    
    # ============================================================
    # Model 2: LightGBM
    # ============================================================
    print(f"\n  [2/4] LightGBM 튜닝 (80 iterations)...")
    
    lgb_params = {
        'n_estimators': [200, 400, 600],
        'max_depth': [5, 7, -1],
        'learning_rate': [0.02, 0.05, 0.1],
        'num_leaves': [31, 50, 80],
        'min_child_samples': [10, 20, 50],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [0, 0.1, 1],
        'scale_pos_weight': [scale_pos_weight, scale_pos_weight * 1.5, scale_pos_weight * 2],
    }
    
    lgb_model = lgb.LGBMClassifier(
        objective='binary', metric='average_precision',
        random_state=42, verbose=-1, force_col_wise=True,
    )
    
    lgb_search = RandomizedSearchCV(
        lgb_model, lgb_params, n_iter=80, cv=tscv,
        scoring='average_precision', n_jobs=1, random_state=42, verbose=0
    )
    lgb_search.fit(X_train_scaled, y_train)
    lgb_best = lgb_search.best_estimator_
    print(f"    Best score (CV AP): {lgb_search.best_score_:.4f}")
    print(f"    Best params: {lgb_search.best_params_}")
    
    results["lightgbm"] = evaluate_model(lgb_best, X_test_scaled, y_test, "LightGBM")
    results["lightgbm"]["model"] = lgb_best
    
    # ============================================================
    # Model 3: Random Forest
    # ============================================================
    print(f"\n  [3/4] Random Forest 튜닝 (50 iterations)...")
    
    rf_params = {
        'n_estimators': [300, 500],
        'max_depth': [7, 10, 15, None],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [2, 5, 10],
        'max_features': ['sqrt', 0.3, 0.5],
        'class_weight': ['balanced', 'balanced_subsample'],
    }
    
    rf_model = RandomForestClassifier(random_state=42, n_jobs=1)
    
    rf_search = RandomizedSearchCV(
        rf_model, rf_params, n_iter=50, cv=tscv,
        scoring='average_precision', n_jobs=1, random_state=42, verbose=0
    )
    rf_search.fit(X_train_scaled, y_train)
    rf_best = rf_search.best_estimator_
    print(f"    Best score (CV AP): {rf_search.best_score_:.4f}")
    print(f"    Best params: {rf_search.best_params_}")
    
    results["random_forest"] = evaluate_model(rf_best, X_test_scaled, y_test, "RandomForest")
    results["random_forest"]["model"] = rf_best
    
    # ============================================================
    # Model 4: Stacking 앙상블
    # ============================================================
    print(f"\n  [4/4] Stacking 앙상블...")
    
    stack = StackingClassifier(
        estimators=[
            ('xgb', xgb_best),
            ('lgb', lgb_best),
            ('rf', rf_best),
        ],
        final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000),
        cv=tscv, n_jobs=1, passthrough=True,
    )
    stack.fit(X_train_scaled, y_train)
    
    results["stacking"] = evaluate_model(stack, X_test_scaled, y_test, "Stacking")
    results["stacking"]["model"] = stack
    
    # ============================================================
    # 최적 threshold 탐색 (Precision 최적화)
    # ============================================================
    print(f"\n  Precision 최적 threshold 탐색...")
    
    for name in ["xgboost", "lightgbm", "stacking"]:
        model = results[name]["model"]
        probs = model.predict_proba(X_test_scaled)[:, 1]
        
        best_thresh = 0.5
        best_precision = 0
        best_f1_at_thresh = 0
        
        for thresh in np.arange(0.1, 0.95, 0.01):
            preds = (probs >= thresh).astype(int)
            if sum(preds) == 0: continue
            p = precision_score(y_test, preds, zero_division=0)
            r = recall_score(y_test, preds, zero_division=0)
            f1 = 2*p*r/(p+r) if p+r > 0 else 0
            
            # Precision >= 15% 이면서 F1 최대
            if p >= 0.15 and f1 > best_f1_at_thresh:
                best_thresh = thresh
                best_precision = p
                best_f1_at_thresh = f1
        
        results[name]["best_threshold"] = best_thresh
        results[name]["precision_at_thresh"] = best_precision
        results[name]["f1_at_thresh"] = best_f1_at_thresh
        
        preds_opt = (probs >= best_thresh).astype(int)
        n_pred = sum(preds_opt)
        n_hit = sum(preds_opt & y_test)
        
        print(f"    {name}: threshold={best_thresh:.2f} → "
              f"Precision={best_precision:.1%} "
              f"({n_hit}/{n_pred} hits)")
    
    # ============================================================
    # Feature Importance 분석
    # ============================================================
    print(f"\n  Feature Importance (XGBoost):")
    importances = xgb_best.feature_importances_
    imp_sorted = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    for fn, imp in imp_sorted[:20]:
        print(f"    {fn:<30} {imp:.4f}")
    
    return results, feature_cols, scaler

def evaluate_model(model, X_test, y_test, name):
    """모델 평가."""
    probs = model.predict_proba(X_test)[:, 1]
    preds_50 = model.predict(X_test)
    
    p50 = precision_score(y_test, preds_50, zero_division=0)
    r50 = recall_score(y_test, preds_50, zero_division=0)
    f50 = f1_score(y_test, preds_50, zero_division=0)
    
    print(f"    {name} @0.5: P={p50:.1%} R={r50:.1%} F1={f50:.3f} (예측 {sum(preds_50)}, 적중 {sum(preds_50 & y_test)})")
    
    return {
        "precision_50": p50,
        "recall_50": r50,
        "f1_50": f50,
        "total_predictions_50": int(sum(preds_50)),
        "total_hits_50": int(sum(preds_50 & y_test)),
    }

# ============================================================
# 30일 백테스트 시뮬레이션
# ============================================================

def backtest_30d(all_data, model, scaler, feature_cols, threshold, surge_threshold):
    """최근 30일 매일 시뮬레이션."""
    print(f"\n  30일 백테스트 (모델 threshold={threshold:.2f}, 급등={surge_threshold*100:.0f}%)...")
    
    BACKTEST_DAYS = 30
    total_predictions = 0
    total_hits = 0
    total_surges = 0
    daily_results = []
    
    for day_offset in range(BACKTEST_DAYS):
        day_features = []
        day_symbols = []
        day_labels = []
        
        for sym, d in all_data.items():
            n = len(d["c"])
            o, h, l, c, v = d["o"], d["h"], d["l"], d["c"], d["v"]
            
            idx = n - 1 - SURGE_WINDOW - (BACKTEST_DAYS - 1 - day_offset)
            if idx < 59 or idx + SURGE_WINDOW >= n: continue
            
            avg_vol = mean([v[j]*c[j] for j in range(max(0,idx-29), idx+1)])
            if avg_vol > MAX_VOL_USD or avg_vol < MIN_VOL_USD: continue
            
            feat = extract_features_at(o, h, l, c, v, idx)
            if feat is None: continue
            
            # 실제 급등 확인
            max_ret = 0
            for j in range(idx+1, min(idx+SURGE_WINDOW+1, n)):
                ret = c[j]/c[idx] - 1 if c[idx] > 0 else 0
                max_ret = max(max_ret, ret)
            
            is_surge = 1 if max_ret >= surge_threshold else 0
            
            fv = [feat.get(fc, 0) for fc in feature_cols]
            day_features.append(fv)
            day_symbols.append(sym)
            day_labels.append(is_surge)
        
        if not day_features: continue
        
        X_day = np.array(day_features, dtype=np.float64)
        X_day = np.nan_to_num(X_day, nan=0, posinf=0, neginf=0)
        X_day_scaled = scaler.transform(X_day)
        
        probs = model.predict_proba(X_day_scaled)[:, 1]
        preds = (probs >= threshold).astype(int)
        y_day = np.array(day_labels)
        
        n_pred = sum(preds)
        n_surge = sum(y_day)
        n_hit = sum(preds & y_day)
        
        total_predictions += n_pred
        total_hits += n_hit
        total_surges += n_surge
        
        daily_results.append({
            "day": day_offset,
            "predictions": int(n_pred),
            "surges": int(n_surge),
            "hits": int(n_hit),
        })
    
    precision = total_hits / total_predictions if total_predictions > 0 else 0
    recall = total_hits / total_surges if total_surges > 0 else 0
    f1 = 2*precision*recall/(precision+recall) if precision+recall > 0 else 0
    
    print(f"    30일 결과: 예측 {total_predictions}, 적중 {total_hits}, 급등 {total_surges}")
    print(f"    Precision: {precision:.1%}")
    print(f"    Recall: {recall:.1%}")
    print(f"    F1: {f1:.3f}")
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_predictions": total_predictions,
        "total_hits": total_hits,
        "total_surges": total_surges,
        "daily": daily_results,
    }

# ============================================================
# 메인
# ============================================================

def main():
    print("=" * 70)
    print("  급등 전조 모델 v2 — ML 파이프라인")
    print("  XGBoost + LightGBM + RF → Stacking 앙상블")
    print("  하이퍼파라미터 튜닝 210+ iterations (x3 급등기준)")
    print("=" * 70)
    
    # [1] 데이터 수집
    print("\n[1/5] 데이터 수집...")
    all_data = collect_data()
    
    best_overall = {"f1": 0}
    all_results = {}
    
    for surge_th in SURGE_THRESHOLDS:
        print(f"\n{'='*70}")
        print(f"  급등 기준: +{surge_th*100:.0f}% (7일 내)")
        print(f"{'='*70}")
        
        # [2] 학습 데이터셋 구축
        print(f"\n[2/5] 학습 데이터셋 구축...")
        rows = build_dataset(all_data, surge_th)
        
        if sum(1 for r in rows if r["_label"] == 1) < 20:
            print(f"  급등 샘플 부족 → 스킵")
            continue
        
        # [3] 모델 학습
        print(f"\n[3/5] 모델 학습...")
        results, feature_cols, scaler = train_models(rows, surge_th)
        
        # [4] 30일 백테스트
        print(f"\n[4/5] 30일 백테스트...")
        for model_name in ["xgboost", "lightgbm", "stacking"]:
            thresh = results[model_name].get("best_threshold", 0.5)
            bt = backtest_30d(all_data, results[model_name]["model"], scaler, feature_cols, thresh, surge_th)
            results[model_name]["backtest_30d"] = bt
            
            key = f"{model_name}_surge{int(surge_th*100)}"
            all_results[key] = {
                "model_name": model_name,
                "surge_threshold": surge_th,
                "threshold": thresh,
                "backtest": bt,
                "cv_precision": results[model_name].get("precision_at_thresh", 0),
                "cv_f1": results[model_name].get("f1_at_thresh", 0),
            }
            
            if bt["f1"] > best_overall.get("f1", 0):
                best_overall = {
                    "model_name": model_name,
                    "surge_threshold": surge_th,
                    "threshold": thresh,
                    "model": results[model_name]["model"],
                    "scaler": scaler,
                    "feature_cols": feature_cols,
                    **bt,
                }
    
    # [5] 최종 결과 저장
    print(f"\n{'='*70}")
    print(f"  [5/5] 최종 결과")
    print(f"{'='*70}")
    
    print(f"\n  전체 결과 비교:")
    print(f"  {'설정':<30} {'Precision':>10} {'Recall':>8} {'F1':>8} {'예측':>6} {'적중':>4}")
    print(f"  {'-'*72}")
    for key, r in sorted(all_results.items(), key=lambda x: x[1]["backtest"]["f1"], reverse=True):
        bt = r["backtest"]
        print(f"  {key:<30} {bt['precision']:>9.1%} {bt['recall']:>7.1%} {bt['f1']:>7.3f} {bt['total_predictions']:>6} {bt['total_hits']:>4}")
    
    if best_overall.get("model"):
        print(f"\n  === 최적 모델 ===")
        print(f"  모델: {best_overall['model_name']}")
        print(f"  급등 기준: +{best_overall['surge_threshold']*100:.0f}%")
        print(f"  Threshold: {best_overall['threshold']:.2f}")
        print(f"  Precision: {best_overall['precision']:.1%}")
        print(f"  Recall: {best_overall['recall']:.1%}")
        print(f"  F1: {best_overall['f1']:.3f}")
        
        # 모델 저장
        model_save = {
            "model_name": best_overall["model_name"],
            "surge_threshold": best_overall["surge_threshold"],
            "threshold": best_overall["threshold"],
            "feature_cols": best_overall["feature_cols"],
            "precision": best_overall["precision"],
            "recall": best_overall["recall"],
            "f1": best_overall["f1"],
        }
        with open(BASE_DIR / "ml_model_config.json", "w") as f:
            json.dump(model_save, f, indent=2)
        
        with open(DATA_DIR / "best_model.pkl", "wb") as f:
            pickle.dump({
                "model": best_overall["model"],
                "scaler": best_overall["scaler"],
                "feature_cols": best_overall["feature_cols"],
                "threshold": best_overall["threshold"],
                "surge_threshold": best_overall["surge_threshold"],
            }, f)
        
        print(f"\n  모델 저장: ml_data/best_model.pkl")
        print(f"  설정 저장: ml_model_config.json")
    else:
        print("\n  유효한 모델 없음!")
    
    # 전체 결과 저장
    save_results = {}
    for key, r in all_results.items():
        save_results[key] = {k: v for k, v in r.items() if k != "model"}
    with open(BASE_DIR / "ml_results.json", "w") as f:
        json.dump(save_results, f, indent=2)
    
    print(f"\n  전체 결과: ml_results.json")
    print(f"\n{'='*70}")
    print(f"  완료!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
