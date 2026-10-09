#!/usr/bin/env python3
"""
급등 전조 모델 v3 — joblib 없는 ML 파이프라인
XGBoost + LightGBM + RF → Soft Voting 앙상블
수동 하이퍼파라미터 튜닝 (RandomizedSearchCV 대신 직접 루프)

모든 병렬 처리 제거 → 안정적 실행
"""

import json, math, os, sys, time, datetime, pickle, warnings, random, itertools
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "ml_data"
DATA_DIR.mkdir(exist_ok=True)

# ============================================================
# 설정
# ============================================================

SURGE_THRESHOLDS = [0.3, 0.5]  # +30%, +50%
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
            req = urllib.request.Request(url, headers={"User-Agent": "ml-builder/3.0"})
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
    if len(closes) < slow + signal: return [0]*len(closes), [0]*len(closes), [0]*len(closes)
    def ema(arr, p):
        k = 2/(p+1); result = [arr[0]]
        for i in range(1, len(arr)): result.append(arr[i]*k + result[-1]*(1-k))
        return result
    ema_fast = ema(closes, fast); ema_slow = ema(closes, slow)
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
# Feature Extraction (50+ features)
# ============================================================

def extract_features_at(o, h, l, c, v, idx):
    if idx < 59: return None
    f = {}
    
    # 가격 리턴
    for d in [1, 2, 3, 5, 7, 10, 14, 21]:
        f[f"ret_{d}d"] = c[idx]/c[idx-d] - 1 if idx >= d and c[idx-d] > 0 else 0
    
    # 이동평균 대비
    for p in [5, 7, 10, 20, 50]:
        if idx >= p-1:
            ma = mean(c[idx-p+1:idx+1])
            f[f"price_vs_ma{p}"] = c[idx]/ma - 1 if ma > 0 else 0
        else:
            f[f"price_vs_ma{p}"] = 0
    
    # MA 크로스
    if idx >= 20:
        ma5 = mean(c[idx-4:idx+1]); ma10 = mean(c[idx-9:idx+1]); ma20 = mean(c[idx-19:idx+1])
        f["ma5_ma10_cross"] = ma5/ma10 - 1 if ma10 > 0 else 0
        f["ma5_ma20_cross"] = ma5/ma20 - 1 if ma20 > 0 else 0
        f["ma10_ma20_cross"] = ma10/ma20 - 1 if ma20 > 0 else 0
    else:
        f["ma5_ma10_cross"] = 0; f["ma5_ma20_cross"] = 0; f["ma10_ma20_cross"] = 0
    
    # 가격 위치
    for window in [14, 30, 60]:
        if idx >= window-1:
            hi = max(h[idx-window+1:idx+1]); lo = min(l[idx-window+1:idx+1])
            rng = hi - lo
            f[f"price_pos_{window}d"] = (c[idx] - lo) / rng if rng > 0 else 0.5
            f[f"price_vs_{window}d_high"] = c[idx] / hi if hi > 0 else 1
        else:
            f[f"price_pos_{window}d"] = 0.5; f[f"price_vs_{window}d_high"] = 1
    
    # RSI
    rsi14 = compute_rsi_series(c[:idx+1], 14)
    rsi7 = compute_rsi_series(c[:idx+1], 7)
    f["rsi_14"] = rsi14[idx] if rsi14[idx] is not None else 50
    f["rsi_7"] = rsi7[idx] if rsi7[idx] is not None else 50
    
    if idx >= 7 and rsi14[idx] is not None and rsi14[idx-7] is not None:
        f["rsi14_change_7d"] = rsi14[idx] - rsi14[idx-7]
    else: f["rsi14_change_7d"] = 0
    if idx >= 3 and rsi14[idx] is not None and rsi14[idx-3] is not None:
        f["rsi14_change_3d"] = rsi14[idx] - rsi14[idx-3]
    else: f["rsi14_change_3d"] = 0
    
    rw = [rsi14[j] for j in range(max(0, idx-13), idx+1) if rsi14[j] is not None]
    f["rsi_oversold_days"] = sum(1 for r in rw if r < 30)
    f["rsi_consol_days"] = sum(1 for r in rw if 40 <= r <= 55)
    f["rsi_overbought_days"] = sum(1 for r in rw if r > 70)
    f["rsi_min_14d"] = min(rw) if rw else 50
    f["rsi_max_14d"] = max(rw) if rw else 50
    f["rsi_range_14d"] = (max(rw) - min(rw)) if rw else 0
    
    # RSI divergence
    if idx >= 7 and rsi14[idx] is not None and rsi14[idx-7] is not None:
        price_dir = 1 if c[idx] > c[idx-7] else -1
        rsi_dir = 1 if rsi14[idx] > rsi14[idx-7] else -1
        f["rsi_divergence"] = 1 if price_dir == -1 and rsi_dir == 1 else (-1 if price_dir == 1 and rsi_dir == -1 else 0)
    else: f["rsi_divergence"] = 0
    
    # MACD
    macd_line, signal_line, histogram = compute_macd(c[:idx+1])
    if idx < len(macd_line):
        pa = c[idx] if c[idx] > 0 else 1
        f["macd_line"] = macd_line[idx] / pa
        f["macd_signal"] = signal_line[idx] / pa
        f["macd_histogram"] = histogram[idx] / pa
        f["macd_cross"] = 1 if idx >= 1 and histogram[idx] > 0 and histogram[idx-1] <= 0 else (
            -1 if idx >= 1 and histogram[idx] < 0 and histogram[idx-1] >= 0 else 0)
        f["macd_hist_slope"] = polyfit_slope([histogram[idx-i] for i in range(min(4,idx+1)-1, -1, -1)]) if idx >= 3 else 0
    else:
        f["macd_line"] = 0; f["macd_signal"] = 0; f["macd_histogram"] = 0
        f["macd_cross"] = 0; f["macd_hist_slope"] = 0
    
    # ATR / 변동성
    atr_s = compute_atr_series(h[:idx+1], l[:idx+1], c[:idx+1], 14)
    a14 = atr_s[idx] if atr_s[idx] else 0
    f["atr_ratio"] = a14/c[idx] if c[idx] > 0 else 0
    a7v = [atr_s[j] for j in range(max(0,idx-6), idx+1) if atr_s[j]]
    a30v = [atr_s[j] for j in range(max(0,idx-29), idx+1) if atr_s[j]]
    f["volatility_ratio_7_30"] = mean(a7v)/mean(a30v) if a30v and mean(a30v) > 0 else 1
    
    if idx >= 29:
        atr_early = [atr_s[j] for j in range(idx-29, idx-14) if atr_s[j]]
        atr_late = [atr_s[j] for j in range(idx-6, idx+1) if atr_s[j]]
        f["volatility_contraction"] = mean(atr_late)/mean(atr_early) if atr_early and mean(atr_early) > 0 else 1
    else: f["volatility_contraction"] = 1
    
    # 볼린저
    if idx >= 19:
        bc = c[idx-19:idx+1]; bm = mean(bc); bs = stdev(bc)
        f["bb_width"] = (2*bs)/bm if bm > 0 else 0
        f["bb_position"] = (c[idx] - (bm - bs)) / (2*bs) if bs > 0 else 0.5
    else: f["bb_width"] = 0; f["bb_position"] = 0.5
    
    # 거래량
    v_7d = mean(v[max(0,idx-6):idx+1])
    v_14d = mean(v[max(0,idx-13):idx+1])
    v_30d = mean(v[max(0,idx-29):idx+1])
    f["vol_ratio_1d_7d"] = v[idx]/v_7d if v_7d > 0 else 1
    f["vol_ratio_3d_14d"] = mean(v[max(0,idx-2):idx+1])/v_14d if v_14d > 0 else 1
    f["vol_ratio_7d_30d"] = v_7d/v_30d if v_30d > 0 else 1
    f["vol_trend_7d"] = polyfit_slope(v[max(0,idx-6):idx+1])
    f["vol_trend_14d"] = polyfit_slope(v[max(0,idx-13):idx+1])
    
    v_last2 = mean(v[max(0,idx-1):idx+1])
    v_prev5 = mean(v[max(0,idx-6):max(1,idx-1)]) if idx >= 3 else v_last2
    f["vol_explosion_2d"] = v_last2/v_prev5 if v_prev5 > 0 else 1
    
    obv = compute_obv(c[max(0,idx-29):idx+1], v[max(0,idx-29):idx+1])
    f["obv_slope_7d"] = polyfit_slope(obv[-7:]) if len(obv) >= 7 else 0
    
    if idx >= 7:
        f["vol_price_divergence"] = polyfit_slope(v[idx-6:idx+1]) - polyfit_slope(c[idx-6:idx+1])
    else: f["vol_price_divergence"] = 0
    
    # 캔들 패턴
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
    
    # Accumulation
    if idx >= 13:
        pc14 = abs(c[idx]/c[idx-13]-1)
        vr = mean(v[idx-6:idx+1]); vp = mean(v[idx-13:idx-6])
        f["accumulation"] = (vr/vp)/(1+pc14*10) if vp > 0 else 1
    else: f["accumulation"] = 1
    
    # 급등 유형별 특화
    if idx >= 30:
        recent_low = min(l[idx-29:idx+1]); peak_30d = max(h[idx-29:idx+1])
        f["drawdown_30d"] = (c[idx] - peak_30d) / peak_30d if peak_30d > 0 else 0
        f["recovery_from_low"] = (c[idx] - recent_low) / recent_low if recent_low > 0 else 0
    else: f["drawdown_30d"] = 0; f["recovery_from_low"] = 0
    
    if idx >= 14:
        ret7 = c[idx]/c[idx-7] - 1 if c[idx-7] > 0 else 0
        ret14 = c[idx]/c[idx-14] - 1 if c[idx-14] > 0 else 0
        f["momentum_acceleration"] = ret7 - (ret14 - ret7)
        up_with_vol = 0
        for j in range(idx, max(idx-7, -1), -1):
            if c[j] >= o[j] and v[j] > v_7d: up_with_vol += 1
            else: break
        f["up_with_volume_streak"] = up_with_vol
    else: f["momentum_acceleration"] = 0; f["up_with_volume_streak"] = 0
    
    if idx >= 30:
        recent_highs = sorted(h[idx-29:idx+1], reverse=True)
        resistance = mean(recent_highs[:3])
        f["distance_to_resistance"] = (resistance - c[idx]) / c[idx] if c[idx] > 0 else 0
        recent_lows = sorted(l[idx-29:idx+1])
        support = mean(recent_lows[:3])
        f["distance_from_support"] = (c[idx] - support) / c[idx] if c[idx] > 0 else 0
    else: f["distance_to_resistance"] = 0; f["distance_from_support"] = 0
    
    if idx >= 7:
        net_move = abs(c[idx] - c[idx-7])
        total_move = sum(abs(c[j] - c[j-1]) for j in range(idx-6, idx+1))
        f["efficiency_ratio_7d"] = net_move / total_move if total_move > 0 else 0
    else: f["efficiency_ratio_7d"] = 0
    
    return f

# ============================================================
# 데이터 수집
# ============================================================

def collect_data():
    cache_file = DATA_DIR / "raw_data.json"
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < 7200:
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
    
    print(f"  소형 알트: {len(small_alts)}개, 1년치 수집...")
    all_data = {}
    for i, sym in enumerate(small_alts):
        if (i+1) % 30 == 0: print(f"    [{i+1}/{len(small_alts)}]...")
        try:
            klines = fetch_all_klines(sym, LOOKBACK_DAYS)
            if len(klines) < MIN_KLINES: continue
            all_data[sym] = {
                "o": [float(k[1]) for k in klines], "h": [float(k[2]) for k in klines],
                "l": [float(k[3]) for k in klines], "c": [float(k[4]) for k in klines],
                "v": [float(k[5]) for k in klines], "ts": [int(k[0]) for k in klines],
            }
            time.sleep(0.05)
        except: continue
    print(f"  수집 완료: {len(all_data)}개")
    with open(cache_file, "w") as f: json.dump(all_data, f)
    return all_data

# ============================================================
# 학습 데이터셋 구축
# ============================================================

def build_dataset(all_data, surge_threshold=0.5):
    print(f"\n  데이터셋 구축 (+{surge_threshold*100:.0f}%)...")
    rows = []; surge_count = 0
    
    for sym, d in all_data.items():
        n = len(d["c"])
        o, h, l, c, v = d["o"], d["h"], d["l"], d["c"], d["v"]
        avg_vol = mean([v[j]*c[j] for j in range(max(0, n-30), n)])
        if avg_vol > MAX_VOL_USD or avg_vol < MIN_VOL_USD: continue
        
        for idx in range(59, n - SURGE_WINDOW):
            feat = extract_features_at(o, h, l, c, v, idx)
            if feat is None: continue
            
            max_ret = 0
            for j in range(idx+1, min(idx+SURGE_WINDOW+1, n)):
                ret = c[j] / c[idx] - 1 if c[idx] > 0 else 0
                max_ret = max(max_ret, ret)
            
            label = 1 if max_ret >= surge_threshold else 0
            feat["_label"] = label
            feat["_ts"] = d["ts"][idx]
            feat["_sym"] = sym
            rows.append(feat)
            if label == 1: surge_count += 1
    
    print(f"  {len(rows)}개 샘플 (급등 {surge_count}, {surge_count/len(rows)*100:.2f}%)")
    return rows

# ============================================================
# 수동 CV + 하이퍼파라미터 튜닝
# ============================================================

def manual_cv_score(model_class, params, X_train, y_train, n_splits=5):
    """TimeSeriesSplit으로 average_precision 계산."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, val_idx in tscv.split(X_train):
        Xt, Xv = X_train[train_idx], X_train[val_idx]
        yt, yv = y_train[train_idx], y_train[val_idx]
        
        model = model_class(**params)
        model.fit(Xt, yt)
        
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(Xv)[:, 1]
        else:
            probs = model.predict(Xv).astype(float)
        
        if sum(yv) > 0:
            scores.append(average_precision_score(yv, probs))
    
    return mean(scores) if scores else 0

def tune_xgboost(X_train, y_train, scale_pos_weight, n_iter=60):
    print(f"    XGBoost 튜닝 ({n_iter} iterations)...")
    
    param_space = {
        'n_estimators': [200, 300, 500],
        'max_depth': [4, 5, 6, 7],
        'learning_rate': [0.02, 0.05, 0.1],
        'min_child_weight': [3, 5, 10],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.6, 0.7, 0.8],
        'gamma': [0, 0.1, 0.5],
        'scale_pos_weight': [scale_pos_weight, scale_pos_weight * 1.5, scale_pos_weight * 2],
    }
    
    keys = list(param_space.keys())
    best_score = 0; best_params = None
    
    random.seed(42)
    for i in range(n_iter):
        params = {k: random.choice(param_space[k]) for k in keys}
        params['objective'] = 'binary:logistic'
        params['eval_metric'] = 'aucpr'
        params['use_label_encoder'] = False
        params['random_state'] = 42
        params['tree_method'] = 'hist'
        params['verbosity'] = 0
        
        try:
            score = manual_cv_score(xgb.XGBClassifier, params, X_train, y_train)
        except: continue
        
        if score > best_score:
            best_score = score
            best_params = params.copy()
            if (i+1) % 20 == 0:
                print(f"      [{i+1}/{n_iter}] best AP={best_score:.4f}")
    
    print(f"      Best AP={best_score:.4f}")
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    return model, best_params, best_score

def tune_lightgbm(X_train, y_train, scale_pos_weight, n_iter=60):
    print(f"    LightGBM 튜닝 ({n_iter} iterations)...")
    
    param_space = {
        'n_estimators': [200, 300, 500],
        'max_depth': [5, 7, -1],
        'learning_rate': [0.02, 0.05, 0.1],
        'num_leaves': [31, 50, 80],
        'min_child_samples': [10, 20, 50],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.6, 0.7, 0.8],
        'scale_pos_weight': [scale_pos_weight, scale_pos_weight * 1.5, scale_pos_weight * 2],
    }
    
    keys = list(param_space.keys())
    best_score = 0; best_params = None
    
    random.seed(42)
    for i in range(n_iter):
        params = {k: random.choice(param_space[k]) for k in keys}
        params['objective'] = 'binary'
        params['metric'] = 'average_precision'
        params['random_state'] = 42
        params['verbose'] = -1
        params['force_col_wise'] = True
        
        try:
            score = manual_cv_score(lgb.LGBMClassifier, params, X_train, y_train)
        except: continue
        
        if score > best_score:
            best_score = score
            best_params = params.copy()
            if (i+1) % 20 == 0:
                print(f"      [{i+1}/{n_iter}] best AP={best_score:.4f}")
    
    print(f"      Best AP={best_score:.4f}")
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train, y_train)
    return model, best_params, best_score

def tune_rf(X_train, y_train, scale_pos_weight, n_iter=30):
    print(f"    RandomForest 튜닝 ({n_iter} iterations)...")
    
    param_space = {
        'n_estimators': [300, 500],
        'max_depth': [7, 10, 15],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 5],
        'max_features': ['sqrt', 0.3],
        'class_weight': ['balanced', 'balanced_subsample'],
    }
    
    keys = list(param_space.keys())
    best_score = 0; best_params = None
    
    random.seed(42)
    for i in range(n_iter):
        params = {k: random.choice(param_space[k]) for k in keys}
        params['random_state'] = 42
        params['n_jobs'] = 1
        
        try:
            score = manual_cv_score(RandomForestClassifier, params, X_train, y_train)
        except: continue
        
        if score > best_score:
            best_score = score
            best_params = params.copy()
    
    print(f"      Best AP={best_score:.4f}")
    model = RandomForestClassifier(**best_params)
    model.fit(X_train, y_train)
    return model, best_params, best_score

# ============================================================
# Soft Voting 앙상블
# ============================================================

class SoftVotingEnsemble:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1.0] * len(models)
    
    def predict_proba(self, X):
        probs = np.zeros((X.shape[0], 2))
        total_w = sum(self.weights)
        for model, w in zip(self.models, self.weights):
            p = model.predict_proba(X)
            probs += p * (w / total_w)
        return probs
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

# ============================================================
# 평가 + threshold 최적화
# ============================================================

def find_best_threshold(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    best_thresh = 0.5; best_f1 = 0; best_p = 0; best_r = 0
    
    for thresh in np.arange(0.05, 0.95, 0.01):
        preds = (probs >= thresh).astype(int)
        if sum(preds) == 0: continue
        p = precision_score(y_test, preds, zero_division=0)
        r = recall_score(y_test, preds, zero_division=0)
        f1 = 2*p*r/(p+r) if p+r > 0 else 0
        
        # Precision >= 10% and maximize F1
        if p >= 0.10 and f1 > best_f1:
            best_thresh = thresh; best_f1 = f1; best_p = p; best_r = r
    
    return best_thresh, best_p, best_r, best_f1

def backtest_30d(all_data, model, scaler, feature_cols, threshold, surge_threshold):
    print(f"\n  30일 백테스트 (threshold={threshold:.2f})...")
    BACKTEST_DAYS = 30
    total_preds = 0; total_hits = 0; total_surges = 0
    
    for day_offset in range(BACKTEST_DAYS):
        day_X = []; day_y = []; day_syms = []
        
        for sym, d in all_data.items():
            n = len(d["c"])
            o, h, l, c, v = d["o"], d["h"], d["l"], d["c"], d["v"]
            idx = n - 1 - SURGE_WINDOW - (BACKTEST_DAYS - 1 - day_offset)
            if idx < 59 or idx + SURGE_WINDOW >= n: continue
            
            avg_vol = mean([v[j]*c[j] for j in range(max(0,idx-29), idx+1)])
            if avg_vol > MAX_VOL_USD or avg_vol < MIN_VOL_USD: continue
            
            feat = extract_features_at(o, h, l, c, v, idx)
            if feat is None: continue
            
            max_ret = 0
            for j in range(idx+1, min(idx+SURGE_WINDOW+1, n)):
                ret = c[j]/c[idx] - 1 if c[idx] > 0 else 0
                max_ret = max(max_ret, ret)
            
            fv = [feat.get(fc, 0) for fc in feature_cols]
            day_X.append(fv)
            day_y.append(1 if max_ret >= surge_threshold else 0)
            day_syms.append(sym)
        
        if not day_X: continue
        X_day = np.nan_to_num(np.array(day_X, dtype=np.float64), nan=0, posinf=0, neginf=0)
        X_day_s = scaler.transform(X_day)
        y_day = np.array(day_y)
        
        probs = model.predict_proba(X_day_s)[:, 1]
        preds = (probs >= threshold).astype(int)
        
        total_preds += sum(preds)
        total_hits += sum(preds & y_day)
        total_surges += sum(y_day)
    
    p = total_hits / total_preds if total_preds > 0 else 0
    r = total_hits / total_surges if total_surges > 0 else 0
    f1 = 2*p*r/(p+r) if p+r > 0 else 0
    
    print(f"    예측 {total_preds}, 적중 {total_hits}, 급등 {total_surges}")
    print(f"    Precision={p:.1%}, Recall={r:.1%}, F1={f1:.3f}")
    return {"precision": p, "recall": r, "f1": f1, "predictions": total_preds, "hits": total_hits, "surges": total_surges}

# ============================================================
# 메인
# ============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("  급등 전조 모델 v3 — No-joblib ML Pipeline")
    print("  XGBoost + LightGBM + RF → Soft Voting")
    print("=" * 70)
    
    print("\n[1] 데이터 수집...")
    all_data = collect_data()
    
    best_overall = {"f1": 0}
    all_results = {}
    
    for surge_th in SURGE_THRESHOLDS:
        print(f"\n{'='*70}")
        print(f"  급등 기준: +{surge_th*100:.0f}%")
        print(f"{'='*70}")
        
        # 데이터셋
        rows = build_dataset(all_data, surge_th)
        n_surge = sum(1 for r in rows if r["_label"] == 1)
        if n_surge < 20:
            print(f"  급등 부족 → 스킵")
            continue
        
        df = pd.DataFrame(rows)
        feature_cols = [c for c in df.columns if not c.startswith("_")]
        X = df[feature_cols].values.astype(np.float64)
        y = df["_label"].values
        ts = df["_ts"].values
        
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        sort_idx = np.argsort(ts)
        X, y, ts = X[sort_idx], y[sort_idx], ts[sort_idx]
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        n_pos = sum(y_train); n_neg = len(y_train) - n_pos
        spw = n_neg / n_pos if n_pos > 0 else 1
        
        print(f"\n  Train: {len(X_train)} ({sum(y_train)} surge)")
        print(f"  Test:  {len(X_test)} ({sum(y_test)} surge)")
        print(f"  불균형: 1:{spw:.0f}")
        
        scaler = RobustScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # 모델 학습
        print(f"\n  [모델 학습]")
        xgb_model, xgb_params, xgb_ap = tune_xgboost(X_train_s, y_train, spw, n_iter=60)
        lgb_model, lgb_params, lgb_ap = tune_lightgbm(X_train_s, y_train, spw, n_iter=60)
        rf_model, rf_params, rf_ap = tune_rf(X_train_s, y_train, spw, n_iter=30)
        
        # 앙상블 (AP 기반 가중치)
        total_ap = xgb_ap + lgb_ap + rf_ap
        weights = [xgb_ap/total_ap, lgb_ap/total_ap, rf_ap/total_ap] if total_ap > 0 else [1/3]*3
        ensemble = SoftVotingEnsemble([xgb_model, lgb_model, rf_model], weights)
        
        print(f"\n    앙상블 가중치: XGB={weights[0]:.2f} LGB={weights[1]:.2f} RF={weights[2]:.2f}")
        
        # 각 모델 + 앙상블 평가
        models = {
            "xgboost": xgb_model,
            "lightgbm": lgb_model,
            "random_forest": rf_model,
            "ensemble": ensemble,
        }
        
        for name, model in models.items():
            thresh, p, r, f1 = find_best_threshold(model, X_test_s, y_test)
            bt = backtest_30d(all_data, model, scaler, feature_cols, thresh, surge_th)
            
            key = f"{name}_surge{int(surge_th*100)}"
            all_results[key] = {
                "model_name": name, "surge_threshold": surge_th,
                "threshold": thresh, "test_precision": p, "test_recall": r,
                "test_f1": f1, "backtest": bt,
            }
            
            if bt["f1"] > best_overall.get("f1", 0):
                best_overall = {
                    "model_name": name, "surge_threshold": surge_th,
                    "threshold": thresh, "model": model,
                    "scaler": scaler, "feature_cols": feature_cols,
                    **bt,
                }
    
    # 최종 결과
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  최종 결과 (소요: {elapsed/60:.1f}분)")
    print(f"{'='*70}")
    
    print(f"\n  {'설정':<35} {'Precision':>10} {'Recall':>8} {'F1':>8} {'예측':>6} {'적중':>4}")
    print(f"  {'-'*75}")
    for key, r in sorted(all_results.items(), key=lambda x: x[1]["backtest"]["f1"], reverse=True):
        bt = r["backtest"]
        print(f"  {key:<35} {bt['precision']:>9.1%} {bt['recall']:>7.1%} {bt['f1']:>7.3f} {bt['predictions']:>6} {bt['hits']:>4}")
    
    if best_overall.get("model"):
        print(f"\n  === 최적 모델 ===")
        print(f"  {best_overall['model_name']} (급등 +{best_overall['surge_threshold']*100:.0f}%)")
        print(f"  Threshold: {best_overall['threshold']:.2f}")
        print(f"  Precision: {best_overall['precision']:.1%}")
        print(f"  Recall: {best_overall['recall']:.1%}")
        print(f"  F1: {best_overall['f1']:.3f}")
        
        # Feature importance (XGB)
        for key, r in all_results.items():
            if "xgboost" in key:
                pass  # importance는 모델에서 뽑아야 하는데 여기선 생략
        
        config = {
            "model_name": best_overall["model_name"],
            "surge_threshold": best_overall["surge_threshold"],
            "threshold": best_overall["threshold"],
            "feature_cols": best_overall["feature_cols"],
            "precision": best_overall["precision"],
            "recall": best_overall["recall"],
            "f1": best_overall["f1"],
            "predictions_30d": best_overall["predictions"],
            "hits_30d": best_overall["hits"],
        }
        with open(BASE_DIR / "ml_model_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        with open(DATA_DIR / "best_model.pkl", "wb") as f:
            pickle.dump({
                "model": best_overall["model"],
                "scaler": best_overall["scaler"],
                "feature_cols": best_overall["feature_cols"],
                "threshold": best_overall["threshold"],
                "surge_threshold": best_overall["surge_threshold"],
            }, f)
        
        print(f"\n  저장: ml_model_config.json + ml_data/best_model.pkl")
    
    save_results = {k: {kk: vv for kk, vv in v.items()} for k, v in all_results.items()}
    with open(BASE_DIR / "ml_results.json", "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    
    print(f"\n  전체 결과: ml_results.json")
    print(f"  총 소요: {elapsed/60:.1f}분")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
