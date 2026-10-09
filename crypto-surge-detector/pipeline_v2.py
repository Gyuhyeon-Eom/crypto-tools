#!/usr/bin/env python3
"""
Crypto Surge Detector Pipeline v2
Improved: level-specific models, relaxed thresholds, better scoring.
"""

import json
import os
import time
import math
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

BINANCE_BASE = "https://api.binance.com"

# ============================================================
# Phase 1: Data Collection (reuse cached data from pipeline.py)
# ============================================================

def api_get(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                return None


def get_top_usdt_pairs(n=100):
    cache_file = DATA_DIR / "top_pairs.json"
    if cache_file.exists():
        with open(cache_file) as f:
            pairs = json.load(f)
        print(f"[Phase 1] Loaded {len(pairs)} cached pairs")
        return pairs

    print("[Phase 1] Fetching top USDT pairs by volume...")
    data = api_get(f"{BINANCE_BASE}/api/v3/ticker/24hr")
    if not data:
        raise RuntimeError("Failed to fetch ticker data")

    usdt_pairs = []
    for t in data:
        sym = t["symbol"]
        if sym.endswith("USDT") and not any(x in sym for x in ["UP", "DOWN", "BEAR", "BULL"]):
            try:
                vol = float(t["quoteVolume"])
                usdt_pairs.append({"symbol": sym, "quoteVolume": vol})
            except (ValueError, KeyError):
                pass

    usdt_pairs.sort(key=lambda x: x["quoteVolume"], reverse=True)
    exclude = {"USDCUSDT", "BUSDUSDT", "TUSDUSDT", "DAIUSDT", "FDUSDUSDT", "EURUSDT", "USTUSDT", "USDPUSDT", "USD1USDT", "RLUSDUSDT"}
    usdt_pairs = [p for p in usdt_pairs if p["symbol"] not in exclude and p["symbol"].isascii()]
    pairs = [p["symbol"] for p in usdt_pairs[:n]]

    with open(cache_file, "w") as f:
        json.dump(pairs, f)
    print(f"[Phase 1] Selected {len(pairs)} pairs")
    return pairs


def fetch_klines(symbol, interval, start_ms, end_ms):
    all_klines = []
    current_start = start_ms
    while current_start < end_ms:
        url = (f"{BINANCE_BASE}/api/v3/klines?"
               f"symbol={symbol}&interval={interval}"
               f"&startTime={current_start}&endTime={end_ms}&limit=1000")
        data = api_get(url)
        if not data or len(data) == 0:
            break
        all_klines.extend(data)
        current_start = data[-1][0] + 1
        time.sleep(0.5)
    return all_klines


def klines_to_df(klines):
    if not klines:
        return pd.DataFrame()
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_vol",
            "taker_buy_quote_vol", "ignore"]
    df = pd.DataFrame(klines, columns=cols)
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    return df[["open_time", "open", "high", "low", "close", "volume", "quote_volume"]]


def collect_data(pairs):
    now_ms = int(time.time() * 1000)
    start_daily = now_ms - 365 * 24 * 3600 * 1000
    start_4h = now_ms - 180 * 24 * 3600 * 1000

    total = len(pairs)
    for i, sym in enumerate(pairs):
        daily_file = DATA_DIR / f"{sym}_1d.parquet"
        h4_file = DATA_DIR / f"{sym}_4h.parquet"

        if daily_file.exists() and h4_file.exists():
            if (i + 1) % 20 == 0:
                print(f"[Phase 1] [{i+1}/{total}] {sym} - cached")
            continue

        print(f"[Phase 1] [{i+1}/{total}] {sym} - downloading...")
        if not daily_file.exists():
            klines = fetch_klines(sym, "1d", start_daily, now_ms)
            df = klines_to_df(klines)
            if not df.empty:
                df.to_parquet(daily_file, index=False)
            time.sleep(0.3)
        if not h4_file.exists():
            klines = fetch_klines(sym, "4h", start_4h, now_ms)
            df = klines_to_df(klines)
            if not df.empty:
                df.to_parquet(h4_file, index=False)
            time.sleep(0.3)

    btc_file = DATA_DIR / "BTCUSDT_1d.parquet"
    if not btc_file.exists():
        print("[Phase 1] Fetching BTC daily data...")
        klines = fetch_klines("BTCUSDT", "1d", start_daily, now_ms)
        df = klines_to_df(klines)
        if not df.empty:
            df.to_parquet(btc_file, index=False)

    print("[Phase 1] Data collection complete.")


# ============================================================
# Phase 2: Surge Event Labeling
# ============================================================

def find_surge_events(df, symbol):
    events = []
    n = len(df)
    if n < 30:
        return events

    highs = df["high"].values
    lows = df["low"].values
    used_indices = set()

    # Level A: 7-day window, low->high >= 100%
    for i in range(n - 7):
        if i in used_indices:
            continue
        window_low = np.min(lows[i:i+7])
        window_high = np.max(highs[i:i+7])
        if window_low > 0 and (window_high / window_low - 1) >= 1.0:
            events.append({
                "symbol": symbol, "start_idx": i, "level": "A",
                "gain": window_high / window_low - 1,
                "date": str(df.iloc[i]["open_time"].date())
            })
            for j in range(max(0, i-7), min(n, i+14)):
                used_indices.add(j)

    # Level B: 7-day window, low->high >= 50%
    for i in range(n - 7):
        if i in used_indices:
            continue
        window_low = np.min(lows[i:i+7])
        window_high = np.max(highs[i:i+7])
        if window_low > 0 and (window_high / window_low - 1) >= 0.5:
            events.append({
                "symbol": symbol, "start_idx": i, "level": "B",
                "gain": window_high / window_low - 1,
                "date": str(df.iloc[i]["open_time"].date())
            })
            for j in range(max(0, i-7), min(n, i+14)):
                used_indices.add(j)

    # Level C: 3-day window, low->high >= 30%
    for i in range(n - 3):
        if i in used_indices:
            continue
        window_low = np.min(lows[i:i+3])
        window_high = np.max(highs[i:i+3])
        if window_low > 0 and (window_high / window_low - 1) >= 0.3:
            events.append({
                "symbol": symbol, "start_idx": i, "level": "C",
                "gain": window_high / window_low - 1,
                "date": str(df.iloc[i]["open_time"].date())
            })
            for j in range(max(0, i-7), min(n, i+14)):
                used_indices.add(j)

    return events


# ============================================================
# Phase 3: Feature Extraction (enhanced)
# ============================================================

def compute_rsi(closes, period=14):
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    rsi = np.full(len(closes), np.nan)
    if len(deltas) < period:
        return rsi
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    if len(gains) >= period:
        fg = np.mean(gains[:period])
        fl = np.mean(losses[:period])
        if fl > 0:
            rsi[period] = 100.0 - 100.0 / (1.0 + fg / fl)
        else:
            rsi[period] = 100.0
    return rsi


def compute_atr(highs, lows, closes, period=14):
    n = len(closes)
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i-1]),
                     abs(lows[i] - closes[i-1]))
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


def extract_features(df, idx, btc_df=None):
    """Extract precursor features. idx is the surge start (or current end for scanning)."""
    if idx < 44:
        return None

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    volumes = df["volume"].values
    opens = df["open"].values

    pre_end = idx
    pre_start = idx - 14
    if pre_start < 0:
        return None

    features = {}

    # === Volume Patterns ===
    vol_7d = np.mean(volumes[pre_end-7:pre_end])
    vol_30d = np.mean(volumes[max(0, pre_end-30):pre_end])
    vol_20d = np.mean(volumes[max(0, pre_end-20):pre_end])

    features["vol_7d_30d_ratio"] = vol_7d / vol_30d if vol_30d > 0 else 1.0
    features["vol_spike_3d"] = np.max(volumes[pre_end-3:pre_end]) / vol_20d if vol_20d > 0 else 1.0

    # Volume trend
    if pre_end >= 7:
        v7 = volumes[pre_end-7:pre_end]
        x = np.arange(7, dtype=float)
        if np.std(v7) > 0:
            slope = np.polyfit(x, v7, 1)[0]
            features["vol_trend_7d"] = slope / np.mean(v7) if np.mean(v7) > 0 else 0
        else:
            features["vol_trend_7d"] = 0.0
    else:
        features["vol_trend_7d"] = 0.0

    # === Price Patterns ===
    atr_full = compute_atr(highs, lows, closes, 14)
    atr_14 = atr_full[pre_end-1] if not np.isnan(atr_full[pre_end-1]) else 0
    features["atr_14d_ratio"] = atr_14 / closes[pre_end-1] if closes[pre_end-1] > 0 else 0

    # Volatility contraction
    atr_vals_7 = [atr_full[j] for j in range(pre_end-7, pre_end) if not np.isnan(atr_full[j])]
    atr_vals_30 = [atr_full[j] for j in range(max(0, pre_end-30), pre_end) if not np.isnan(atr_full[j])]
    atr_7 = np.mean(atr_vals_7) if atr_vals_7 else atr_14
    atr_30 = np.mean(atr_vals_30) if atr_vals_30 else atr_14
    features["volatility_contraction"] = atr_7 / atr_30 if atr_30 > 0 else 1.0

    # Bollinger Band width
    if pre_end >= 20:
        bb_closes = closes[pre_end-20:pre_end]
        bb_mean = np.mean(bb_closes)
        bb_std = np.std(bb_closes)
        features["bb_width"] = (2 * bb_std) / bb_mean if bb_mean > 0 else 0
    else:
        features["bb_width"] = 0

    # Price position vs 60d high
    if pre_end >= 60:
        high_60d = np.max(highs[pre_end-60:pre_end])
        features["price_vs_60d_high"] = closes[pre_end-1] / high_60d if high_60d > 0 else 1.0
    else:
        features["price_vs_60d_high"] = 1.0

    # Price momentum: 7d return
    if pre_end >= 8:
        features["price_ret_7d"] = closes[pre_end-1] / closes[pre_end-8] - 1 if closes[pre_end-8] > 0 else 0
    else:
        features["price_ret_7d"] = 0

    # Price distance from 20d MA
    if pre_end >= 20:
        ma20 = np.mean(closes[pre_end-20:pre_end])
        features["price_vs_ma20"] = closes[pre_end-1] / ma20 - 1 if ma20 > 0 else 0
    else:
        features["price_vs_ma20"] = 0

    # === RSI Patterns ===
    rsi_full = compute_rsi(closes, 14)
    rsi_val = rsi_full[pre_end-1]
    features["rsi_14"] = rsi_val if not np.isnan(rsi_val) else 50.0

    # RSI consolidation: days in 40-55 range in last 14 days
    rsi_window = rsi_full[pre_start:pre_end]
    valid_rsi = rsi_window[~np.isnan(rsi_window)]
    features["rsi_consolidation_days"] = int(np.sum((valid_rsi >= 40) & (valid_rsi <= 55)))

    # RSI trend (7d change)
    if pre_end >= 8 and not np.isnan(rsi_full[pre_end-1]) and not np.isnan(rsi_full[pre_end-8]):
        features["rsi_trend_7d"] = rsi_full[pre_end-1] - rsi_full[pre_end-8]
    else:
        features["rsi_trend_7d"] = 0.0

    # RSI oversold days (< 30) in last 14 days
    features["rsi_oversold_days"] = int(np.sum(valid_rsi < 30)) if len(valid_rsi) > 0 else 0

    # === Candle Patterns ===
    consec_red = 0
    for j in range(pre_end-1, max(pre_start-1, -1), -1):
        if closes[j] < opens[j]:
            consec_red += 1
        else:
            break
    features["consecutive_red"] = consec_red

    green_count = sum(1 for j in range(pre_end-3, pre_end) if closes[j] >= opens[j])
    features["green_ratio_3d"] = green_count / 3.0

    # Doji/Hammer in last 3 days
    doji_hammer = 0
    for j in range(pre_end-3, pre_end):
        body = abs(closes[j] - opens[j])
        total_range = highs[j] - lows[j]
        if total_range > 0:
            body_ratio = body / total_range
            lower_shadow = min(opens[j], closes[j]) - lows[j]
            if body_ratio < 0.1:
                doji_hammer += 1
            elif lower_shadow > 2 * body and lower_shadow > 0.5 * total_range:
                doji_hammer += 1
    features["doji_hammer_count"] = doji_hammer

    # Body size ratio (avg body / avg range) - small bodies = indecision
    bodies = []
    ranges = []
    for j in range(max(0, pre_end-7), pre_end):
        bodies.append(abs(closes[j] - opens[j]))
        ranges.append(highs[j] - lows[j])
    avg_body = np.mean(bodies) if bodies else 0
    avg_range = np.mean(ranges) if ranges else 1
    features["body_range_ratio"] = avg_body / avg_range if avg_range > 0 else 0

    # === Relative Strength ===
    if btc_df is not None and pre_end >= 8:
        coin_dates = df["open_time"].dt.date.values
        btc_dates = btc_df["open_time"].dt.date.values
        coin_date = coin_dates[pre_end-1]
        coin_date_7d = coin_dates[pre_end-8]

        btc_price_now = btc_price_7d = None
        for bi in range(len(btc_dates)-1, -1, -1):
            if btc_dates[bi] <= coin_date and btc_price_now is None:
                btc_price_now = btc_df["close"].values[bi]
            if btc_dates[bi] <= coin_date_7d and btc_price_7d is None:
                btc_price_7d = btc_df["close"].values[bi]
            if btc_price_now is not None and btc_price_7d is not None:
                break

        coin_ret_7d = closes[pre_end-1] / closes[pre_end-8] - 1 if closes[pre_end-8] > 0 else 0
        if btc_price_now and btc_price_7d and btc_price_7d > 0:
            btc_ret_7d = btc_price_now / btc_price_7d - 1
            features["vs_btc_7d"] = coin_ret_7d - btc_ret_7d
        else:
            features["vs_btc_7d"] = 0.0
        features["coin_ret_7d"] = coin_ret_7d
    else:
        features["vs_btc_7d"] = 0.0
        features["coin_ret_7d"] = 0.0

    # === Accumulation Signal ===
    # Volume increase while price flat (accumulation)
    if pre_end >= 14:
        price_change_14d = abs(closes[pre_end-1] / closes[pre_end-14] - 1)
        vol_change_14d = np.mean(volumes[pre_end-7:pre_end]) / np.mean(volumes[pre_end-14:pre_end-7]) if np.mean(volumes[pre_end-14:pre_end-7]) > 0 else 1
        features["accumulation_signal"] = vol_change_14d / (1 + price_change_14d * 10)
    else:
        features["accumulation_signal"] = 1.0

    return features


def extract_random_non_surge_features(df, surge_indices, btc_df, n_samples=5):
    n = len(df)
    if n < 60:
        return []
    blocked = set()
    for si in surge_indices:
        for j in range(max(0, si - 14), min(n, si + 8)):
            blocked.add(j)
    candidates = [i for i in range(44, n) if i not in blocked]
    if len(candidates) < n_samples:
        return []
    step = max(1, len(candidates) // n_samples)
    sampled = [candidates[i * step] for i in range(min(n_samples, len(candidates)))]
    results = []
    for idx in sampled:
        feat = extract_features(df, idx, btc_df)
        if feat:
            results.append(feat)
    return results


# ============================================================
# Phase 4: Statistical Validation
# ============================================================

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (m1 - m2) / pooled_std


def mann_whitney_u(x, y):
    nx, ny = len(x), len(y)
    if nx < 5 or ny < 5:
        return 1.0
    combined = [(v, 0) for v in x] + [(v, 1) for v in y]
    combined.sort(key=lambda t: t[0])
    ranks = np.zeros(len(combined))
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    r1 = sum(ranks[k] for k in range(len(combined)) if combined[k][1] == 0)
    u1 = r1 - nx * (nx + 1) / 2.0
    mu = nx * ny / 2.0
    sigma = math.sqrt(nx * ny * (nx + ny + 1) / 12.0)
    if sigma == 0:
        return 1.0
    z = (u1 - mu) / sigma
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return p


def validate_features(surge_features, normal_features):
    if not surge_features or not normal_features:
        return {}
    feature_names = [k for k in surge_features[0].keys() if not k.startswith("_")]
    results = {}
    for fname in feature_names:
        surge_vals = np.array([f[fname] for f in surge_features if fname in f], dtype=float)
        normal_vals = np.array([f[fname] for f in normal_features if fname in f], dtype=float)
        surge_vals = surge_vals[~np.isnan(surge_vals)]
        normal_vals = normal_vals[~np.isnan(normal_vals)]
        if len(surge_vals) < 5 or len(normal_vals) < 5:
            continue
        d = cohens_d(surge_vals, normal_vals)
        p = mann_whitney_u(surge_vals, normal_vals)
        results[fname] = {
            "surge_mean": float(np.mean(surge_vals)),
            "surge_std": float(np.std(surge_vals)),
            "surge_median": float(np.median(surge_vals)),
            "normal_mean": float(np.mean(normal_vals)),
            "normal_std": float(np.std(normal_vals)),
            "normal_median": float(np.median(normal_vals)),
            "cohens_d": float(d),
            "p_value": float(p),
            "n_surge": int(len(surge_vals)),
            "n_normal": int(len(normal_vals))
        }
    return results


def build_scoring_model(validation_results, min_d=0.15, max_p=0.05):
    """Build scoring model from validated features.
    Use relaxed thresholds: |d| >= 0.15 and p < 0.05.
    """
    model_features = {}
    for fname, info in validation_results.items():
        if abs(info["cohens_d"]) >= min_d and info["p_value"] < max_p:
            direction = "higher" if info["cohens_d"] > 0 else "lower"
            # Weight = effect size, capped at 2
            weight = min(abs(info["cohens_d"]), 2.0)
            model_features[fname] = {
                "weight": weight,
                "direction": direction,
                "surge_mean": info["surge_mean"],
                "surge_std": info["surge_std"],
                "normal_mean": info["normal_mean"],
                "normal_std": info["normal_std"],
                "cohens_d": info["cohens_d"],
                "p_value": info["p_value"]
            }
    return model_features


def compute_composite_score(features, thresholds):
    if not features or not thresholds:
        return 0.0
    score = 0.0
    max_score = 0.0
    for fname, info in thresholds.items():
        if fname not in features:
            continue
        weight = info["weight"]
        max_score += weight
        val = features[fname]
        direction = info["direction"]
        surge_mean = info["surge_mean"]
        normal_mean = info["normal_mean"]
        if direction == "higher":
            if surge_mean > normal_mean:
                ratio = (val - normal_mean) / (surge_mean - normal_mean)
            else:
                ratio = 0
        else:
            if surge_mean < normal_mean:
                ratio = (normal_mean - val) / (normal_mean - surge_mean)
            else:
                ratio = 0
        ratio = max(0, min(1.0, ratio))
        score += weight * ratio
    return (score / max_score * 100) if max_score > 0 else 0.0


def compute_pr(surge_scores, normal_scores, thresholds_list):
    results = []
    for thr in thresholds_list:
        tp = sum(1 for s in surge_scores if s >= thr)
        fp = sum(1 for s in normal_scores if s >= thr)
        fn = sum(1 for s in surge_scores if s < thr)
        tn = sum(1 for s in normal_scores if s < thr)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        results.append({
            "threshold": thr, "precision": precision, "recall": recall,
            "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn
        })
    return results


# ============================================================
# Main Pipeline
# ============================================================

def run_pipeline():
    print("=" * 60)
    print("  Crypto Surge Precursor Detection Pipeline v2")
    print("=" * 60)

    # Phase 1
    pairs = get_top_usdt_pairs(100)
    collect_data(pairs)

    btc_file = DATA_DIR / "BTCUSDT_1d.parquet"
    btc_df = pd.read_parquet(btc_file) if btc_file.exists() else None

    # Phase 2 & 3
    print("\n[Phase 2-3] Finding surge events and extracting features...")
    all_surge_features = {"A": [], "B": [], "C": [], "all": []}
    all_normal_features = []
    all_events = []

    for sym in pairs:
        daily_file = DATA_DIR / f"{sym}_1d.parquet"
        if not daily_file.exists():
            continue
        df = pd.read_parquet(daily_file)
        if len(df) < 60:
            continue

        events = find_surge_events(df, sym)
        all_events.extend(events)
        surge_indices = [e["start_idx"] for e in events]

        for evt in events:
            feat = extract_features(df, evt["start_idx"], btc_df)
            if feat:
                feat["_level"] = evt["level"]
                feat["_symbol"] = sym
                feat["_date"] = evt["date"]
                feat["_gain"] = evt["gain"]
                all_surge_features[evt["level"]].append(feat)
                all_surge_features["all"].append(feat)

        n_surge = len(surge_indices)
        n_normal = max(8, n_surge * 3)
        normal_feats = extract_random_non_surge_features(df, surge_indices, btc_df, n_normal)
        all_normal_features.extend(normal_feats)

    print(f"  Surge events: A={len(all_surge_features['A'])}, "
          f"B={len(all_surge_features['B'])}, C={len(all_surge_features['C'])}")
    print(f"  Total surge features: {len(all_surge_features['all'])}")
    print(f"  Normal samples: {len(all_normal_features)}")

    # Phase 4: Statistical Validation - per level
    print("\n[Phase 4] Statistical validation per level...")

    models = {}
    level_validations = {}

    for level in ["A", "B", "C", "all"]:
        surge_feats = all_surge_features[level]
        if len(surge_feats) < 20:
            print(f"  Level {level}: too few events ({len(surge_feats)}), skipping")
            continue

        vr = validate_features(surge_feats, all_normal_features)
        level_validations[level] = vr

        # Build model with relaxed thresholds
        model = build_scoring_model(vr, min_d=0.15, max_p=0.05)

        print(f"\n  === Level {level} ({len(surge_feats)} events) ===")
        print(f"  Significant features (|d|>=0.15, p<0.05): {len(model)}")
        for fname in sorted(model.keys(), key=lambda x: abs(model[x]["cohens_d"]), reverse=True):
            info = model[fname]
            print(f"    {fname}: d={info['cohens_d']:.3f}, p={info['p_value']:.4f}, "
                  f"surge={info['surge_mean']:.4f}, normal={info['normal_mean']:.4f}")

        # Compute scores for PR analysis
        surge_scores = []
        for feat in surge_feats:
            clean = {k: v for k, v in feat.items() if not k.startswith("_")}
            surge_scores.append(compute_composite_score(clean, model))
        normal_scores = []
        for feat in all_normal_features:
            normal_scores.append(compute_composite_score(feat, model))

        pr = compute_pr(surge_scores, normal_scores, [30, 40, 50, 60, 70, 80])
        print(f"\n  Precision/Recall for Level {level}:")
        for p in pr:
            print(f"    Score >= {p['threshold']}: P={p['precision']:.3f}, R={p['recall']:.3f}, F1={p['f1']:.3f}")

        models[level] = {"features": model, "pr": pr}

    # Combine: weighted ensemble of level-specific models
    # Prioritize Level A (strongest signals) with Level B/C as secondary
    print("\n[Phase 4] Building combined ensemble model...")

    # Use Level A as primary, add features from B and all that aren't in A
    combined_model = {}
    if "A" in models:
        for fname, info in models["A"]["features"].items():
            combined_model[fname] = info.copy()
            combined_model[fname]["source"] = "A"

    for level in ["B", "all"]:
        if level in models:
            for fname, info in models[level]["features"].items():
                if fname not in combined_model:
                    info_copy = info.copy()
                    info_copy["weight"] *= 0.7  # lower weight for non-A features
                    info_copy["source"] = level
                    combined_model[fname] = info_copy

    print(f"  Combined model: {len(combined_model)} features")
    for fname in sorted(combined_model.keys(), key=lambda x: combined_model[x]["weight"], reverse=True):
        info = combined_model[fname]
        print(f"    {fname}: w={info['weight']:.3f}, d={info['cohens_d']:.3f}, src={info['source']}")

    # Final PR on combined model
    all_surge_scores = []
    for feat in all_surge_features["all"]:
        clean = {k: v for k, v in feat.items() if not k.startswith("_")}
        all_surge_scores.append(compute_composite_score(clean, combined_model))
    all_normal_scores = []
    for feat in all_normal_features:
        all_normal_scores.append(compute_composite_score(feat, combined_model))

    final_pr = compute_pr(all_surge_scores, all_normal_scores, [30, 40, 50, 55, 60, 65, 70, 75, 80])
    print("\n  Combined model P/R:")
    for p in final_pr:
        print(f"    Score >= {p['threshold']}: P={p['precision']:.3f}, R={p['recall']:.3f}, F1={p['f1']:.3f} (TP={p['tp']}, FP={p['fp']})")

    # Save full model
    full_model = {
        "combined_model": combined_model,
        "level_models": {k: v["features"] for k, v in models.items()},
        "level_validations": level_validations,
        "level_pr": {k: v["pr"] for k, v in models.items()},
        "combined_pr": final_pr,
        "event_counts": {
            "A": len(all_surge_features["A"]),
            "B": len(all_surge_features["B"]),
            "C": len(all_surge_features["C"]),
            "total": len(all_surge_features["all"]),
            "normal": len(all_normal_features)
        },
        "generated_at": datetime.now(timezone.utc).isoformat()
    }

    model_file = DATA_DIR / "model_v2.json"
    with open(model_file, "w") as f:
        json.dump(full_model, f, indent=2, default=str)

    # Save thresholds for scanner
    thresholds_file = DATA_DIR / "thresholds.json"
    with open(thresholds_file, "w") as f:
        json.dump(combined_model, f, indent=2, default=str)

    print(f"\n  Model saved to {model_file}")

    # Phase 5: Current Market Scan
    print("\n[Phase 5] Scanning current market...")
    scan_results = []
    for sym in pairs:
        daily_file = DATA_DIR / f"{sym}_1d.parquet"
        if not daily_file.exists():
            continue
        df = pd.read_parquet(daily_file)
        if len(df) < 60:
            continue
        idx = len(df)
        feat = extract_features(df, idx, btc_df)
        if not feat:
            continue
        score = compute_composite_score(feat, combined_model)

        # Also compute per-level scores
        level_scores = {}
        for level in ["A", "B"]:
            if level in models:
                level_scores[level] = compute_composite_score(feat, models[level]["features"])

        scan_results.append({
            "symbol": sym,
            "score": score,
            "level_scores": level_scores,
            "features": feat,
            "last_close": float(df["close"].values[-1]),
            "last_date": str(df["open_time"].values[-1])[:10]
        })

    scan_results.sort(key=lambda x: x["score"], reverse=True)

    # Generate reports
    generate_analysis_report(full_model, all_events, models, combined_model)
    generate_scan_report(scan_results, combined_model, models)
    generate_scanner_script()

    print("\n" + "=" * 60)
    print("  Pipeline v2 complete!")
    print(f"  - analysis_report.md")
    print(f"  - scan_result.md")
    print(f"  - surge_scanner.py")
    print("=" * 60)


def generate_analysis_report(full_model, all_events, models, combined_model):
    """Generate analysis_report.md."""
    counts = full_model["event_counts"]
    lv = full_model["level_validations"]

    lines = []
    lines.append("# 크립토 급등 전조증상 분석 리포트")
    lines.append("")
    lines.append("생성일시: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S KST')))
    lines.append("")

    lines.append("## 1. 데이터 개요")
    lines.append("")
    lines.append("- 분석 대상: USDT 페어 거래량 상위 100개 코인 (Binance)")
    lines.append("- 분석 기간: 최근 365일 일봉 데이터")
    lines.append("- 급등 이벤트:")
    lines.append("  - Level A (7일 내 +100% 이상): {}건".format(counts["A"]))
    lines.append("  - Level B (7일 내 +50% 이상): {}건".format(counts["B"]))
    lines.append("  - Level C (3일 내 +30% 이상): {}건".format(counts["C"]))
    lines.append("  - 합계: {}건".format(counts["total"]))
    lines.append("- 비급등(일반) 기간 샘플: {}건".format(counts["normal"]))
    lines.append("")

    feature_desc = {
        "vol_7d_30d_ratio": "7일/30일 평균 거래량 비율",
        "vol_spike_3d": "3일 내 최대 거래량 / 20일 평균",
        "vol_trend_7d": "7일 거래량 추세 (정규화 기울기)",
        "atr_14d_ratio": "14일 ATR / 현재가 (변동성)",
        "volatility_contraction": "7일 ATR / 30일 ATR (변동성 수축도)",
        "bb_width": "볼린저 밴드 폭 (20일, 2std/mean)",
        "price_vs_60d_high": "현재가 / 60일 고점",
        "price_ret_7d": "7일 수익률",
        "price_vs_ma20": "현재가 vs 20일 이동평균 괴리율",
        "rsi_14": "14일 RSI",
        "rsi_consolidation_days": "RSI 40~55 체류 일수 (14일 중)",
        "rsi_trend_7d": "7일간 RSI 변화량",
        "rsi_oversold_days": "RSI 30 미만 일수 (14일 중)",
        "consecutive_red": "직전 연속 음봉 수",
        "green_ratio_3d": "최근 3일 양봉 비율",
        "doji_hammer_count": "최근 3일 Doji/Hammer 수",
        "body_range_ratio": "7일 평균 봉 body/range 비율",
        "vs_btc_7d": "BTC 대비 7일 초과수익률",
        "coin_ret_7d": "코인 7일 수익률",
        "accumulation_signal": "축적 시그널 (거래량 증가 + 가격 횡보)"
    }

    # Per-level detailed results
    lines.append("## 2. 레벨별 전조증상 분석")
    lines.append("")

    for level, level_name in [("A", "Level A: 7일 내 +100% 급등"),
                               ("B", "Level B: 7일 내 +50% 급등"),
                               ("C", "Level C: 3일 내 +30% 급등")]:
        if level not in lv:
            continue
        lines.append("### {}".format(level_name))
        lines.append("")
        lines.append("이벤트 수: {}건".format(counts[level]))
        lines.append("")
        lines.append("| Feature | 급등 전 평균 | 일반 평균 | Cohen's d | p-value | 유의미 |")
        lines.append("|---------|-------------|----------|-----------|---------|--------|")

        sorted_feats = sorted(lv[level].items(), key=lambda x: abs(x[1]["cohens_d"]), reverse=True)
        for fname, info in sorted_feats:
            sig = "O" if abs(info["cohens_d"]) >= 0.15 and info["p_value"] < 0.05 else "X"
            desc = feature_desc.get(fname, fname)
            lines.append("| {} | {:.4f} | {:.4f} | {:+.3f} | {:.4f} | {} |".format(
                desc, info["surge_mean"], info["normal_mean"],
                info["cohens_d"], info["p_value"], sig))
        lines.append("")

        # Interpretation
        sig_feats = [(f, i) for f, i in sorted_feats if abs(i["cohens_d"]) >= 0.15 and i["p_value"] < 0.05]
        if sig_feats:
            lines.append("**주요 발견:**")
            for fname, info in sig_feats:
                desc = feature_desc.get(fname, fname)
                if info["cohens_d"] > 0:
                    lines.append("- {}: 급등 전에 일반 대비 더 **높은** 값 (급등 전 {:.4f} vs 일반 {:.4f})".format(
                        desc, info["surge_mean"], info["normal_mean"]))
                else:
                    lines.append("- {}: 급등 전에 일반 대비 더 **낮은** 값 (급등 전 {:.4f} vs 일반 {:.4f})".format(
                        desc, info["surge_mean"], info["normal_mean"]))
            lines.append("")

    # Combined model description
    lines.append("## 3. 복합 스코어링 모델")
    lines.append("")
    lines.append("Level A의 강한 신호를 우선시하고, Level B/전체에서 추가 유의미 feature를 보완한 앙상블 모델.")
    lines.append("")
    lines.append("### 사용된 Feature 및 가중치")
    lines.append("")
    lines.append("| Feature | 가중치 | 방향 | 효과크기(d) | 출처 |")
    lines.append("|---------|--------|------|------------|------|")
    for fname in sorted(combined_model.keys(), key=lambda x: combined_model[x]["weight"], reverse=True):
        info = combined_model[fname]
        desc = feature_desc.get(fname, fname)
        dir_text = "급등 전 높음" if info["direction"] == "higher" else "급등 전 낮음"
        lines.append("| {} | {:.3f} | {} | {:+.3f} | {} |".format(
            desc, info["weight"], dir_text, info["cohens_d"], info.get("source", "?")))
    lines.append("")

    # PR analysis
    lines.append("### Precision/Recall 분석 (복합 모델)")
    lines.append("")
    lines.append("| 점수 기준 | Precision | Recall | F1 | TP | FP |")
    lines.append("|----------|-----------|--------|-----|----|----|")
    for p in full_model["combined_pr"]:
        lines.append("| >= {} | {:.3f} | {:.3f} | {:.3f} | {} | {} |".format(
            p["threshold"], p["precision"], p["recall"], p["f1"], p["tp"], p["fp"]))
    lines.append("")

    lines.append("## 4. 핵심 발견 요약")
    lines.append("")

    if "A" in lv:
        a_feats = [(f, i) for f, i in lv["A"].items() if abs(i["cohens_d"]) >= 0.3 and i["p_value"] < 0.01]
        if a_feats:
            lines.append("### Level A (100%+ 폭등) 전 가장 강한 신호:")
            lines.append("")
            for fname, info in sorted(a_feats, key=lambda x: abs(x[1]["cohens_d"]), reverse=True):
                desc = feature_desc.get(fname, fname)
                d_val = abs(info["cohens_d"])
                effect = "큰 효과" if d_val >= 0.8 else "중간 효과" if d_val >= 0.5 else "작은 효과"
                lines.append("1. **{}** (d={:+.3f}, {})".format(desc, info["cohens_d"], effect))
                lines.append("   - 급등 전 평균: {:.4f}, 일반: {:.4f}".format(info["surge_mean"], info["normal_mean"]))
            lines.append("")

    lines.append("## 5. 해석 가이드")
    lines.append("")
    lines.append("- **Cohen's d 해석**: |d| >= 0.2 작은 효과, >= 0.5 중간, >= 0.8 큰 효과")
    lines.append("- **점수 해석**: 0~100 범위, 높을수록 과거 급등 전 패턴과 유사")
    lines.append("- **주의**: 과거 패턴이 미래를 보장하지 않음. 시장 구조 변화, 규제, 거시경제 미반영")
    lines.append("- **활용**: 관심 목록 필터링 용도. 단독 매매 시그널로 사용 금지")
    lines.append("- Level A 모델은 극단적 급등(100%+)에 특화, 일반적 상승은 Level B/C 참고")
    lines.append("")

    report_path = BASE_DIR / "analysis_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Analysis report saved to {report_path}")


def generate_scan_report(scan_results, combined_model, models):
    """Generate scan_result.md."""
    feature_desc = {
        "vol_7d_30d_ratio": "거래량(7d/30d)",
        "vol_spike_3d": "볼륨 스파이크",
        "vol_trend_7d": "거래량 추세",
        "atr_14d_ratio": "ATR/가격",
        "volatility_contraction": "변동성 수축",
        "bb_width": "BB 폭",
        "price_vs_60d_high": "60일 고점 대비",
        "price_ret_7d": "7일 수익률",
        "price_vs_ma20": "MA20 괴리",
        "rsi_14": "RSI(14)",
        "rsi_consolidation_days": "RSI 횡보 일수",
        "rsi_trend_7d": "RSI 추세",
        "rsi_oversold_days": "RSI 과매도 일수",
        "consecutive_red": "연속 음봉",
        "green_ratio_3d": "3일 양봉비율",
        "doji_hammer_count": "Doji/Hammer",
        "body_range_ratio": "Body/Range 비율",
        "vs_btc_7d": "BTC 대비",
        "coin_ret_7d": "7일 수익률",
        "accumulation_signal": "축적 시그널"
    }

    lines = []
    lines.append("# 현재 시장 급등 전조 스캔 결과")
    lines.append("")
    lines.append("스캔 시각: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S KST')))
    lines.append("")

    lines.append("## 상위 20개 코인 (복합 점수)")
    lines.append("")

    top20 = scan_results[:20]
    if not top20:
        lines.append("스캔 결과 없음")
    else:
        # Check if level scores exist
        has_level = any("A" in r.get("level_scores", {}) for r in top20)

        header = "| 순위 | 코인 | 복합점수 |"
        sep = "|------|------|---------|"
        if has_level:
            header += " A점수 | B점수 |"
            sep += "-------|-------|"
        header += " 현재가 |"
        sep += "--------|"

        lines.append(header)
        lines.append(sep)
        for i, r in enumerate(top20):
            coin = r["symbol"].replace("USDT", "")
            row = "| {} | {} | {:.1f} |".format(i+1, coin, r["score"])
            if has_level:
                a_score = r.get("level_scores", {}).get("A", 0)
                b_score = r.get("level_scores", {}).get("B", 0)
                row += " {:.1f} | {:.1f} |".format(a_score, b_score)
            row += " ${:.4f} |".format(r["last_close"])
            lines.append(row)
        lines.append("")

    lines.append("## 상위 10개 코인 상세 분석")
    lines.append("")

    for i, r in enumerate(scan_results[:10]):
        coin = r["symbol"].replace("USDT", "")
        lines.append("### {}. {} (복합: {:.1f})".format(i+1, coin, r["score"]))
        if r.get("level_scores"):
            parts = []
            for lv, sc in sorted(r["level_scores"].items()):
                parts.append("Level {}: {:.1f}".format(lv, sc))
            lines.append("레벨별: {}".format(" / ".join(parts)))
        lines.append("")
        lines.append("| 지표 | 현재값 | 급등 전 평균 | 일반 평균 | 상태 |")
        lines.append("|------|--------|-------------|----------|------|")

        for fname in sorted(combined_model.keys(), key=lambda x: combined_model[x]["weight"], reverse=True):
            if fname in r["features"]:
                val = r["features"][fname]
                surge_m = combined_model[fname]["surge_mean"]
                normal_m = combined_model[fname]["normal_mean"]
                desc = feature_desc.get(fname, fname)
                direction = combined_model[fname]["direction"]
                mid = (surge_m + normal_m) / 2
                if direction == "higher":
                    if val >= surge_m:
                        status = "++ (급등 패턴 초과)"
                    elif val >= mid:
                        status = "+ (급등 패턴 근접)"
                    else:
                        status = "- (일반 수준)"
                else:
                    if val <= surge_m:
                        status = "++ (급등 패턴 초과)"
                    elif val <= mid:
                        status = "+ (급등 패턴 근접)"
                    else:
                        status = "- (일반 수준)"
                lines.append("| {} | {:.4f} | {:.4f} | {:.4f} | {} |".format(
                    desc, val, surge_m, normal_m, status))
        lines.append("")

    # Score distribution
    lines.append("## 점수 분포")
    lines.append("")
    scores = [r["score"] for r in scan_results]
    if scores:
        lines.append("- 전체 코인 수: {}".format(len(scores)))
        lines.append("- 평균 점수: {:.1f}".format(np.mean(scores)))
        lines.append("- 중앙값: {:.1f}".format(np.median(scores)))
        lines.append("- 최대값: {:.1f}".format(np.max(scores)))
        for thr in [80, 70, 60, 50]:
            cnt = sum(1 for s in scores if s >= thr)
            lines.append("- 점수 >= {}: {}개".format(thr, cnt))
    lines.append("")
    lines.append("---")
    lines.append("주의: 이 스캔 결과는 과거 패턴 기반 통계적 유사도 점수이며, 투자 조언이 아닙니다.")

    report_path = BASE_DIR / "scan_result.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Scan report saved to {report_path}")


def generate_scanner_script():
    """Write surge_scanner.py as a standalone script that loads thresholds.json."""
    scanner_path = BASE_DIR / "surge_scanner.py"
    # Read our own source to extract relevant functions
    # Instead, write a clean standalone script
    code = '''#!/usr/bin/env python3
"""
Crypto Surge Scanner - Standalone
Loads validated model from data/thresholds.json and scans cached market data.
Run pipeline_v2.py first to generate data and model.
"""
import json, os, time, math
import urllib.request, urllib.error
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

def load_thresholds():
    p = DATA_DIR / "thresholds.json"
    if not p.exists():
        print("Error: thresholds.json not found. Run pipeline_v2.py first.")
        return {}
    with open(p) as f:
        return json.load(f)

def compute_rsi(closes, period=14):
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    rsi = np.full(len(closes), np.nan)
    if len(deltas) < period: return rsi
    ag = np.mean(gains[:period]); al = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        ag = (ag*(period-1)+gains[i])/period; al = (al*(period-1)+losses[i])/period
        rsi[i+1] = 100.0 if al==0 else 100.0-100.0/(1.0+ag/al)
    fg=np.mean(gains[:period]); fl=np.mean(losses[:period])
    rsi[period] = 100.0 if fl==0 else 100.0-100.0/(1.0+fg/fl)
    return rsi

def compute_atr(highs, lows, closes, period=14):
    n=len(closes); tr=np.zeros(n); tr[0]=highs[0]-lows[0]
    for i in range(1,n): tr[i]=max(highs[i]-lows[i],abs(highs[i]-closes[i-1]),abs(lows[i]-closes[i-1]))
    atr=np.full(n,np.nan)
    if n>=period:
        atr[period-1]=np.mean(tr[:period])
        for i in range(period,n): atr[i]=(atr[i-1]*(period-1)+tr[i])/period
    return atr

def extract_features(df, idx, btc_df=None):
    if idx<44: return None
    c=df["close"].values; h=df["high"].values; l=df["low"].values; v=df["volume"].values; o=df["open"].values
    pe=idx; ps=idx-14
    if ps<0: return None
    f={}
    v7=np.mean(v[pe-7:pe]); v30=np.mean(v[max(0,pe-30):pe]); v20=np.mean(v[max(0,pe-20):pe])
    f["vol_7d_30d_ratio"]=v7/v30 if v30>0 else 1.0
    f["vol_spike_3d"]=np.max(v[pe-3:pe])/v20 if v20>0 else 1.0
    if pe>=7:
        vv=v[pe-7:pe]; x=np.arange(7,dtype=float)
        f["vol_trend_7d"]=(np.polyfit(x,vv,1)[0]/np.mean(vv)) if np.std(vv)>0 and np.mean(vv)>0 else 0.0
    else: f["vol_trend_7d"]=0.0
    af=compute_atr(h,l,c,14)
    a14=af[pe-1] if not np.isnan(af[pe-1]) else 0
    f["atr_14d_ratio"]=a14/c[pe-1] if c[pe-1]>0 else 0
    a7v=[af[j] for j in range(pe-7,pe) if not np.isnan(af[j])]
    a30v=[af[j] for j in range(max(0,pe-30),pe) if not np.isnan(af[j])]
    a7=np.mean(a7v) if a7v else a14; a30=np.mean(a30v) if a30v else a14
    f["volatility_contraction"]=a7/a30 if a30>0 else 1.0
    if pe>=20:
        bc=c[pe-20:pe]; bm=np.mean(bc); bs=np.std(bc)
        f["bb_width"]=(2*bs)/bm if bm>0 else 0
    else: f["bb_width"]=0
    f["price_vs_60d_high"]=c[pe-1]/np.max(h[pe-60:pe]) if pe>=60 and np.max(h[pe-60:pe])>0 else 1.0
    f["price_ret_7d"]=c[pe-1]/c[pe-8]-1 if pe>=8 and c[pe-8]>0 else 0
    f["price_vs_ma20"]=c[pe-1]/np.mean(c[pe-20:pe])-1 if pe>=20 and np.mean(c[pe-20:pe])>0 else 0
    rf=compute_rsi(c,14); rv=rf[pe-1]
    f["rsi_14"]=rv if not np.isnan(rv) else 50.0
    rw=rf[ps:pe]; vr=rw[~np.isnan(rw)]
    f["rsi_consolidation_days"]=int(np.sum((vr>=40)&(vr<=55)))
    f["rsi_trend_7d"]=rf[pe-1]-rf[pe-8] if pe>=8 and not np.isnan(rf[pe-1]) and not np.isnan(rf[pe-8]) else 0.0
    f["rsi_oversold_days"]=int(np.sum(vr<30)) if len(vr)>0 else 0
    cr=0
    for j in range(pe-1,max(ps-1,-1),-1):
        if c[j]<o[j]: cr+=1
        else: break
    f["consecutive_red"]=cr
    f["green_ratio_3d"]=sum(1 for j in range(pe-3,pe) if c[j]>=o[j])/3.0
    dh=0
    for j in range(pe-3,pe):
        bd=abs(c[j]-o[j]); tr2=h[j]-l[j]
        if tr2>0:
            if bd/tr2<0.1: dh+=1
            elif (min(o[j],c[j])-l[j])>2*bd and (min(o[j],c[j])-l[j])>0.5*tr2: dh+=1
    f["doji_hammer_count"]=dh
    bodies=[abs(c[j]-o[j]) for j in range(max(0,pe-7),pe)]
    ranges=[h[j]-l[j] for j in range(max(0,pe-7),pe)]
    f["body_range_ratio"]=np.mean(bodies)/np.mean(ranges) if np.mean(ranges)>0 else 0
    if btc_df is not None and pe>=8:
        cd=df["open_time"].dt.date.values; bd=btc_df["open_time"].dt.date.values
        bpn=bp7=None
        for bi in range(len(bd)-1,-1,-1):
            if bd[bi]<=cd[pe-1] and bpn is None: bpn=btc_df["close"].values[bi]
            if bd[bi]<=cd[pe-8] and bp7 is None: bp7=btc_df["close"].values[bi]
            if bpn is not None and bp7 is not None: break
        cr7=c[pe-1]/c[pe-8]-1 if c[pe-8]>0 else 0
        f["vs_btc_7d"]=(cr7-(bpn/bp7-1)) if bpn and bp7 and bp7>0 else 0.0
        f["coin_ret_7d"]=cr7
    else: f["vs_btc_7d"]=0.0; f["coin_ret_7d"]=0.0
    if pe>=14:
        pc14=abs(c[pe-1]/c[pe-14]-1); vc14=np.mean(v[pe-7:pe])/np.mean(v[pe-14:pe-7]) if np.mean(v[pe-14:pe-7])>0 else 1
        f["accumulation_signal"]=vc14/(1+pc14*10)
    else: f["accumulation_signal"]=1.0
    return f

def compute_score(features, thresholds):
    if not features or not thresholds: return 0.0
    sc=0.0; mx=0.0
    for fn,info in thresholds.items():
        if fn not in features: continue
        w=info["weight"]; mx+=w; val=features[fn]
        d=info["direction"]; sm=info["surge_mean"]; nm=info["normal_mean"]
        if d=="higher": r=(val-nm)/(sm-nm) if sm>nm else 0
        else: r=(nm-val)/(nm-sm) if nm>sm else 0
        sc+=w*max(0,min(1.5,r))
    return (sc/mx*100) if mx>0 else 0.0

def main():
    print("="*50)
    print("  Crypto Surge Precursor Scanner")
    print("="*50)
    TH=load_thresholds()
    if not TH: return
    pf=DATA_DIR/"top_pairs.json"
    if not pf.exists(): print("No pairs. Run pipeline_v2.py first."); return
    with open(pf) as ff: pairs=json.load(ff)
    bf=DATA_DIR/"BTCUSDT_1d.parquet"
    btc=pd.read_parquet(bf) if bf.exists() else None
    results=[]
    for sym in pairs:
        df_path=DATA_DIR/(sym+"_1d.parquet")
        if not df_path.exists(): continue
        df=pd.read_parquet(df_path)
        if len(df)<60: continue
        feat=extract_features(df,len(df),btc)
        if not feat: continue
        sc=compute_score(feat,TH)
        results.append({"symbol":sym,"score":sc,"features":feat,"last_close":float(df["close"].values[-1])})
    results.sort(key=lambda x:x["score"],reverse=True)
    print("\\nScanned {} coins\\n".format(len(results)))
    print("{:<5} {:<12} {:<8} {:<12}".format("Rank","Coin","Score","Price"))
    print("-"*40)
    for i,r in enumerate(results[:20]):
        cn=r["symbol"].replace("USDT","")
        print("{:<5} {:<12} {:<8.1f} ${:<12.4f}".format(i+1,cn,r["score"],r["last_close"]))
    print("\\n--- Top 10 Detail ---\\n")
    for i,r in enumerate(results[:10]):
        cn=r["symbol"].replace("USDT","")
        print("{}. {} (Score: {:.1f})".format(i+1,cn,r["score"]))
        for fn in sorted(TH.keys(),key=lambda x:TH[x]["weight"],reverse=True):
            if fn in r["features"]:
                print("   {}: {:.4f} (surge: {:.4f}, normal: {:.4f})".format(
                    fn,r["features"][fn],TH[fn]["surge_mean"],TH[fn]["normal_mean"]))
        print()

if __name__=="__main__": main()
'''
    with open(scanner_path, "w", encoding="utf-8") as f:
        f.write(code)
    os.chmod(scanner_path, 0o755)
    print(f"  Scanner script saved to {scanner_path}")


if __name__ == "__main__":
    run_pipeline()
