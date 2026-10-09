#!/usr/bin/env python3
"""
Crypto Surge Detector Pipeline
Phase 1: Data Collection (Binance API)
Phase 2: Surge Event Labeling
Phase 3: Precursor Feature Extraction
Phase 4: Statistical Validation
Phase 5: Current Market Scan
"""

import json
import os
import time
import math
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

BINANCE_BASE = "https://api.binance.com"

# ============================================================
# Phase 1: Data Collection
# ============================================================

def api_get(url, max_retries=3):
    """GET request with retry logic."""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  Retry {attempt+1}/{max_retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                return None


def get_top_usdt_pairs(n=100):
    """Get top N USDT pairs by 24h quote volume."""
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
    # Exclude stablecoins and non-ASCII symbols
    exclude = {"USDCUSDT", "BUSDUSDT", "TUSDUSDT", "DAIUSDT", "FDUSDUSDT", "EURUSDT", "USTUSDT", "USDPUSDT", "USD1USDT", "RLUSDUSDT"}
    usdt_pairs = [p for p in usdt_pairs if p["symbol"] not in exclude and p["symbol"].isascii()]
    pairs = [p["symbol"] for p in usdt_pairs[:n]]

    with open(cache_file, "w") as f:
        json.dump(pairs, f)
    print(f"[Phase 1] Selected {len(pairs)} pairs")
    return pairs


def fetch_klines(symbol, interval, start_ms, end_ms):
    """Fetch klines from Binance, handling pagination."""
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
        current_start = data[-1][0] + 1  # next ms after last candle
        time.sleep(0.5)  # rate limit

    return all_klines


def klines_to_df(klines):
    """Convert raw klines to DataFrame."""
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
    """Collect daily (365d) and 4h (180d) data for all pairs."""
    now_ms = int(time.time() * 1000)
    start_daily = now_ms - 365 * 24 * 3600 * 1000
    start_4h = now_ms - 180 * 24 * 3600 * 1000

    total = len(pairs)
    for i, sym in enumerate(pairs):
        daily_file = DATA_DIR / f"{sym}_1d.parquet"
        h4_file = DATA_DIR / f"{sym}_4h.parquet"

        if daily_file.exists() and h4_file.exists():
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

    # Always fetch BTC daily for relative strength
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
    """Find surge events in daily OHLCV data."""
    events = []
    n = len(df)
    if n < 30:
        return events

    closes = df["close"].values
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
# Phase 3: Feature Extraction
# ============================================================

def compute_rsi(closes, period=14):
    """Compute RSI array."""
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
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    # Fill initial RSI
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        first_avg_gain = np.mean(gains[:period])
        first_avg_loss = np.mean(losses[:period])
        if first_avg_loss > 0:
            rsi[period] = 100.0 - 100.0 / (1.0 + first_avg_gain / first_avg_loss)

    return rsi


def compute_atr(highs, lows, closes, period=14):
    """Compute ATR array."""
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
    """Extract precursor features for a given index (14 days before surge)."""
    if idx < 44:  # Need at least 44 days of history (30d vol avg + 14d lookback)
        return None

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    volumes = df["volume"].values
    opens = df["open"].values

    # Pre-surge window: idx-14 to idx-1 (the 14 days before surge start)
    pre_end = idx  # exclusive
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

    # Volume trend (linear regression slope over 7 days, normalized)
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

    # 14-day ATR / price ratio
    atr_14 = atr_full[pre_end-1] if not np.isnan(atr_full[pre_end-1]) else 0
    features["atr_14d_ratio"] = atr_14 / closes[pre_end-1] if closes[pre_end-1] > 0 else 0

    # Volatility contraction: 7d ATR / 30d ATR
    atr_7 = np.mean([atr_full[j] for j in range(pre_end-7, pre_end) if not np.isnan(atr_full[j])]) if pre_end >= 7 else atr_14
    atr_30 = np.mean([atr_full[j] for j in range(max(0, pre_end-30), pre_end) if not np.isnan(atr_full[j])]) if pre_end >= 30 else atr_14
    features["volatility_contraction"] = atr_7 / atr_30 if atr_30 > 0 else 1.0

    # Bollinger Band width (20d)
    if pre_end >= 20:
        bb_closes = closes[pre_end-20:pre_end]
        bb_mean = np.mean(bb_closes)
        bb_std = np.std(bb_closes)
        features["bb_width"] = (2 * bb_std) / bb_mean if bb_mean > 0 else 0
    else:
        features["bb_width"] = 0

    # Price position: current vs 60d high
    if pre_end >= 60:
        high_60d = np.max(highs[pre_end-60:pre_end])
        features["price_vs_60d_high"] = closes[pre_end-1] / high_60d if high_60d > 0 else 1.0
    else:
        features["price_vs_60d_high"] = 1.0

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

    # === Candle Patterns ===
    # Consecutive red candles before surge
    consec_red = 0
    for j in range(pre_end-1, max(pre_start-1, -1), -1):
        if closes[j] < opens[j]:
            consec_red += 1
        else:
            break
    features["consecutive_red"] = consec_red

    # Green candle ratio in last 3 days
    green_count = sum(1 for j in range(pre_end-3, pre_end) if closes[j] >= opens[j])
    features["green_ratio_3d"] = green_count / 3.0

    # Doji/Hammer detection in last 3 days
    doji_hammer = 0
    for j in range(pre_end-3, pre_end):
        body = abs(closes[j] - opens[j])
        total_range = highs[j] - lows[j]
        if total_range > 0:
            body_ratio = body / total_range
            lower_shadow = min(opens[j], closes[j]) - lows[j]
            # Doji: very small body
            if body_ratio < 0.1:
                doji_hammer += 1
            # Hammer: lower shadow > 2x body, body in upper third
            elif lower_shadow > 2 * body and (min(opens[j], closes[j]) - lows[j]) > 0.6 * total_range:
                doji_hammer += 1
    features["doji_hammer_count"] = doji_hammer

    # === Relative Strength ===
    if btc_df is not None and pre_end >= 8:
        # Align by date
        coin_dates = df["open_time"].dt.date.values
        btc_dates = btc_df["open_time"].dt.date.values
        coin_date = coin_dates[pre_end-1]
        coin_date_7d = coin_dates[pre_end-8]

        btc_price_now = None
        btc_price_7d = None
        for bi in range(len(btc_dates)-1, -1, -1):
            if btc_dates[bi] <= coin_date and btc_price_now is None:
                btc_price_now = btc_df["close"].values[bi]
            if btc_dates[bi] <= coin_date_7d and btc_price_7d is None:
                btc_price_7d = btc_df["close"].values[bi]
            if btc_price_now is not None and btc_price_7d is not None:
                break

        coin_ret_7d = (closes[pre_end-1] / closes[pre_end-8] - 1) if closes[pre_end-8] > 0 else 0
        if btc_price_now and btc_price_7d and btc_price_7d > 0:
            btc_ret_7d = btc_price_now / btc_price_7d - 1
            features["vs_btc_7d"] = coin_ret_7d - btc_ret_7d
        else:
            features["vs_btc_7d"] = 0.0
        features["coin_ret_7d"] = coin_ret_7d
    else:
        features["vs_btc_7d"] = 0.0
        features["coin_ret_7d"] = 0.0

    return features


def extract_random_non_surge_features(df, surge_indices, btc_df, n_samples=5):
    """Extract features from non-surge periods for comparison."""
    n = len(df)
    if n < 60:
        return []

    # Blocked indices: within 14 days before or 7 days after any surge
    blocked = set()
    for si in surge_indices:
        for j in range(max(0, si - 14), min(n, si + 8)):
            blocked.add(j)

    candidates = [i for i in range(44, n) if i not in blocked]
    if len(candidates) < n_samples:
        return []

    # Deterministic sampling: evenly spaced
    step = len(candidates) // n_samples
    sampled = [candidates[i * step] for i in range(n_samples)]

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
    """Compute Cohen's d effect size."""
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
    """Simple Mann-Whitney U test (normal approximation for p-value)."""
    nx, ny = len(x), len(y)
    if nx < 5 or ny < 5:
        return 1.0  # not enough data

    combined = [(v, 0) for v in x] + [(v, 1) for v in y]
    combined.sort(key=lambda t: t[0])

    # Assign ranks (handle ties by average rank)
    ranks = np.zeros(len(combined))
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-based
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
    # Two-tailed p-value from z (normal approximation)
    p = 2 * (1 - _norm_cdf(abs(z)))
    return p


def _norm_cdf(x):
    """Standard normal CDF approximation."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def validate_features(surge_features, normal_features):
    """Compare surge precursor features vs normal period features."""
    if not surge_features or not normal_features:
        return {}

    feature_names = [k for k in surge_features[0].keys() if not k.startswith("_")]
    results = {}

    for fname in feature_names:
        surge_vals = np.array([f[fname] for f in surge_features if fname in f], dtype=float)
        normal_vals = np.array([f[fname] for f in normal_features if fname in f], dtype=float)

        # Remove NaN
        surge_vals = surge_vals[~np.isnan(surge_vals)]
        normal_vals = normal_vals[~np.isnan(normal_vals)]

        if len(surge_vals) < 5 or len(normal_vals) < 5:
            continue

        d = cohens_d(surge_vals, normal_vals)
        p = mann_whitney_u(surge_vals, normal_vals)

        results[fname] = {
            "surge_mean": float(np.mean(surge_vals)),
            "surge_std": float(np.std(surge_vals)),
            "normal_mean": float(np.mean(normal_vals)),
            "normal_std": float(np.std(normal_vals)),
            "cohens_d": float(d),
            "p_value": float(p),
            "significant": p < 0.05 and abs(d) > 0.3
        }

    return results


def compute_composite_score(features, thresholds):
    """Compute a surge precursor score (0-100) based on validated thresholds."""
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
        direction = info["direction"]  # "higher" or "lower" means surge group has higher/lower value
        surge_mean = info["surge_mean"]
        normal_mean = info["normal_mean"]

        if direction == "higher":
            # Value closer to surge_mean (higher) scores more
            if normal_mean < surge_mean:
                ratio = (val - normal_mean) / (surge_mean - normal_mean) if surge_mean != normal_mean else 0
            else:
                ratio = 0
        else:
            # Value closer to surge_mean (lower) scores more
            if normal_mean > surge_mean:
                ratio = (normal_mean - val) / (normal_mean - surge_mean) if surge_mean != normal_mean else 0
            else:
                ratio = 0

        ratio = max(0, min(2, ratio))  # cap at 2x
        score += weight * min(ratio, 1.0)

    return (score / max_score * 100) if max_score > 0 else 0.0


# ============================================================
# Main Pipeline
# ============================================================

def run_pipeline():
    print("=" * 60)
    print("  Crypto Surge Precursor Detection Pipeline")
    print("=" * 60)

    # Phase 1
    pairs = get_top_usdt_pairs(100)
    collect_data(pairs)

    # Load BTC data
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

        # Sample non-surge features (more samples per coin for better stats)
        n_surge = len(surge_indices)
        n_normal = max(5, n_surge * 3)
        normal_feats = extract_random_non_surge_features(df, surge_indices, btc_df, n_normal)
        all_normal_features.extend(normal_feats)

    print(f"  Surge events: A={len(all_surge_features['A'])}, "
          f"B={len(all_surge_features['B'])}, C={len(all_surge_features['C'])}")
    print(f"  Normal samples: {len(all_normal_features)}")

    # Phase 4: Statistical Validation
    print("\n[Phase 4] Statistical validation...")

    # Validate across all surge levels combined
    validation_results = validate_features(all_surge_features["all"], all_normal_features)

    # Per-level validation
    level_results = {}
    for level in ["A", "B", "C"]:
        if len(all_surge_features[level]) >= 10:
            level_results[level] = validate_features(
                all_surge_features[level], all_normal_features
            )

    # Build threshold model from significant features
    significant_features = {}
    for fname, info in validation_results.items():
        if info["significant"]:
            direction = "higher" if info["surge_mean"] > info["normal_mean"] else "lower"
            # Weight by effect size
            weight = min(abs(info["cohens_d"]), 2.0)
            significant_features[fname] = {
                "weight": weight,
                "direction": direction,
                "surge_mean": info["surge_mean"],
                "surge_std": info["surge_std"],
                "normal_mean": info["normal_mean"],
                "normal_std": info["normal_std"],
                "cohens_d": info["cohens_d"],
                "p_value": info["p_value"]
            }

    print(f"  Significant features: {len(significant_features)}/{len(validation_results)}")
    for fname, info in sorted(significant_features.items(), key=lambda x: abs(x[1]["cohens_d"]), reverse=True):
        print(f"    {fname}: d={info['cohens_d']:.3f}, p={info['p_value']:.4f}, "
              f"surge={info['surge_mean']:.4f}, normal={info['normal_mean']:.4f}")

    # Compute precision/recall for composite score thresholds
    print("\n[Phase 4] Precision/Recall analysis...")
    all_surge_scores = []
    for feat in all_surge_features["all"]:
        clean_feat = {k: v for k, v in feat.items() if not k.startswith("_")}
        score = compute_composite_score(clean_feat, significant_features)
        all_surge_scores.append(score)

    all_normal_scores = []
    for feat in all_normal_features:
        score = compute_composite_score(feat, significant_features)
        all_normal_scores.append(score)

    pr_results = []
    for threshold in [30, 40, 50, 60, 70, 80]:
        tp = sum(1 for s in all_surge_scores if s >= threshold)
        fp = sum(1 for s in all_normal_scores if s >= threshold)
        fn = sum(1 for s in all_surge_scores if s < threshold)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        pr_results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "tp": tp, "fp": fp, "fn": fn
        })
        print(f"    Score >= {threshold}: precision={precision:.3f}, recall={recall:.3f} (TP={tp}, FP={fp})")

    # Save model
    model = {
        "significant_features": significant_features,
        "validation_results": {k: v for k, v in validation_results.items()},
        "level_results": level_results,
        "precision_recall": pr_results,
        "event_counts": {
            "A": len(all_surge_features["A"]),
            "B": len(all_surge_features["B"]),
            "C": len(all_surge_features["C"]),
            "total": len(all_surge_features["all"]),
            "normal": len(all_normal_features)
        },
        "generated_at": datetime.utcnow().isoformat()
    }
    model_file = DATA_DIR / "model.json"
    with open(model_file, "w") as f:
        json.dump(model, f, indent=2, default=str)
    print(f"\n  Model saved to {model_file}")

    # Generate analysis report
    generate_analysis_report(model, all_events, all_surge_features)

    # Phase 5: Current Market Scan
    print("\n[Phase 5] Scanning current market...")
    scan_results = scan_current_market(pairs, significant_features, btc_df)

    # Generate scan result report
    generate_scan_report(scan_results, significant_features)

    # Generate standalone scanner script
    generate_scanner_script(significant_features)

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print(f"  - analysis_report.md")
    print(f"  - scan_result.md")
    print(f"  - surge_scanner.py")
    print("=" * 60)


def scan_current_market(pairs, thresholds, btc_df):
    """Scan all pairs and compute surge precursor scores."""
    results = []

    for sym in pairs:
        daily_file = DATA_DIR / f"{sym}_1d.parquet"
        if not daily_file.exists():
            continue
        df = pd.read_parquet(daily_file)
        if len(df) < 60:
            continue

        # Extract features at the latest point
        idx = len(df)
        feat = extract_features(df, idx, btc_df)
        if not feat:
            continue

        score = compute_composite_score(feat, thresholds)
        results.append({
            "symbol": sym,
            "score": score,
            "features": feat,
            "last_close": float(df["close"].values[-1]),
            "last_date": str(df["open_time"].values[-1])[:10]
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def generate_analysis_report(model, all_events, all_surge_features):
    """Generate analysis_report.md in Korean."""
    sig = model["significant_features"]
    val = model["validation_results"]
    counts = model["event_counts"]
    pr = model["precision_recall"]
    level_results = model.get("level_results", {})

    lines = []
    lines.append("# 크립토 급등 전조증상 분석 리포트")
    lines.append("")
    lines.append(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S KST')}")
    lines.append("")

    lines.append("## 1. 데이터 개요")
    lines.append("")
    lines.append(f"- 분석 대상: USDT 페어 거래량 상위 100개 코인")
    lines.append(f"- 분석 기간: 최근 365일 일봉 데이터")
    lines.append(f"- 급등 이벤트 탐지 수:")
    lines.append(f"  - Level A (7일 내 +100%): {counts['A']}건")
    lines.append(f"  - Level B (7일 내 +50%): {counts['B']}건")
    lines.append(f"  - Level C (3일 내 +30%): {counts['C']}건")
    lines.append(f"  - 합계: {counts['total']}건")
    lines.append(f"- 비급등(일반) 기간 샘플: {counts['normal']}건")
    lines.append("")

    lines.append("## 2. Feature별 통계적 검증 결과")
    lines.append("")
    lines.append("| Feature | 급등 전 평균 | 일반 기간 평균 | Cohen's d | p-value | 유의미 |")
    lines.append("|---------|-------------|--------------|-----------|---------|--------|")

    for fname in sorted(val.keys(), key=lambda x: abs(val[x]["cohens_d"]), reverse=True):
        info = val[fname]
        sig_mark = "O" if info["significant"] else "X"
        lines.append(
            f"| {fname} | {info['surge_mean']:.4f} | {info['normal_mean']:.4f} | "
            f"{info['cohens_d']:.3f} | {info['p_value']:.4f} | {sig_mark} |"
        )
    lines.append("")

    lines.append("## 3. 유의미한 전조증상 패턴")
    lines.append("")

    if not sig:
        lines.append("통계적으로 유의미한 패턴이 발견되지 않았습니다.")
    else:
        lines.append(f"총 {len(sig)}개의 유의미한 feature가 발견되었습니다 (p < 0.05, |d| > 0.3).")
        lines.append("")

        feature_descriptions = {
            "vol_7d_30d_ratio": "7일 평균 거래량 / 30일 평균 거래량 비율",
            "vol_spike_3d": "3일 내 최대 거래량 / 20일 평균 거래량",
            "vol_trend_7d": "7일간 거래량 추세 (정규화된 기울기)",
            "atr_14d_ratio": "14일 ATR / 현재가 비율 (변동성)",
            "volatility_contraction": "7일 ATR / 30일 ATR (변동성 수축도)",
            "bb_width": "볼린저 밴드 폭 (20일)",
            "price_vs_60d_high": "현재가 / 60일 고점 비율",
            "rsi_14": "14일 RSI",
            "rsi_consolidation_days": "RSI 40~55 구간 체류 일수 (14일 중)",
            "rsi_trend_7d": "7일간 RSI 변화량",
            "consecutive_red": "연속 음봉 수",
            "green_ratio_3d": "최근 3일 양봉 비율",
            "doji_hammer_count": "최근 3일 Doji/Hammer 캔들 수",
            "vs_btc_7d": "BTC 대비 7일 초과 수익률",
            "coin_ret_7d": "7일 수익률"
        }

        for fname in sorted(sig.keys(), key=lambda x: abs(sig[x]["cohens_d"]), reverse=True):
            info = sig[fname]
            desc = feature_descriptions.get(fname, fname)
            direction_text = "높음" if info["direction"] == "higher" else "낮음"
            lines.append(f"### {desc}")
            lines.append(f"- 급등 전 평균: {info['surge_mean']:.4f} (std: {info['surge_std']:.4f})")
            lines.append(f"- 일반 기간 평균: {info['normal_mean']:.4f} (std: {info['normal_std']:.4f})")
            lines.append(f"- 효과 크기 (Cohen's d): {info['cohens_d']:.3f}")
            lines.append(f"- p-value: {info['p_value']:.6f}")
            lines.append(f"- 급등 전 값이 일반 대비 {direction_text}")
            lines.append(f"- 모델 가중치: {info['weight']:.3f}")
            lines.append("")

    lines.append("## 4. 복합 점수 Precision/Recall 분석")
    lines.append("")
    lines.append("유의미한 feature들을 가중 결합한 복합 점수(0~100)의 성능:")
    lines.append("")
    lines.append("| 점수 기준 | Precision | Recall | TP | FP | FN |")
    lines.append("|----------|-----------|--------|----|----|-----|")
    for p in pr:
        lines.append(
            f"| >= {p['threshold']} | {p['precision']:.3f} | {p['recall']:.3f} | "
            f"{p['tp']} | {p['fp']} | {p['fn']} |"
        )
    lines.append("")

    lines.append("## 5. 레벨별 분석")
    lines.append("")
    for level in ["A", "B", "C"]:
        level_name = {"A": "Level A (7일 내 +100%)", "B": "Level B (7일 내 +50%)", "C": "Level C (3일 내 +30%)"}
        if level in level_results and level_results[level]:
            lines.append(f"### {level_name[level]}")
            lines.append(f"- 이벤트 수: {counts[level]}건")
            lr = level_results[level]
            sig_feats = [f for f, v in lr.items() if v.get("significant")]
            lines.append(f"- 유의미 feature 수: {len(sig_feats)}")
            if sig_feats:
                lines.append(f"- 주요 패턴: {', '.join(sig_feats)}")
            lines.append("")

    lines.append("## 6. 해석 및 주의사항")
    lines.append("")
    lines.append("- 본 분석은 과거 데이터 기반 통계적 패턴 탐지이며, 미래 수익을 보장하지 않음")
    lines.append("- Cohen's d > 0.5는 중간 효과, > 0.8은 큰 효과로 해석")
    lines.append("- p-value < 0.05를 통계적 유의미 기준으로 사용")
    lines.append("- 복합 점수가 높을수록 과거 급등 전 패턴과 유사하나, false positive 존재")
    lines.append("- 시장 구조 변화, 규제, 거시경제 요인은 반영되지 않음")
    lines.append("- 투자 결정의 유일한 근거로 사용하지 말 것")
    lines.append("")

    report_path = BASE_DIR / "analysis_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Analysis report saved to {report_path}")


def generate_scan_report(scan_results, thresholds):
    """Generate scan_result.md."""
    lines = []
    lines.append("# 현재 시장 급등 전조 스캔 결과")
    lines.append("")
    lines.append(f"스캔 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S KST')}")
    lines.append("")

    feature_descriptions = {
        "vol_7d_30d_ratio": "거래량 비율(7d/30d)",
        "vol_spike_3d": "3일 볼륨 스파이크",
        "vol_trend_7d": "거래량 추세",
        "atr_14d_ratio": "변동성(ATR/가격)",
        "volatility_contraction": "변동성 수축",
        "bb_width": "BB 폭",
        "price_vs_60d_high": "60일 고점 대비",
        "rsi_14": "RSI(14)",
        "rsi_consolidation_days": "RSI 횡보 일수",
        "rsi_trend_7d": "RSI 추세",
        "consecutive_red": "연속 음봉",
        "green_ratio_3d": "3일 양봉비율",
        "doji_hammer_count": "Doji/Hammer",
        "vs_btc_7d": "BTC 대비 성과",
        "coin_ret_7d": "7일 수익률"
    }

    lines.append("## 상위 20개 코인")
    lines.append("")

    top_n = scan_results[:20]
    if not top_n:
        lines.append("스캔 결과 없음")
    else:
        lines.append("| 순위 | 코인 | 점수 | 현재가 |")
        lines.append("|------|------|------|--------|")
        for i, r in enumerate(top_n):
            lines.append(f"| {i+1} | {r['symbol'].replace('USDT', '')} | {r['score']:.1f} | ${r['last_close']:.4f} |")
        lines.append("")

        lines.append("## 상위 10개 코인 상세 분석")
        lines.append("")
        for i, r in enumerate(scan_results[:10]):
            coin = r["symbol"].replace("USDT", "")
            lines.append(f"### {i+1}. {coin} (점수: {r['score']:.1f})")
            lines.append("")
            lines.append("| 지표 | 현재값 | 급등 전 평균 | 일반 평균 | 기여 |")
            lines.append("|------|--------|-------------|----------|------|")

            for fname in sorted(thresholds.keys(), key=lambda x: thresholds[x]["weight"], reverse=True):
                if fname in r["features"]:
                    val = r["features"][fname]
                    surge_m = thresholds[fname]["surge_mean"]
                    normal_m = thresholds[fname]["normal_mean"]
                    desc = feature_descriptions.get(fname, fname)

                    # Determine if this feature contributes positively
                    direction = thresholds[fname]["direction"]
                    if direction == "higher":
                        contrib = "+" if val >= (surge_m + normal_m) / 2 else "-"
                    else:
                        contrib = "+" if val <= (surge_m + normal_m) / 2 else "-"

                    lines.append(f"| {desc} | {val:.4f} | {surge_m:.4f} | {normal_m:.4f} | {contrib} |")
            lines.append("")

    # Summary
    lines.append("## 점수 분포")
    lines.append("")
    scores = [r["score"] for r in scan_results]
    if scores:
        lines.append(f"- 전체 코인 수: {len(scores)}")
        lines.append(f"- 평균 점수: {np.mean(scores):.1f}")
        lines.append(f"- 중앙값: {np.median(scores):.1f}")
        lines.append(f"- 점수 >= 70: {sum(1 for s in scores if s >= 70)}개")
        lines.append(f"- 점수 >= 60: {sum(1 for s in scores if s >= 60)}개")
        lines.append(f"- 점수 >= 50: {sum(1 for s in scores if s >= 50)}개")
    lines.append("")

    lines.append("---")
    lines.append("주의: 이 스캔 결과는 과거 패턴 기반 통계적 유사도 점수이며, 투자 조언이 아닙니다.")

    report_path = BASE_DIR / "scan_result.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Scan report saved to {report_path}")


def generate_scanner_script(thresholds):
    """Generate standalone surge_scanner.py by writing model to JSON and referencing it."""
    # Save thresholds as JSON for the scanner to load
    thresholds_file = DATA_DIR / "thresholds.json"
    with open(thresholds_file, "w") as f:
        json.dump(thresholds, f, indent=2, default=str)

    # Write scanner script using regular string (no nested f-strings)
    scanner_path = BASE_DIR / "surge_scanner.py"
    with open(scanner_path, "w", encoding="utf-8") as fw:
        fw.write(SCANNER_TEMPLATE)
    os.chmod(scanner_path, 0o755)
    print(f"  Scanner script saved to {scanner_path}")


SCANNER_TEMPLATE = r'''#!/usr/bin/env python3
"""
Crypto Surge Scanner - Standalone Script
Scans current market using validated precursor patterns.
Uses cached data from data/ directory for deterministic results.
"""

import json
import os
import time
import math
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
BINANCE_BASE = "https://api.binance.com"


def load_thresholds():
    tf = DATA_DIR / "thresholds.json"
    if not tf.exists():
        print("No thresholds.json found. Run pipeline.py first.")
        return {}
    with open(tf) as f:
        return json.load(f)


def api_get(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None


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
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        fg = np.mean(gains[:period])
        fl = np.mean(losses[:period])
        if fl > 0:
            rsi[period] = 100.0 - 100.0 / (1.0 + fg / fl)
    return rsi


def compute_atr(highs, lows, closes, period=14):
    n = len(closes)
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


def extract_features(df, idx, btc_df=None):
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

    vol_7d = np.mean(volumes[pre_end-7:pre_end])
    vol_30d = np.mean(volumes[max(0, pre_end-30):pre_end])
    vol_20d = np.mean(volumes[max(0, pre_end-20):pre_end])
    features["vol_7d_30d_ratio"] = vol_7d / vol_30d if vol_30d > 0 else 1.0
    features["vol_spike_3d"] = np.max(volumes[pre_end-3:pre_end]) / vol_20d if vol_20d > 0 else 1.0

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

    atr_full = compute_atr(highs, lows, closes, 14)
    atr_14 = atr_full[pre_end-1] if not np.isnan(atr_full[pre_end-1]) else 0
    features["atr_14d_ratio"] = atr_14 / closes[pre_end-1] if closes[pre_end-1] > 0 else 0
    atr_7 = np.nanmean([atr_full[j] for j in range(pre_end-7, pre_end)])
    atr_30 = np.nanmean([atr_full[j] for j in range(max(0, pre_end-30), pre_end)])
    features["volatility_contraction"] = atr_7 / atr_30 if atr_30 > 0 else 1.0

    if pre_end >= 20:
        bb_closes = closes[pre_end-20:pre_end]
        bb_mean = np.mean(bb_closes)
        bb_std = np.std(bb_closes)
        features["bb_width"] = (2 * bb_std) / bb_mean if bb_mean > 0 else 0
    else:
        features["bb_width"] = 0

    if pre_end >= 60:
        high_60d = np.max(highs[pre_end-60:pre_end])
        features["price_vs_60d_high"] = closes[pre_end-1] / high_60d if high_60d > 0 else 1.0
    else:
        features["price_vs_60d_high"] = 1.0

    rsi_full = compute_rsi(closes, 14)
    rsi_val = rsi_full[pre_end-1]
    features["rsi_14"] = rsi_val if not np.isnan(rsi_val) else 50.0
    rsi_window = rsi_full[pre_start:pre_end]
    valid_rsi = rsi_window[~np.isnan(rsi_window)]
    features["rsi_consolidation_days"] = int(np.sum((valid_rsi >= 40) & (valid_rsi <= 55)))
    if pre_end >= 8 and not np.isnan(rsi_full[pre_end-1]) and not np.isnan(rsi_full[pre_end-8]):
        features["rsi_trend_7d"] = rsi_full[pre_end-1] - rsi_full[pre_end-8]
    else:
        features["rsi_trend_7d"] = 0.0

    consec_red = 0
    for j in range(pre_end-1, max(pre_start-1, -1), -1):
        if closes[j] < opens[j]:
            consec_red += 1
        else:
            break
    features["consecutive_red"] = consec_red
    green_count = sum(1 for j in range(pre_end-3, pre_end) if closes[j] >= opens[j])
    features["green_ratio_3d"] = green_count / 3.0

    doji_hammer = 0
    for j in range(pre_end-3, pre_end):
        body = abs(closes[j] - opens[j])
        total_range = highs[j] - lows[j]
        if total_range > 0:
            body_ratio = body / total_range
            lower_shadow = min(opens[j], closes[j]) - lows[j]
            if body_ratio < 0.1:
                doji_hammer += 1
            elif lower_shadow > 2 * body and (min(opens[j], closes[j]) - lows[j]) > 0.6 * total_range:
                doji_hammer += 1
    features["doji_hammer_count"] = doji_hammer

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
        coin_ret_7d = (closes[pre_end-1] / closes[pre_end-8] - 1) if closes[pre_end-8] > 0 else 0
        if btc_price_now and btc_price_7d and btc_price_7d > 0:
            btc_ret_7d = btc_price_now / btc_price_7d - 1
            features["vs_btc_7d"] = coin_ret_7d - btc_ret_7d
        else:
            features["vs_btc_7d"] = 0.0
        features["coin_ret_7d"] = coin_ret_7d
    else:
        features["vs_btc_7d"] = 0.0
        features["coin_ret_7d"] = 0.0

    return features


def compute_score(features, thresholds):
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
            if normal_mean < surge_mean:
                ratio = (val - normal_mean) / (surge_mean - normal_mean) if surge_mean != normal_mean else 0
            else:
                ratio = 0
        else:
            if normal_mean > surge_mean:
                ratio = (normal_mean - val) / (normal_mean - surge_mean) if surge_mean != normal_mean else 0
            else:
                ratio = 0
        ratio = max(0, min(2, ratio))
        score += weight * min(ratio, 1.0)
    return (score / max_score * 100) if max_score > 0 else 0.0


def main():
    print("=" * 50)
    print("  Crypto Surge Precursor Scanner")
    print("=" * 50)

    THRESHOLDS = load_thresholds()
    if not THRESHOLDS:
        return

    pairs_file = DATA_DIR / "top_pairs.json"
    if not pairs_file.exists():
        print("No pairs cache. Run pipeline.py first.")
        return
    with open(pairs_file) as f:
        pairs = json.load(f)

    btc_file = DATA_DIR / "BTCUSDT_1d.parquet"
    btc_df = pd.read_parquet(btc_file) if btc_file.exists() else None

    results = []
    for sym in pairs:
        daily_file = DATA_DIR / (sym + "_1d.parquet")
        if not daily_file.exists():
            continue
        df = pd.read_parquet(daily_file)
        if len(df) < 60:
            continue
        idx = len(df)
        feat = extract_features(df, idx, btc_df)
        if not feat:
            continue
        score = compute_score(feat, THRESHOLDS)
        results.append({
            "symbol": sym,
            "score": score,
            "features": feat,
            "last_close": float(df["close"].values[-1])
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    print("\nScanned {} coins\n".format(len(results)))
    print("{:<5} {:<12} {:<8} {:<12}".format("Rank", "Coin", "Score", "Price"))
    print("-" * 40)
    for i, r in enumerate(results[:20]):
        coin = r["symbol"].replace("USDT", "")
        print("{:<5} {:<12} {:<8.1f} ${:<12.4f}".format(i+1, coin, r["score"], r["last_close"]))

    print("\n--- Top 10 Detail ---\n")
    for i, r in enumerate(results[:10]):
        coin = r["symbol"].replace("USDT", "")
        print("{}. {} (Score: {:.1f})".format(i+1, coin, r["score"]))
        for fname in sorted(THRESHOLDS.keys(), key=lambda x: THRESHOLDS[x]["weight"], reverse=True):
            if fname in r["features"]:
                val = r["features"][fname]
                surge_m = THRESHOLDS[fname]["surge_mean"]
                normal_m = THRESHOLDS[fname]["normal_mean"]
                print("   {}: {:.4f} (surge avg: {:.4f}, normal avg: {:.4f})".format(fname, val, surge_m, normal_m))
        print()


if __name__ == "__main__":
    main()
'''


if __name__ == "__main__":
    run_pipeline()
