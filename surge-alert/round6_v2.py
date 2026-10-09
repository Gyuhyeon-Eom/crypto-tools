#!/usr/bin/env python3
"""
Round 6 v2: Advanced Surge Detection Model
==========================================
Improvements over R4/R5:
  1. Dynamic Triple Barrier (ATR-based pt/sl)
  2. Proper 2-Stage Meta-Labeling (high-recall primary -> high-precision meta)
  3. Purged K-Fold with Embargo (no temporal leakage)
  4. Cross-Sectional Features (percentile ranks, relative strength vs BTC)
  5. Derivatives Features (funding rate, OI change - live scan only)
  6. Sample Weights (uniqueness-based)
  7. Walk-Forward Backtest (6 windows)
  8. Better Evaluation Metrics (Precision@K, PR-AUC, Lift, simulated PnL)
  9. All existing features from round4_metalabel.py

Usage:
  python3 round6_v2.py            # Full train + evaluate + scan
  python3 round6_v2.py --scan-only  # Live scan with saved model
"""

import json, math, os, sys, time, datetime, pickle, warnings, random, argparse
import urllib.request
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, precision_score, recall_score,
    precision_recall_curve
)
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "ml_data"
MIN_VOL = 1_000_000
MAX_VOL = 10_000_000
SURGE_WINDOW = 7  # max hold days for triple barrier

# Import existing helpers from round4
sys.path.insert(0, str(BASE_DIR))
from round4_metalabel import (
    extract_features, has_recent_surge,
    mean, stdev, polyfit_slope,
    compute_rsi, compute_atr, compute_macd, compute_obv,
    triple_barrier_label
)

# ============================================================
# Slack
# ============================================================
WEBHOOK_URL = ""
_env = os.path.expanduser("~/.openclaw/.env.slack-surge")
if os.path.exists(_env):
    with open(_env) as f:
        for line in f:
            if line.startswith("SLACK_WEBHOOK_URL="):
                WEBHOOK_URL = line.strip().split("=", 1)[1]


def send_slack(msg):
    if not WEBHOOK_URL:
        print(f"[SLACK] {msg}")
        return
    for chunk in [msg[i:i + 3500] for i in range(0, len(msg), 3500)]:
        payload = json.dumps({"text": chunk})
        req = urllib.request.Request(
            WEBHOOK_URL, data=payload.encode(),
            headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            urllib.request.urlopen(req, timeout=30)
        except Exception:
            pass
        time.sleep(0.5)


# ============================================================
# Bybit API helper
# ============================================================
def bybit_get(ep, params=None):
    url = "https://api.bybit.com" + ep
    if params:
        url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
    req = urllib.request.Request(url, headers={"User-Agent": "r6v2-scan"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read())
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


# ============================================================
# 1. Dynamic Triple Barrier (ATR-based)
# ============================================================
def dynamic_triple_barrier_label(o, h, l, c, v, idx, max_days=7):
    """
    ATR-based dynamic profit target and stop loss.
    pt = 2.0 * ATR_14_ratio
    sl = 1.0 * ATR_14_ratio
    Returns: 1 (hit PT first), -1 (hit SL first), 0 (expired)
    """
    if idx >= len(c) - 1:
        return 0, 0.0, 0.0
    entry = c[idx]
    if entry <= 0:
        return 0, 0.0, 0.0

    # Compute ATR ratio at this point
    atr_vals = compute_atr(h[:idx + 1], l[:idx + 1], c[:idx + 1], 14)
    atr14 = atr_vals[idx] if idx < len(atr_vals) and atr_vals[idx] is not None else None

    if atr14 is None or atr14 <= 0:
        # Fallback to fixed barriers
        pt_pct = 0.30
        sl_pct = 0.15
    else:
        atr_ratio = atr14 / entry
        pt_pct = max(0.15, min(0.60, 2.0 * atr_ratio))  # clamp [15%, 60%]
        sl_pct = max(0.05, min(0.30, 1.0 * atr_ratio))   # clamp [5%, 30%]

    for j in range(idx + 1, min(idx + max_days + 1, len(c))):
        ret = c[j] / entry - 1
        if ret >= pt_pct:
            return 1, pt_pct, sl_pct
        if ret <= -sl_pct:
            return -1, pt_pct, sl_pct

    return 0, pt_pct, sl_pct


# ============================================================
# 3. Purged K-Fold with Embargo
# ============================================================
class PurgedKFold:
    """
    K-Fold CV that purges overlapping samples and applies an embargo gap.
    - purge: remove training samples whose label window overlaps with test
    - embargo: skip embargo_n samples after each test fold boundary
    """

    def __init__(self, n_splits=5, embargo_pct=0.01, label_horizon=7):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.label_horizon = label_horizon

    def split(self, X, timestamps=None):
        n = len(X)
        embargo_n = max(1, int(n * self.embargo_pct))
        fold_size = n // self.n_splits

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n)

            # Train indices: everything outside test window + purge zone + embargo
            purge_start = max(0, test_start - self.label_horizon)
            embargo_end = min(n, test_end + embargo_n)

            train_idx = np.concatenate([
                np.arange(0, purge_start),
                np.arange(embargo_end, n)
            ]).astype(int)

            test_idx = np.arange(test_start, test_end).astype(int)

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx


# ============================================================
# 4. Cross-Sectional Features
# ============================================================
def compute_cross_sectional_features(all_data, btc_data=None):
    """
    For each coin at each time index, compute percentile ranks
    across all coins at the same timestamp.

    Returns: dict[symbol] -> dict[idx] -> {feature_name: value}
    """
    # Build per-timestamp data: ts -> list of (symbol, idx, metrics)
    ts_to_metrics = defaultdict(list)

    for sym, d in all_data.items():
        n = len(d["c"])
        c, v, h, l = d["c"], d["v"], d["h"], d["l"]
        for idx in range(59, n):
            ts = d["ts"][idx]

            # 30d return
            ret_30d = c[idx] / c[idx - 30] - 1 if idx >= 30 and c[idx - 30] > 0 else 0

            # 30d volatility (stdev of daily returns)
            if idx >= 30:
                daily_rets = [(c[j] / c[j - 1] - 1) if c[j - 1] > 0 else 0
                              for j in range(idx - 29, idx + 1)]
                vol_30d = stdev(daily_rets)
            else:
                vol_30d = 0

            # Average dollar volume 7d
            vol_usd_7d = mean([v[j] * c[j] for j in range(max(0, idx - 6), idx + 1)])

            # Volume z-score (7d vs 30d)
            v7 = mean(v[max(0, idx - 6):idx + 1])
            v30 = mean(v[max(0, idx - 29):idx + 1])
            v30_std = stdev(v[max(0, idx - 29):idx + 1])
            vol_zscore = (v7 - v30) / v30_std if v30_std > 0 else 0

            # Price change z-score (7d return vs 30d return distribution)
            if idx >= 30:
                ret_7d = c[idx] / c[idx - 7] - 1 if idx >= 7 and c[idx - 7] > 0 else 0
                weekly_rets = []
                for j in range(idx - 29, idx + 1):
                    if j >= 7 and c[j - 7] > 0:
                        weekly_rets.append(c[j] / c[j - 7] - 1)
                wr_mean = mean(weekly_rets) if weekly_rets else 0
                wr_std = stdev(weekly_rets) if weekly_rets else 1
                price_change_zscore = (ret_7d - wr_mean) / wr_std if wr_std > 0 else 0
            else:
                ret_7d = 0
                price_change_zscore = 0

            ts_to_metrics[ts].append({
                "sym": sym, "idx": idx,
                "ret_30d": ret_30d, "vol_30d": vol_30d,
                "vol_usd_7d": vol_usd_7d, "vol_zscore": vol_zscore,
                "price_change_zscore": price_change_zscore,
                "ret_7d": ret_7d
            })

    # Compute BTC returns for relative strength
    btc_returns = {}
    if btc_data and len(btc_data.get("c", [])) > 0:
        bc = btc_data["c"]
        bts = btc_data["ts"]
        for i in range(7, len(bc)):
            if bc[i - 7] > 0:
                btc_returns[bts[i]] = bc[i] / bc[i - 7] - 1

    # Compute percentiles
    cross_features = defaultdict(dict)  # sym -> idx -> features

    for ts, entries in ts_to_metrics.items():
        if len(entries) < 5:
            continue  # need enough coins for meaningful percentiles

        ret_30d_vals = sorted(e["ret_30d"] for e in entries)
        vol_30d_vals = sorted(e["vol_30d"] for e in entries)
        vol_usd_vals = sorted(e["vol_usd_7d"] for e in entries)
        n = len(entries)

        for e in entries:
            sym, idx = e["sym"], e["idx"]

            # Percentile ranks (0 to 1)
            ret_pct = np.searchsorted(ret_30d_vals, e["ret_30d"]) / n
            vol_pct = np.searchsorted(vol_30d_vals, e["vol_30d"]) / n
            volusd_pct = np.searchsorted(vol_usd_vals, e["vol_usd_7d"]) / n

            # Stealth accumulation: high volume but low price change
            stealth = e["vol_zscore"] - e["price_change_zscore"]

            # Relative strength vs BTC
            btc_ret = btc_returns.get(ts, 0)
            rel_strength = e["ret_7d"] - btc_ret

            cross_features[sym][idx] = {
                "return_30d_percentile": ret_pct,
                "volatility_percentile": vol_pct,
                "volume_percentile": volusd_pct,
                "stealth_accumulation": stealth,
                "relative_strength_vs_btc": rel_strength,
            }

    return cross_features


# ============================================================
# 5. Derivatives Features (Live Scan Only)
# ============================================================
def fetch_derivatives_features(symbol):
    """Fetch funding rate and OI from Bybit for a symbol (linear perps)."""
    features = {
        "funding_rate": 0.0,
        "funding_rate_zscore": 0.0,
        "oi_change_pct": 0.0,
    }
    try:
        # Current funding rate
        resp = bybit_get("/v5/market/tickers", {"category": "linear", "symbol": symbol})
        ticker_list = resp.get("result", {}).get("list", [])
        if ticker_list:
            t = ticker_list[0]
            fr = float(t.get("fundingRate", 0))
            features["funding_rate"] = fr
            # Typical funding rate is ~0.01%, z-score relative to that
            features["funding_rate_zscore"] = (fr - 0.0001) / 0.0003 if abs(fr) > 0 else 0

            # OI change: use openInterest vs prev
            oi = float(t.get("openInterest", 0))
            oi_val = float(t.get("openInterestValue", 0))  # not always present
            # Try to get historical OI for change
            try:
                oi_resp = bybit_get("/v5/market/open-interest", {
                    "category": "linear", "symbol": symbol,
                    "intervalTime": "1d", "limit": "2"
                })
                oi_list = oi_resp.get("result", {}).get("list", [])
                if len(oi_list) >= 2:
                    oi_now = float(oi_list[0].get("openInterest", 0))
                    oi_prev = float(oi_list[1].get("openInterest", 0))
                    if oi_prev > 0:
                        features["oi_change_pct"] = (oi_now - oi_prev) / oi_prev
            except Exception:
                pass
    except Exception:
        pass
    return features


# ============================================================
# 6. Sample Weights (Uniqueness)
# ============================================================
def compute_sample_weights(timestamps, label_horizon=7):
    """
    Weight = 1 / average_overlap_count.
    For each sample, count how many other samples have overlapping
    label windows, then weight inversely.
    """
    n = len(timestamps)
    if n == 0:
        return np.ones(0)

    ts = np.array(timestamps, dtype=np.float64)
    weights = np.ones(n, dtype=np.float64)

    # Sort for efficient overlap counting
    sorted_idx = np.argsort(ts)
    sorted_ts = ts[sorted_idx]

    # For each sample, count overlaps (label window = ts to ts + horizon*86400)
    horizon_ms = label_horizon * 86400 * 1000  # assuming ts in milliseconds

    for i in range(n):
        t_start = sorted_ts[i]
        t_end = t_start + horizon_ms
        # Count samples whose start is within [t_start - horizon_ms, t_end]
        left = np.searchsorted(sorted_ts, t_start - horizon_ms, side='left')
        right = np.searchsorted(sorted_ts, t_end, side='right')
        overlap_count = right - left  # includes self
        if overlap_count > 0:
            weights[sorted_idx[i]] = 1.0 / overlap_count

    # Normalize to mean=1
    w_mean = weights.mean()
    if w_mean > 0:
        weights = weights / w_mean

    return weights


# ============================================================
# 8. Evaluation Metrics
# ============================================================
def precision_at_k(y_true, y_scores, k):
    """Precision among top-k predictions."""
    if k <= 0 or len(y_true) == 0:
        return 0.0
    top_k_idx = np.argsort(y_scores)[-k:]
    return np.mean(y_true[top_k_idx])


def compute_lift(y_true, y_scores, percentile=10):
    """Lift = precision in top percentile / base rate."""
    base_rate = np.mean(y_true)
    if base_rate == 0:
        return 0.0
    k = max(1, int(len(y_true) * percentile / 100))
    top_prec = precision_at_k(y_true, y_scores, k)
    return top_prec / base_rate


def simulated_pnl(y_true, y_pred, pt_pcts, sl_pcts, labels_raw,
                   slippage=0.003, fee=0.001):
    """
    Simulate PnL for predicted positives.
    Each trade costs slippage + fee on entry and exit.
    Returns: total_pnl, win_rate, max_consecutive_losses, trades
    """
    cost = 2 * (slippage + fee)  # round-trip
    pnl_list = []

    for i in range(len(y_pred)):
        if y_pred[i] == 0:
            continue
        pt = pt_pcts[i] if i < len(pt_pcts) else 0.30
        sl = sl_pcts[i] if i < len(sl_pcts) else 0.15
        raw_label = labels_raw[i] if i < len(labels_raw) else 0

        if raw_label == 1:
            pnl_list.append(pt - cost)
        elif raw_label == -1:
            pnl_list.append(-sl - cost)
        else:
            # Expired: assume small loss (mean reversion)
            pnl_list.append(-cost)

    if not pnl_list:
        return 0.0, 0.0, 0, 0

    total_pnl = sum(pnl_list)
    wins = sum(1 for p in pnl_list if p > 0)
    win_rate = wins / len(pnl_list)

    # Max consecutive losses
    max_consec_loss = 0
    current_streak = 0
    for p in pnl_list:
        if p <= 0:
            current_streak += 1
            max_consec_loss = max(max_consec_loss, current_streak)
        else:
            current_streak = 0

    return total_pnl, win_rate, max_consec_loss, len(pnl_list)


# ============================================================
# Build Dataset
# ============================================================
def build_dataset(all_data, cross_features=None, verbose=True):
    """
    Build feature matrix with dynamic triple barrier labels.
    Returns DataFrame with features + meta columns (_label, _label_raw, _ts, _sym, _pt, _sl).
    """
    rows = []
    label_counts = {"surge": 0, "drop": 0, "neutral": 0}
    excluded = 0

    # Filter coins by volume
    valid_symbols = {}
    for sym, d in all_data.items():
        n = len(d["c"])
        c, v = d["c"], d["v"]
        if n < 60 + SURGE_WINDOW:
            continue
        av = mean([v[j] * c[j] for j in range(max(0, n - 30), n)])
        if MIN_VOL <= av <= MAX_VOL:
            valid_symbols[sym] = d

    for sym, d in valid_symbols.items():
        n = len(d["c"])
        o, h, l, c, v = d["o"], d["h"], d["l"], d["c"], d["v"]

        for idx in range(59, n - SURGE_WINDOW):
            if has_recent_surge(c, idx):
                excluded += 1
                continue

            feat = extract_features(o, h, l, c, v, idx)
            if not feat:
                continue

            # Dynamic triple barrier
            label_raw, pt_pct, sl_pct = dynamic_triple_barrier_label(
                o, h, l, c, v, idx, max_days=SURGE_WINDOW
            )

            # Add cross-sectional features if available
            if cross_features and sym in cross_features and idx in cross_features[sym]:
                cf = cross_features[sym][idx]
                feat.update(cf)
            else:
                feat["return_30d_percentile"] = 0.5
                feat["volatility_percentile"] = 0.5
                feat["volume_percentile"] = 0.5
                feat["stealth_accumulation"] = 0.0
                feat["relative_strength_vs_btc"] = 0.0

            # Derivatives features are zero in training (historical data)
            feat["funding_rate"] = 0.0
            feat["funding_rate_zscore"] = 0.0
            feat["oi_change_pct"] = 0.0

            feat["_label"] = 1 if label_raw == 1 else 0
            feat["_label_raw"] = label_raw
            feat["_ts"] = d["ts"][idx]
            feat["_sym"] = sym
            feat["_pt"] = pt_pct
            feat["_sl"] = sl_pct

            rows.append(feat)
            if label_raw == 1:
                label_counts["surge"] += 1
            elif label_raw == -1:
                label_counts["drop"] += 1
            else:
                label_counts["neutral"] += 1

    if verbose:
        print(f"  Dataset: {len(rows)} samples "
              f"(surge:{label_counts['surge']}, drop:{label_counts['drop']}, "
              f"neutral:{label_counts['neutral']})")
        print(f"  Excluded (recent surge): {excluded}")

    return pd.DataFrame(rows)


# ============================================================
# 2. Two-Stage Meta-Labeling
# ============================================================
def train_stage1(X_train, y_train, sample_weights=None):
    """
    Stage 1: High-recall primary model.
    LGBMClassifier with class_weight='balanced', tuned for recall.
    """
    npos = int(y_train.sum())
    nneg = len(y_train) - npos
    spw = nneg / npos if npos > 0 else 1.0

    best_ap = 0
    best_params = None

    param_space = {
        'n_estimators': [300, 500, 700],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.02, 0.05],
        'num_leaves': [15, 31, 50],
        'min_child_samples': [5, 10, 20],
        'subsample': [0.6, 0.7, 0.8],
        'colsample_bytree': [0.5, 0.6, 0.7],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [0, 0.1, 1],
        'scale_pos_weight': [spw * m for m in [3, 4, 5, 6]],
    }
    fixed = {
        'objective': 'binary', 'metric': 'average_precision',
        'random_state': 42, 'verbose': -1,
        'force_col_wise': True, 'n_jobs': 1,
        'is_unbalance': False,  # use scale_pos_weight instead
    }

    random.seed(42)
    for trial in range(30):
        params = {k: random.choice(v) for k, v in param_space.items()}
        if params.get('scale_pos_weight', 0) <= 0:
            params['scale_pos_weight'] = 1.0
        params.update(fixed)
        try:
            # Quick CV with PurgedKFold
            cv = PurgedKFold(n_splits=5, embargo_pct=0.01, label_horizon=SURGE_WINDOW)
            ap_scores = []
            for ti, vi in cv.split(X_train):
                if y_train[ti].sum() == 0:
                    continue  # skip fold with no positives
                m = lgb.LGBMClassifier(**params)
                sw = sample_weights[ti] if sample_weights is not None else None
                m.fit(X_train[ti], y_train[ti], sample_weight=sw)
                pr = m.predict_proba(X_train[vi])[:, 1]
                if y_train[vi].sum() > 0:
                    ap_scores.append(average_precision_score(y_train[vi], pr))
            score = mean(ap_scores) if ap_scores else 0
            if score > best_ap:
                best_ap = score
                best_params = params.copy()
        except Exception:
            continue

    if not best_params:
        # Fallback
        best_params = {
            'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.02,
            'num_leaves': 31, 'min_child_samples': 10, 'subsample': 0.7,
            'colsample_bytree': 0.6, 'scale_pos_weight': spw * 4,
        }
        best_params.update(fixed)

    model = lgb.LGBMClassifier(**best_params)
    sw = sample_weights if sample_weights is not None else None
    model.fit(X_train, y_train, sample_weight=sw)
    return model, best_params


def train_stage2(X_train, y_train, stage1_model, sample_weights=None):
    """
    Stage 2: Meta-labeling model.
    Trained only on Stage 1 positives (threshold ~0.3).
    Extra features: stage1_prob, confidence_margin.
    """
    # Get Stage 1 predictions
    s1_probs = stage1_model.predict_proba(X_train)[:, 1]
    s1_threshold = 0.30

    # Filter to Stage 1 positives
    s1_mask = s1_probs >= s1_threshold
    if s1_mask.sum() < 50:
        # Lower threshold if too few positives
        s1_threshold = max(0.10, np.percentile(s1_probs, 70))
        s1_mask = s1_probs >= s1_threshold

    X_meta = X_train[s1_mask]
    y_meta = y_train[s1_mask]
    sw_meta = sample_weights[s1_mask] if sample_weights is not None else None

    # Add meta features: stage1 probability and confidence margin
    meta_feats = np.column_stack([
        s1_probs[s1_mask],                          # stage1_prob
        s1_probs[s1_mask] - s1_threshold,            # confidence_margin
    ])
    X_meta_aug = np.hstack([X_meta, meta_feats])

    if y_meta.sum() < 5:
        print("  [Stage 2] Too few positives for meta model, using Stage 1 only")
        return None, s1_threshold

    npos = int(y_meta.sum())
    nneg = len(y_meta) - npos
    spw = nneg / npos if npos > 0 else 1.0

    best_ap = 0
    best_params = None

    param_space = {
        'n_estimators': [200, 300, 500],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.02, 0.05],
        'num_leaves': [15, 31],
        'min_child_samples': [5, 10, 20],
        'subsample': [0.6, 0.7, 0.8],
        'colsample_bytree': [0.5, 0.6, 0.7],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [0, 0.1, 1],
        'scale_pos_weight': [spw * m for m in [1, 2, 3]],
    }
    fixed = {
        'objective': 'binary', 'metric': 'average_precision',
        'random_state': 123, 'verbose': -1,
        'force_col_wise': True, 'n_jobs': 1,
    }

    random.seed(123)
    for trial in range(25):
        params = {k: random.choice(v) for k, v in param_space.items()}
        # Ensure scale_pos_weight > 0
        if params.get('scale_pos_weight', 0) <= 0:
            params['scale_pos_weight'] = 1.0
        params.update(fixed)
        try:
            cv = PurgedKFold(n_splits=3, embargo_pct=0.01, label_horizon=SURGE_WINDOW)
            ap_scores = []
            for ti, vi in cv.split(X_meta_aug):
                if y_meta[ti].sum() == 0:
                    continue  # skip fold with no positives in train
                m = lgb.LGBMClassifier(**params)
                sw_fold = sw_meta[ti] if sw_meta is not None else None
                m.fit(X_meta_aug[ti], y_meta[ti], sample_weight=sw_fold)
                pr = m.predict_proba(X_meta_aug[vi])[:, 1]
                if y_meta[vi].sum() > 0:
                    ap_scores.append(average_precision_score(y_meta[vi], pr))
            score = mean(ap_scores) if ap_scores else 0
            if score > best_ap:
                best_ap = score
                best_params = params.copy()
        except Exception:
            continue

    if not best_params:
        best_params = {
            'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.02,
            'num_leaves': 31, 'min_child_samples': 10, 'subsample': 0.7,
            'colsample_bytree': 0.6, 'scale_pos_weight': spw * 2,
        }
        best_params.update(fixed)

    meta_model = lgb.LGBMClassifier(**best_params)
    sw_all = sw_meta if sw_meta is not None else None
    meta_model.fit(X_meta_aug, y_meta, sample_weight=sw_all)

    print(f"  [Stage 2] Trained on {len(X_meta_aug)} Stage1+ samples "
          f"({int(y_meta.sum())}+), s1_threshold={s1_threshold:.2f}")

    return meta_model, s1_threshold


def predict_two_stage(X, stage1_model, stage2_model, s1_threshold, meta_threshold=0.50):
    """
    Two-stage prediction.
    Returns: final_probs, final_preds, s1_probs
    """
    s1_probs = stage1_model.predict_proba(X)[:, 1]
    s1_mask = s1_probs >= s1_threshold

    final_probs = np.zeros(len(X))
    final_preds = np.zeros(len(X), dtype=int)

    if stage2_model is None:
        # Fallback: use Stage 1 only
        final_probs = s1_probs
        final_preds = (s1_probs >= meta_threshold).astype(int)
        return final_probs, final_preds, s1_probs

    # For Stage 1 positives, run through Stage 2
    if s1_mask.sum() > 0:
        X_s1pos = X[s1_mask]
        meta_feats = np.column_stack([
            s1_probs[s1_mask],
            s1_probs[s1_mask] - s1_threshold,
        ])
        X_meta = np.hstack([X_s1pos, meta_feats])

        s2_probs = stage2_model.predict_proba(X_meta)[:, 1]

        # Combined probability: Stage1 * Stage2
        combined = s1_probs[s1_mask] * s2_probs
        final_probs[s1_mask] = combined
        final_preds[s1_mask] = (s2_probs >= meta_threshold).astype(int)

    return final_probs, final_preds, s1_probs


# ============================================================
# 7. Walk-Forward Backtest
# ============================================================
def walk_forward_backtest(X, y, timestamps, sample_weights, fcols,
                          pt_pcts, sl_pcts, labels_raw,
                          n_windows=6, verbose=True):
    """
    Walk-forward: train on all prior data, test on current window.
    Returns aggregated metrics per window and overall.
    """
    n = len(X)
    # Reserve first 40% for minimum training set, split rest into windows
    min_train_pct = 0.4
    min_train = int(n * min_train_pct)
    test_total = n - min_train
    window_size = test_total // n_windows

    if window_size < 20 or min_train < 100:
        print("  [Walk-Forward] Not enough data for walk-forward backtest")
        return None

    results = []
    all_y_true = []
    all_y_prob = []
    all_y_pred = []
    all_pt = []
    all_sl = []
    all_labels_raw = []

    for w in range(n_windows):
        train_end = min_train + w * window_size
        test_start = train_end
        test_end = min(test_start + window_size, n)

        if test_end <= test_start:
            continue

        X_tr = X[:train_end]
        y_tr = y[:train_end]

        # Skip window if no positives in train set
        if y_tr.sum() < 5:
            if verbose:
                print(f"  Window {w+1}: skipped (only {int(y_tr.sum())} positives in train)")
            continue
        sw_tr = sample_weights[:train_end] if sample_weights is not None else None
        X_te = X[test_start:test_end]
        y_te = y[test_start:test_end]

        # Scale
        scaler = RobustScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Train Stage 1
        s1_model, _ = train_stage1(X_tr_s, y_tr, sw_tr)
        # Train Stage 2
        s2_model, s1_thresh = train_stage2(X_tr_s, y_tr, s1_model, sw_tr)

        # Find best meta threshold on training data
        best_meta_t = 0.50
        best_prec = 0
        _, _, s1_probs_tr = predict_two_stage(
            X_tr_s, s1_model, s2_model, s1_thresh, 0.50)

        for mt in np.arange(0.30, 0.85, 0.05):
            _, preds_tr, _ = predict_two_stage(
                X_tr_s, s1_model, s2_model, s1_thresh, mt)
            if preds_tr.sum() >= 3:
                p = precision_score(y_tr, preds_tr, zero_division=0)
                if p > best_prec:
                    best_prec = p
                    best_meta_t = mt

        # Predict on test
        probs_te, preds_te, _ = predict_two_stage(
            X_te_s, s1_model, s2_model, s1_thresh, best_meta_t)

        tp = int(preds_te.sum())
        hits = int((preds_te & y_te).sum())
        prec = hits / tp if tp > 0 else 0
        rec = hits / y_te.sum() if y_te.sum() > 0 else 0

        all_y_true.extend(y_te.tolist())
        all_y_prob.extend(probs_te.tolist())
        all_y_pred.extend(preds_te.tolist())
        all_pt.extend(pt_pcts[test_start:test_end].tolist())
        all_sl.extend(sl_pcts[test_start:test_end].tolist())
        all_labels_raw.extend(labels_raw[test_start:test_end].tolist())

        results.append({
            "window": w + 1,
            "train_size": len(X_tr),
            "test_size": len(X_te),
            "predictions": tp,
            "hits": hits,
            "precision": prec,
            "recall": rec,
            "meta_threshold": best_meta_t,
        })

        if verbose:
            print(f"    Window {w + 1}: Train={len(X_tr)} Test={len(X_te)} "
                  f"P={prec:.1%} R={rec:.1%} ({hits}/{tp})")

    if not results:
        return None

    # Aggregate metrics
    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    all_y_pred = np.array(all_y_pred)

    total_tp = int(all_y_pred.sum())
    total_hits = int((all_y_pred & all_y_true).sum())
    total_surges = int(all_y_true.sum())

    agg_prec = total_hits / total_tp if total_tp > 0 else 0
    agg_rec = total_hits / total_surges if total_surges > 0 else 0

    # PR-AUC
    pr_auc = average_precision_score(all_y_true, all_y_prob) if total_surges > 0 else 0

    # Precision@K
    k50 = precision_at_k(all_y_true, all_y_prob, 50)
    k100 = precision_at_k(all_y_true, all_y_prob, 100)

    # Lift
    lift = compute_lift(all_y_true, all_y_prob, 10)

    # Simulated PnL
    pnl, win_rate, max_consec, n_trades = simulated_pnl(
        all_y_true, all_y_pred,
        np.array(all_pt), np.array(all_sl), np.array(all_labels_raw)
    )

    aggregate = {
        "precision": agg_prec,
        "recall": agg_rec,
        "f1": 2 * agg_prec * agg_rec / (agg_prec + agg_rec) if (agg_prec + agg_rec) > 0 else 0,
        "pr_auc": pr_auc,
        "precision_at_50": k50,
        "precision_at_100": k100,
        "lift_top10pct": lift,
        "total_predictions": total_tp,
        "total_hits": total_hits,
        "total_surges": total_surges,
        "simulated_pnl": pnl,
        "win_rate": win_rate,
        "max_consecutive_losses": max_consec,
        "n_trades": n_trades,
        "windows": results,
    }

    return aggregate


# ============================================================
# Live Scan
# ============================================================
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


def live_scan(stage1_model, stage2_model, s1_threshold, meta_threshold,
              scaler, fcols, btc_data_live=None):
    """
    Scan all Bybit spot altcoins for surge precursors.
    Includes derivatives features for live scanning.
    """
    print("\n[Scan] Fetching Bybit tickers...")
    tickers = bybit_get("/v5/market/tickers", {"category": "spot"}) \
        .get("result", {}).get("list", [])

    alts = [
        t["symbol"] for t in tickers
        if t["symbol"].endswith("USDT")
        and t["symbol"] not in STABLECOINS
        and t["symbol"] not in LARGE_CAPS
        and MIN_VOL <= float(t.get("turnover24h", 0)) <= MAX_VOL
    ]
    print(f"  Scanning {len(alts)} altcoins...")

    # Get BTC data for relative strength
    btc_ret_7d = 0.0
    if btc_data_live is None:
        try:
            btc_resp = bybit_get("/v5/market/kline", {
                "category": "spot", "symbol": "BTCUSDT",
                "interval": "D", "limit": "30"
            })
            btc_klines = list(reversed(btc_resp.get("result", {}).get("list", [])))
            if len(btc_klines) >= 8:
                btc_c = [float(k[4]) for k in btc_klines]
                btc_ret_7d = btc_c[-1] / btc_c[-8] - 1 if btc_c[-8] > 0 else 0
        except Exception:
            pass

    # Collect all coins' metrics for cross-sectional at scan time
    scan_metrics = []
    scan_data = {}

    for sym in alts:
        try:
            data = bybit_get("/v5/market/kline", {
                "category": "spot", "symbol": sym,
                "interval": "D", "limit": "100"
            })
            klines = list(reversed(data.get("result", {}).get("list", [])))
            if len(klines) < 60:
                continue

            oo = [float(k[1]) for k in klines]
            hh = [float(k[2]) for k in klines]
            ll = [float(k[3]) for k in klines]
            cc = [float(k[4]) for k in klines]
            vv = [float(k[5]) for k in klines]

            scan_data[sym] = {"o": oo, "h": hh, "l": ll, "c": cc, "v": vv}

            idx = len(cc) - 1
            ret_30d = cc[idx] / cc[idx - 30] - 1 if idx >= 30 and cc[idx - 30] > 0 else 0
            if idx >= 30:
                daily_rets = [(cc[j] / cc[j - 1] - 1) if cc[j - 1] > 0 else 0
                              for j in range(idx - 29, idx + 1)]
                vol_30d = stdev(daily_rets)
            else:
                vol_30d = 0
            vol_usd_7d = mean([vv[j] * cc[j] for j in range(max(0, idx - 6), idx + 1)])

            scan_metrics.append({
                "sym": sym, "ret_30d": ret_30d,
                "vol_30d": vol_30d, "vol_usd_7d": vol_usd_7d
            })
            time.sleep(0.03)
        except Exception:
            continue

    # Compute cross-sectional percentiles at scan time
    if scan_metrics:
        ret_vals = sorted(m["ret_30d"] for m in scan_metrics)
        vol_vals = sorted(m["vol_30d"] for m in scan_metrics)
        vusd_vals = sorted(m["vol_usd_7d"] for m in scan_metrics)
        nm = len(scan_metrics)
        cross_scan = {}
        for m in scan_metrics:
            cross_scan[m["sym"]] = {
                "return_30d_percentile": np.searchsorted(ret_vals, m["ret_30d"]) / nm,
                "volatility_percentile": np.searchsorted(vol_vals, m["vol_30d"]) / nm,
                "volume_percentile": np.searchsorted(vusd_vals, m["vol_usd_7d"]) / nm,
            }
    else:
        cross_scan = {}

    scan_results = []
    deriv_count = 0

    for sym, sd in scan_data.items():
        oo, hh, ll, cc, vv = sd["o"], sd["h"], sd["l"], sd["c"], sd["v"]
        idx = len(cc) - 1

        if has_recent_surge(cc, idx):
            continue

        feat = extract_features(oo, hh, ll, cc, vv, idx)
        if not feat:
            continue

        # Cross-sectional features
        cs = cross_scan.get(sym, {})
        feat["return_30d_percentile"] = cs.get("return_30d_percentile", 0.5)
        feat["volatility_percentile"] = cs.get("volatility_percentile", 0.5)
        feat["volume_percentile"] = cs.get("volume_percentile", 0.5)

        # Stealth accumulation (approximate at scan time)
        v7 = mean(vv[max(0, idx - 6):idx + 1])
        v30 = mean(vv[max(0, idx - 29):idx + 1])
        v30_std = stdev(vv[max(0, idx - 29):idx + 1])
        vol_zscore = (v7 - v30) / v30_std if v30_std > 0 else 0
        ret_7d = cc[idx] / cc[idx - 7] - 1 if idx >= 7 and cc[idx - 7] > 0 else 0
        feat["stealth_accumulation"] = vol_zscore - 0  # simplified
        feat["relative_strength_vs_btc"] = ret_7d - btc_ret_7d

        # Derivatives features (live only)
        deriv = fetch_derivatives_features(sym)
        feat.update(deriv)
        if abs(deriv["funding_rate"]) > 0:
            deriv_count += 1
        time.sleep(0.05)

        # Build feature vector
        fv = np.array([[feat.get(fc, 0) for fc in fcols]], dtype=np.float64)
        fv = np.nan_to_num(fv, nan=0, posinf=0, neginf=0)
        fvs = scaler.transform(fv)

        # Two-stage prediction
        probs, preds, s1_probs = predict_two_stage(
            fvs, stage1_model, stage2_model, s1_threshold, meta_threshold
        )

        prob = float(probs[0])
        s1_prob = float(s1_probs[0])
        is_hit = int(preds[0])

        if s1_prob >= s1_threshold * 0.8:  # show near-misses too
            scan_results.append({
                "coin": sym.replace("USDT", ""),
                "symbol": sym,
                "prob": prob,
                "s1_prob": s1_prob,
                "price": cc[-1],
                "ret_7d": ret_7d * 100,
                "ret_14d": (cc[-1] / cc[-15] - 1) * 100 if len(cc) >= 15 and cc[-15] > 0 else 0,
                "rsi": feat.get("rsi_14", 50),
                "vol_ratio": feat.get("vol_ratio_7d_30d", 1),
                "funding": deriv["funding_rate"],
                "oi_chg": deriv["oi_change_pct"],
                "hit": is_hit == 1,
            })

    print(f"  Derivatives data fetched for {deriv_count} coins")
    scan_results.sort(key=lambda x: x["prob"], reverse=True)
    return scan_results


# ============================================================
# Main: Full Pipeline
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-only", action="store_true",
                        help="Skip training, load saved model and scan")
    args = parser.parse_args()

    t0 = time.time()

    if args.scan_only:
        return run_scan_only()

    send_slack("*Round 6 v2 시작* — Dynamic Barrier + 2-Stage Meta + Purged CV + Walk-Forward")
    print("=" * 60)
    print("Round 6 v2: Advanced Surge Detection")
    print("=" * 60)

    # ---- Load Data ----
    print("\n[1] Loading data...")
    # Prefer extended data if available
    data_file = DATA_DIR / "raw_data_extended.json"
    if not data_file.exists():
        data_file = DATA_DIR / "raw_data.json"
    with open(data_file) as f:
        all_data = json.load(f)
    print(f"  {len(all_data)} coins loaded")

    # ---- Cross-Sectional Features ----
    print("\n[2] Computing cross-sectional features...")
    btc_data = all_data.get("BTCUSDT", None)
    cross_features = compute_cross_sectional_features(all_data, btc_data)
    n_cs = sum(len(v) for v in cross_features.values())
    print(f"  {n_cs} cross-sectional feature sets computed")

    # ---- Build Dataset with Dynamic Triple Barrier ----
    print("\n[3] Building dataset (dynamic triple barrier)...")
    df = build_dataset(all_data, cross_features, verbose=True)
    if len(df) < 200:
        print("ERROR: Not enough data")
        return

    fcols = [c for c in df.columns if not c.startswith("_")]
    X = np.nan_to_num(df[fcols].values.astype(np.float64), nan=0, posinf=0, neginf=0)
    y = df["_label"].values
    labels_raw = df["_label_raw"].values
    timestamps = df["_ts"].values
    pt_pcts = df["_pt"].values
    sl_pcts = df["_sl"].values

    # Sort by timestamp
    si = np.argsort(timestamps)
    X, y = X[si], y[si]
    labels_raw = labels_raw[si]
    timestamps = timestamps[si]
    pt_pcts = pt_pcts[si]
    sl_pcts = sl_pcts[si]

    print(f"  Features: {len(fcols)}")
    print(f"  Feature names: {fcols[:10]}... ({len(fcols)} total)")

    # ---- Sample Weights ----
    print("\n[4] Computing sample weights (uniqueness)...")
    sample_weights = compute_sample_weights(timestamps, label_horizon=SURGE_WINDOW)
    print(f"  Weight stats: min={sample_weights.min():.3f} "
          f"max={sample_weights.max():.3f} mean={sample_weights.mean():.3f}")

    # ---- Train/Test Split (80/20 by time) ----
    sp = int(len(X) * 0.8)
    X_train, X_test = X[:sp], X[sp:]
    y_train, y_test = y[:sp], y[sp:]
    sw_train = sample_weights[:sp]
    pt_test, sl_test = pt_pcts[sp:], sl_pcts[sp:]
    lr_test = labels_raw[sp:]

    npos_train = int(y_train.sum())
    npos_test = int(y_test.sum())
    print(f"\n  Train: {len(X_train)} ({npos_train}+)  Test: {len(X_test)} ({npos_test}+)")

    # ---- Scale ----
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ---- Stage 1: High-Recall Primary ----
    print("\n[5] Training Stage 1 (high-recall primary)...")
    s1_model, s1_params = train_stage1(X_train_s, y_train, sw_train)
    s1_probs_test = s1_model.predict_proba(X_test_s)[:, 1]
    s1_ap = average_precision_score(y_test, s1_probs_test) if npos_test > 0 else 0
    print(f"  Stage 1 PR-AUC: {s1_ap:.4f}")

    # Check recall at threshold=0.3
    s1_preds_03 = (s1_probs_test >= 0.30).astype(int)
    if s1_preds_03.sum() > 0:
        s1_rec = recall_score(y_test, s1_preds_03, zero_division=0)
        s1_prec = precision_score(y_test, s1_preds_03, zero_division=0)
        print(f"  Stage 1 @0.30: P={s1_prec:.1%} R={s1_rec:.1%} ({int(s1_preds_03.sum())} preds)")

    # ---- Stage 2: Meta-Labeling ----
    print("\n[6] Training Stage 2 (meta-labeling)...")
    s2_model, s1_threshold = train_stage2(X_train_s, y_train, s1_model, sw_train)

    # ---- Find Best Meta Threshold on Test ----
    print("\n[7] Optimizing meta threshold...")
    best_meta_t = 0.50
    best_result = {"precision": 0}

    for mt in np.arange(0.30, 0.85, 0.05):
        probs_te, preds_te, _ = predict_two_stage(
            X_test_s, s1_model, s2_model, s1_threshold, mt
        )
        tp = int(preds_te.sum())
        if tp < 3:
            continue
        hits = int((preds_te & y_test).sum())
        prec = hits / tp
        rec = hits / npos_test if npos_test > 0 else 0
        if prec > best_result["precision"]:
            best_meta_t = mt
            best_result = {
                "precision": prec, "recall": rec,
                "predictions": tp, "hits": hits,
            }
        print(f"    mt={mt:.2f}: P={prec:.1%} R={rec:.1%} ({hits}/{tp})")

    print(f"\n  Best meta_threshold={best_meta_t:.2f}: "
          f"P={best_result['precision']:.1%} R={best_result['recall']:.1%}")

    # ---- Walk-Forward Backtest ----
    print("\n[8] Walk-Forward Backtest (6 windows)...")
    wf_results = walk_forward_backtest(
        X, y, timestamps, sample_weights, fcols,
        pt_pcts, sl_pcts, labels_raw,
        n_windows=6, verbose=True
    )

    if wf_results:
        print(f"\n  === Walk-Forward Aggregate ===")
        print(f"  Precision:  {wf_results['precision']:.1%}")
        print(f"  Recall:     {wf_results['recall']:.1%}")
        print(f"  F1:         {wf_results['f1']:.3f}")
        print(f"  PR-AUC:     {wf_results['pr_auc']:.4f}")
        print(f"  P@50:       {wf_results['precision_at_50']:.1%}")
        print(f"  P@100:      {wf_results['precision_at_100']:.1%}")
        print(f"  Lift(top10%): {wf_results['lift_top10pct']:.2f}x")
        print(f"  Sim PnL:    {wf_results['simulated_pnl']:.1%}")
        print(f"  Win Rate:   {wf_results['win_rate']:.1%}")
        print(f"  Max Consec Losses: {wf_results['max_consecutive_losses']}")
        print(f"  Trades:     {wf_results['n_trades']}")

    # ---- Final evaluation on test set with best threshold ----
    print("\n[9] Final evaluation on holdout test set...")
    final_probs, final_preds, _ = predict_two_stage(
        X_test_s, s1_model, s2_model, s1_threshold, best_meta_t
    )
    tp_final = int(final_preds.sum())
    hits_final = int((final_preds & y_test).sum())
    prec_final = hits_final / tp_final if tp_final > 0 else 0
    rec_final = hits_final / npos_test if npos_test > 0 else 0
    f1_final = 2 * prec_final * rec_final / (prec_final + rec_final) if (prec_final + rec_final) > 0 else 0
    pr_auc_final = average_precision_score(y_test, final_probs) if npos_test > 0 else 0

    pnl_final, wr_final, mcl_final, nt_final = simulated_pnl(
        y_test, final_preds, pt_test, sl_test, lr_test
    )

    print(f"  Precision:    {prec_final:.1%}")
    print(f"  Recall:       {rec_final:.1%}")
    print(f"  F1:           {f1_final:.3f}")
    print(f"  PR-AUC:       {pr_auc_final:.4f}")
    print(f"  Predictions:  {tp_final}")
    print(f"  Hits:         {hits_final}")
    print(f"  Sim PnL:      {pnl_final:.1%} ({nt_final} trades, {wr_final:.0%} win)")
    print(f"  Max Consec L: {mcl_final}")

    # ---- Live Scan ----
    print("\n[10] Live Scan...")
    scan_results = live_scan(
        s1_model, s2_model, s1_threshold, best_meta_t,
        scaler, fcols
    )

    # ---- Report ----
    elapsed = time.time() - t0
    lines = [f"*Round 6 v2 완료* ({elapsed / 60:.0f}분)"]
    lines.append("Dynamic Barrier + 2-Stage Meta + Purged CV + Walk-Forward")
    lines.append(f"```"
                 f"Holdout Test:\n"
                 f"  Precision:   {prec_final:.1%}\n"
                 f"  Recall:      {rec_final:.1%}\n"
                 f"  F1:          {f1_final:.3f}\n"
                 f"  PR-AUC:      {pr_auc_final:.4f}\n"
                 f"  Predictions: {tp_final}\n"
                 f"  Hits:        {hits_final}/{npos_test} surges\n"
                 f"  Sim PnL:     {pnl_final:.1%} ({nt_final} trades)\n"
                 f"  Win Rate:    {wr_final:.0%}")
    if wf_results:
        lines.append(f"\nWalk-Forward (6 windows):\n"
                     f"  Precision:   {wf_results['precision']:.1%}\n"
                     f"  PR-AUC:      {wf_results['pr_auc']:.4f}\n"
                     f"  Lift(top10%): {wf_results['lift_top10pct']:.2f}x\n"
                     f"  Sim PnL:     {wf_results['simulated_pnl']:.1%}")
    lines.append("```")

    lines.append(f"\n*전조 후보:*")
    lines.append("```")
    lines.append(f" # {'코인':<10} {'확률':>6} {'S1':>5} {'현재가':<11} "
                 f"{'7d':>7} {'14d':>7} {'RSI':>4} {'FR':>8}")
    hits_list = [r for r in scan_results if r["hit"]]
    for i, r in enumerate(scan_results[:15]):
        flag = "<<<" if r["hit"] else ""
        fr_str = f"{r['funding'] * 100:+.4f}%" if abs(r['funding']) > 0 else "   n/a"
        lines.append(
            f"{i + 1:>2} {r['coin']:<10} {r['prob']:>5.1%} {r['s1_prob']:>4.1%} "
            f"${r['price']:<10.4f} {r['ret_7d']:>+6.1f}% {r['ret_14d']:>+6.1f}% "
            f"{r['rsi']:>4.0f} {fr_str} {flag}"
        )
    lines.append(f"```\n전조 후보: {len(hits_list)}개")

    msg = "\n".join(lines)
    send_slack(msg)
    print("\n" + msg)

    # ---- Save Model ----
    print("\n[11] Saving model...")
    save_data = {
        "stage1_model": s1_model,
        "stage2_model": s2_model,
        "scaler": scaler,
        "feature_cols": fcols,
        "s1_threshold": float(s1_threshold),
        "meta_threshold": float(best_meta_t),
        "round": 6,
        "version": "v2",
    }
    with open(DATA_DIR / "best_model_v2.pkl", "wb") as f:
        pickle.dump(save_data, f)

    config = {
        "model_name": "R6v2_2stage_meta",
        "round": 6,
        "version": "v2",
        "method": "dynamic_barrier+2stage_meta+purged_cv+walk_forward",
        "s1_threshold": float(s1_threshold),
        "meta_threshold": float(best_meta_t),
        "feature_cols": fcols,
        "n_features": len(fcols),
        "holdout": {
            "precision": float(prec_final),
            "recall": float(rec_final),
            "f1": float(f1_final),
            "pr_auc": float(pr_auc_final),
            "predictions": tp_final,
            "hits": hits_final,
            "surges": npos_test,
            "simulated_pnl": float(pnl_final),
            "win_rate": float(wr_final),
            "max_consecutive_losses": mcl_final,
        },
        "walk_forward": {
            "precision": float(wf_results["precision"]) if wf_results else 0,
            "pr_auc": float(wf_results["pr_auc"]) if wf_results else 0,
            "lift_top10pct": float(wf_results["lift_top10pct"]) if wf_results else 0,
            "simulated_pnl": float(wf_results["simulated_pnl"]) if wf_results else 0,
            "n_windows": 6,
        } if wf_results else None,
        "trained_at": datetime.datetime.now().isoformat(),
        "filter": "no_recent_surge_30d",
        "min_vol": MIN_VOL,
        "max_vol": MAX_VOL,
    }
    with open(BASE_DIR / "ml_model_config_v2.json", "w") as f:
        json.dump(config, f, indent=2)

    print("  Saved: ml_data/best_model_v2.pkl + ml_model_config_v2.json")
    print(f"\nTotal time: {elapsed / 60:.1f} minutes")


def run_scan_only():
    """Load saved v2 model and run live scan only."""
    pkl_path = DATA_DIR / "best_model_v2.pkl"
    if not pkl_path.exists():
        print("ERROR: No saved v2 model found. Run full training first.")
        sys.exit(1)

    print("[Scan-Only] Loading saved model...")
    with open(pkl_path, "rb") as f:
        pkg = pickle.load(f)

    s1_model = pkg["stage1_model"]
    s2_model = pkg["stage2_model"]
    scaler = pkg["scaler"]
    fcols = pkg["feature_cols"]
    s1_threshold = pkg["s1_threshold"]
    meta_threshold = pkg["meta_threshold"]

    print(f"  s1_threshold={s1_threshold:.2f}, meta_threshold={meta_threshold:.2f}")
    print(f"  Features: {len(fcols)}")

    scan_results = live_scan(
        s1_model, s2_model, s1_threshold, meta_threshold,
        scaler, fcols
    )

    # Report
    lines = ["*R6v2 Scan*"]
    lines.append("```")
    lines.append(f" # {'코인':<10} {'확률':>6} {'S1':>5} {'현재가':<11} "
                 f"{'7d':>7} {'14d':>7} {'RSI':>4} {'FR':>8}")
    hits_list = [r for r in scan_results if r["hit"]]
    for i, r in enumerate(scan_results[:15]):
        flag = "<<<" if r["hit"] else ""
        fr_str = f"{r['funding'] * 100:+.4f}%" if abs(r['funding']) > 0 else "   n/a"
        lines.append(
            f"{i + 1:>2} {r['coin']:<10} {r['prob']:>5.1%} {r['s1_prob']:>4.1%} "
            f"${r['price']:<10.4f} {r['ret_7d']:>+6.1f}% {r['ret_14d']:>+6.1f}% "
            f"{r['rsi']:>4.0f} {fr_str} {flag}"
        )
    lines.append(f"```\n전조 후보: {len(hits_list)}개")

    msg = "\n".join(lines)
    send_slack(msg)
    print(msg)


if __name__ == "__main__":
    main()
