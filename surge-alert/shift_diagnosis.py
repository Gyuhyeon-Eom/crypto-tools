#!/usr/bin/env python3
"""
Distribution Shift 진단 + Ablation Study
1. Adversarial Validation (train vs test 분포 차이 측정)
2. PSI (피처별 분포 안정성)
3. Ablation Study (R7 컴포넌트별 영향 분리)
"""

import json, os, sys, time, pickle, warnings, random
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_score, recall_score, average_precision_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "ml_data"
SURGE_WINDOW = 7

sys.path.insert(0, str(BASE_DIR))
from round4_metalabel import extract_features, has_recent_surge, mean, stdev, compute_atr

# Slack
WEBHOOK_URL = ""
_env = os.path.expanduser("~/.openclaw/.env.slack-surge")
if os.path.exists(_env):
    with open(_env) as f:
        for line in f:
            if line.startswith("SLACK_WEBHOOK_URL="):
                WEBHOOK_URL = line.strip().split("=", 1)[1]

def send_slack(msg):
    return  # Retired: replaced by momentum-alert; historical analysis remains available.
    if not WEBHOOK_URL:
        print(msg); return
    for chunk in [msg[i:i+3500] for i in range(0, len(msg), 3500)]:
        payload = json.dumps({"text": chunk})
        req = urllib.request.Request(WEBHOOK_URL, data=payload.encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try: urllib.request.urlopen(req, timeout=30)
        except: pass
        time.sleep(0.5)


# ============================================================
# Dynamic Triple Barrier (from R7)
# ============================================================
def dynamic_triple_barrier_label(o, h, l, c, v, idx, max_days=7):
    if idx >= len(c) - 1: return 0
    entry = c[idx]
    if entry <= 0: return 0
    atr_vals = compute_atr(h[:idx+1], l[:idx+1], c[:idx+1], 14)
    atr14 = atr_vals[idx] if idx < len(atr_vals) and atr_vals[idx] else None
    if atr14 is None or atr14 <= 0:
        pt_pct, sl_pct = 0.30, 0.15
    else:
        atr_ratio = atr14 / entry
        pt_pct = max(0.15, min(0.60, 2.0 * atr_ratio))
        sl_pct = max(0.05, min(0.30, 1.0 * atr_ratio))
    for j in range(idx+1, min(idx+max_days+1, len(c))):
        ret = c[j] / entry - 1
        if ret >= pt_pct: return 1
        if ret <= -sl_pct: return -1
    return 0


# ============================================================
# Cross-sectional features
# ============================================================
def build_xs_features(all_data, base_features_by_sym):
    """Build cross-sectional percentile features"""
    # Group by timestamp
    ts_groups = {}
    for sym, feat_list in base_features_by_sym.items():
        for feat_dict in feat_list:
            ts = feat_dict["_ts"]
            if ts not in ts_groups:
                ts_groups[ts] = []
            ts_groups[ts].append((sym, feat_dict))
    
    # For each timestamp, compute percentiles
    top20 = ['momentum_acceleration', 'ma5_ma10_cross', 'avg_upper_wick', 'ma20_slope',
             'volatility_ratio_7_30', 'price_vs_60d_high', 'price_pos_60d', 'recovery_from_low',
             'avg_lower_wick', 'vol_trend_14d', 'macd_histogram', 'rsi_range_14d',
             'bb_width', 'rsi_max_14d', 'volatility_contraction', 'range_squeeze',
             'vol_ratio_7d_30d', 'rsi_min_14d', 'vol_usd_7d', 'atr_ratio']
    
    xs_lookup = {}  # (sym, ts) -> xs_features
    
    for ts, items in ts_groups.items():
        if len(items) < 5:
            continue
        for feat_name in top20:
            vals = [it[1].get(feat_name, 0) for it in items]
            sorted_vals = sorted(vals)
            n = len(sorted_vals)
            for sym, fd in items:
                v = fd.get(feat_name, 0)
                rank = sum(1 for x in sorted_vals if x <= v) / n
                key = (sym, ts)
                if key not in xs_lookup:
                    xs_lookup[key] = {}
                xs_lookup[key][f"{feat_name}_xs_pct"] = rank
    
    # stealth score
    for ts, items in ts_groups.items():
        if len(items) < 5:
            continue
        rets = sorted([it[1].get("ret_21d", 0) for it in items])
        vols = sorted([it[1].get("atr_ratio", 0) for it in items])
        n = len(rets)
        for sym, fd in items:
            key = (sym, ts)
            if key not in xs_lookup:
                xs_lookup[key] = {}
            ret_pct = sum(1 for x in rets if x <= fd.get("ret_21d", 0)) / n
            vol_pct = sum(1 for x in vols if x <= fd.get("atr_ratio", 0)) / n
            xs_lookup[key]["stealth_score"] = int(ret_pct < 0.3) + int(vol_pct < 0.4)
    
    return xs_lookup, top20


# ============================================================
# Build dataset
# ============================================================
def build_dataset(all_data, include_xs=True):
    """Build features + labels"""
    MIN_VOL, MAX_VOL = 1_000_000, 10_000_000
    
    # First pass: extract base features
    base_features_by_sym = {}
    rows_base = []
    
    for sym, d in all_data.items():
        n = len(d["c"])
        o, h, l, c, v = d["o"], d["h"], d["l"], d["c"], d["v"]
        av = mean([v[j]*c[j] for j in range(max(0, n-30), n)])
        if av > MAX_VOL or av < MIN_VOL:
            continue
        
        sym_feats = []
        for idx in range(59, n - SURGE_WINDOW):
            if has_recent_surge(c, idx):
                continue
            feat = extract_features(o, h, l, c, v, idx)
            if not feat:
                continue
            label = dynamic_triple_barrier_label(o, h, l, c, v, idx)
            feat["_label"] = 1 if label == 1 else 0
            feat["_ts"] = d["ts"][idx]
            feat["_sym"] = sym
            sym_feats.append(feat)
            rows_base.append(feat)
        
        if sym_feats:
            base_features_by_sym[sym] = sym_feats
    
    # XS features
    xs_lookup = {}
    xs_feat_names = []
    if include_xs:
        xs_lookup, top20 = build_xs_features(all_data, base_features_by_sym)
        xs_feat_names = [f"{f}_xs_pct" for f in top20] + ["stealth_score"]
    
    # Build final rows
    for row in rows_base:
        key = (row["_sym"], row["_ts"])
        if key in xs_lookup:
            row.update(xs_lookup[key])
    
    df = pd.DataFrame(rows_base)
    base_fcols = [c for c in df.columns if not c.startswith("_")]
    
    if not include_xs:
        base_fcols = [c for c in base_fcols if "_xs_pct" not in c and c != "stealth_score"]
    
    X = np.nan_to_num(df[base_fcols].values.astype(np.float64), nan=0, posinf=0, neginf=0)
    y = df["_label"].values
    ts_arr = df["_ts"].values
    
    # Sort by time
    si = np.argsort(ts_arr)
    X, y, ts_arr = X[si], y[si], ts_arr[si]
    
    sp = int(len(X) * 0.8)
    return X[:sp], X[sp:], y[:sp], y[sp:], ts_arr[:sp], ts_arr[sp:], base_fcols


# ============================================================
# PurgedKFold
# ============================================================
class PurgedKFold:
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
            test_end = min(test_start + fold_size, n)
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
# 1. ADVERSARIAL VALIDATION
# ============================================================
def adversarial_validation(X_train, X_test, fcols):
    print("\n" + "="*60)
    print("  ADVERSARIAL VALIDATION")
    print("="*60)
    
    X_combined = np.vstack([X_train, X_test])
    is_test = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    scores = cross_val_score(clf, X_combined, is_test, scoring='roc_auc', cv=5)
    auc = scores.mean()
    
    print(f"\n  AUC (train vs test 구분 능력): {auc:.3f}")
    if auc < 0.6:
        print("  → SAFE: 분포 거의 동일. Shift 없음.")
    elif auc < 0.75:
        print("  → MILD: 약간의 shift. 주의 필요.")
    elif auc < 0.9:
        print("  → SIGNIFICANT: 상당한 shift. 피처 정리 필요.")
    else:
        print("  → SEVERE: 심각한 shift. 현재 데이터로 안정 모델 어려움.")
    
    # Feature importance for shift
    clf.fit(X_combined, is_test)
    importances = pd.Series(clf.feature_importances_, index=fcols)
    top_shift = importances.sort_values(ascending=False).head(20)
    
    print(f"\n  Shift 주범 피처 TOP 20:")
    for feat, imp in top_shift.items():
        flag = " ← XS" if "_xs_pct" in feat or feat == "stealth_score" else ""
        print(f"    {feat:<35} {imp:.4f}{flag}")
    
    xs_total = sum(imp for feat, imp in importances.items() if "_xs_pct" in feat or feat == "stealth_score")
    base_total = sum(imp for feat, imp in importances.items() if "_xs_pct" not in feat and feat != "stealth_score")
    print(f"\n  XS 피처 총 importance: {xs_total:.3f}")
    print(f"  Base 피처 총 importance: {base_total:.3f}")
    
    return auc, top_shift


# ============================================================
# 2. PSI (Population Stability Index)
# ============================================================
def calculate_psi(train, test, bins=10):
    eps = 1e-9
    breakpoints = np.percentile(train, np.linspace(0, 100, bins+1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        return 0.0
    train_dist = np.histogram(train, breakpoints)[0] / len(train) + eps
    test_dist = np.histogram(test, breakpoints)[0] / len(test) + eps
    psi = np.sum((test_dist - train_dist) * np.log(test_dist / train_dist))
    return psi


def psi_analysis(X_train, X_test, fcols):
    print("\n" + "="*60)
    print("  PSI ANALYSIS (Population Stability Index)")
    print("="*60)
    
    psi_scores = {}
    for i, feat in enumerate(fcols):
        psi = calculate_psi(X_train[:, i], X_test[:, i])
        psi_scores[feat] = psi
    
    psi_df = pd.Series(psi_scores).sort_values(ascending=False)
    
    severe = psi_df[psi_df > 0.25]
    moderate = psi_df[(psi_df > 0.1) & (psi_df <= 0.25)]
    stable = psi_df[psi_df <= 0.1]
    
    print(f"\n  전체: {len(fcols)}개 피처")
    print(f"  심각 (PSI > 0.25): {len(severe)}개 ← 제거 대상")
    print(f"  경미 (0.1~0.25): {len(moderate)}개")
    print(f"  안정 (< 0.1): {len(stable)}개")
    
    if len(severe) > 0:
        print(f"\n  심각 피처 (PSI > 0.25):")
        for feat, psi in severe.items():
            flag = " ← XS" if "_xs_pct" in feat or feat == "stealth_score" else ""
            print(f"    {feat:<35} PSI={psi:.3f}{flag}")
    
    if len(moderate) > 0:
        print(f"\n  경미 피처 (0.1~0.25) TOP 10:")
        for feat, psi in moderate.head(10).items():
            flag = " ← XS" if "_xs_pct" in feat or feat == "stealth_score" else ""
            print(f"    {feat:<35} PSI={psi:.3f}{flag}")
    
    # XS vs Base PSI comparison
    xs_psis = [v for k, v in psi_scores.items() if "_xs_pct" in k or k == "stealth_score"]
    base_psis = [v for k, v in psi_scores.items() if "_xs_pct" not in k and k != "stealth_score"]
    print(f"\n  XS 피처 평균 PSI: {np.mean(xs_psis):.3f}" if xs_psis else "")
    print(f"  Base 피처 평균 PSI: {np.mean(base_psis):.3f}")
    
    return psi_df


# ============================================================
# 3. ABLATION STUDY
# ============================================================
def train_and_eval(X_tr, y_tr, X_te, y_te, fcols, config, seed=42):
    """Train pipeline with given config, return metrics"""
    random.seed(seed)
    np.random.seed(seed)
    
    scaler = RobustScaler()
    Xtrs = scaler.fit_transform(X_tr)
    Xtes = scaler.transform(X_te)
    
    npos = int(y_tr.sum())
    nneg = len(y_tr) - npos
    spw = max(nneg / npos if npos > 0 else 1.0, 0.1)
    
    # Stage 1: LGBM
    s1 = lgb.LGBMClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.02,
        num_leaves=31, min_child_samples=10, subsample=0.7,
        colsample_bytree=0.6, scale_pos_weight=spw * 4,
        objective='binary', metric='average_precision',
        random_state=seed, verbose=-1, force_col_wise=True, n_jobs=1
    )
    s1.fit(Xtrs, y_tr)
    s1_probs = s1.predict_proba(Xtes)[:, 1]
    
    if not config.get("stage2", False):
        # No Stage 2 — just use Stage 1
        best_p, best_t, best_preds = 0, 0.5, None
        for t in np.arange(0.30, 0.95, 0.05):
            preds = (s1_probs >= t).astype(int)
            if preds.sum() >= 3:
                p = precision_score(y_te, preds, zero_division=0)
                if p > best_p:
                    best_p = p
                    best_t = t
                    best_preds = preds
        if best_preds is None:
            best_preds = np.zeros(len(y_te), dtype=int)
        return {
            "precision": best_p,
            "recall": recall_score(y_te, best_preds, zero_division=0),
            "preds": int(best_preds.sum()),
            "hits": int((best_preds & y_te).sum()),
            "pr_auc": average_precision_score(y_te, s1_probs) if y_te.sum() > 0 else 0,
        }
    
    # Stage 2
    s1_threshold = 0.30
    s1_mask_tr = s1.predict_proba(Xtrs)[:, 1] >= s1_threshold
    
    if s1_mask_tr.sum() < 10 or y_tr[s1_mask_tr].sum() < 3:
        # Not enough Stage 1 positives for Stage 2
        return {"precision": 0, "recall": 0, "preds": 0, "hits": 0, "pr_auc": 0}
    
    # Stage 2 features
    s1_probs_tr = s1.predict_proba(Xtrs)[:, 1]
    
    if config.get("stage2_model") == "logreg":
        # LogReg Stage 2 with different features
        s2_feats_tr = np.column_stack([
            s1_probs_tr[s1_mask_tr],
            s1_probs_tr[s1_mask_tr] - s1_threshold,
        ])
        s2_feats_te_mask = s1_probs >= s1_threshold
        if s2_feats_te_mask.sum() == 0:
            return {"precision": 0, "recall": 0, "preds": 0, "hits": 0, "pr_auc": 0}
        s2_feats_te = np.column_stack([
            s1_probs[s2_feats_te_mask],
            s1_probs[s2_feats_te_mask] - s1_threshold,
        ])
        s2 = LogisticRegression(C=0.1, random_state=seed, max_iter=1000)
        s2.fit(s2_feats_tr, y_tr[s1_mask_tr])
        s2_probs = s2.predict_proba(s2_feats_te)[:, 1]
    else:
        # LGBM Stage 2
        s2_feats_tr = np.column_stack([
            Xtrs[s1_mask_tr],
            s1_probs_tr[s1_mask_tr].reshape(-1, 1),
        ])
        s2_feats_te_mask = s1_probs >= s1_threshold
        if s2_feats_te_mask.sum() == 0:
            return {"precision": 0, "recall": 0, "preds": 0, "hits": 0, "pr_auc": 0}
        s2_feats_te = np.column_stack([
            Xtes[s2_feats_te_mask],
            s1_probs[s2_feats_te_mask].reshape(-1, 1),
        ])
        npos2 = int(y_tr[s1_mask_tr].sum())
        nneg2 = s1_mask_tr.sum() - npos2
        spw2 = max(nneg2 / npos2 if npos2 > 0 else 1.0, 0.1)
        s2 = lgb.LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.02,
            num_leaves=31, min_child_samples=10, subsample=0.7,
            colsample_bytree=0.6, scale_pos_weight=spw2 * 2,
            objective='binary', random_state=seed, verbose=-1, n_jobs=1
        )
        s2.fit(s2_feats_tr, y_tr[s1_mask_tr])
        s2_probs = s2.predict_proba(s2_feats_te)[:, 1]
    
    # Find best threshold
    best_p, best_t = 0, 0.5
    for t in np.arange(0.30, 0.90, 0.05):
        preds = (s2_probs >= t).astype(int)
        if preds.sum() >= 3:
            p = precision_score(y_te[s2_feats_te_mask], preds, zero_division=0)
            if p > best_p:
                best_p = p
                best_t = t
    
    final_preds = np.zeros(len(y_te), dtype=int)
    if s2_feats_te_mask.sum() > 0:
        final_preds[s2_feats_te_mask] = (s2_probs >= best_t).astype(int)
    
    return {
        "precision": best_p,
        "recall": recall_score(y_te, final_preds, zero_division=0),
        "preds": int(final_preds.sum()),
        "hits": int((final_preds & y_te).sum()),
        "pr_auc": average_precision_score(y_te, s1_probs) if y_te.sum() > 0 else 0,
    }


def ablation_study(all_data):
    print("\n" + "="*60)
    print("  ABLATION STUDY")
    print("="*60)
    
    configs = {
        "R5_baseline (no XS, no S2)": {"xs": False, "stage2": False},
        "R7_full (XS + LogReg S2)": {"xs": True, "stage2": True, "stage2_model": "logreg"},
        "R7_no_xs (LogReg S2 only)": {"xs": False, "stage2": True, "stage2_model": "logreg"},
        "R7_no_stage2 (XS only)": {"xs": True, "stage2": False},
        "XS + LGBM S2": {"xs": True, "stage2": True, "stage2_model": "lgbm"},
        "no XS + LGBM S2": {"xs": False, "stage2": True, "stage2_model": "lgbm"},
        "PSI-pruned + no S2": {"xs": True, "stage2": False, "psi_prune": True},
        "PSI-pruned + LGBM S2": {"xs": True, "stage2": True, "stage2_model": "lgbm", "psi_prune": True},
    }
    
    # Build datasets once
    print("\n  Building datasets...")
    X_tr_xs, X_te_xs, y_tr, y_te, ts_tr, ts_te, fcols_xs = build_dataset(all_data, include_xs=True)
    X_tr_base, X_te_base, _, _, _, _, fcols_base = build_dataset(all_data, include_xs=False)
    
    # PSI pruning
    psi_scores = {}
    for i, feat in enumerate(fcols_xs):
        psi_scores[feat] = calculate_psi(X_tr_xs[:, i], X_te_xs[:, i])
    stable_mask = np.array([psi_scores.get(f, 0) <= 0.25 for f in fcols_xs])
    X_tr_pruned = X_tr_xs[:, stable_mask]
    X_te_pruned = X_te_xs[:, stable_mask]
    fcols_pruned = [f for f, m in zip(fcols_xs, stable_mask) if m]
    print(f"  PSI pruning: {len(fcols_xs)} → {len(fcols_pruned)} features")
    
    results = {}
    for name, cfg in configs.items():
        print(f"\n  [{name}]...")
        
        if cfg.get("psi_prune"):
            X_tr, X_te, fcols = X_tr_pruned, X_te_pruned, fcols_pruned
        elif cfg.get("xs"):
            X_tr, X_te, fcols = X_tr_xs, X_te_xs, fcols_xs
        else:
            X_tr, X_te, fcols = X_tr_base, X_te_base, fcols_base
        
        r = train_and_eval(X_tr, y_tr, X_te, y_te, fcols, cfg)
        results[name] = r
        print(f"    P={r['precision']:.1%} R={r['recall']:.1%} ({r['hits']}/{r['preds']}) PR-AUC={r['pr_auc']:.3f}")
    
    # Print comparison table
    print("\n" + "="*60)
    print("  ABLATION RESULTS")
    print("="*60)
    print(f"\n  {'Config':<35} {'Prec':>7} {'Recall':>7} {'Preds':>6} {'Hits':>5} {'PR-AUC':>7}")
    print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*6} {'-'*5} {'-'*7}")
    for name, r in sorted(results.items(), key=lambda x: -x[1]["precision"]):
        print(f"  {name:<35} {r['precision']:>6.1%} {r['recall']:>6.1%} {r['preds']:>6} {r['hits']:>5} {r['pr_auc']:>7.3f}")
    
    return results


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    print("="*60)
    print("  DISTRIBUTION SHIFT DIAGNOSIS")
    print("="*60)
    
    # Load data
    data_file = DATA_DIR / "raw_data_extended.json"
    if not data_file.exists():
        data_file = DATA_DIR / "raw_data.json"
    with open(data_file) as f:
        all_data = json.load(f)
    print(f"\n  {len(all_data)} coins loaded")
    
    # Build dataset with XS features
    print("\n  Building full dataset (with XS)...")
    X_tr, X_te, y_tr, y_te, ts_tr, ts_te, fcols = build_dataset(all_data, include_xs=True)
    print(f"  Train: {len(X_tr)} ({int(y_tr.sum())}+)  Test: {len(X_te)} ({int(y_te.sum())}+)")
    print(f"  Features: {len(fcols)}")
    
    # 1. Adversarial Validation
    adv_auc, shift_features = adversarial_validation(X_tr, X_te, fcols)
    
    # 2. PSI
    psi_df = psi_analysis(X_tr, X_te, fcols)
    
    # 3. Ablation Study
    abl_results = ablation_study(all_data)
    
    elapsed = time.time() - t0
    
    # Slack report
    lines = [
        f":microscope: *Distribution Shift 진단 완료* ({elapsed/60:.0f}분)",
        "",
        f"*1. Adversarial Validation*",
        f"```AUC: {adv_auc:.3f}",
    ]
    if adv_auc >= 0.9:
        lines.append("판정: SEVERE — 심각한 shift")
    elif adv_auc >= 0.75:
        lines.append("판정: SIGNIFICANT — 상당한 shift")
    elif adv_auc >= 0.6:
        lines.append("판정: MILD — 약간의 shift")
    else:
        lines.append("판정: SAFE — shift 없음")
    
    lines.append(f"\nShift 주범 TOP 5:")
    for feat, imp in shift_features.head(5).items():
        flag = " (XS)" if "_xs_pct" in feat or feat == "stealth_score" else ""
        lines.append(f"  {feat}: {imp:.4f}{flag}")
    lines.append("```")
    
    severe_psi = psi_df[psi_df > 0.25]
    lines.append(f"\n*2. PSI Analysis*")
    lines.append(f"```심각 (>0.25): {len(severe_psi)}개")
    lines.append(f"경미 (0.1~0.25): {len(psi_df[(psi_df > 0.1) & (psi_df <= 0.25)])}개")
    lines.append(f"안정 (<0.1): {len(psi_df[psi_df <= 0.1])}개```")
    
    lines.append(f"\n*3. Ablation Study*")
    lines.append("```")
    lines.append(f"{'Config':<32} {'Prec':>6} {'Recall':>6} {'Preds':>5} {'Hits':>4}")
    for name, r in sorted(abl_results.items(), key=lambda x: -x[1]["precision"]):
        short = name[:32]
        lines.append(f"{short:<32} {r['precision']:>5.1%} {r['recall']:>5.1%} {r['preds']:>5} {r['hits']:>4}")
    lines.append("```")
    
    send_slack("\n".join(lines))
    print(f"\n{'='*60}")
    print(f"  완료: {elapsed/60:.1f}분")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
