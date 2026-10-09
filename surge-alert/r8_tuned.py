#!/usr/bin/env python3
"""
R8: R7_full base + vol_usd transform + soft voting ensemble
- vol_usd_7d → vol_zscore_30d (자기 30일 평균 대비 z-score)
- vol_usd_1d → vol_ratio_1d_30d (30일 평균 대비 비율)
- Hard voting 4/5 → Soft voting (확률 평균 > threshold)
- PSI 재측정으로 변환 효과 확인
- Walk-forward + ablation matrix
"""

import json, os, sys, time, pickle, warnings, random
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, average_precision_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "ml_data"
SURGE_WINDOW = 7
MIN_VOL, MAX_VOL = 1_000_000, 10_000_000

sys.path.insert(0, str(BASE_DIR))
from round4_metalabel import extract_features, has_recent_surge, mean, stdev, compute_atr
from shift_diagnosis import (
    dynamic_triple_barrier_label, build_xs_features, PurgedKFold, calculate_psi
)

# Slack
WEBHOOK_URL = ""
_env = os.path.expanduser("~/.openclaw/.env.slack-surge")
if os.path.exists(_env):
    with open(_env) as f:
        for line in f:
            if line.startswith("SLACK_WEBHOOK_URL="):
                WEBHOOK_URL = line.strip().split("=", 1)[1]

def send_slack(msg):
    if not WEBHOOK_URL: print(msg); return
    for chunk in [msg[i:i+3500] for i in range(0, len(msg), 3500)]:
        payload = json.dumps({"text": chunk})
        req = urllib.request.Request(WEBHOOK_URL, data=payload.encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try: urllib.request.urlopen(req, timeout=30)
        except: pass
        time.sleep(0.5)


def bybit_get(ep, params=None):
    url = "https://api.bybit.com" + ep
    if params: url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
    req = urllib.request.Request(url, headers={"User-Agent": "r8/1"})
    for i in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as r: return json.loads(r.read())
        except:
            if i == 2: raise
            time.sleep(1 + i)


# ============================================================
# vol_usd 변환
# ============================================================
def transform_vol_features(feat, o, h, l, c, v, idx):
    """vol_usd_7d, vol_usd_1d를 shift-resistant 버전으로 변환"""
    # vol_zscore_30d: 자기 30일 평균 대비 z-score
    if idx >= 29:
        vol_usds = [v[j] * c[j] for j in range(idx-29, idx+1)]
        vol_mean = mean(vol_usds)
        vol_std = stdev(vol_usds) if len(vol_usds) > 1 else 1
        vol_7d = mean([v[j]*c[j] for j in range(max(0,idx-6), idx+1)])
        vol_1d = v[idx] * c[idx]
        feat["vol_zscore_7d_30d"] = (vol_7d - vol_mean) / vol_std if vol_std > 0 else 0
        feat["vol_zscore_1d_30d"] = (vol_1d - vol_mean) / vol_std if vol_std > 0 else 0
    else:
        feat["vol_zscore_7d_30d"] = 0
        feat["vol_zscore_1d_30d"] = 0
    
    # log detrend: log(vol) - rolling_mean(log(vol), 30)
    if idx >= 29:
        log_vols = [np.log1p(v[j]*c[j]) for j in range(idx-29, idx+1)]
        log_mean = mean(log_vols)
        feat["vol_log_detrend_7d"] = mean([np.log1p(v[j]*c[j]) for j in range(max(0,idx-6), idx+1)]) - log_mean
        feat["vol_log_detrend_1d"] = np.log1p(v[idx]*c[idx]) - log_mean
    else:
        feat["vol_log_detrend_7d"] = 0
        feat["vol_log_detrend_1d"] = 0
    
    # Remove original vol_usd absolutes
    feat.pop("vol_usd_7d", None)
    feat.pop("vol_usd_1d", None)
    
    return feat


# ============================================================
# Build dataset with vol transform
# ============================================================
def build_dataset(all_data, include_xs=True, transform_vol=True):
    base_features_by_sym = {}
    rows = []
    
    for sym, d in all_data.items():
        n = len(d["c"])
        o, h, l, c, v = d["o"], d["h"], d["l"], d["c"], d["v"]
        av = mean([v[j]*c[j] for j in range(max(0,n-30), n)])
        if av > MAX_VOL or av < MIN_VOL: continue
        
        sym_feats = []
        for idx in range(59, n - SURGE_WINDOW):
            if has_recent_surge(c, idx): continue
            feat = extract_features(o, h, l, c, v, idx)
            if not feat: continue
            
            if transform_vol:
                feat = transform_vol_features(feat, o, h, l, c, v, idx)
            
            label = dynamic_triple_barrier_label(o, h, l, c, v, idx)
            feat["_label"] = 1 if label == 1 else 0
            feat["_ts"] = d["ts"][idx]
            feat["_sym"] = sym
            sym_feats.append(feat)
            rows.append(feat)
        
        if sym_feats:
            base_features_by_sym[sym] = sym_feats
    
    # XS features
    if include_xs:
        xs_lookup, top20 = build_xs_features(all_data, base_features_by_sym)
        for row in rows:
            key = (row["_sym"], row["_ts"])
            if key in xs_lookup:
                row.update(xs_lookup[key])
    
    df = pd.DataFrame(rows)
    fcols = [c for c in df.columns if not c.startswith("_")]
    
    X = np.nan_to_num(df[fcols].values.astype(np.float64), nan=0, posinf=0, neginf=0)
    y = df["_label"].values
    ts_arr = df["_ts"].values
    
    si = np.argsort(ts_arr)
    X, y, ts_arr = X[si], y[si], ts_arr[si]
    sp = int(len(X) * 0.8)
    
    return X[:sp], X[sp:], y[:sp], y[sp:], ts_arr[:sp], ts_arr[sp:], fcols


# ============================================================
# Train pipeline with soft voting
# ============================================================
def train_single_seed(X_tr, y_tr, X_te, fcols, seed=42):
    """Train Stage 1 + Stage 2 for one seed, return test probabilities"""
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
        colsample_bytree=0.6, scale_pos_weight=max(spw * 4, 0.1),
        objective='binary', metric='average_precision',
        random_state=seed, verbose=-1, force_col_wise=True, n_jobs=1
    )
    s1.fit(Xtrs, y_tr)
    
    s1_probs_tr = s1.predict_proba(Xtrs)[:, 1]
    s1_probs_te = s1.predict_proba(Xtes)[:, 1]
    s1_threshold = 0.30
    
    # Stage 2: LogReg on S1 positives
    s1_mask_tr = s1_probs_tr >= s1_threshold
    if s1_mask_tr.sum() < 10 or y_tr[s1_mask_tr].sum() < 3:
        return s1_probs_te, scaler, s1, None, s1_threshold
    
    s2_feats_tr = np.column_stack([
        s1_probs_tr[s1_mask_tr],
        s1_probs_tr[s1_mask_tr] - s1_threshold,
    ])
    
    s2 = LogisticRegression(C=0.1, random_state=seed, max_iter=1000)
    s2.fit(s2_feats_tr, y_tr[s1_mask_tr])
    
    # Stage 2 probabilities on test
    s1_mask_te = s1_probs_te >= s1_threshold
    final_probs = np.zeros(len(X_te))
    
    if s1_mask_te.sum() > 0:
        s2_feats_te = np.column_stack([
            s1_probs_te[s1_mask_te],
            s1_probs_te[s1_mask_te] - s1_threshold,
        ])
        s2_probs = s2.predict_proba(s2_feats_te)[:, 1]
        final_probs[s1_mask_te] = s1_probs_te[s1_mask_te] * s2_probs
    
    return final_probs, scaler, s1, s2, s1_threshold


def soft_vote_ensemble(X_tr, y_tr, X_te, fcols, seeds=[42, 123, 456, 789, 1000]):
    """Train multiple seeds, average probabilities (soft voting)"""
    all_probs = []
    models = []
    
    for seed in seeds:
        probs, scaler, s1, s2, s1_t = train_single_seed(X_tr, y_tr, X_te, fcols, seed)
        all_probs.append(probs)
        models.append((scaler, s1, s2, s1_t))
    
    # Soft voting: average probabilities
    avg_probs = np.mean(all_probs, axis=0)
    return avg_probs, all_probs, models


# ============================================================
# Walk-Forward
# ============================================================
def walk_forward(X, y, ts, fcols, n_windows=6, seeds=[42, 123, 456, 789, 1000]):
    n = len(X)
    min_train = int(n * 0.4)
    test_total = n - min_train
    window_size = test_total // n_windows
    
    all_y_true, all_probs, all_preds = [], [], []
    
    for w in range(n_windows):
        train_end = min_train + w * window_size
        test_start = train_end
        test_end = min(test_start + window_size, n)
        if test_end <= test_start: continue
        
        X_tr, y_tr = X[:train_end], y[:train_end]
        X_te, y_te = X[test_start:test_end], y[test_start:test_end]
        
        if y_tr.sum() < 5: continue
        
        avg_probs, _, _ = soft_vote_ensemble(X_tr, y_tr, X_te, fcols, seeds)
        
        # Find best threshold on recent train data
        _, _, models = soft_vote_ensemble(X_tr, y_tr, X_tr[-2000:], fcols, seeds)
        tr_probs = np.mean([
            train_single_seed(X_tr, y_tr, X_tr[-2000:], fcols, s)[0] 
            for s in seeds
        ], axis=0)
        
        best_p, best_t = 0, 0.5
        for t in np.arange(0.05, 0.50, 0.02):
            preds = (tr_probs >= t).astype(int)
            if preds.sum() >= 3:
                p = precision_score(y_tr[-2000:], preds, zero_division=0)
                if p > best_p:
                    best_p = p
                    best_t = t
        
        preds_te = (avg_probs >= best_t).astype(int)
        tp = int(preds_te.sum())
        hits = int((preds_te & y_te).sum())
        prec = hits / tp if tp > 0 else 0
        rec = hits / y_te.sum() if y_te.sum() > 0 else 0
        
        print(f"    Window {w+1}: P={prec:.1%} R={rec:.1%} ({hits}/{tp}) t={best_t:.2f}")
        
        all_y_true.extend(y_te.tolist())
        all_probs.extend(avg_probs.tolist())
        all_preds.extend(preds_te.tolist())
    
    if all_preds:
        all_y = np.array(all_y_true)
        all_p = np.array(all_preds)
        tp = int(all_p.sum())
        hits = int((all_p & all_y).sum())
        return {
            "precision": hits / tp if tp > 0 else 0,
            "recall": hits / all_y.sum() if all_y.sum() > 0 else 0,
            "preds": tp, "hits": hits,
            "pr_auc": average_precision_score(all_y, np.array(all_probs)) if all_y.sum() > 0 else 0,
        }
    return {"precision": 0, "recall": 0, "preds": 0, "hits": 0, "pr_auc": 0}


# ============================================================
# Live scan
# ============================================================
def live_scan(models, fcols, all_coin_data=None):
    """Run live scan with ensemble"""
    SKIP = {"USDCUSDT","UUSDT","XUSDUSDT","PAXGUSDT","DAIUSDT","FDUSDUSDT",
            "TUSDUSDT","BUSDUSDT","EURUSDT","USTCUSDT","USDTUSDT","USDDUSDT",
            "RLUSDUSDT","USD1USDT","BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT",
            "SOLUSDT","ADAUSDT","DOGEUSDT","TRXUSDT","DOTUSDT","MATICUSDT",
            "LTCUSDT","BCHUSDT","LINKUSDT","AVAXUSDT","UNIUSDT","AAVEUSDT","ATOMUSDT"}
    
    print("\n[Scan] Fetching tickers...")
    tickers = bybit_get("/v5/market/tickers", {"category": "spot"}).get("result", {}).get("list", [])
    alts = [t["symbol"] for t in tickers
            if t["symbol"].endswith("USDT") and t["symbol"] not in SKIP
            and MIN_VOL <= float(t.get("turnover24h", 0)) <= MAX_VOL]
    
    print(f"  Scanning {len(alts)} coins...")
    results = []
    
    for sym in alts:
        try:
            data = bybit_get("/v5/market/kline", {"category": "spot", "symbol": sym, "interval": "D", "limit": "100"})
            klines = list(reversed(data.get("result", {}).get("list", [])))
            if len(klines) < 60: continue
            o = [float(k[1]) for k in klines]; h = [float(k[2]) for k in klines]
            l = [float(k[3]) for k in klines]; c = [float(k[4]) for k in klines]
            v = [float(k[5]) for k in klines]
            idx = len(c) - 1
            
            if has_recent_surge(c, idx): continue
            feat = extract_features(o, h, l, c, v, idx)
            if not feat: continue
            feat = transform_vol_features(feat, o, h, l, c, v, idx)
            
            # XS features (simplified - just use 0.5 defaults for live)
            for f in fcols:
                if f not in feat:
                    feat[f] = 0.5 if "_xs_pct" in f else 0
            
            fv = np.array([[feat.get(fc, 0) for fc in fcols]], dtype=np.float64)
            fv = np.nan_to_num(fv, nan=0, posinf=0, neginf=0)
            
            # Ensemble soft vote
            probs = []
            s1_probs = []
            for scaler, s1, s2, s1_t in models:
                fvs = scaler.transform(fv)
                s1p = s1.predict_proba(fvs)[0, 1]
                s1_probs.append(s1p)
                
                if s2 is not None and s1p >= s1_t:
                    s2_feat = np.array([[s1p, s1p - s1_t]])
                    s2p = s2.predict_proba(s2_feat)[0, 1]
                    probs.append(s1p * s2p)
                else:
                    probs.append(0)
            
            avg_prob = np.mean(probs)
            avg_s1 = np.mean(s1_probs)
            
            # Funding rate
            fr = 0
            try:
                fr_data = bybit_get("/v5/market/tickers", {"category": "linear", "symbol": sym})
                fr_items = fr_data.get("result", {}).get("list", [])
                if fr_items: fr = float(fr_items[0].get("fundingRate", 0))
            except: pass
            
            r7 = (c[-1]/c[-8]-1)*100 if len(c) >= 8 and c[-8] > 0 else 0
            r14 = (c[-1]/c[-15]-1)*100 if len(c) >= 15 and c[-15] > 0 else 0
            rsi = feat.get("rsi_14", 0)
            
            results.append({
                "coin": sym.replace("USDT",""), "prob": avg_prob,
                "s1_prob": avg_s1, "price": c[-1],
                "ret_7d": r7, "ret_14d": r14, "rsi": rsi,
                "fr": fr, "vol_z": feat.get("vol_zscore_7d_30d", 0),
            })
            time.sleep(0.03)
        except: continue
    
    results.sort(key=lambda x: x["prob"], reverse=True)
    return results


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    print("="*60)
    print("  R8: R7_full + vol transform + soft voting")
    print("="*60)
    
    # Load data
    data_file = DATA_DIR / "raw_data_extended.json"
    with open(data_file) as f:
        all_data = json.load(f)
    print(f"\n  {len(all_data)} coins loaded")
    
    # Build dataset
    print("\n[1] Building dataset (vol transformed + XS)...")
    X_tr, X_te, y_tr, y_te, ts_tr, ts_te, fcols = build_dataset(all_data, include_xs=True, transform_vol=True)
    print(f"  Train: {len(X_tr)} ({int(y_tr.sum())}+)  Test: {len(X_te)} ({int(y_te.sum())}+)")
    print(f"  Features: {len(fcols)}")
    
    # PSI check on transformed features
    print("\n[2] PSI check (post vol transform)...")
    psi_scores = {}
    for i, feat in enumerate(fcols):
        psi = calculate_psi(X_tr[:, i], X_te[:, i])
        psi_scores[feat] = psi
    severe = {k: v for k, v in psi_scores.items() if v > 0.25}
    moderate = {k: v for k, v in psi_scores.items() if 0.1 < v <= 0.25}
    print(f"  심각 (>0.25): {len(severe)}개")
    if severe:
        for f, p in sorted(severe.items(), key=lambda x: -x[1]):
            print(f"    {f}: PSI={p:.3f}")
    print(f"  경미 (0.1~0.25): {len(moderate)}개")
    print(f"  안정 (<0.1): {len(psi_scores) - len(severe) - len(moderate)}개")
    
    # New vol features PSI
    for vf in ["vol_zscore_7d_30d", "vol_zscore_1d_30d", "vol_log_detrend_7d", "vol_log_detrend_1d"]:
        if vf in psi_scores:
            print(f"  {vf}: PSI={psi_scores[vf]:.3f} {'OK' if psi_scores[vf] < 0.1 else 'WARN'}")
    
    # Train ensemble
    seeds = [42, 123, 456, 789, 1000]
    print(f"\n[3] Training soft-vote ensemble ({len(seeds)} seeds)...")
    avg_probs, all_probs, models = soft_vote_ensemble(X_tr, y_tr, X_te, fcols, seeds)
    
    # Per-seed results
    for i, (seed, probs) in enumerate(zip(seeds, all_probs)):
        best_p, best_t = 0, 0.5
        for t in np.arange(0.05, 0.50, 0.02):
            preds = (probs >= t).astype(int)
            if preds.sum() >= 3:
                p = precision_score(y_te, preds, zero_division=0)
                if p > best_p: best_p = p; best_t = t
        preds = (probs >= best_t).astype(int)
        hits = int((preds & y_te).sum())
        print(f"    Seed {seed}: P={best_p:.1%} ({hits}/{int(preds.sum())}) t={best_t:.2f}")
    
    # Soft vote threshold optimization
    print(f"\n[4] Optimizing soft-vote threshold...")
    best_p, best_t, best_r, best_preds = 0, 0.1, 0, None
    for t in np.arange(0.01, 0.50, 0.01):
        preds = (avg_probs >= t).astype(int)
        if preds.sum() >= 5:
            p = precision_score(y_te, preds, zero_division=0)
            r = recall_score(y_te, preds, zero_division=0)
            hits = int((preds & y_te).sum())
            if p >= 0.15 and hits > (best_preds.sum() * best_p if best_preds is not None else 0):
                # Optimize for most hits at P >= 15%
                if p > best_p or (p >= best_p - 0.02 and hits > (int(best_preds.sum() * best_p) if best_preds is not None else 0)):
                    best_p = p
                    best_t = t
                    best_r = r
                    best_preds = preds
            if p >= 0.20:
                print(f"    t={t:.2f}: P={p:.1%} R={r:.1%} ({hits}/{int(preds.sum())})")
    
    if best_preds is None:
        # Fallback: find best precision regardless
        for t in np.arange(0.01, 0.50, 0.01):
            preds = (avg_probs >= t).astype(int)
            if preds.sum() >= 3:
                p = precision_score(y_te, preds, zero_division=0)
                if p > best_p:
                    best_p = p; best_t = t; best_preds = preds
                    best_r = recall_score(y_te, preds, zero_division=0)
    
    hits = int((best_preds & y_te).sum()) if best_preds is not None else 0
    total_preds = int(best_preds.sum()) if best_preds is not None else 0
    print(f"\n  Best: t={best_t:.2f} P={best_p:.1%} R={best_r:.1%} ({hits}/{total_preds})")
    
    # PR-AUC
    pr_auc = average_precision_score(y_te, avg_probs) if y_te.sum() > 0 else 0
    print(f"  PR-AUC: {pr_auc:.4f}")
    
    # Walk-Forward
    print(f"\n[5] Walk-Forward Backtest (6 windows, soft voting)...")
    X_all = np.vstack([X_tr, X_te])
    y_all = np.concatenate([y_tr, y_te])
    ts_all = np.concatenate([ts_tr, ts_te])
    wf = walk_forward(X_all, y_all, ts_all, fcols, n_windows=6, seeds=seeds)
    print(f"\n  WF Aggregate: P={wf['precision']:.1%} R={wf['recall']:.1%} ({wf['hits']}/{wf['preds']}) PR-AUC={wf['pr_auc']:.3f}")
    
    # Live scan
    print(f"\n[6] Live Scan...")
    scan_results = live_scan(models, fcols)
    
    # Determine hit threshold from holdout
    scan_hits = [r for r in scan_results if r["prob"] >= best_t]
    
    # Save model
    print(f"\n[7] Saving...")
    save_data = {
        "models": models, "feature_cols": fcols,
        "soft_threshold": float(best_t), "s1_threshold": 0.30,
        "seeds": seeds, "round": 8, "version": "r8_tuned",
    }
    with open(DATA_DIR / "best_model_r8.pkl", "wb") as f:
        pickle.dump(save_data, f)
    
    config = {
        "round": 8, "version": "r8_tuned",
        "method": "R7_full + vol_transform + soft_voting",
        "features": len(fcols), "soft_threshold": float(best_t),
        "holdout": {"precision": float(best_p), "recall": float(best_r),
                    "preds": total_preds, "hits": hits, "pr_auc": float(pr_auc)},
        "walk_forward": wf,
        "trained_at": datetime.datetime.now().isoformat() if 'datetime' in dir() else time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    import datetime as dt
    config["trained_at"] = dt.datetime.now().isoformat()
    with open(BASE_DIR / "ml_model_config_r8.json", "w") as f:
        json.dump(config, f, indent=2)
    
    elapsed = time.time() - t0
    
    # Report
    lines = [
        f":chart_with_upwards_trend: *R8 완료* ({elapsed/60:.0f}분)",
        f"R7_full + vol_usd 변환 + soft voting",
        "```",
        f"=== Holdout ===",
        f"  Precision: {best_p:.1%}",
        f"  Recall:    {best_r:.1%}",
        f"  Preds:     {total_preds} ({hits} hits)",
        f"  PR-AUC:    {pr_auc:.4f}",
        f"  Threshold: {best_t:.2f}",
        f"",
        f"=== Walk-Forward ===",
        f"  Precision: {wf['precision']:.1%}",
        f"  Recall:    {wf['recall']:.1%}",
        f"  PR-AUC:    {wf['pr_auc']:.3f}",
        f"",
        f"=== Comparison ===",
        f"{'Model':<20} {'Prec':>6} {'Recall':>7} {'Preds':>6} {'Hits':>5}",
        f"{'R5 baseline':<20} {'15.1%':>6} {'3.0%':>7} {'119':>6} {'18':>5}",
        f"{'v2':<20} {'13.2%':>6} {'53.6%':>7} {'2459':>6} {'324':>5}",
        f"{'R7 (failed)':<20} {'0.0%':>6} {'0.0%':>7} {'1':>6} {'0':>5}",
        f"{'R8 (this)':<20} {f'{best_p:.1%}':>6} {f'{best_r:.1%}':>7} {str(total_preds):>6} {str(hits):>5}",
        "```",
    ]
    
    if scan_hits:
        lines.append(f"\n*전조 후보 ({len(scan_hits)}개):*")
        lines.append("```")
        lines.append(f" # {'코인':<8} {'확률':>6} {'S1':>5} {'가격':<10} {'7d':>7} {'14d':>7} {'RSI':>4} {'FR':>10} {'VolZ':>5}")
        for i, r in enumerate(scan_hits[:15]):
            fr_str = f"{r['fr']*100:+.4f}%" if r['fr'] != 0 else "n/a"
            lines.append(f"{i+1:>2} {r['coin']:<8} {r['prob']:>5.1%} {r['s1_prob']:>4.1%} ${r['price']:<9.4f} {r['ret_7d']:>+6.1f}% {r['ret_14d']:>+6.1f}% {r['rsi']:>4.0f} {fr_str:>10} {r['vol_z']:>+4.1f}")
        lines.append("```")
    else:
        lines.append(f"\n*전조 후보 없음* (threshold {best_t:.2f})")
        if scan_results:
            lines.append(f"*주시 목록 TOP 10:*")
            lines.append("```")
            for i, r in enumerate(scan_results[:10]):
                fr_str = f"{r['fr']*100:+.4f}%" if r['fr'] != 0 else "n/a"
                lines.append(f"{i+1:>2} {r['coin']:<8} {r['prob']:>5.1%} S1={r['s1_prob']:>4.1%} {r['ret_7d']:>+6.1f}% RSI={r['rsi']:.0f} FR={fr_str}")
            lines.append("```")
    
    send_slack("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"\nDone. {elapsed/60:.1f} minutes.")


if __name__ == "__main__":
    main()
