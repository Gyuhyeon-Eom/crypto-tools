#!/usr/bin/env python3
"""
급등 전조 모델 백테스트
최근 30일 동안 매일 모델을 돌렸다면:
1. 어떤 코인이 리스트에 올랐을까?
2. 그 중 실제 급등한 코인은?
3. Precision / Recall은?
4. 놓친 급등은?

순수 stdlib
"""

import json, math, os, sys, time, datetime
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).parent

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

# ============================================================
# RSI / ATR
# ============================================================

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

# ============================================================
# Feature extraction (at specific index)
# ============================================================

def extract_features_at(o, h, l, c, v, idx):
    pe = idx + 1
    if pe < 45: return None

    f = {}
    f["ret_1d"] = c[idx]/c[idx-1]-1 if c[idx-1] > 0 else 0
    f["ret_3d"] = c[idx]/c[idx-3]-1 if idx >= 3 and c[idx-3] > 0 else 0
    f["ret_7d"] = c[idx]/c[idx-7]-1 if idx >= 7 and c[idx-7] > 0 else 0
    f["ret_14d"] = c[idx]/c[idx-14]-1 if idx >= 14 and c[idx-14] > 0 else 0

    ma7 = mean(c[idx-6:idx+1]) if idx >= 6 else c[idx]
    ma20 = mean(c[idx-19:idx+1]) if idx >= 19 else c[idx]
    f["price_vs_ma7"] = c[idx]/ma7 - 1
    f["price_vs_ma20"] = c[idx]/ma20 - 1

    if idx >= 59:
        f["price_vs_60d_high"] = c[idx] / max(h[idx-59:idx+1])
        mn = min(l[idx-59:idx+1])
        f["price_vs_60d_low"] = c[idx] / mn if mn > 0 else 1
    else:
        f["price_vs_60d_high"] = 1.0; f["price_vs_60d_low"] = 1.0

    rsi_s = compute_rsi_series(c[:idx+1], 14)
    f["rsi_14"] = rsi_s[idx] if rsi_s[idx] is not None else 50
    f["rsi_trend_7d"] = (rsi_s[idx] - rsi_s[idx-7]) if idx >= 7 and rsi_s[idx] and rsi_s[idx-7] else 0

    rw = [rsi_s[j] for j in range(max(0, idx-13), idx+1) if rsi_s[j] is not None]
    f["rsi_consol_days"] = sum(1 for r in rw if 40 <= r <= 55)
    f["rsi_oversold_days"] = sum(1 for r in rw if r < 30)
    f["rsi_min_14d"] = min(rw) if rw else 50

    f["green_ratio_3d"] = sum(1 for j in range(idx-2, idx+1) if c[j] >= o[j]) / 3.0 if idx >= 2 else 0.5
    f["green_ratio_7d"] = sum(1 for j in range(idx-6, idx+1) if c[j] >= o[j]) / 7.0 if idx >= 6 else 0.5

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

    v7 = mean(v[idx-6:idx+1]) if idx >= 6 else v[idx]
    v30 = mean(v[max(0,idx-29):idx+1])
    f["vol_ratio_7d_30d"] = v7/v30 if v30 > 0 else 1

    f["vol_trend_7d"] = polyfit_slope(v[idx-6:idx+1]) if idx >= 6 else 0

    atr_s = compute_atr_series(h[:idx+1], l[:idx+1], c[:idx+1], 14)
    a14 = atr_s[idx] if atr_s[idx] else 0
    f["atr_14d_ratio"] = a14/c[idx] if c[idx] > 0 else 0
    a7v = [atr_s[j] for j in range(max(0,idx-6), idx+1) if atr_s[j]]
    a30v = [atr_s[j] for j in range(max(0,idx-29), idx+1) if atr_s[j]]
    f["volatility_ratio"] = mean(a7v)/mean(a30v) if a30v and mean(a30v) > 0 else 1

    if idx >= 19:
        bc = c[idx-19:idx+1]; bm = mean(bc); bs = stdev(bc)
        f["bb_width"] = (2*bs)/bm if bm > 0 else 0
        f["bb_position"] = (c[idx] - (bm - bs)) / (2*bs) if bs > 0 else 0.5
    else:
        f["bb_width"] = 0; f["bb_position"] = 0.5

    bodies = []; upwicks = []; lowwicks = []
    for j in range(max(0,idx-6), idx+1):
        body = abs(c[j]-o[j]); full = h[j]-l[j]
        if full > 0:
            bodies.append(body/full)
            upwicks.append((h[j]-max(c[j],o[j]))/full)
            lowwicks.append((min(c[j],o[j])-l[j])/full)
    f["avg_body_ratio_7d"] = mean(bodies) if bodies else 0.5
    f["avg_upper_wick_7d"] = mean(upwicks) if upwicks else 0

    if idx >= 13:
        pc14 = abs(c[idx]/c[idx-13]-1)
        vr = mean(v[idx-6:idx+1]); vp = mean(v[idx-13:idx-6])
        f["accumulation"] = (vr/vp)/(1+pc14*10) if vp > 0 else 1
    else:
        f["accumulation"] = 1

    return f

# ============================================================
# Gaussian scoring
# ============================================================

def compute_score(features, TH):
    if not features or not TH: return 0.0
    sc = 0.0; mx = 0.0
    for fn, info in TH.items():
        if fn not in features: continue
        w = info["weight"]; mx += w; val = features[fn]
        sm = info["surge_mean"]; ss = info["surge_std"]
        nm = info["normal_mean"]; d = info["direction"]
        if ss > 0:
            z = (val - sm) / ss; gauss = math.exp(-0.5 * z * z)
        else:
            gauss = 1.0 if val == sm else 0.0
        if d == "higher" and val < nm: gauss *= 0.3
        elif d == "lower" and val > nm: gauss *= 0.3
        sc += w * gauss
    return (sc / mx * 100) if mx > 0 else 0.0

# ============================================================
# Bybit API
# ============================================================

def bybit_get(endpoint, params=None):
    url = "https://api.bybit.com" + endpoint
    if params:
        url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "backtest/1.0")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())

def fetch_klines(symbol, interval="D", limit=200):
    data = bybit_get("/v5/market/kline", {
        "category": "spot", "symbol": symbol, "interval": interval, "limit": str(limit)
    })
    rows = data.get("result", {}).get("list", [])
    return list(reversed(rows)) if rows else []

# ============================================================
# Main backtest
# ============================================================

def main():
    print("=" * 65)
    print("  급등 전조 모델 백테스트")
    print("  최근 30일 시뮬레이션 → Precision / Recall 측정")
    print("=" * 65)

    with open(BASE_DIR / "thresholds.json") as f:
        TH = json.load(f)
    with open(BASE_DIR / "top_pairs.json") as f:
        pairs = json.load(f)

    BACKTEST_DAYS = 30
    SURGE_THRESHOLD = 0.5    # 7일 내 +50% (100%는 너무 드물어서 50%로)
    SURGE_WINDOW = 7
    SCORE_THRESHOLDS = [55, 60, 65, 70, 75]  # 여러 기준 테스트
    TOP_N = 10

    print(f"\n[1] 데이터 수집: {len(pairs)}개 코인 x {BACKTEST_DAYS+60}일...")

    # 모든 코인 데이터 수집
    all_data = {}
    for i, sym in enumerate(pairs):
        if (i+1) % 30 == 0:
            print(f"  [{i+1}/{len(pairs)}]...")
        try:
            klines = fetch_klines(sym, "D", BACKTEST_DAYS + 80)
            if len(klines) < 60: continue
            all_data[sym] = {
                "o": [float(k[1]) for k in klines],
                "h": [float(k[2]) for k in klines],
                "l": [float(k[3]) for k in klines],
                "c": [float(k[4]) for k in klines],
                "v": [float(k[5]) for k in klines],
                "ts": [int(k[0]) for k in klines],
            }
            time.sleep(0.04)
        except:
            continue

    print(f"  수집 완료: {len(all_data)}개")

    # 백테스트 기간 계산
    # 각 코인별로 가용 기간이 다름 → 코인별로 개별 처리
    # 전체 기간은 BACKTEST_DAYS로 고정
    n_days = BACKTEST_DAYS
    print(f"  백테스트 기간: {n_days}일 (최근 {SURGE_WINDOW}일 제외)")

    # [2] 매일 시뮬레이션
    print(f"\n[2] 매일 스캔 시뮬레이션...")

    daily_predictions = {}
    actual_surges = {}
    all_surges = set()

    for day_offset in range(n_days):
        predictions = {}
        surges = set()

        for sym, d in all_data.items():
            total_len = len(d["c"])
            # 이 날의 인덱스: 끝에서 역산
            # day_offset=0 → n_days+SURGE_WINDOW일 전, day_offset=n_days-1 → SURGE_WINDOW+1일 전
            idx = total_len - 1 - SURGE_WINDOW - (n_days - 1 - day_offset)

            if idx < 45 or idx + SURGE_WINDOW >= total_len:
                continue

            # 30일 평균 거래대금 (이 시점 기준)
            recent_vol = mean([d["v"][j] * d["c"][j] for j in range(max(0, idx-29), idx+1)])
            if recent_vol > 10_000_000 or recent_vol < 50_000:
                continue

            # Feature + Score
            feat = extract_features_at(d["o"], d["h"], d["l"], d["c"], d["v"], idx)
            if not feat: continue
            score = compute_score(feat, TH)
            predictions[sym] = score

            # 실제 급등 확인 (향후 7일)
            max_ret = 0
            for j in range(idx+1, min(idx+SURGE_WINDOW+1, total_len)):
                ret = d["c"][j] / d["c"][idx] - 1 if d["c"][idx] > 0 else 0
                max_ret = max(max_ret, ret)
            if max_ret >= SURGE_THRESHOLD:
                surges.add(sym)
                all_surges.add((sym, day_offset))

        daily_predictions[day_offset] = predictions
        actual_surges[day_offset] = surges

    # [3] 성능 평가
    print(f"\n[3] 성능 평가 (급등 기준: 7일 내 +{SURGE_THRESHOLD*100:.0f}%)")
    print(f"    전체 급등 이벤트: {len(all_surges)}건\n")

    print(f"{'점수기준':>8} {'TOP':>4} {'예측수':>6} {'적중':>4} {'Precision':>10} {'Recall':>8} {'급등총':>6}")
    print("-" * 55)

    best_config = None
    best_f1 = 0

    for threshold in SCORE_THRESHOLDS:
        for top_n in [5, 10, 20]:
            total_predictions = 0
            total_hits = 0
            total_surges_count = 0

            for day in range(n_days):
                preds = daily_predictions.get(day, {})
                surges = actual_surges.get(day, set())
                total_surges_count += len(surges)

                # 이 날의 예측: 점수 threshold 이상 + 상위 top_n
                qualified = {s: sc for s, sc in preds.items() if sc >= threshold}
                top_coins = sorted(qualified.items(), key=lambda x: x[1], reverse=True)[:top_n]
                predicted_set = set(s for s, _ in top_coins)

                total_predictions += len(predicted_set)
                hits = predicted_set & surges
                total_hits += len(hits)

            precision = total_hits / total_predictions if total_predictions > 0 else 0
            # Recall: 전체 급등 중 몇 개를 잡았나
            total_unique_surges = sum(len(actual_surges.get(d, set())) for d in range(n_days))
            recall = total_hits / total_unique_surges if total_unique_surges > 0 else 0

            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_config = (threshold, top_n, precision, recall, f1)

            if top_n == 10:  # 기본 출력
                print(f"{threshold:>8} {top_n:>4} {total_predictions:>6} {total_hits:>4} {precision:>9.1%} {recall:>7.1%} {total_unique_surges:>6}")

    if best_config:
        print(f"\n최적 설정: 점수 {best_config[0]}+ TOP {best_config[1]}")
        print(f"  Precision: {best_config[2]:.1%}")
        print(f"  Recall:    {best_config[3]:.1%}")
        print(f"  F1 Score:  {best_config[4]:.3f}")
    else:
        print("\n급등 이벤트가 없어서 최적 설정 산출 불가")
        best_config = (60, 10, 0, 0, 0)

    # [4] 최근 급등 사례 분석
    print(f"\n[4] 최근 급등 사례 — 모델이 잡았나?")
    print(f"{'코인':<12} {'날짜':>6} {'급등률':>7} {'당시점수':>8} {'TOP10?':>7}")
    print("-" * 45)

    surge_list = sorted(all_surges, key=lambda x: x[1], reverse=True)
    for sym, day in surge_list[:20]:
        cn = sym.replace("USDT", "")
        d = all_data[sym]
        total_len = len(d["c"])
        idx = total_len - 1 - SURGE_WINDOW - (n_days - 1 - day)

        # 실제 급등률
        max_ret = 0
        for j in range(idx+1, min(idx+8, total_len)):
            ret = d["c"][j] / d["c"][idx] - 1 if d["c"][idx] > 0 else 0
            max_ret = max(max_ret, ret)

        score = daily_predictions.get(day, {}).get(sym, 0)

        # TOP 10 안에 있었나?
        preds = daily_predictions.get(day, {})
        qualified = {s: sc for s, sc in preds.items() if sc >= 55}
        top10 = [s for s, _ in sorted(qualified.items(), key=lambda x: x[1], reverse=True)[:10]]
        in_top10 = "O" if sym in top10 else "X"

        # 날짜 계산
        ts = d["ts"][idx]
        date_str = datetime.datetime.fromtimestamp(ts/1000).strftime("%m/%d")

        print(f"{cn:<12} {date_str:>6} {max_ret*100:>+6.0f}% {score:>8.1f} {in_top10:>7}")

    # [5] 놓친 급등
    print(f"\n[5] 놓친 급등 (TOP 10 밖)")
    missed = [(sym, day) for sym, day in surge_list if
              sym not in [s for s, _ in sorted(
                  {s: sc for s, sc in daily_predictions.get(day, {}).items() if sc >= 55}.items(),
                  key=lambda x: x[1], reverse=True)[:10]]]

    for sym, day in missed[:10]:
        cn = sym.replace("USDT", "")
        score = daily_predictions.get(day, {}).get(sym, 0)
        d = all_data[sym]
        idx = len(d["c"]) - 1 - SURGE_WINDOW - (n_days - 1 - day)
        max_ret = 0
        for j in range(idx+1, min(idx+8, len(d["c"]))):
            ret = d["c"][j] / d["c"][idx] - 1 if d["c"][idx] > 0 else 0
            max_ret = max(max_ret, ret)
        ts = d["ts"][idx]
        date_str = datetime.datetime.fromtimestamp(ts/1000).strftime("%m/%d")
        feat = extract_features_at(d["o"], d["h"], d["l"], d["c"], d["v"], idx)
        ret7 = feat.get("ret_7d", 0)*100 if feat else 0
        rsi = feat.get("rsi_14", 0) if feat else 0
        print(f"  {cn:<12} {date_str} +{max_ret*100:.0f}% 점수={score:.0f} ret7d={ret7:+.0f}% RSI={rsi:.0f}")

    # 저장
    report = {
        "backtest_days": n_days,
        "surge_threshold": SURGE_THRESHOLD,
        "total_surges": len(all_surges),
        "best_config": {
            "score_threshold": best_config[0],
            "top_n": best_config[1],
            "precision": best_config[2],
            "recall": best_config[3],
            "f1": best_config[4],
        },
    }
    with open(BASE_DIR / "backtest_result.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n결과 저장: backtest_result.json")

if __name__ == "__main__":
    main()
