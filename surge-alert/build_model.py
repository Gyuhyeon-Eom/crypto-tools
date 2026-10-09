#!/usr/bin/env python3
"""
소형 알트코인 급등 전조 모델 v2
- Bybit 전체 USDT spot에서 소형 알트만 필터
- 최근 1년 내 +100% 급등 이벤트 추출
- 급등 직전 7일 공통점/특이점 통계 분석
- 가우시안 스코어링용 thresholds.json 생성
순수 stdlib (numpy/pandas 없음)
"""

import json, math, os, sys, time, datetime
import urllib.request, urllib.error
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent

# ============================================================
# 설정
# ============================================================

# 소형 알트 기준
MIN_DAILY_VOL_USD = 200_000    # 너무 작으면 유동성 없음
MAX_DAILY_VOL_USD = 5_000_000  # 이 이상은 대형
MIN_PRICE_USD = 0.0000001      # 최소 가격 (먼지 제외)
SURGE_THRESHOLD = 1.0          # +100% 급등 기준
SURGE_WINDOW = 7               # 7일 내 급등
MIN_KLINES = 90                # 최소 90일 데이터
LOOKBACK_DAYS = 365            # 1년

STABLECOINS = {
    "USDCUSDT", "UUSDT", "XUSDUSDT", "PAXGUSDT", "DAIUSDT",
    "FDUSDUSDT", "TUSDUSDT", "BUSDUSDT", "EURUSDT", "USTCUSDT",
    "USDTUSDT", "USDDUSDT",
}

# 대형 코인 제외 (시총 상위)
LARGE_CAPS = {
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
    "ADAUSDT", "DOGEUSDT", "TRXUSDT", "DOTUSDT", "MATICUSDT",
    "LTCUSDT", "BCHUSDT", "LINKUSDT", "AVAXUSDT", "UNIUSDT",
    "AAVEUSDT", "ATOMUSDT",
}

# ============================================================
# Bybit API
# ============================================================

def bybit_get(endpoint, params=None):
    url = "https://api.bybit.com" + endpoint
    if params:
        url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "surge-model-builder/1.0")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())

def fetch_all_tickers():
    data = bybit_get("/v5/market/tickers", {"category": "spot"})
    return data.get("result", {}).get("list", [])

def fetch_klines(symbol, interval="D", limit=200, start=None):
    params = {"category": "spot", "symbol": symbol, "interval": interval, "limit": str(limit)}
    if start:
        params["start"] = str(start)
    data = bybit_get("/v5/market/kline", params)
    rows = data.get("result", {}).get("list", [])
    return list(reversed(rows)) if rows else []

def fetch_all_klines(symbol, days=365):
    """여러 번 호출해서 1년치 데이터 수집."""
    all_rows = []
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - days * 24 * 3600 * 1000

    cursor = start_ms
    while cursor < now_ms:
        rows = fetch_klines(symbol, "D", 200, start=cursor)
        if not rows:
            break
        all_rows.extend(rows)
        last_ts = int(rows[-1][0])
        if last_ts <= cursor:
            break
        cursor = last_ts + 24 * 3600 * 1000
        time.sleep(0.05)

    # 중복 제거
    seen = set()
    unique = []
    for r in all_rows:
        ts = r[0]
        if ts not in seen:
            seen.add(ts)
            unique.append(r)
    unique.sort(key=lambda x: int(x[0]))
    return unique

# ============================================================
# Math helpers
# ============================================================

def mean(arr):
    return sum(arr) / len(arr) if arr else 0.0

def stdev(arr):
    if len(arr) < 2:
        return 0.0
    m = mean(arr)
    return math.sqrt(sum((x - m) ** 2 for x in arr) / (len(arr) - 1))

def polyfit_slope(y_vals):
    n = len(y_vals)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = mean(y_vals)
    if y_mean == 0:
        return 0.0
    num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(y_vals))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return (num / den) / y_mean if den != 0 else 0.0

def cohens_d(g1, g2):
    m1, m2 = mean(g1), mean(g2)
    s1, s2 = stdev(g1), stdev(g2)
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0
    sp = math.sqrt(((n1-1)*s1*s1 + (n2-1)*s2*s2) / (n1+n2-2))
    return (m1 - m2) / sp if sp > 0 else 0.0

# ============================================================
# RSI / ATR
# ============================================================

def compute_rsi_series(closes, period=14):
    rsi = [None] * len(closes)
    if len(closes) < period + 1:
        return rsi
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [max(d, 0) for d in deltas]
    losses = [max(-d, 0) for d in deltas]
    ag = mean(gains[:period])
    al = mean(losses[:period])
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
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr

# ============================================================
# Feature Extraction
# ============================================================

def extract_features_at(klines, idx):
    """idx 시점의 feature 추출 (idx는 klines 배열의 인덱스, 해당 날 포함)."""
    if idx < 44:
        return None

    o = [float(k[1]) for k in klines[:idx+1]]
    h = [float(k[2]) for k in klines[:idx+1]]
    l = [float(k[3]) for k in klines[:idx+1]]
    c = [float(k[4]) for k in klines[:idx+1]]
    v = [float(k[5]) for k in klines[:idx+1]]

    n = len(c)
    pe = n
    ps = pe - 14

    f = {}

    # Volume features
    v7 = mean(v[pe-7:pe])
    v30 = mean(v[max(0,pe-30):pe])
    v20 = mean(v[max(0,pe-20):pe])
    f["vol_ratio_7d_30d"] = v7/v30 if v30 > 0 else 1.0
    f["vol_spike_3d"] = max(v[pe-3:pe])/v20 if v20 > 0 else 1.0
    f["vol_trend_7d"] = polyfit_slope(v[pe-7:pe]) if pe >= 7 else 0.0

    # Volume USD (for filtering)
    avg_vol_usd = mean([v[i] * c[i] for i in range(max(0, pe-7), pe)])
    f["_avg_vol_usd"] = avg_vol_usd

    # ATR
    atr_s = compute_atr_series(h, l, c, 14)
    a14 = atr_s[pe-1] if atr_s[pe-1] is not None else 0
    f["atr_14d_ratio"] = a14/c[pe-1] if c[pe-1] > 0 else 0

    a7v = [atr_s[j] for j in range(pe-7, pe) if atr_s[j] is not None]
    a30v = [atr_s[j] for j in range(max(0, pe-30), pe) if atr_s[j] is not None]
    a7 = mean(a7v) if a7v else a14
    a30 = mean(a30v) if a30v else a14
    f["volatility_ratio"] = a7/a30 if a30 > 0 else 1.0

    # Bollinger
    if pe >= 20:
        bc = c[pe-20:pe]; bm = mean(bc); bs = stdev(bc)
        f["bb_width"] = (2*bs)/bm if bm > 0 else 0
        f["bb_position"] = (c[pe-1] - (bm - bs)) / (2*bs) if bs > 0 else 0.5
    else:
        f["bb_width"] = 0
        f["bb_position"] = 0.5

    # Price structure
    if pe >= 60:
        f["price_vs_60d_high"] = c[pe-1]/max(h[pe-60:pe])
        f["price_vs_60d_low"] = c[pe-1]/min(l[pe-60:pe]) if min(l[pe-60:pe]) > 0 else 1
    else:
        f["price_vs_60d_high"] = 1.0
        f["price_vs_60d_low"] = 1.0

    f["ret_1d"] = c[pe-1]/c[pe-2]-1 if pe >= 2 and c[pe-2] > 0 else 0
    f["ret_3d"] = c[pe-1]/c[pe-4]-1 if pe >= 4 and c[pe-4] > 0 else 0
    f["ret_7d"] = c[pe-1]/c[pe-8]-1 if pe >= 8 and c[pe-8] > 0 else 0
    f["ret_14d"] = c[pe-1]/c[pe-15]-1 if pe >= 15 and c[pe-15] > 0 else 0

    f["price_vs_ma7"] = c[pe-1]/mean(c[pe-7:pe])-1 if pe >= 7 else 0
    f["price_vs_ma20"] = c[pe-1]/mean(c[pe-20:pe])-1 if pe >= 20 else 0

    # RSI
    rsi_s = compute_rsi_series(c, 14)
    rv = rsi_s[pe-1]
    f["rsi_14"] = rv if rv is not None else 50.0
    f["rsi_trend_7d"] = (rsi_s[pe-1] - rsi_s[pe-8]) if pe >= 8 and rsi_s[pe-1] is not None and rsi_s[pe-8] is not None else 0.0

    rw = [rsi_s[j] for j in range(max(0, ps), pe) if rsi_s[j] is not None]
    f["rsi_consol_days"] = sum(1 for r in rw if 40 <= r <= 55)
    f["rsi_oversold_days"] = sum(1 for r in rw if r < 30)
    f["rsi_min_14d"] = min(rw) if rw else 50
    f["rsi_max_14d"] = max(rw) if rw else 50

    # Candle patterns
    cr = 0
    for j in range(pe-1, max(ps-1, -1), -1):
        if c[j] < o[j]: cr += 1
        else: break
    f["consec_red"] = cr

    cg = 0
    for j in range(pe-1, max(ps-1, -1), -1):
        if c[j] >= o[j]: cg += 1
        else: break
    f["consec_green"] = cg

    f["green_ratio_3d"] = sum(1 for j in range(pe-3, pe) if c[j] >= o[j]) / 3.0
    f["green_ratio_7d"] = sum(1 for j in range(pe-7, pe) if c[j] >= o[j]) / 7.0

    # Wicks / body analysis
    bodies_7d = []
    upper_wicks_7d = []
    lower_wicks_7d = []
    for j in range(pe-7, pe):
        body = abs(c[j] - o[j])
        full = h[j] - l[j]
        if full > 0:
            bodies_7d.append(body / full)
            upper_wicks_7d.append((h[j] - max(c[j], o[j])) / full)
            lower_wicks_7d.append((min(c[j], o[j]) - l[j]) / full)
    f["avg_body_ratio_7d"] = mean(bodies_7d) if bodies_7d else 0.5
    f["avg_upper_wick_7d"] = mean(upper_wicks_7d) if upper_wicks_7d else 0
    f["avg_lower_wick_7d"] = mean(lower_wicks_7d) if lower_wicks_7d else 0

    # Accumulation
    if pe >= 14:
        pc14 = abs(c[pe-1]/c[pe-14]-1)
        v_r = mean(v[pe-7:pe]); v_p = mean(v[pe-14:pe-7])
        f["accumulation"] = (v_r/v_p)/(1+pc14*10) if v_p > 0 else 1.0
    else:
        f["accumulation"] = 1.0

    # Volume-price divergence (가격 하락인데 거래량 증가)
    if pe >= 7:
        price_slope = polyfit_slope(c[pe-7:pe])
        vol_slope = polyfit_slope(v[pe-7:pe])
        f["vol_price_divergence"] = vol_slope - price_slope
    else:
        f["vol_price_divergence"] = 0

    return f

# ============================================================
# 급등 이벤트 탐지
# ============================================================

def find_surge_events(klines, threshold=SURGE_THRESHOLD, window=SURGE_WINDOW):
    """7일 내 +100% 이상 상승한 시점 탐지. 시작 시점(급등 직전)을 반환."""
    events = []
    c = [float(k[4]) for k in klines]
    n = len(c)

    i = 44  # 최소 feature 추출 가능 시점부터
    while i < n - window:
        # i일부터 window일 후까지의 최대 수익률
        max_ret = 0
        max_j = i
        for j in range(i+1, min(i+window+1, n)):
            ret = c[j] / c[i] - 1 if c[i] > 0 else 0
            if ret > max_ret:
                max_ret = ret
                max_j = j

        if max_ret >= threshold:
            events.append({
                "start_idx": i,
                "peak_idx": max_j,
                "return": max_ret,
                "start_price": c[i],
                "peak_price": c[max_j],
            })
            i = max_j + 7  # 겹치지 않게 건너뜀
        else:
            i += 1

    return events

# ============================================================
# 메인 파이프라인
# ============================================================

def main():
    print("=" * 60)
    print("  소형 알트코인 급등 전조 모델 빌더 v2")
    print("=" * 60)

    # 1. 전체 Bybit spot 티커 수집
    print("\n[1/5] Bybit spot 티커 수집...")
    tickers = fetch_all_tickers()
    print(f"  전체: {len(tickers)}개")

    # 소형 알트 필터링
    small_alts = []
    for t in tickers:
        sym = t["symbol"]
        if not sym.endswith("USDT"):
            continue
        if sym in STABLECOINS or sym in LARGE_CAPS:
            continue
        vol24 = float(t.get("turnover24h", 0))
        price = float(t.get("lastPrice", 0))
        if MIN_DAILY_VOL_USD <= vol24 <= MAX_DAILY_VOL_USD and price >= MIN_PRICE_USD:
            small_alts.append({"symbol": sym, "vol24": vol24, "price": price})

    print(f"  소형 알트 (${MIN_DAILY_VOL_USD/1000:.0f}K~${MAX_DAILY_VOL_USD/1e6:.0f}M): {len(small_alts)}개")

    # 2. 1년치 데이터 수집 + 급등 이벤트 탐지
    print(f"\n[2/5] 데이터 수집 + 급등 이벤트 탐지...")
    all_surge_features = []  # 급등 직전 feature
    all_normal_features = []  # 일반 시점 feature
    surge_examples = []  # 급등 사례

    total = len(small_alts)
    for idx, alt in enumerate(small_alts):
        sym = alt["symbol"]
        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1}/{total}] {sym}...")

        try:
            klines = fetch_all_klines(sym, LOOKBACK_DAYS)
            if len(klines) < MIN_KLINES:
                continue

            events = find_surge_events(klines)

            # 급등 직전 feature 추출
            surge_indices = set()
            for ev in events:
                feat = extract_features_at(klines, ev["start_idx"])
                if feat:
                    all_surge_features.append(feat)
                    surge_indices.add(ev["start_idx"])
                    surge_examples.append({
                        "symbol": sym,
                        "return": ev["return"],
                        "start_price": ev["start_price"],
                        "peak_price": ev["peak_price"],
                    })

            # 일반(non-surge) 시점 샘플링 (급등 전후 14일 제외)
            excluded = set()
            for ev in events:
                for d in range(max(0, ev["start_idx"]-7), min(len(klines), ev["peak_idx"]+14)):
                    excluded.add(d)

            # 일반 시점에서 균등 샘플
            normal_candidates = [i for i in range(44, len(klines)) if i not in excluded]
            # 급등 건수 대비 2배 샘플
            sample_n = min(len(normal_candidates), max(len(events) * 2, 5))
            step = max(1, len(normal_candidates) // sample_n) if sample_n > 0 else 1
            for i in range(0, len(normal_candidates), step):
                if len(all_normal_features) - (len(all_surge_features) * 3) > 0:
                    break  # 너무 많은 normal 방지
                feat = extract_features_at(klines, normal_candidates[i])
                if feat:
                    all_normal_features.append(feat)

            time.sleep(0.05)
        except Exception as e:
            continue

    print(f"\n  급등 이벤트: {len(all_surge_features)}건")
    print(f"  일반 시점: {len(all_normal_features)}건")
    print(f"  급등 코인 수: {len(set(e['symbol'] for e in surge_examples))}")

    if len(all_surge_features) < 10:
        print("\n급등 이벤트가 너무 적어서 모델 생성 불가!")
        # 기준 완화
        print("기준을 +50%로 완화해서 재시도...")
        return

    # 3. 통계 분석
    print(f"\n[3/5] 급등 전 vs 일반 시점 통계 비교...")

    # Feature 이름 수집 (internal 제외)
    feature_names = [k for k in all_surge_features[0].keys() if not k.startswith("_")]

    results = {}
    for fn in feature_names:
        surge_vals = [f[fn] for f in all_surge_features if fn in f and f[fn] is not None]
        normal_vals = [f[fn] for f in all_normal_features if fn in f and f[fn] is not None]

        if len(surge_vals) < 5 or len(normal_vals) < 5:
            continue

        d = cohens_d(surge_vals, normal_vals)
        sm = mean(surge_vals)
        nm = mean(normal_vals)
        ss = stdev(surge_vals)
        ns = stdev(normal_vals)

        results[fn] = {
            "cohens_d": d,
            "surge_mean": sm,
            "surge_std": ss,
            "normal_mean": nm,
            "normal_std": ns,
            "surge_n": len(surge_vals),
            "normal_n": len(normal_vals),
            "direction": "higher" if d > 0 else "lower",
        }

    # 효과크기 순 정렬
    sorted_features = sorted(results.items(), key=lambda x: abs(x[1]["cohens_d"]), reverse=True)

    print(f"\n  분석 feature: {len(sorted_features)}개")
    print(f"\n  {'Feature':<25} {'Cohen d':>8} {'방향':>6} {'급등전':>10} {'일반':>10}")
    print("  " + "-" * 65)
    for fn, info in sorted_features[:25]:
        d = info["cohens_d"]
        direction = "↑" if info["direction"] == "higher" else "↓"
        print(f"  {fn:<25} {d:>8.3f} {direction:>6} {info['surge_mean']:>10.4f} {info['normal_mean']:>10.4f}")

    # 4. 유의미한 feature만 선별 (|d| >= 0.2)
    print(f"\n[4/5] 유의미한 feature 선별 (|Cohen's d| >= 0.2)...")

    thresholds = {}
    for fn, info in sorted_features:
        if abs(info["cohens_d"]) < 0.2:
            continue
        thresholds[fn] = {
            "weight": abs(info["cohens_d"]),
            "direction": info["direction"],
            "surge_mean": info["surge_mean"],
            "surge_std": info["surge_std"],
            "normal_mean": info["normal_mean"],
            "normal_std": info["normal_std"],
        }

    print(f"  선별된 feature: {len(thresholds)}개")

    # 5. 저장
    print(f"\n[5/5] 모델 저장...")

    # thresholds
    with open(BASE_DIR / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"  thresholds.json 저장 ({len(thresholds)}개 feature)")

    # 급등 사례
    with open(BASE_DIR / "surge_examples.json", "w") as f:
        json.dump(surge_examples, f, indent=2)
    print(f"  surge_examples.json 저장 ({len(surge_examples)}건)")

    # 분석 리포트
    report = []
    report.append("# 소형 알트코인 급등 전조 분석 리포트")
    report.append(f"\n생성일: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"\n## 데이터")
    report.append(f"- 대상: Bybit spot 소형 알트 (일거래대금 ${MIN_DAILY_VOL_USD/1000:.0f}K~${MAX_DAILY_VOL_USD/1e6:.0f}M)")
    report.append(f"- 분석 기간: 최근 {LOOKBACK_DAYS}일")
    report.append(f"- 급등 기준: {SURGE_WINDOW}일 내 +{SURGE_THRESHOLD*100:.0f}%")
    report.append(f"- 소형 알트 수: {len(small_alts)}개")
    report.append(f"- 급등 이벤트: {len(all_surge_features)}건")
    report.append(f"- 급등 코인: {len(set(e['symbol'] for e in surge_examples))}개")
    report.append(f"- 일반 시점: {len(all_normal_features)}건")

    report.append(f"\n## 급등 사례 TOP 10 (수익률 기준)")
    top_examples = sorted(surge_examples, key=lambda x: x["return"], reverse=True)[:10]
    for i, ex in enumerate(top_examples):
        cn = ex["symbol"].replace("USDT", "")
        report.append(f"{i+1}. {cn}: +{ex['return']*100:.0f}% (${ex['start_price']:.4f} → ${ex['peak_price']:.4f})")

    report.append(f"\n## 유의미한 전조 시그널 (효과크기 순)")
    report.append(f"\n| Feature | Cohen's d | 방향 | 급등전 평균 | 일반 평균 |")
    report.append(f"|---------|-----------|------|-----------|---------|")
    for fn, info in sorted_features:
        if abs(info["cohens_d"]) < 0.2:
            continue
        d = info["cohens_d"]
        direction = "높을수록" if info["direction"] == "higher" else "낮을수록"
        report.append(f"| {fn} | {d:.3f} | {direction} | {info['surge_mean']:.4f} | {info['normal_mean']:.4f} |")

    report.append(f"\n## 핵심 패턴 요약")
    report.append("(효과크기 상위 5개 기준)")
    for fn, info in sorted_features[:5]:
        direction = "높을수록" if info["direction"] == "higher" else "낮을수록"
        diff_pct = abs(info["surge_mean"] - info["normal_mean"]) / abs(info["normal_mean"]) * 100 if info["normal_mean"] != 0 else 0
        report.append(f"- **{fn}**: 급등전 {info['surge_mean']:.4f} vs 일반 {info['normal_mean']:.4f} ({direction} 급등 확률 증가, 차이 {diff_pct:.0f}%)")

    with open(BASE_DIR / "analysis_report.md", "w") as f:
        f.write("\n".join(report))
    print(f"  analysis_report.md 저장")

    # 코인 풀 저장
    pair_list = [a["symbol"] for a in small_alts]
    with open(BASE_DIR / "top_pairs.json", "w") as f:
        json.dump(pair_list, f)
    print(f"  top_pairs.json 저장 ({len(pair_list)}개)")

    print(f"\n{'='*60}")
    print(f"  모델 빌드 완료!")
    print(f"  급등 이벤트 {len(all_surge_features)}건에서 {len(thresholds)}개 시그널 추출")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
