#!/usr/bin/env python3
"""
급등 전조 스캐너 v2 — LaunchAgent 상시 실행
6시간마다 Bybit spot 스캔 → Slack DM
순수 stdlib (numpy/pandas 없음)

결과 일관성 보장:
- 오늘 캔들 포함 (현재 상태 반영)
- 코인 풀 고정 (top_pairs.json) → 매번 같은 대상
- Deterministic scoring (가우시안 분포 기반)
"""

import json, math, os, sys, time, datetime
import urllib.request, urllib.error
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ============================================================
# 설정
# ============================================================

BASE_DIR = Path(__file__).parent
SCAN_INTERVAL = 1 * 60 * 60   # 1시간
MAX_VOL_USD = 10_000_000       # 소형캡만
MIN_SCORE = 60
TOP_N = 10
STABLECOINS = {"UUSDT", "XUSDUSDT", "PAXGUSDT", "USDCUSDT", "DAIUSDT",
               "FDUSDUSDT", "TUSDUSDT", "BUSDUSDT", "EURUSDT", "USTCUSDT"}

# Slack webhook (멀티 웹훅 지원)
SLACK_WEBHOOK_URLS = []
_env_path = os.path.expanduser("~/.openclaw/.env.slack-surge")
if os.path.exists(_env_path):
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("SLACK_WEBHOOK_URL") and "=" in line:
                val = line.split("=", 1)[1]
                if val and val not in SLACK_WEBHOOK_URLS:
                    SLACK_WEBHOOK_URLS.append(val)
SLACK_WEBHOOK_URL = SLACK_WEBHOOK_URLS[0] if SLACK_WEBHOOK_URLS else ""

# 코인 풀 파일 (없으면 자동 생성)
PAIRS_FILE = BASE_DIR / "top_pairs.json"
THRESHOLDS_FILE = BASE_DIR / "thresholds.json"
PREV_RESULTS_FILE = BASE_DIR / "prev_results.json"

# ============================================================
# Bybit API
# ============================================================

def bybit_get(endpoint, params=None):
    url = "https://api.bybit.com" + endpoint
    if params:
        url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "surge-alert/2.0")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())

def fetch_klines(symbol, interval="D", limit=200):
    data = bybit_get("/v5/market/kline", {
        "category": "spot", "symbol": symbol, "interval": interval, "limit": str(limit)
    })
    rows = data.get("result", {}).get("list", [])
    if not rows:
        return None
    rows = list(reversed(rows))  # oldest first
    return rows  # 오늘 캔들 포함 (현재 상태 반영)

def refresh_pairs():
    """Bybit에서 USDT 페어 목록 갱신 (일주일에 한번)."""
    data = bybit_get("/v5/market/tickers", {"category": "spot"})
    tickers = data.get("result", {}).get("list", [])
    pairs = []
    for t in tickers:
        sym = t["symbol"]
        if not sym.endswith("USDT"):
            continue
        if sym in STABLECOINS:
            continue
        vol24 = float(t.get("turnover24h", 0))
        if vol24 > 100000:  # 최소 거래량
            pairs.append(sym)
    pairs.sort()
    with open(PAIRS_FILE, "w") as f:
        json.dump(pairs, f)
    return pairs

def load_pairs():
    if PAIRS_FILE.exists():
        mtime = os.path.getmtime(PAIRS_FILE)
        age_hours = (time.time() - mtime) / 3600
        if age_hours < 168:  # 7일
            with open(PAIRS_FILE) as f:
                return json.load(f)
    return refresh_pairs()

# ============================================================
# Math helpers (pure stdlib)
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
    if den == 0:
        return 0.0
    return (num / den) / y_mean

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

def extract_features(klines):
    n = len(klines)
    if n < 60:
        return None

    o = [float(k[1]) for k in klines]
    h = [float(k[2]) for k in klines]
    l = [float(k[3]) for k in klines]
    c = [float(k[4]) for k in klines]
    v = [float(k[5]) for k in klines]

    pe = n
    ps = pe - 14

    f = {}

    # Volume
    v7 = mean(v[pe-7:pe]); v30 = mean(v[max(0,pe-30):pe]); v20 = mean(v[max(0,pe-20):pe])
    f["vol_7d_30d_ratio"] = v7/v30 if v30 > 0 else 1.0
    f["vol_spike_3d"] = max(v[pe-3:pe])/v20 if v20 > 0 else 1.0
    f["vol_trend_7d"] = polyfit_slope(v[pe-7:pe]) if pe >= 7 else 0.0

    # ATR
    atr_s = compute_atr_series(h, l, c, 14)
    a14 = atr_s[pe-1] if atr_s[pe-1] is not None else 0
    f["atr_14d_ratio"] = a14/c[pe-1] if c[pe-1] > 0 else 0
    a7v = [atr_s[j] for j in range(pe-7, pe) if atr_s[j] is not None]
    a30v = [atr_s[j] for j in range(max(0, pe-30), pe) if atr_s[j] is not None]
    a7 = mean(a7v) if a7v else a14
    a30 = mean(a30v) if a30v else a14
    f["volatility_contraction"] = a7/a30 if a30 > 0 else 1.0

    # Bollinger
    if pe >= 20:
        bc = c[pe-20:pe]; bm = mean(bc); bs = stdev(bc)
        f["bb_width"] = (2*bs)/bm if bm > 0 else 0
    else:
        f["bb_width"] = 0

    # Price
    f["price_vs_60d_high"] = c[pe-1]/max(h[pe-60:pe]) if pe >= 60 and max(h[pe-60:pe]) > 0 else 1.0
    f["price_ret_7d"] = c[pe-1]/c[pe-8]-1 if pe >= 8 and c[pe-8] > 0 else 0
    f["price_vs_ma20"] = c[pe-1]/mean(c[pe-20:pe])-1 if pe >= 20 and mean(c[pe-20:pe]) > 0 else 0

    # RSI
    rsi_s = compute_rsi_series(c, 14)
    rv = rsi_s[pe-1]
    f["rsi_14"] = rv if rv is not None else 50.0
    rw = [rsi_s[j] for j in range(ps, pe) if rsi_s[j] is not None]
    f["rsi_consolidation_days"] = sum(1 for r in rw if 40 <= r <= 55)
    f["rsi_trend_7d"] = (rsi_s[pe-1] - rsi_s[pe-8]) if pe >= 8 and rsi_s[pe-1] is not None and rsi_s[pe-8] is not None else 0.0

    # Candle patterns
    cr = 0
    for j in range(pe-1, max(ps-1, -1), -1):
        if c[j] < o[j]: cr += 1
        else: break
    f["consecutive_red"] = cr
    f["green_ratio_3d"] = sum(1 for j in range(pe-3, pe) if c[j] >= o[j]) / 3.0
    f["coin_ret_7d"] = f["price_ret_7d"]

    # Accumulation
    if pe >= 14:
        pc14 = abs(c[pe-1]/c[pe-14]-1)
        v_r = mean(v[pe-7:pe]); v_p = mean(v[pe-14:pe-7])
        f["accumulation_signal"] = (v_r/v_p)/(1+pc14*10) if v_p > 0 else 1.0
    else:
        f["accumulation_signal"] = 1.0

    return f

# ============================================================
# Gaussian Scoring
# ============================================================

def compute_score(features, thresholds):
    if not features or not thresholds:
        return 0.0
    sc = 0.0; mx = 0.0
    for fn, info in thresholds.items():
        if fn not in features: continue
        w = info["weight"]; mx += w
        val = features[fn]
        sm = info["surge_mean"]; ss = info["surge_std"]
        nm = info["normal_mean"]; d = info["direction"]
        if ss > 0:
            z = (val - sm) / ss
            gauss = math.exp(-0.5 * z * z)
        else:
            gauss = 1.0 if val == sm else 0.0
        if d == "higher" and val < nm: gauss *= 0.3
        elif d == "lower" and val > nm: gauss *= 0.3
        sc += w * gauss
    return (sc / mx * 100) if mx > 0 else 0.0

def count_fit(features, thresholds):
    fit = total = 0
    for fn, info in thresholds.items():
        if fn not in features: continue
        total += 1
        val = features[fn]
        sm = info["surge_mean"]; ss = info["surge_std"]
        nm = info["normal_mean"]; d = info["direction"]
        if ss > 0:
            z = (val - sm) / ss
            gauss = math.exp(-0.5 * z * z)
        else:
            gauss = 1.0 if val == sm else 0.0
        if d == "higher" and val < nm: gauss *= 0.3
        elif d == "lower" and val > nm: gauss *= 0.3
        if gauss > 0.5: fit += 1
    return fit, total

# ============================================================
# Slack
# ============================================================

def send_slack(msg):
    if not SLACK_WEBHOOK_URL:
        print(f"[NO SLACK]\n{msg}\n")
        return
    MAX_LEN = 3500
    chunks = [msg] if len(msg) <= MAX_LEN else []
    if not chunks:
        lines = msg.split("\n")
        chunk = []; chunk_len = 0
        for line in lines:
            if chunk_len + len(line) + 1 > MAX_LEN and chunk:
                chunks.append("\n".join(chunk))
                chunk = [line]; chunk_len = len(line)
            else:
                chunk.append(line); chunk_len += len(line) + 1
        if chunk: chunks.append("\n".join(chunk))
    for c in chunks:
        payload = json.dumps({
            "text": c,
            "username": "Surge Alert",
            "icon_emoji": ":mag:",
        }).encode()
        for wh_url in SLACK_WEBHOOK_URLS:
            req = urllib.request.Request(
                wh_url, data=payload,
                headers={"Content-Type": "application/json"}, method="POST")
            try:
                urllib.request.urlopen(req, timeout=30)
            except Exception as e:
                print(f"Slack error ({wh_url[:50]}...): {e}")
        time.sleep(1)

# ============================================================
# 변동 감지 (이전 결과와 비교)
# ============================================================

def load_prev_results():
    if PREV_RESULTS_FILE.exists():
        with open(PREV_RESULTS_FILE) as f:
            return json.load(f)
    return {}

def save_prev_results(results):
    data = {r["symbol"]: {"score": r["score"], "price": r["price"]} for r in results}
    with open(PREV_RESULTS_FILE, "w") as f:
        json.dump(data, f)

# ============================================================
# 메인 스캔
# ============================================================

def run_scan():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"[{now}] 스캔 시작...")

    if not THRESHOLDS_FILE.exists():
        print("thresholds.json not found!")
        return
    with open(THRESHOLDS_FILE) as f:
        TH = json.load(f)

    pairs = load_pairs()
    print(f"  {len(pairs)}개 페어 로드")

    prev = load_prev_results()
    results = []
    errors = 0

    for sym in pairs:
        try:
            klines = fetch_klines(sym, "D", 200)
            if not klines or len(klines) < 60:
                continue

            # 30일 평균 일거래대금
            recent = klines[-30:]
            avg_vol_usd = mean([float(k[5]) * float(k[4]) for k in recent])
            if avg_vol_usd > MAX_VOL_USD:
                continue

            last_price = float(klines[-1][4])
            feat = extract_features(klines)
            if not feat:
                continue
            sc = compute_score(feat, TH)
            if sc >= MIN_SCORE:
                fit, total = count_fit(feat, TH)
                results.append({
                    "symbol": sym,
                    "score": sc,
                    "price": last_price,
                    "vol_usd": avg_vol_usd,
                    "features": feat,
                    "fit": fit,
                    "total": total,
                })
            time.sleep(0.05)
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  {sym}: {e}")
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[:TOP_N]

    print(f"[{now}] 스캔 완료: {len(results)}개 통과, {errors}개 에러")

    if not top:
        print("  시그널 없음")
        return

    # 변동 감지
    new_entries = []
    for r in top[:5]:
        cn = r["symbol"].replace("USDT", "")
        if r["symbol"] not in prev:
            new_entries.append(cn)

    # Slack 메시지 포맷
    lines = [f":mag: *급등 전조 스캐너* ({now})"]
    lines.append(f"확정 일봉 기준 | 점수 {MIN_SCORE}+ | 소형캡 (<${MAX_VOL_USD/1e6:.0f}M)")

    if new_entries:
        lines.append(f":new: 신규 진입: {', '.join(new_entries)}")

    lines.append("")
    lines.append("```")
    lines.append(f"{'#':>2} {'코인':<10} {'점수':>5} {'적합':>5} {'현재가':<11} {'거래대금':<8}")
    lines.append("-" * 48)
    for i, r in enumerate(top):
        cn = r["symbol"].replace("USDT", "")
        vol_str = f"${r['vol_usd']/1e6:.1f}M"
        # 이전 대비 변화
        prev_sc = prev.get(r["symbol"], {}).get("score", 0)
        delta = r["score"] - prev_sc if prev_sc > 0 else 0
        delta_str = f" ({delta:+.0f})" if delta != 0 else ""
        lines.append(f"{i+1:>2} {cn:<10} {r['score']:>5.1f}{delta_str:<6} {r['fit']}/{r['total']:<3} ${r['price']:<10.4f} {vol_str:<8}")
    lines.append("```")

    # TOP 3 상세
    lines.append("")
    for i, r in enumerate(top[:3]):
        cn = r["symbol"].replace("USDT", "")
        ret7 = r["features"].get("price_ret_7d", 0) * 100
        rsi = r["features"].get("rsi_14", 0)
        vol_t = r["features"].get("vol_trend_7d", 0) * 100
        rsi_t = r["features"].get("rsi_trend_7d", 0)
        lines.append(f"*{i+1}. {cn}* ({r['score']:.1f}점, 적합 {r['fit']}/{r['total']})")
        lines.append(f"  7일수익률 {ret7:+.1f}% | RSI {rsi:.0f} | 거래량추세 {vol_t:+.1f}% | RSI변화 {rsi_t:+.1f}")

    lines.append("")
    lines.append("_가우시안 분포 | 6시간 간격 | 소형캡 전용_")

    msg = "\n".join(lines)
    send_slack(msg)
    save_prev_results(results)
    print(f"[{now}] Slack 전송 완료")

# ============================================================
# 루프
# ============================================================

def main():
    print("=" * 50)
    print("  급등 전조 스캐너 v2")
    print(f"  스캔 간격: {SCAN_INTERVAL // 3600}시간")
    print(f"  기준: 점수 {MIN_SCORE}+, 거래대금 <${MAX_VOL_USD/1e6:.0f}M")
    print(f"  오늘 캔들 포함 (현재 상태 반영)")
    print("=" * 50)

    try:
        run_scan()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    while True:
        time.sleep(SCAN_INTERVAL)
        try:
            run_scan()
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
