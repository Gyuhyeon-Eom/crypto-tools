#!/usr/bin/env python3
"""
소형 알트코인 급등 탐지기 v3 — 2단계 시스템
순수 stdlib (numpy/pandas 없음)

[Layer 1] 일봉 기반 — 10분마다 실행
  급등 가능 후보군 선별 (바닥 다지기 + 구조적 패턴)

[Layer 2] 시간봉/15분봉 기반 — 5분마다 실행
  후보군의 실시간 거래량 스파이크 + 가격 모멘텀 감지

[Layer 3] 멀티타임프레임 컨플루언스
  일봉 점수 + 시간봉 스파이크 + 15분봉 모멘텀 종합

알림:
  - 즉시 알림: Layer 2에서 스파이크 감지 시
  - 정기 리포트: 1시간마다 후보 리스트
"""

import json, math, os, sys, time, datetime, hashlib
import urllib.request, urllib.error
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ============================================================
# 설정
# ============================================================

BASE_DIR = Path(__file__).parent
THRESHOLDS_FILE = BASE_DIR / "thresholds.json"
PAIRS_FILE = BASE_DIR / "top_pairs.json"
STATE_FILE = BASE_DIR / "detector_state.json"

# 타이밍
DAILY_SCAN_INTERVAL = 600       # 일봉 스캔: 10분
REALTIME_SCAN_INTERVAL = 300    # 실시간 스캔: 5분
REPORT_INTERVAL = 3600          # 정기 리포트: 1시간
PAIR_REFRESH_INTERVAL = 86400   # 페어 갱신: 24시간

# 필터
MAX_VOL_USD = 10_000_000
MIN_VOL_USD = 50_000
MIN_DAILY_SCORE = 55            # 일봉 후보 기준 (넉넉하게)
TOP_WATCHLIST = 30              # 실시간 감시 대상
TOP_ALERT = 10                  # 알림 대상

# 실시간 스파이크 기준
VOL_SPIKE_THRESHOLD = 3.0       # 최근 1h 거래량 / 24h 평균 >= 3배
VOL_MEGA_SPIKE = 5.0            # 5배 이상이면 즉시 알림
PRICE_MOMENTUM_1H = 0.03        # 1시간 +3% 이상
PRICE_MOMENTUM_4H = 0.05        # 4시간 +5% 이상
BREAKOUT_THRESHOLD = 0.02       # 24h 고점 돌파 +2%

STABLECOINS = {
    "USDCUSDT", "UUSDT", "XUSDUSDT", "PAXGUSDT", "DAIUSDT",
    "FDUSDUSDT", "TUSDUSDT", "BUSDUSDT", "EURUSDT", "USTCUSDT",
}
LARGE_CAPS = {
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
    "ADAUSDT", "DOGEUSDT", "TRXUSDT", "DOTUSDT", "MATICUSDT",
    "LTCUSDT", "BCHUSDT", "LINKUSDT", "AVAXUSDT", "UNIUSDT",
    "AAVEUSDT", "ATOMUSDT",
}

# Slack (멀티 웹훅 지원)
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

# ============================================================
# Math helpers
# ============================================================

def mean(arr):
    return sum(arr) / len(arr) if arr else 0.0

def stdev(arr):
    if len(arr) < 2: return 0.0
    m = mean(arr)
    return math.sqrt(sum((x - m) ** 2 for x in arr) / (len(arr) - 1))

def median(arr):
    s = sorted(arr)
    n = len(s)
    if n == 0: return 0
    if n % 2 == 1: return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2

def polyfit_slope(y_vals):
    n = len(y_vals)
    if n < 2: return 0.0
    xm = (n - 1) / 2.0
    ym = mean(y_vals)
    if ym == 0: return 0.0
    num = sum((i - xm) * (y - ym) for i, y in enumerate(y_vals))
    den = sum((i - xm) ** 2 for i in range(n))
    return (num / den) / ym if den else 0.0

def ema(arr, period):
    if not arr: return []
    k = 2 / (period + 1)
    result = [arr[0]]
    for i in range(1, len(arr)):
        result.append(arr[i] * k + result[-1] * (1 - k))
    return result

# ============================================================
# Bybit API
# ============================================================

def bybit_get(endpoint, params=None, retries=3):
    url = "https://api.bybit.com" + endpoint
    if params:
        url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "surge-detector/3.0")
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(1)

def fetch_klines(symbol, interval="D", limit=200):
    data = bybit_get("/v5/market/kline", {
        "category": "spot", "symbol": symbol, "interval": interval, "limit": str(limit)
    })
    rows = data.get("result", {}).get("list", [])
    return list(reversed(rows)) if rows else []

def fetch_tickers():
    data = bybit_get("/v5/market/tickers", {"category": "spot"})
    return {t["symbol"]: t for t in data.get("result", {}).get("list", [])}

# ============================================================
# RSI
# ============================================================

def compute_rsi_series(closes, period=14):
    rsi = [None] * len(closes)
    if len(closes) < period + 1: return rsi
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [max(d, 0) for d in deltas]
    losses = [max(-d, 0) for d in deltas]
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
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr

# ============================================================
# Layer 1: 일봉 기반 구조적 분석
# ============================================================

def extract_daily_features(klines):
    n = len(klines)
    if n < 45: return None

    o = [float(k[1]) for k in klines]
    h = [float(k[2]) for k in klines]
    l = [float(k[3]) for k in klines]
    c = [float(k[4]) for k in klines]
    v = [float(k[5]) for k in klines]
    pe = n

    f = {}

    # 가격 구조
    f["ret_1d"] = c[-1]/c[-2]-1 if c[-2] > 0 else 0
    f["ret_3d"] = c[-1]/c[-4]-1 if pe >= 4 and c[-4] > 0 else 0
    f["ret_7d"] = c[-1]/c[-8]-1 if pe >= 8 and c[-8] > 0 else 0
    f["ret_14d"] = c[-1]/c[-15]-1 if pe >= 15 and c[-15] > 0 else 0

    ma7 = mean(c[-7:]) if pe >= 7 else c[-1]
    ma20 = mean(c[-20:]) if pe >= 20 else c[-1]
    f["price_vs_ma7"] = c[-1]/ma7 - 1
    f["price_vs_ma20"] = c[-1]/ma20 - 1

    if pe >= 60:
        f["price_vs_60d_high"] = c[-1] / max(h[-60:])
        f["price_vs_60d_low"] = c[-1] / min(l[-60:]) if min(l[-60:]) > 0 else 1
    else:
        f["price_vs_60d_high"] = 1.0
        f["price_vs_60d_low"] = 1.0

    # RSI
    rsi_s = compute_rsi_series(c, 14)
    f["rsi_14"] = rsi_s[-1] if rsi_s[-1] is not None else 50
    f["rsi_trend_7d"] = (rsi_s[-1] - rsi_s[-8]) if pe >= 8 and rsi_s[-1] and rsi_s[-8] else 0

    rw = [r for r in rsi_s[-14:] if r is not None]
    f["rsi_consol_days"] = sum(1 for r in rw if 40 <= r <= 55)
    f["rsi_oversold_days"] = sum(1 for r in rw if r < 30)
    f["rsi_min_14d"] = min(rw) if rw else 50

    # 캔들
    f["green_ratio_3d"] = sum(1 for j in range(-3, 0) if c[j] >= o[j]) / 3.0
    f["green_ratio_7d"] = sum(1 for j in range(-7, 0) if c[j] >= o[j]) / 7.0

    cr = 0
    for j in range(-1, max(-15, -pe), -1):
        if c[j] < o[j]: cr += 1
        else: break
    f["consec_red"] = cr

    cg = 0
    for j in range(-1, max(-15, -pe), -1):
        if c[j] >= o[j]: cg += 1
        else: break
    f["consec_green"] = cg

    # 거래량
    v7 = mean(v[-7:]); v30 = mean(v[max(-30, -pe):])
    f["vol_ratio_7d_30d"] = v7/v30 if v30 > 0 else 1
    f["vol_trend_7d"] = polyfit_slope(v[-7:]) if pe >= 7 else 0

    # ATR / 변동성
    atr_s = compute_atr_series(h, l, c, 14)
    a14 = atr_s[-1] if atr_s[-1] else 0
    f["atr_14d_ratio"] = a14/c[-1] if c[-1] > 0 else 0
    a7v = [atr_s[j] for j in range(-7, 0) if atr_s[j]]
    a30v = [atr_s[j] for j in range(max(-30, -pe), 0) if atr_s[j]]
    f["volatility_ratio"] = mean(a7v)/mean(a30v) if a30v and mean(a30v) > 0 else 1

    # BB
    if pe >= 20:
        bm = mean(c[-20:]); bs = stdev(c[-20:])
        f["bb_width"] = (2*bs)/bm if bm > 0 else 0
        f["bb_position"] = (c[-1] - (bm - bs)) / (2*bs) if bs > 0 else 0.5
    else:
        f["bb_width"] = 0; f["bb_position"] = 0.5

    # Wicks
    bodies = []; upwicks = []; lowwicks = []
    for j in range(-7, 0):
        body = abs(c[j] - o[j]); full = h[j] - l[j]
        if full > 0:
            bodies.append(body/full)
            upwicks.append((h[j] - max(c[j], o[j])) / full)
            lowwicks.append((min(c[j], o[j]) - l[j]) / full)
    f["avg_body_ratio_7d"] = mean(bodies) if bodies else 0.5
    f["avg_upper_wick_7d"] = mean(upwicks) if upwicks else 0
    f["avg_lower_wick_7d"] = mean(lowwicks) if lowwicks else 0

    # Accumulation
    if pe >= 14:
        pc14 = abs(c[-1]/c[-14]-1)
        vr = mean(v[-7:]); vp = mean(v[-14:-7])
        f["accumulation"] = (vr/vp)/(1+pc14*10) if vp > 0 else 1
    else:
        f["accumulation"] = 1

    # Volume-price divergence
    f["vol_price_divergence"] = polyfit_slope(v[-7:]) - polyfit_slope(c[-7:]) if pe >= 7 else 0

    # === 추가 feature: 급등 직전 특화 ===

    # 최근 급락 후 안정화 (7일 내 최대 하락폭)
    if pe >= 10:
        drops = []
        for j in range(-7, 0):
            window = c[max(-10, -pe):pe+j+1]
            if window and max(window) > 0:
                drops.append(c[pe+j] / max(window) - 1)
        f["max_drop_7d"] = min(drops) if drops else 0
    else:
        f["max_drop_7d"] = 0

    # 거래량 폭발 시작 신호 (최근 1~2일 거래량 vs 7일 평균)
    v_last2 = mean(v[-2:])
    v_prev5 = mean(v[-7:-2]) if pe >= 7 else v_last2
    f["vol_explosion_2d"] = v_last2 / v_prev5 if v_prev5 > 0 else 1

    # Squeeze: BB 폭 축소 후 확장 시작
    if pe >= 30:
        bb_widths = []
        for j in range(-10, 0):
            start = pe + j - 19
            end = pe + j + 1
            if start >= 0 and end <= pe:
                bc = c[start:end]
                if len(bc) >= 20:
                    bm = mean(bc); bs = stdev(bc)
                    bb_widths.append((2*bs)/bm if bm > 0 else 0)
        if len(bb_widths) >= 5:
            early = bb_widths[:5]
            late = bb_widths[-3:]
            max_late = max(late) if late else 1
            f["bb_squeeze"] = min(early) / max_late if max_late > 0 else 1
        else:
            f["bb_squeeze"] = 1
    else:
        f["bb_squeeze"] = 1

    return f

def compute_daily_score(features, thresholds):
    if not features or not thresholds: return 0.0
    sc = 0.0; mx = 0.0
    for fn, info in thresholds.items():
        if fn not in features: continue
        w = info["weight"]; mx += w; val = features[fn]
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

# ============================================================
# Layer 2: 실시간 거래량/모멘텀 분석
# ============================================================

def analyze_realtime(symbol):
    """시간봉 + 15분봉 분석."""
    result = {"symbol": symbol, "signals": [], "score": 0, "urgency": 0}

    try:
        # 1시간봉 (48시간)
        h1 = fetch_klines(symbol, "60", 48)
        if len(h1) < 24:
            return result

        h1_v = [float(k[5]) for k in h1]
        h1_c = [float(k[4]) for k in h1]
        h1_h = [float(k[2]) for k in h1]
        h1_o = [float(k[1]) for k in h1]

        current_price = h1_c[-1]
        if current_price <= 0:
            return result

        # --- 거래량 스파이크 ---
        vol_24h_avg = mean(h1_v[-24:])
        vol_last_1h = h1_v[-1]
        vol_last_3h = mean(h1_v[-3:])

        vol_spike_1h = vol_last_1h / vol_24h_avg if vol_24h_avg > 0 else 0
        vol_spike_3h = vol_last_3h / vol_24h_avg if vol_24h_avg > 0 else 0

        if vol_spike_1h >= VOL_MEGA_SPIKE:
            result["signals"].append(f"VOL_MEGA_1H: {vol_spike_1h:.1f}x")
            result["urgency"] += 3
        elif vol_spike_1h >= VOL_SPIKE_THRESHOLD:
            result["signals"].append(f"VOL_SPIKE_1H: {vol_spike_1h:.1f}x")
            result["urgency"] += 2

        if vol_spike_3h >= VOL_SPIKE_THRESHOLD:
            result["signals"].append(f"VOL_SPIKE_3H: {vol_spike_3h:.1f}x")
            result["urgency"] += 1

        # --- 가격 모멘텀 ---
        ret_1h = h1_c[-1] / h1_c[-2] - 1 if h1_c[-2] > 0 else 0
        ret_4h = h1_c[-1] / h1_c[-5] - 1 if len(h1_c) >= 5 and h1_c[-5] > 0 else 0

        if ret_1h >= PRICE_MOMENTUM_1H:
            result["signals"].append(f"MOM_1H: {ret_1h*100:+.1f}%")
            result["urgency"] += 2
        if ret_4h >= PRICE_MOMENTUM_4H:
            result["signals"].append(f"MOM_4H: {ret_4h*100:+.1f}%")
            result["urgency"] += 1

        # --- 24h 고점 돌파 ---
        high_24h = max(h1_h[-24:])
        if current_price > high_24h * (1 - BREAKOUT_THRESHOLD):
            if current_price >= high_24h:
                result["signals"].append(f"BREAKOUT_24H: ${current_price:.4f}")
                result["urgency"] += 2

        # --- 거래량 가속 패턴 (시간별 증가 추세) ---
        if len(h1_v) >= 6:
            recent_3h = h1_v[-3:]
            if all(recent_3h[i] > recent_3h[i-1] for i in range(1, len(recent_3h))):
                if recent_3h[-1] > vol_24h_avg * 1.5:
                    result["signals"].append("VOL_ACCELERATING")
                    result["urgency"] += 2

        # --- 15분봉 분석 (더 빠른 감지) ---
        m15 = fetch_klines(symbol, "15", 48)
        if len(m15) >= 12:
            m15_v = [float(k[5]) for k in m15]
            m15_c = [float(k[4]) for k in m15]
            m15_h = [float(k[2]) for k in m15]

            # 15분 거래량 스파이크
            v15_avg = mean(m15_v[-48:]) if len(m15_v) >= 48 else mean(m15_v)
            v15_last = m15_v[-1]
            v15_spike = v15_last / v15_avg if v15_avg > 0 else 0

            if v15_spike >= 5:
                result["signals"].append(f"VOL_SPIKE_15M: {v15_spike:.1f}x")
                result["urgency"] += 2

            # 15분 연속 양봉 + 거래량 증가
            m15_o = [float(k[1]) for k in m15]
            consec_green_15m = 0
            for j in range(-1, max(-8, -len(m15_c)), -1):
                if m15_c[j] >= m15_o[j]:
                    consec_green_15m += 1
                else:
                    break
            if consec_green_15m >= 4:
                result["signals"].append(f"GREEN_STREAK_15M: {consec_green_15m}")
                result["urgency"] += 1

            # 15분 모멘텀
            ret_15m = m15_c[-1] / m15_c[-2] - 1 if m15_c[-2] > 0 else 0
            ret_1h_15m = m15_c[-1] / m15_c[-5] - 1 if len(m15_c) >= 5 and m15_c[-5] > 0 else 0

            if ret_15m >= 0.02:  # 15분에 +2%
                result["signals"].append(f"MOM_15M: {ret_15m*100:+.1f}%")
                result["urgency"] += 1

        # --- Bid/Ask 모멘텀 (거래량 * 가격변화 방향) ---
        buy_vol = sum(h1_v[j] for j in range(-3, 0) if h1_c[j] >= h1_o[j])
        sell_vol = sum(h1_v[j] for j in range(-3, 0) if h1_c[j] < h1_o[j])
        if sell_vol > 0:
            buy_sell_ratio = buy_vol / sell_vol
            if buy_sell_ratio >= 2:
                result["signals"].append(f"BUY_PRESSURE: {buy_sell_ratio:.1f}x")
                result["urgency"] += 1

        # 종합 점수
        result["vol_spike_1h"] = vol_spike_1h
        result["vol_spike_3h"] = vol_spike_3h
        result["ret_1h"] = ret_1h
        result["ret_4h"] = ret_4h
        result["price"] = current_price
        result["vol_usd_1h"] = vol_last_1h * current_price

        # 실시간 점수 계산
        rt_score = 0
        rt_score += min(30, vol_spike_1h * 5)
        rt_score += min(20, vol_spike_3h * 4)
        rt_score += min(20, max(0, ret_1h * 200))
        rt_score += min(15, max(0, ret_4h * 100))
        rt_score += min(15, result["urgency"] * 3)
        result["score"] = rt_score

    except Exception as e:
        result["error"] = str(e)

    return result

# ============================================================
# Layer 3: 멀티타임프레임 종합
# ============================================================

def compute_combined_score(daily_score, realtime_result):
    """일봉 점수 + 실시간 점수 종합."""
    rt_score = realtime_result.get("score", 0)
    urgency = realtime_result.get("urgency", 0)

    # 가중 결합: 일봉 40% + 실시간 60% (실시간이 더 중요)
    combined = daily_score * 0.4 + rt_score * 0.6

    # 긴급도 보너스
    combined += urgency * 3

    # 시너지 보너스: 일봉 높고 + 실시간도 높으면 추가점
    if daily_score >= 65 and rt_score >= 40:
        combined += 10

    return combined

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
        chunk = []; cl = 0
        for line in lines:
            if cl + len(line) + 1 > MAX_LEN and chunk:
                chunks.append("\n".join(chunk))
                chunk = [line]; cl = len(line)
            else:
                chunk.append(line); cl += len(line) + 1
        if chunk: chunks.append("\n".join(chunk))
    for c in chunks:
        payload = json.dumps({"text": c, "username": "Surge Alert", "icon_emoji": ":mag:"}).encode()
        for wh_url in SLACK_WEBHOOK_URLS:
            req = urllib.request.Request(
                wh_url, data=payload,
                headers={"Content-Type": "application/json"}, method="POST")
            try:
                urllib.request.urlopen(req, timeout=30)
            except Exception as e:
                print(f"Slack error ({wh_url[:50]}...): {e}")
        time.sleep(0.5)

def send_urgent_alert(coin_data, daily_score, rt_result):
    """즉시 알림: 스파이크 감지."""
    cn = coin_data["symbol"].replace("USDT", "")
    signals = " | ".join(rt_result["signals"])
    combined = compute_combined_score(daily_score, rt_result)

    lines = [f":rotating_light: *{cn} 급등 시그널 감지*"]
    lines.append(f"종합점수: {combined:.0f} (일봉 {daily_score:.0f} + 실시간 {rt_result['score']:.0f})")
    lines.append(f"현재가: ${rt_result['price']:.4f}")
    lines.append(f"시그널: {signals}")

    if rt_result.get("vol_spike_1h", 0) >= 2:
        lines.append(f"1H 거래량: {rt_result['vol_spike_1h']:.1f}x (24h 평균 대비)")
    if rt_result.get("ret_1h", 0) != 0:
        lines.append(f"1H 수익률: {rt_result['ret_1h']*100:+.1f}%")
    if rt_result.get("ret_4h", 0) != 0:
        lines.append(f"4H 수익률: {rt_result['ret_4h']*100:+.1f}%")

    lines.append(f"긴급도: {'!' * min(5, rt_result['urgency'])}")

    send_slack("\n".join(lines))

# ============================================================
# State management
# ============================================================

def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "watchlist": {},
        "last_daily_scan": 0,
        "last_report": 0,
        "last_pair_refresh": 0,
        "alerted": {},  # symbol -> last alert timestamp
    }

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

# ============================================================
# Pair management
# ============================================================

def refresh_pairs():
    tickers = fetch_tickers()
    pairs = []
    for sym, t in tickers.items():
        if not sym.endswith("USDT"): continue
        if sym in STABLECOINS or sym in LARGE_CAPS: continue
        vol24 = float(t.get("turnover24h", 0))
        if MIN_VOL_USD <= vol24 <= MAX_VOL_USD:
            pairs.append(sym)
    pairs.sort()
    with open(PAIRS_FILE, "w") as f:
        json.dump(pairs, f)
    return pairs

def load_pairs(state):
    now = time.time()
    if now - state.get("last_pair_refresh", 0) > PAIR_REFRESH_INTERVAL:
        pairs = refresh_pairs()
        state["last_pair_refresh"] = now
        return pairs
    if PAIRS_FILE.exists():
        with open(PAIRS_FILE) as f:
            return json.load(f)
    return refresh_pairs()

# ============================================================
# Main loop
# ============================================================

def run_daily_scan(pairs, state, TH):
    """Layer 1: 일봉 기반 후보 선별."""
    now_str = datetime.datetime.now().strftime("%H:%M")
    print(f"  [L1 {now_str}] 일봉 스캔 ({len(pairs)}개)...")

    watchlist = {}
    for sym in pairs:
        try:
            klines = fetch_klines(sym, "D", 100)
            if len(klines) < 45: continue

            recent = klines[-30:]
            avg_vol = mean([float(k[5]) * float(k[4]) for k in recent])
            if avg_vol > MAX_VOL_USD or avg_vol < MIN_VOL_USD: continue

            feat = extract_daily_features(klines)
            if not feat: continue

            score = compute_daily_score(feat, TH)
            if score >= MIN_DAILY_SCORE:
                watchlist[sym] = {
                    "daily_score": score,
                    "price": float(klines[-1][4]),
                    "vol_usd": avg_vol,
                    "features": feat,
                }
            time.sleep(0.03)
        except:
            continue

    # Top N만 유지
    sorted_wl = sorted(watchlist.items(), key=lambda x: x[1]["daily_score"], reverse=True)
    state["watchlist"] = {k: v for k, v in sorted_wl[:TOP_WATCHLIST]}
    state["last_daily_scan"] = time.time()

    print(f"  [L1 {now_str}] 후보 {len(state['watchlist'])}개 선별")
    return state

def run_realtime_scan(state):
    """Layer 2: 실시간 거래량/모멘텀 감지."""
    now_str = datetime.datetime.now().strftime("%H:%M")
    watchlist = state.get("watchlist", {})
    if not watchlist:
        return []

    results = []
    for sym, info in watchlist.items():
        try:
            rt = analyze_realtime(sym)
            if rt["signals"]:  # 시그널이 하나라도 있으면
                combined = compute_combined_score(info["daily_score"], rt)
                results.append({
                    "symbol": sym,
                    "daily_score": info["daily_score"],
                    "rt_result": rt,
                    "combined_score": combined,
                    "price": rt.get("price", info["price"]),
                    "vol_usd": info["vol_usd"],
                })

                # 즉시 알림 (긴급도 높은 경우)
                now_ts = time.time()
                last_alert = state.get("alerted", {}).get(sym, 0)
                if rt["urgency"] >= 4 and now_ts - last_alert > 1800:  # 30분 쿨다운
                    send_urgent_alert({"symbol": sym}, info["daily_score"], rt)
                    state.setdefault("alerted", {})[sym] = now_ts
                    print(f"  [L2 {now_str}] ALERT: {sym} urgency={rt['urgency']}")

            time.sleep(0.1)
        except:
            continue

    results.sort(key=lambda x: x["combined_score"], reverse=True)
    return results

def send_report(state, rt_results):
    """정기 리포트."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    watchlist = state.get("watchlist", {})

    lines = [f":mag: *급등 전조 스캐너 v3* ({now})"]
    lines.append(f"감시 {len(watchlist)}개 | 시그널 {len(rt_results)}개")
    lines.append("")

    # 실시간 시그널 있는 코인 우선
    if rt_results:
        lines.append("*:zap: 실시간 시그널 감지*")
        lines.append("```")
        lines.append(f"{'#':>2} {'코인':<10} {'종합':>5} {'일봉':>4} {'RT':>4} {'현재가':<11} {'시그널'}")
        lines.append("-" * 65)
        for i, r in enumerate(rt_results[:10]):
            cn = r["symbol"].replace("USDT", "")
            signals = ", ".join(r["rt_result"]["signals"][:3])
            lines.append(f"{i+1:>2} {cn:<10} {r['combined_score']:>5.0f} {r['daily_score']:>4.0f} {r['rt_result']['score']:>4.0f} ${r['price']:<10.4f} {signals}")
        lines.append("```")
        lines.append("")

    # 일봉 기반 워치리스트 (시그널 없어도)
    sorted_wl = sorted(watchlist.items(), key=lambda x: x[1]["daily_score"], reverse=True)
    lines.append("*:chart_with_upwards_trend: 일봉 후보 TOP 10*")
    lines.append("```")
    lines.append(f"{'#':>2} {'코인':<10} {'점수':>5} {'현재가':<11} {'7d':>7} {'RSI':>4} {'60dH':>5}")
    lines.append("-" * 52)
    for i, (sym, info) in enumerate(sorted_wl[:10]):
        cn = sym.replace("USDT", "")
        feat = info.get("features", {})
        ret7 = feat.get("ret_7d", 0) * 100
        rsi = feat.get("rsi_14", 0)
        vs60 = feat.get("price_vs_60d_high", 0) * 100
        lines.append(f"{i+1:>2} {cn:<10} {info['daily_score']:>5.1f} ${info['price']:<10.4f} {ret7:>+6.1f}% {rsi:>4.0f} {vs60:>4.0f}%")
    lines.append("```")

    lines.append("")
    lines.append("_v3: 일봉 구조 + 시간봉 거래량 + 15분봉 모멘텀_")

    send_slack("\n".join(lines))
    state["last_report"] = time.time()

def main():
    print("=" * 60)
    print("  급등 전조 탐지기 v3 — 2단계 시스템")
    print(f"  일봉 스캔: {DAILY_SCAN_INTERVAL}s | 실시간: {REALTIME_SCAN_INTERVAL}s")
    print(f"  리포트: {REPORT_INTERVAL}s | 후보: {TOP_WATCHLIST}개")
    print("=" * 60)

    if not THRESHOLDS_FILE.exists():
        print("thresholds.json not found!")
        return

    with open(THRESHOLDS_FILE) as f:
        TH = json.load(f)

    state = load_state()

    # 초기 실행
    pairs = load_pairs(state)
    state = run_daily_scan(pairs, state, TH)
    rt_results = run_realtime_scan(state)
    send_report(state, rt_results)
    save_state(state)

    last_daily = time.time()
    last_rt = time.time()

    while True:
        try:
            now = time.time()

            # Layer 1: 일봉 스캔
            if now - last_daily >= DAILY_SCAN_INTERVAL:
                pairs = load_pairs(state)
                state = run_daily_scan(pairs, state, TH)
                last_daily = now
                save_state(state)

            # Layer 2: 실시간 스캔
            if now - last_rt >= REALTIME_SCAN_INTERVAL:
                rt_results = run_realtime_scan(state)
                last_rt = now

                # 정기 리포트
                if now - state.get("last_report", 0) >= REPORT_INTERVAL:
                    send_report(state, rt_results)
                    save_state(state)

            time.sleep(30)  # 메인 루프 30초

        except KeyboardInterrupt:
            print("\n종료")
            save_state(state)
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(60)

if __name__ == "__main__":
    main()
