#!/usr/bin/env python3
"""
Pre-Breakout Scanner — "연기 감지기"
아직 터지지 않았지만 곧 터질 것 같은 종목을 찾는다.

핵심 신호:
  1. BB Squeeze — 볼린저밴드 수축 (변동성 압축)
  2. Slow Drift — 완만한 우상향 (급등 아닌 느린 상승)
  3. Volume Build — 거래량 점진 증가 (폭증 아닌 서서히)
  4. Range Tightening — 가격 레인지 수축 (삼각수렴)
  5. Neutral FR — 펀딩비 중립 (아직 쏠림 없음)
  6. OI Accumulation — OI 조용히 증가 (스마트머니 매집)

Usage:
    python3 prebreakout_scanner.py
"""

import json, math, os, sys, time
import urllib.request
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "prebreakout_log"
STATE_FILE = BASE_DIR / "prebreakout_state.json"
BYBIT_BASE = "https://api.bybit.com"

STABLECOINS = {
    "USDCUSDT", "UUSDT", "XUSDUSDT", "PAXGUSDT", "DAIUSDT", "FDUSDUSDT",
    "TUSDUSDT", "BUSDUSDT", "EURUSDT", "USTCUSDT", "USDTUSDT", "USDDUSDT",
    "RLUSDUSDT", "USD1USDT",
}
LARGE_CAPS = {
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT",
}
SKIP_SYMBOLS = STABLECOINS | LARGE_CAPS

# Slack webhook
WEBHOOK_URL = ""
_env = os.path.expanduser("~/.openclaw/.env.slack-surge")
if os.path.exists(_env):
    with open(_env) as f:
        for line in f:
            if line.startswith("SLACK_WEBHOOK_URL="):
                WEBHOOK_URL = line.strip().split("=", 1)[1]

sys.stdout.reconfigure(line_buffering=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bybit_get(ep, params=None, retries=3):
    url = BYBIT_BASE + ep
    if params:
        url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
    req = urllib.request.Request(url, headers={"User-Agent": "prebreakout/1"})
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read())
        except Exception as e:
            if i == retries - 1:
                return None
            time.sleep(1 + i)
    return None


def send_slack(msg):
    if not WEBHOOK_URL:
        print(msg)
        return
    for chunk in [msg[i:i+3500] for i in range(0, len(msg), 3500)]:
        payload = json.dumps({"text": chunk})
        req = urllib.request.Request(
            WEBHOOK_URL, data=payload.encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=30)
        except Exception as e:
            print(f"  [WARN] Slack: {e}")
        time.sleep(0.5)


def mean(arr):
    return sum(arr) / len(arr) if arr else 0

def stdev(arr):
    if len(arr) < 2:
        return 0
    m = mean(arr)
    return math.sqrt(sum((x - m) ** 2 for x in arr) / (len(arr) - 1))

def safe_div(a, b, default=0.0):
    if b == 0 or b is None:
        return default
    r = a / b
    return default if math.isnan(r) or math.isinf(r) else r

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-10, min(10, x))))

def fetch_klines(symbol, interval, limit, category="spot"):
    data = bybit_get("/v5/market/kline", {
        "category": category, "symbol": symbol,
        "interval": interval, "limit": str(limit),
    })
    if not data:
        return []
    klines = data.get("result", {}).get("list", [])
    return list(reversed(klines))


# ---------------------------------------------------------------------------
# Signal 1: BB Squeeze (볼린저밴드 수축)
# ---------------------------------------------------------------------------

def compute_bb_squeeze(klines_daily):
    """
    BB width가 최근 N일 중 최저 수준이면 squeeze.
    Returns: (squeeze_score 0~1, bb_width, bb_width_pct)
    """
    if len(klines_daily) < 30:
        return 0, 0, 0

    closes = [float(k[4]) for k in klines_daily]

    # Current BB width (20-period)
    c20 = closes[-20:]
    bb_mean = mean(c20)
    bb_std = stdev(c20)
    bb_width = (2 * bb_std) / bb_mean if bb_mean > 0 else 999

    # Historical BB widths (rolling 20-period over last 30 days)
    widths = []
    for i in range(20, len(closes)):
        window = closes[i-20:i]
        m = mean(window)
        s = stdev(window)
        w = (2 * s) / m if m > 0 else 0
        widths.append(w)

    if not widths:
        return 0, bb_width, 0

    # Percentile: 현재 width가 과거 대비 얼마나 좁은가
    rank = sum(1 for w in widths if w <= bb_width) / len(widths)
    squeeze_pct = 1 - rank  # 좁을수록 1에 가까움

    # Score: 상위 20% 이내 좁기면 강한 squeeze
    score = sigmoid((squeeze_pct - 0.6) * 8)
    return score, round(bb_width, 5), round(squeeze_pct, 3)


# ---------------------------------------------------------------------------
# Signal 2: Slow Drift (완만한 우상향)
# ---------------------------------------------------------------------------

def compute_slow_drift(klines_daily):
    """
    최근 7~14일간 꾸준히 올라가는 느린 상승 감지.
    급등은 페널티. 하루 0.5~2% 수준의 완만한 상승이 이상적.
    Returns: (drift_score 0~1, daily_avg_return, consistency)
    """
    if len(klines_daily) < 10:
        return 0, 0, 0

    closes = [float(k[4]) for k in klines_daily[-10:]]
    daily_returns = [(closes[i] / closes[i-1] - 1) for i in range(1, len(closes))]

    avg_ret = mean(daily_returns)
    pos_count = sum(1 for r in daily_returns if r > 0)
    consistency = pos_count / len(daily_returns)  # 양봉 비율

    # 이상적: 하루 평균 +0.3~2%, 양봉 60%+
    # 급등(>5%/day)이면 이미 터진 거라 페널티
    if avg_ret > 0.05:
        score = 0.2  # 이미 급등
    elif avg_ret > 0.003 and consistency >= 0.5:
        # Sweet spot: 천천히 꾸준히
        drift_quality = sigmoid((avg_ret - 0.003) * 100) * sigmoid((consistency - 0.5) * 8)
        score = drift_quality
    else:
        score = 0.1

    return score, round(avg_ret * 100, 3), round(consistency, 2)


# ---------------------------------------------------------------------------
# Signal 3: Volume Build (거래량 점진 증가)
# ---------------------------------------------------------------------------

def compute_volume_build(klines_1h):
    """
    거래량이 '폭증'이 아니라 '점진적으로' 증가하는 패턴.
    최근 24h 평균 vs 이전 7일 평균 비교.
    급격한 스파이크는 오히려 감점.
    """
    if len(klines_1h) < 72:
        return 0, 0, 0

    vols = [float(k[5]) * float(k[4]) for k in klines_1h]  # USD volume

    recent_24h = vols[-24:]
    prior = vols[:-24]

    avg_recent = mean(recent_24h)
    avg_prior = mean(prior) if prior else 1

    # Volume ratio (recent / prior)
    vol_ratio = safe_div(avg_recent, avg_prior, 1.0)

    # Spike detection: 최근 24h에서 max가 평균의 5배 이상이면 spike
    max_recent = max(recent_24h) if recent_24h else 0
    has_spike = max_recent > avg_recent * 5

    # 이상적: 1.2x~2.5x 점진 증가. 3x 이상은 이미 터진 거.
    if has_spike or vol_ratio > 3.0:
        score = 0.15  # 이미 폭증 — pre-breakout이 아님
    elif 1.2 <= vol_ratio <= 3.0:
        score = sigmoid((vol_ratio - 1.2) * 3)
    else:
        score = 0.1

    # Volume trend (선형 회귀 기울기)
    # 24h 거래량을 4h 버킷으로 나눠서 추세 확인
    buckets = []
    for i in range(0, len(recent_24h), 4):
        chunk = recent_24h[i:i+4]
        if chunk:
            buckets.append(mean(chunk))

    vol_trend = 0
    if len(buckets) >= 3:
        # Simple: last bucket / first bucket
        vol_trend = safe_div(buckets[-1], buckets[0], 1.0) - 1

    return score, round(vol_ratio, 2), round(vol_trend, 3)


# ---------------------------------------------------------------------------
# Signal 4: Range Tightening (레인지 수축)
# ---------------------------------------------------------------------------

def compute_range_tightening(klines_daily):
    """
    최근 7일 high-low 레인지가 이전 대비 수축하는 패턴.
    삼각수렴/쐐기 패턴 감지.
    """
    if len(klines_daily) < 21:
        return 0, 0

    # 최근 7일 daily range (%)
    recent_ranges = []
    for k in klines_daily[-7:]:
        h, l = float(k[2]), float(k[3])
        c = float(k[4])
        if c > 0:
            recent_ranges.append((h - l) / c)

    # 이전 14일 daily range (%)
    prior_ranges = []
    for k in klines_daily[-21:-7]:
        h, l = float(k[2]), float(k[3])
        c = float(k[4])
        if c > 0:
            prior_ranges.append((h - l) / c)

    if not recent_ranges or not prior_ranges:
        return 0, 0

    avg_recent = mean(recent_ranges)
    avg_prior = mean(prior_ranges)

    # Compression ratio: 낮을수록 수축
    compression = safe_div(avg_recent, avg_prior, 1.0)

    # 이상적: 0.4~0.8 (최근이 이전 대비 40~80% 수준)
    if compression < 0.8:
        score = sigmoid((0.8 - compression) * 8)
    else:
        score = 0.1

    return score, round(compression, 3)


# ---------------------------------------------------------------------------
# Signal 5: Neutral Funding Rate (펀딩비 중립)
# ---------------------------------------------------------------------------

def compute_neutral_fr(symbol):
    """
    펀딩비가 중립(-0.01% ~ +0.01%) 근처면 아직 포지션 쏠림 없음.
    극단적 펀딩비는 감점.
    """
    data = bybit_get("/v5/market/tickers", {"category": "linear", "symbol": symbol})
    if not data:
        return 0.5, 0

    items = data.get("result", {}).get("list", [])
    if not items:
        return 0.5, 0

    fr = float(items[0].get("fundingRate", 0))

    # 중립 근처(|FR| < 0.01%)면 1.0, 멀어질수록 감소
    neutrality = math.exp(-abs(fr) * 10000)  # |FR|=0.01% → ~0.37, |FR|=0 → 1.0
    score = neutrality

    return score, round(fr * 100, 4)


# ---------------------------------------------------------------------------
# Signal 6: OI Accumulation (조용한 OI 증가)
# ---------------------------------------------------------------------------

def compute_oi_accumulation(symbol):
    """
    OI가 조용히 증가하면 스마트머니 매집 가능성.
    가격 변동 없이 OI만 오르면 강한 신호.
    """
    data = bybit_get("/v5/market/open-interest", {
        "category": "linear", "symbol": symbol,
        "intervalTime": "1h", "limit": "25",
    })
    if not data:
        return 0, 0

    oi_list = data.get("result", {}).get("list", [])
    if len(oi_list) < 10:
        return 0, 0

    oi_list = list(reversed(oi_list))
    ois = [float(o.get("openInterest", 0)) for o in oi_list]

    recent = mean(ois[-6:])
    prior = mean(ois[:-6]) if len(ois) > 6 else 1

    oi_change = safe_div(recent - prior, prior, 0)

    # 이상적: OI +3~15% 증가 (조용한 매집)
    if 0.03 <= oi_change <= 0.15:
        score = sigmoid((oi_change - 0.03) * 30)
    elif oi_change > 0.15:
        score = 0.5  # 너무 많이 올랐으면 이미 눈에 띄는 상태
    else:
        score = 0.1

    return score, round(oi_change * 100, 2)


# ---------------------------------------------------------------------------
# Composite Score
# ---------------------------------------------------------------------------

def compute_prebreakout_score(bb, drift, vol_build, range_tight, fr_neutral, oi_accum):
    """
    종합 Pre-Breakout 점수.
    가중치: BB Squeeze가 가장 중요 (변동성 압축 = 폭발 임박)
    """
    score = (
        0.25 * bb +            # 볼린저밴드 수축 (핵심)
        0.20 * drift +         # 완만한 우상향
        0.15 * vol_build +     # 거래량 점진 증가
        0.15 * range_tight +   # 레인지 수축
        0.10 * fr_neutral +    # 펀딩비 중립
        0.15 * oi_accum        # OI 매집
    )
    return score


# ---------------------------------------------------------------------------
# Disqualification filters (이미 터진 것 제외)
# ---------------------------------------------------------------------------

def is_already_exploded(klines_1h, klines_daily):
    """이미 급등/급락한 종목 제외."""
    if len(klines_1h) < 6:
        return True

    # 최근 6h에 5% 이상 급등이면 이미 터진 거
    c_now = float(klines_1h[-1][4])
    c_6h = float(klines_1h[-6][4]) if len(klines_1h) >= 6 else c_now
    ret_6h = safe_div(c_now - c_6h, c_6h, 0)
    if abs(ret_6h) > 0.05:
        return True

    # 24h에 15% 이상이면 제외
    if len(klines_1h) >= 24:
        c_24h = float(klines_1h[-24][4])
        ret_24h = safe_div(c_now - c_24h, c_24h, 0)
        if abs(ret_24h) > 0.15:
            return True

    return False


# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------

def run_scan():
    t0 = time.time()
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%dT%H:%M:%S")

    LOG_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("  PRE-BREAKOUT SCANNER (연기 감지기)")
    print(f"  {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load state (연속 출현 추적)
    state = {}
    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Step 1: Get all linear pairs
    print("[1/5] Fetching linear tickers...")
    linear_data = bybit_get("/v5/market/tickers", {"category": "linear"})
    if not linear_data:
        print("  [ERROR] Failed to fetch tickers")
        return

    tickers = linear_data.get("result", {}).get("list", [])
    filtered = []
    ticker_map = {}
    for t in tickers:
        sym = t["symbol"]
        if not sym.endswith("USDT"):
            continue
        if sym in SKIP_SYMBOLS:
            continue
        turnover = float(t.get("turnover24h", 0))
        if turnover < 500_000:  # 최소 50만 USDT
            continue
        filtered.append(sym)
        ticker_map[sym] = t

    print(f"  {len(filtered)} symbols after filter")

    # Step 2: Fetch klines
    print(f"[2/5] Fetching klines...")
    coin_data = {}
    for i, sym in enumerate(filtered):
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(filtered)}")
        try:
            klines_1h = fetch_klines(sym, "60", 168, "linear")
            if len(klines_1h) < 48:
                continue
            time.sleep(0.02)

            klines_daily = fetch_klines(sym, "D", 60, "linear")
            if len(klines_daily) < 21:
                continue
            time.sleep(0.02)

            # 이미 터진 거 제외
            if is_already_exploded(klines_1h, klines_daily):
                continue

            coin_data[sym] = {
                "klines_1h": klines_1h,
                "klines_daily": klines_daily,
                "ticker": ticker_map[sym],
            }
        except Exception:
            continue

    print(f"  {len(coin_data)} coins with data (exploded excluded)")

    # Step 3: Compute signals
    print("[3/5] Computing pre-breakout signals...")
    results = []

    for sym, cd in coin_data.items():
        klines_1h = cd["klines_1h"]
        klines_daily = cd["klines_daily"]
        ticker = cd["ticker"]
        price = float(klines_1h[-1][4])

        # Signal 1: BB Squeeze
        bb_score, bb_width, bb_squeeze_pct = compute_bb_squeeze(klines_daily)

        # Signal 2: Slow Drift
        drift_score, daily_avg_ret, consistency = compute_slow_drift(klines_daily)

        # Signal 3: Volume Build
        vol_score, vol_ratio, vol_trend = compute_volume_build(klines_1h)

        # Signal 4: Range Tightening
        range_score, compression = compute_range_tightening(klines_daily)

        # Signal 5 & 6: Only call API for top candidates (save time)
        quick_score = 0.25 * bb_score + 0.20 * drift_score + 0.15 * vol_score + 0.15 * range_score
        if quick_score < 0.20:
            continue

        fr_score, fr_pct = compute_neutral_fr(sym)
        oi_score, oi_change_pct = compute_oi_accumulation(sym)
        time.sleep(0.03)

        # Composite
        total = compute_prebreakout_score(
            bb_score, drift_score, vol_score, range_score, fr_score, oi_score
        )

        # 연속 출현 보너스
        coin = sym.replace("USDT", "")
        prev = state.get(coin, {})
        appear_count = prev.get("count", 0) + 1
        # 3회 이상 연속이면 +10% 보너스
        if appear_count >= 3:
            total *= 1.10

        # 7d return (context)
        ret_7d = 0
        if len(klines_daily) >= 8:
            c7 = float(klines_daily[-8][4])
            ret_7d = safe_div(price - c7, c7, 0)

        turnover = float(ticker.get("turnover24h", 0))

        results.append({
            "symbol": sym,
            "coin": coin,
            "price": price,
            "score": round(total, 4),
            "bb_squeeze": round(bb_score, 3),
            "bb_width": bb_width,
            "bb_pct": bb_squeeze_pct,
            "drift": round(drift_score, 3),
            "daily_ret": daily_avg_ret,
            "consistency": consistency,
            "vol_build": round(vol_score, 3),
            "vol_ratio": vol_ratio,
            "vol_trend": vol_trend,
            "range": round(range_score, 3),
            "compression": compression,
            "fr": round(fr_score, 3),
            "fr_pct": fr_pct,
            "oi": round(oi_score, 3),
            "oi_chg": oi_change_pct,
            "ret_7d": round(ret_7d * 100, 1),
            "turnover": turnover,
            "appear": appear_count,
        })

    # Step 4: Rank & filter
    print("[4/5] Ranking...")
    results.sort(key=lambda x: x["score"], reverse=True)
    top = [r for r in results if r["score"] >= 0.30][:15]

    # Update state
    new_state = {}
    for r in results:
        if r["score"] >= 0.25:
            new_state[r["coin"]] = {
                "count": r["appear"],
                "last_seen": now_str,
                "last_score": r["score"],
            }
    # Decay: 6시간 이상 안 보이면 제거
    cutoff = (now - timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%S")
    for coin, v in state.items():
        if coin not in new_state and v.get("last_seen", "") > cutoff:
            new_state[coin] = v  # 유지
    with open(STATE_FILE, "w") as f:
        json.dump(new_state, f, indent=2)

    elapsed = time.time() - t0

    # Step 5: Alert
    print(f"[5/5] Results: {len(top)} candidates | {elapsed:.0f}s")

    # Log
    log_file = LOG_DIR / f"{now.strftime('%Y-%m-%d')}.jsonl"
    for r in top:
        r_log = {**r, "timestamp": now_str}
        with open(log_file, "a") as f:
            f.write(json.dumps(r_log, ensure_ascii=False) + "\n")

    if top:
        lines = [f":eyes: *PRE-BREAKOUT* ({now.strftime('%m/%d %H:%M')}) — 연기 감지"]
        lines.append("`BB=밴드수축 Drift=우상향 Vol=거래량증 Range=레인지축 FR=펀비중립 OI=매집`")
        lines.append("```")
        lines.append(f" # {'코인':<8} {'점수':>5} {'BB':>4} {'Drft':>4} {'Vol':>4} {'Rng':>4} {'FR':>4} {'OI':>4} {'7d':>6} {'특징'}")

        for i, r in enumerate(top):
            # Notes
            notes = []
            if r["bb_pct"] >= 0.8:
                notes.append("극압축")
            elif r["bb_pct"] >= 0.6:
                notes.append("압축")
            if r["consistency"] >= 0.7:
                notes.append("꾸준")
            if r["vol_ratio"] >= 1.5:
                notes.append(f"vol{r['vol_ratio']}x")
            if r["compression"] < 0.6:
                notes.append("수렴")
            if r["oi_chg"] >= 5:
                notes.append(f"OI+{r['oi_chg']:.0f}%")
            if r["appear"] >= 3:
                notes.append(f"{r['appear']}연속")
            note_str = " ".join(notes) if notes else ""

            lines.append(
                f"{i+1:>2} {r['coin']:<8} {r['score']:.2f} "
                f"{r['bb_squeeze']:.2f} {r['drift']:.2f} {r['vol_build']:.2f} "
                f"{r['range']:.2f} {r['fr']:.2f} {r['oi']:.2f} "
                f"{r['ret_7d']:>+5.1f}% {note_str}"
            )

        lines.append("```")
        lines.append(f"_스캔: {len(filtered)}개 → {len(coin_data)}개 → {len(top)}개 | {elapsed:.0f}초_")
        msg = "\n".join(lines)
        send_slack(msg)
        print(msg)
    else:
        msg = f":mag: *Pre-Breakout* ({now.strftime('%m/%d %H:%M')}) — 후보 없음 | {elapsed:.0f}초"
        print(msg)

    return top


if __name__ == "__main__":
    run_scan()
