#!/usr/bin/env python3
"""
Ignition Scanner — Production ignition score alert system
Combines short-term signals (V, D, P, M) with R8 model probability
for early detection of crypto price surges on Bybit.

Usage:
    python3 ignition_scanner.py                 # Normal scan (30min interval)
    python3 ignition_scanner.py --update-watchlist  # Also update R8 watchlist
    python3 ignition_scanner.py --track-only    # Only run result tracking
"""

import json, math, os, sys, time, pickle, argparse
import urllib.request
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "ml_data"
LOG_DIR = BASE_DIR / "ignition_log"
ALERT_HISTORY_FILE = BASE_DIR / "alert_history.json"
SCORE_HISTORY_FILE = BASE_DIR / "score_history.json"  # EMA smoothing용
R8_WATCHLIST_FILE = BASE_DIR / "r8_watchlist.json"
RESULTS_FILE = BASE_DIR / "ignition_results.jsonl"

BYBIT_BASE = "https://api.bybit.com"

STABLECOINS = {
    "USDCUSDT", "UUSDT", "XUSDUSDT", "PAXGUSDT", "DAIUSDT", "FDUSDUSDT",
    "TUSDUSDT", "BUSDUSDT", "EURUSDT", "USTCUSDT", "USDTUSDT", "USDDUSDT",
    "RLUSDUSDT", "USD1USDT",
}
LARGE_CAPS = {
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT",
    "DOGEUSDT", "TRXUSDT", "DOTUSDT", "MATICUSDT", "LTCUSDT", "BCHUSDT",
    "LINKUSDT", "AVAXUSDT", "UNIUSDT", "AAVEUSDT", "ATOMUSDT",
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
# Imports from existing modules
# ---------------------------------------------------------------------------
sys.path.insert(0, str(BASE_DIR))
from round4_metalabel import (
    extract_features, has_recent_surge, mean, stdev, compute_atr, compute_rsi,
)
from r8_tuned import transform_vol_features

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sigmoid(x):
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


def bybit_get(ep, params=None, retries=3):
    """GET request to Bybit API with retries."""
    url = BYBIT_BASE + ep
    if params:
        url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
    req = urllib.request.Request(url, headers={"User-Agent": "ignition/1"})
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read())
        except Exception as e:
            if i == retries - 1:
                print(f"  [WARN] API failed: {ep} — {e}")
                return None
            time.sleep(1 + i)
    return None


def send_slack(msg):
    """Send message to Slack webhook (chunked for length limit)."""
    return  # Retired: replaced by momentum-alert; historical analysis remains available.
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
            print(f"  [WARN] Slack send failed: {e}")
        time.sleep(0.5)


def safe_div(a, b, default=0.0):
    """Safe division avoiding ZeroDivisionError and NaN."""
    if b == 0 or b is None or math.isnan(b):
        return default
    result = a / b
    return default if math.isnan(result) or math.isinf(result) else result


def percentile_rank(values):
    """Convert a list of values to percentile ranks (0~1)."""
    n = len(values)
    if n == 0:
        return []
    arr = np.array(values, dtype=np.float64)
    # Use rankdata-style: average rank / n
    order = arr.argsort().argsort()  # rank 0-indexed
    return ((order + 1) / n).tolist()


def load_json(path, default=None):
    """Load JSON file safely."""
    if default is None:
        default = {}
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(path, data):
    """Save JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Alert cooldown
# ---------------------------------------------------------------------------

def load_alert_history():
    return load_json(ALERT_HISTORY_FILE, {})


def save_alert_history(history):
    save_json(ALERT_HISTORY_FILE, history)


def is_in_cooldown(coin, history, cooldown_hours=24):
    """Check if coin was alerted within cooldown period."""
    if coin not in history:
        return False
    try:
        last_alert = datetime.fromisoformat(history[coin])
        return datetime.now() - last_alert < timedelta(hours=cooldown_hours)
    except (ValueError, TypeError):
        return False


def cleanup_alert_history(history, max_age_days=7):
    """Remove entries older than max_age_days."""
    cutoff = datetime.now() - timedelta(days=max_age_days)
    cleaned = {}
    for coin, ts_str in history.items():
        try:
            if datetime.fromisoformat(ts_str) > cutoff:
                cleaned[coin] = ts_str
        except (ValueError, TypeError):
            pass
    return cleaned


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_all_spot_tickers():
    """Fetch all Bybit spot tickers."""
    data = bybit_get("/v5/market/tickers", {"category": "spot"})
    if not data:
        return []
    return data.get("result", {}).get("list", [])


def fetch_klines(symbol, interval, limit, category="spot"):
    """Fetch kline data from Bybit. Returns list in chronological order."""
    data = bybit_get("/v5/market/kline", {
        "category": category, "symbol": symbol,
        "interval": interval, "limit": str(limit),
    })
    if not data:
        return []
    klines = data.get("result", {}).get("list", [])
    return list(reversed(klines))  # Bybit returns newest first


def fetch_linear_ticker(symbol):
    """Fetch linear (perpetual) ticker for funding rate and OI."""
    data = bybit_get("/v5/market/tickers", {"category": "linear", "symbol": symbol})
    if not data:
        return None
    items = data.get("result", {}).get("list", [])
    return items[0] if items else None


def fetch_funding_history(symbol, limit=56):
    """Fetch funding rate history (8h intervals, 56 = ~7 days)."""
    data = bybit_get("/v5/market/funding/history", {
        "category": "linear", "symbol": symbol, "limit": str(limit),
    })
    if not data:
        return []
    return list(reversed(data.get("result", {}).get("list", [])))


def fetch_open_interest_history(symbol, interval="1h", limit=25):
    """Fetch OI history."""
    data = bybit_get("/v5/market/open-interest", {
        "category": "linear", "symbol": symbol,
        "intervalTime": interval, "limit": str(limit),
    })
    if not data:
        return []
    return list(reversed(data.get("result", {}).get("list", [])))


def parse_klines(klines):
    """Parse kline data into OHLCV arrays."""
    o = [float(k[1]) for k in klines]
    h = [float(k[2]) for k in klines]
    l = [float(k[3]) for k in klines]
    c = [float(k[4]) for k in klines]
    v = [float(k[5]) for k in klines]
    return o, h, l, c, v


# ---------------------------------------------------------------------------
# Signal computations: V, D, P, M
# ---------------------------------------------------------------------------

def fetch_taker_buy_ratio(sym):
    """Fetch recent trades and compute taker buy ratio."""
    try:
        data = bybit_get("/v5/market/recent-trade", {"category": "spot", "symbol": sym, "limit": "500"})
        trades = data.get("result", {}).get("list", [])
        if not trades:
            return 0.5, 0, 0
        total_vol = 0
        buy_vol = 0
        large_count = 0  # trades > $10K
        for t in trades:
            size = float(t.get("size", 0))
            price = float(t.get("price", 0))
            usd = size * price
            total_vol += usd
            if t.get("side", "").lower() == "buy":
                buy_vol += usd
            if usd > 10000:
                large_count += 1
        ratio = buy_vol / total_vol if total_vol > 0 else 0.5
        return ratio, large_count, len(trades)
    except:
        return 0.5, 0, 0


def compute_V(klines_1h, klines_15m, taker_data=None):
    """
    Volume explosion signal.
    klines_1h: 168 bars (7 days of 1h)
    klines_15m: 96 bars (24h of 15m)
    taker_data: (taker_buy_ratio, large_trade_count, total_trades) or None
    """
    raw = {}

    # -- 1h volume z-score --
    if len(klines_1h) < 24:
        return 0.5, raw

    vols_1h = [float(k[5]) * float(k[4]) for k in klines_1h]  # volume * close = USD volume
    current_1h_vol = vols_1h[-1] if vols_1h else 0
    mean_168h = mean(vols_1h) if vols_1h else 1
    std_168h = stdev(vols_1h) if len(vols_1h) > 1 else 1
    v_zscore_1h = safe_div(current_1h_vol - mean_168h, std_168h)

    # -- 15m volume z-score --
    v_zscore_15m = 0.0
    if len(klines_15m) >= 10:
        vols_15m = [float(k[5]) * float(k[4]) for k in klines_15m]
        current_15m_vol = vols_15m[-1] if vols_15m else 0
        mean_96_15m = mean(vols_15m) if vols_15m else 1
        std_96_15m = stdev(vols_15m) if len(vols_15m) > 1 else 1
        v_zscore_15m = safe_div(current_15m_vol - mean_96_15m, std_96_15m)

    # -- Volume acceleration --
    last_24_zscores = []
    for i in range(max(0, len(vols_1h) - 24), len(vols_1h)):
        z = safe_div(vols_1h[i] - mean_168h, std_168h)
        last_24_zscores.append(z)
    v_acceleration = v_zscore_1h - (mean(last_24_zscores) if last_24_zscores else 0)

    # -- Taker buy ratio (trade flow) --
    taker_buy_ratio = 0.5
    large_trade_count = 0
    if taker_data:
        taker_buy_ratio, large_trade_count, _ = taker_data

    # Taker buy signal: ratio > 0.6 = active buying
    taker_signal = sigmoid((taker_buy_ratio - 0.55) * 15)
    # Large trade signal: more large trades = whale activity
    large_trade_signal = sigmoid((large_trade_count - 3) / 3)

    raw["v_zscore_1h"] = round(v_zscore_1h, 3)
    raw["v_zscore_15m"] = round(v_zscore_15m, 3)
    raw["v_acceleration"] = round(v_acceleration, 3)
    raw["taker_buy_ratio"] = round(taker_buy_ratio, 3)
    raw["large_trade_count"] = large_trade_count
    raw["taker_signal"] = round(float(taker_signal), 3)

    V = float(max(
        sigmoid((v_zscore_1h - 2) / 2),
        sigmoid((v_zscore_15m - 3) / 2),
        sigmoid(v_acceleration / 2),
    )) * 0.7 + 0.15 * float(taker_signal) + 0.15 * float(large_trade_signal)
    return V, raw


def compute_D(linear_ticker, funding_history, oi_history, price_change_24h):
    """
    Derivatives signal.
    """
    raw = {}
    fr_zscore = 0.0
    oi_change_1h = 0.0
    oi_price_div = 0.0
    funding_flip = 0
    funding_rate_now = 0.0

    # Funding rate z-score
    if funding_history and len(funding_history) >= 3:
        rates = [float(f.get("fundingRate", 0)) for f in funding_history]
        funding_rate_now = rates[-1]
        mean_7d = mean(rates)
        std_7d = stdev(rates) if len(rates) > 1 else 0.0001
        fr_zscore = safe_div(funding_rate_now - mean_7d, std_7d) if std_7d > 0 else 0

        # Funding flip: was negative 24h ago, now positive
        # 24h ago = ~3 funding periods back (8h intervals)
        if len(rates) >= 4:
            funding_24h_ago = rates[-4] if len(rates) >= 4 else rates[0]
            if funding_24h_ago < -0.0001 and funding_rate_now > 0:
                funding_flip = 1

    # OI change 1h
    if oi_history and len(oi_history) >= 2:
        try:
            current_oi = float(oi_history[-1].get("openInterest", 0))
            oi_1h_ago = float(oi_history[-2].get("openInterest", 0))
            oi_change_1h = safe_div(current_oi - oi_1h_ago, oi_1h_ago)
        except (ValueError, KeyError, IndexError):
            pass

    # OI-price divergence (24h)
    oi_change_24h = 0.0
    if oi_history and len(oi_history) >= 24:
        try:
            current_oi = float(oi_history[-1].get("openInterest", 0))
            oi_24h_ago = float(oi_history[-24].get("openInterest", 0))
            oi_change_24h = safe_div(current_oi - oi_24h_ago, oi_24h_ago)
        except (ValueError, KeyError, IndexError):
            pass
    elif oi_history and len(oi_history) >= 2:
        try:
            current_oi = float(oi_history[-1].get("openInterest", 0))
            oi_first = float(oi_history[0].get("openInterest", 0))
            oi_change_24h = safe_div(current_oi - oi_first, oi_first)
        except (ValueError, KeyError, IndexError):
            pass

    oi_price_div = oi_change_24h - price_change_24h

    raw["fr_zscore"] = round(fr_zscore, 3)
    raw["oi_change_1h"] = round(oi_change_1h, 4)
    raw["oi_price_div"] = round(oi_price_div, 4)
    raw["funding_flip"] = funding_flip
    raw["funding_rate"] = round(funding_rate_now, 6)

    D = (
        0.10 * float(sigmoid(abs(fr_zscore) / 2))      # FR: 노이즈 많아서 낮게
        + 0.40 * float(sigmoid((oi_change_1h - 0.05) / 0.05))  # OI 변화: 실질 자금흐름
        + 0.35 * float(sigmoid(oi_price_div / 0.1))     # OI-가격 다이버전스: 조용한 매집
        + 0.15 * funding_flip                            # Funding flip: 전환 신호
    )
    return D, raw


def compute_P(price, klines_1h, klines_daily):
    """
    Price pattern signal — enhanced with volatility signature.
    """
    raw = {}

    # Price position 7d
    high_7d = price
    low_7d = price
    if len(klines_daily) >= 7:
        last7 = klines_daily[-7:]
        high_7d = max(float(k[2]) for k in last7)
        low_7d = min(float(k[3]) for k in last7)
    price_position_7d = safe_div(price - low_7d, high_7d - low_7d, 0.5)

    # Volatility compression (ATR-based)
    current_vol = 0.0
    avg_vol_30d = 1.0
    if len(klines_daily) >= 7:
        recent_ranges = [float(k[2]) - float(k[3]) for k in klines_daily[-7:]]
        current_vol = mean(recent_ranges) if recent_ranges else 0
    if len(klines_daily) >= 30:
        all_ranges = [float(k[2]) - float(k[3]) for k in klines_daily[-30:]]
        avg_vol_30d = mean(all_ranges) if all_ranges else 1
    volatility_compression = 1 - safe_div(current_vol, avg_vol_30d, 1.0)

    # Bollinger Band Squeeze
    bb_squeeze = 0.0
    if len(klines_daily) >= 20:
        closes_20 = [float(k[4]) for k in klines_daily[-20:]]
        bb_mean = mean(closes_20)
        bb_std = stdev(closes_20) if len(closes_20) > 1 else 1
        bb_width = (2 * bb_std) / bb_mean if bb_mean > 0 else 0
        # BB width의 30일 백분위 (좁을수록 squeeze)
        if len(klines_daily) >= 30:
            bb_widths = []
            for j in range(10, min(31, len(klines_daily))):
                c_slice = [float(k[4]) for k in klines_daily[-j-10:-j+10] if -j+10 <= 0] or closes_20
                bm = mean(c_slice)
                bs = stdev(c_slice) if len(c_slice) > 1 else 1
                bw = (2 * bs) / bm if bm > 0 else 0
                bb_widths.append(bw)
            if bb_widths:
                bb_squeeze = 1 - (sum(1 for w in bb_widths if w <= bb_width) / len(bb_widths))

    # ATR compression duration: 연속으로 compression > 0.3인 일수
    compression_duration = 0
    if len(klines_daily) >= 30:
        for j in range(1, min(15, len(klines_daily) - 7)):
            idx = len(klines_daily) - j
            if idx < 7:
                break
            rec = [float(k[2]) - float(k[3]) for k in klines_daily[idx-6:idx+1]]
            cv = mean(rec) if rec else 0
            comp = 1 - safe_div(cv, avg_vol_30d, 1.0)
            if comp > 0.3:
                compression_duration += 1
            else:
                break
    compression_duration_signal = sigmoid((compression_duration - 3) / 3)

    # Parkinson volatility (HL-based, more accurate than close-to-close)
    parkinson_vol = 0.0
    if len(klines_daily) >= 14:
        log_hl = [np.log(float(k[2]) / float(k[3])) ** 2 if float(k[3]) > 0 else 0
                   for k in klines_daily[-14:]]
        parkinson_vol = np.sqrt(mean(log_hl) / (4 * np.log(2)))

    # Breakout
    high_24h = price
    if len(klines_1h) >= 24:
        high_24h = max(float(k[2]) for k in klines_1h[-25:-1])
    breakout = 1 if price > high_24h else 0

    # === 과매도 반등 신호 (급락 후 V자 반등 패턴) ===
    oversold_bounce = 0.0
    consec_drop = 0
    drop_7d = 0.0
    rsi_low = False
    vol_recovering = False

    if len(klines_daily) >= 8:
        closes = [float(k[4]) for k in klines_daily]
        # 7일간 낙폭
        drop_7d = (closes[-1] / closes[-8] - 1) if closes[-8] > 0 else 0
        # 연속 하락일 수
        for j in range(1, min(8, len(closes))):
            if closes[-j] < closes[-j-1] if j+1 <= len(closes) else False:
                consec_drop += 1
            else:
                break
        # RSI 저점 체크 (간이 계산)
        if len(closes) >= 15:
            gains, losses = [], []
            for j in range(1, min(15, len(closes))):
                d = closes[-j] - closes[-j-1] if j+1 <= len(closes) else 0
                gains.append(max(d, 0))
                losses.append(max(-d, 0))
            ag = mean(gains) if gains else 0
            al = mean(losses) if losses else 0.001
            rsi = 100 - (100 / (1 + ag / al)) if al > 0 else 50
            rsi_low = rsi < 35
        # 거래량 회복 (최근 2일 vs 이전 5일)
        if len(klines_daily) >= 7:
            vols = [float(k[5]) * float(k[4]) for k in klines_daily]
            recent_vol = mean(vols[-2:]) if len(vols) >= 2 else 0
            prior_vol = mean(vols[-7:-2]) if len(vols) >= 7 else 1
            vol_recovering = recent_vol > prior_vol * 0.8

    # 조건: 7d -15% 이상 하락 + 연속 2일+ 하락 + RSI 바닥 + 거래량 회복 조짐
    if drop_7d < -0.15 and consec_drop >= 2 and rsi_low:
        oversold_bounce = 0.6
        if vol_recovering:
            oversold_bounce = 0.9  # 거래량까지 회복이면 강한 신호

    raw["price_position_7d"] = round(price_position_7d, 3)
    raw["volatility_compression"] = round(volatility_compression, 3)
    raw["bb_squeeze"] = round(bb_squeeze, 3)
    raw["compression_duration"] = compression_duration
    raw["parkinson_vol"] = round(parkinson_vol, 4)
    raw["breakout"] = breakout
    raw["oversold_bounce"] = round(oversold_bounce, 3)
    raw["drop_7d"] = round(drop_7d, 3)
    raw["consec_drop"] = consec_drop

    P = (
        0.20 * breakout
        + 0.15 * float(sigmoid((volatility_compression - 0.3) * 5))
        + 0.15 * (1 - price_position_7d)
        + 0.10 * float(sigmoid(bb_squeeze * 3))       # BB squeeze 신호
        + 0.10 * float(compression_duration_signal)     # 압축 지속 기간
        + 0.20 * oversold_bounce                        # 과매도 반등 (급락 후 V자)
        + 0.10 * float(sigmoid((0.02 - parkinson_vol) * 50))  # 낮은 Parkinson vol = 압축
    )
    return P, raw


def compute_T(klines_1h, klines_daily):
    """
    Multi-Timeframe alignment signal.
    1h, 4h, 1d 추세가 같은 방향이면 강한 신호.
    """
    raw = {}

    def trend_score(closes, short=3, long=10):
        """MA cross 기반 추세: +1=상승, -1=하락, 0=횡보"""
        if len(closes) < long:
            return 0
        ma_short = mean(closes[-short:])
        ma_long = mean(closes[-long:])
        if ma_long == 0:
            return 0
        ratio = ma_short / ma_long - 1
        if ratio > 0.005:
            return 1
        elif ratio < -0.005:
            return -1
        return 0

    # 1h trend (from 1h klines)
    trend_1h = 0
    if len(klines_1h) >= 12:
        closes_1h = [float(k[4]) for k in klines_1h]
        trend_1h = trend_score(closes_1h, 3, 12)

    # 4h trend (aggregate 1h into 4h)
    trend_4h = 0
    if len(klines_1h) >= 40:
        # Take every 4th close as 4h proxy
        closes_4h = [float(klines_1h[i][4]) for i in range(3, len(klines_1h), 4)]
        trend_4h = trend_score(closes_4h, 3, 10)

    # 1d trend (from daily klines)
    trend_1d = 0
    if len(klines_daily) >= 20:
        closes_1d = [float(k[4]) for k in klines_daily]
        trend_1d = trend_score(closes_1d, 5, 20)

    # Volume acceleration across timeframes
    vol_accel_1h = 0
    if len(klines_1h) >= 24:
        vols = [float(k[5]) * float(k[4]) for k in klines_1h]
        recent = mean(vols[-6:])
        prior = mean(vols[-24:-6]) if len(vols) >= 24 else mean(vols)
        vol_accel_1h = 1 if recent > prior * 1.3 else 0

    vol_accel_1d = 0
    if len(klines_daily) >= 14:
        vols_d = [float(k[5]) * float(k[4]) for k in klines_daily]
        recent_d = mean(vols_d[-3:])
        prior_d = mean(vols_d[-14:-3]) if len(vols_d) >= 14 else mean(vols_d)
        vol_accel_1d = 1 if recent_d > prior_d * 1.3 else 0

    # Alignment: all bullish = 1, mixed = 0.5, all bearish = 0
    trends = [trend_1h, trend_4h, trend_1d]
    bullish_count = sum(1 for t in trends if t > 0)
    bearish_count = sum(1 for t in trends if t < 0)

    # MTF alignment score
    if bullish_count == 3:
        mtf_alignment = 1.0
    elif bullish_count == 2 and bearish_count == 0:
        mtf_alignment = 0.7
    elif bearish_count >= 2:
        mtf_alignment = 0.1
    else:
        mtf_alignment = 0.4

    # Volume confirmation across TFs
    vol_confirm = (vol_accel_1h + vol_accel_1d) / 2

    raw["trend_1h"] = trend_1h
    raw["trend_4h"] = trend_4h
    raw["trend_1d"] = trend_1d
    raw["mtf_alignment"] = round(mtf_alignment, 2)
    raw["vol_accel_1h"] = vol_accel_1h
    raw["vol_accel_1d"] = vol_accel_1d

    T = 0.6 * mtf_alignment + 0.4 * vol_confirm
    return T, raw


def compute_M(klines_1h, klines_daily):
    """
    Momentum signal.
    """
    raw = {}
    return_1h = 0.0
    return_4h = 0.0
    return_24h = 0.0

    if len(klines_1h) >= 2:
        c_now = float(klines_1h[-1][4])
        c_1h = float(klines_1h[-2][4])
        return_1h = safe_div(c_now - c_1h, c_1h)

    if len(klines_1h) >= 5:
        c_now = float(klines_1h[-1][4])
        c_4h = float(klines_1h[-5][4])
        return_4h = safe_div(c_now - c_4h, c_4h)

    if len(klines_1h) >= 25:
        c_now = float(klines_1h[-1][4])
        c_24h = float(klines_1h[-25][4])
        return_24h = safe_div(c_now - c_24h, c_24h)

    raw["return_1h"] = round(return_1h, 5)
    raw["return_4h"] = round(return_4h, 5)
    raw["return_24h"] = round(return_24h, 5)

    M = (
        0.5 * float(sigmoid(return_1h * 50))
        + 0.3 * float(sigmoid(return_4h * 25))
        + 0.2 * float(sigmoid(return_24h * 15))
    )
    return M, raw


# ---------------------------------------------------------------------------
# R8 model scoring
# ---------------------------------------------------------------------------

def load_r8_model():
    """Load the R8 model from pickle."""
    model_path = DATA_DIR / "best_model_r8.pkl"
    if not model_path.exists():
        print(f"  [WARN] R8 model not found at {model_path}")
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)


def compute_r8_proba(r8_model, klines_daily):
    """
    Compute R8 probability for a coin using daily klines.
    Returns probability (0~1).
    """
    if not r8_model or len(klines_daily) < 60:
        return 0.0

    o, h, l, c, v = parse_klines(klines_daily)
    idx = len(c) - 1

    # Extract features using the same pipeline as R8
    feat = extract_features(o, h, l, c, v, idx)
    if not feat:
        return 0.0

    feat = transform_vol_features(feat, o, h, l, c, v, idx)

    models = r8_model["models"]
    feature_cols = r8_model["feature_cols"]

    # Fill missing features with defaults
    for fc in feature_cols:
        if fc not in feat:
            feat[fc] = 0.5 if "_xs_pct" in fc else 0.0

    fv = np.array([[feat.get(fc, 0) for fc in feature_cols]], dtype=np.float64)
    fv = np.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0)

    # Soft-vote: average probabilities across ensemble
    probs = []
    for scaler, s1, s2, s1_threshold in models:
        fvs = scaler.transform(fv)
        s1_prob = s1.predict_proba(fvs)[0, 1]

        if s2 is not None and s1_prob >= s1_threshold:
            s2_feat = np.array([[s1_prob, s1_prob - s1_threshold]])
            s2_prob = s2.predict_proba(s2_feat)[0, 1]
            probs.append(s1_prob * s2_prob)
        else:
            probs.append(0.0)

    return float(np.mean(probs))


# ---------------------------------------------------------------------------
# Main scan logic
# ---------------------------------------------------------------------------

def run_scan(update_watchlist=False):
    """Execute the full ignition scan."""
    t0 = time.time()
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%dT%H:%M:%S")
    today_str = now.strftime("%Y-%m-%d")

    # Ensure log directory exists
    LOG_DIR.mkdir(exist_ok=True)

    # Load R8 model
    print("[1/7] Loading R8 model...")
    r8_model = load_r8_model()
    if r8_model:
        print(f"  R8 model loaded: {len(r8_model['models'])} ensemble members, "
              f"{len(r8_model['feature_cols'])} features")
    else:
        print("  R8 model unavailable — will use 0.0 for R8_proba")

    # Load alert history & watchlist
    alert_history = load_alert_history()
    alert_history = cleanup_alert_history(alert_history)
    r8_watchlist = load_json(R8_WATCHLIST_FILE, {"coins": [], "updated": ""})

    # Check if we need to update watchlist (first run of day or flag)
    if update_watchlist or r8_watchlist.get("updated", "")[:10] != today_str:
        update_watchlist = True

    # ---------------------------------------------------------------
    # Step 1: Fetch all spot tickers
    # ---------------------------------------------------------------
    print("[2/7] Fetching Bybit spot tickers + linear pairs...")
    tickers = fetch_all_spot_tickers()
    if not tickers:
        print("  [ERROR] Failed to fetch tickers!")
        return

    # Fetch linear (futures) pairs to filter only tradeable coins
    linear_symbols = set()
    try:
        linear_data = bybit_get("/v5/market/tickers", {"category": "linear"})
        linear_symbols = set(
            t["symbol"] for t in linear_data.get("result", {}).get("list", [])
            if t["symbol"].endswith("USDT")
        )
        print(f"  Linear(선물) pairs: {len(linear_symbols)}")
    except Exception:
        print("  [WARN] Failed to fetch linear tickers, skipping futures filter")

    # Quick filter: USDT pairs, turnover >= $1M, not stablecoin/large-cap, must have futures
    ticker_map = {}  # symbol -> ticker data
    filtered_symbols = []
    for t in tickers:
        sym = t["symbol"]
        if not sym.endswith("USDT"):
            continue
        if sym in SKIP_SYMBOLS:
            continue
        # Must have linear (futures) pair
        if linear_symbols and sym not in linear_symbols:
            continue
        turnover = float(t.get("turnover24h", 0))
        if turnover < 200_000:
            continue
        ticker_map[sym] = t
        filtered_symbols.append(sym)

    total_spot = sum(1 for t in tickers if t["symbol"].endswith("USDT"))
    print(f"  Total USDT pairs: {total_spot}, futures filter: {len(filtered_symbols)}")

    # ---------------------------------------------------------------
    # Step 2: Fetch kline data for all filtered coins
    # ---------------------------------------------------------------
    print(f"[3/7] Fetching klines for {len(filtered_symbols)} coins...")
    coin_data = {}  # symbol -> {klines_1h, klines_15m, klines_daily, ticker}
    batch_count = 0

    for i, sym in enumerate(filtered_symbols):
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(filtered_symbols)}")

        try:
            # Fetch 1h klines (168 = 7 days)
            klines_1h = fetch_klines(sym, "60", 168)
            if len(klines_1h) < 24:
                continue
            time.sleep(0.02)

            # Fetch 15m klines (96 = 24h)
            klines_15m = fetch_klines(sym, "15", 96)
            time.sleep(0.02)

            # Fetch daily klines (60 days)
            klines_daily = fetch_klines(sym, "D", 60)
            if len(klines_daily) < 14:
                continue
            time.sleep(0.02)

            coin_data[sym] = {
                "klines_1h": klines_1h,
                "klines_15m": klines_15m,
                "klines_daily": klines_daily,
                "ticker": ticker_map[sym],
            }
            batch_count += 1
        except Exception as e:
            continue

    print(f"  Successfully fetched data for {batch_count} coins")

    # ---------------------------------------------------------------
    # Step 3: Compute V, D, P, M for all coins
    # ---------------------------------------------------------------
    print("[4/7] Computing signals (V, D, P, M)...")
    coin_signals = {}  # symbol -> {V, D, P, M, raw, ...}

    for sym, cd in coin_data.items():
        klines_1h = cd["klines_1h"]
        klines_15m = cd["klines_15m"]
        klines_daily = cd["klines_daily"]
        ticker = cd["ticker"]

        price = float(klines_1h[-1][4]) if klines_1h else 0
        if price <= 0:
            continue

        # V (Volume explosion) — taker data deferred to heavy phase
        V, v_raw = compute_V(klines_1h, klines_15m, taker_data=None)

        # Price change 24h for D computation
        price_24h_ago = float(klines_1h[-25][4]) if len(klines_1h) >= 25 else price
        price_change_24h = safe_div(price - price_24h_ago, price_24h_ago)

        # D (Derivatives) — fetch funding & OI
        # We defer heavy API calls: only fetch for coins passing volume filter
        # Store placeholder for now
        D = 0.5
        d_raw = {}

        # P (Price pattern)
        P, p_raw = compute_P(price, klines_1h, klines_daily)

        # M (Momentum)
        M, m_raw = compute_M(klines_1h, klines_daily)

        # T (Multi-TF alignment)
        T, t_raw = compute_T(klines_1h, klines_daily)

        # Returns for filtering
        return_7d = 0.0
        return_30d = 0.0
        if len(klines_daily) >= 8:
            c7 = float(klines_daily[-8][4])
            return_7d = safe_div(price - c7, c7) if c7 > 0 else 0
        if len(klines_daily) >= 31:
            c30 = float(klines_daily[-31][4])
            return_30d = safe_div(price - c30, c30) if c30 > 0 else 0

        # Price position 30d
        price_position_30d = 0.5
        if len(klines_daily) >= 30:
            high_30d = max(float(k[2]) for k in klines_daily[-30:])
            low_30d = min(float(k[3]) for k in klines_daily[-30:])
            price_position_30d = safe_div(price - low_30d, high_30d - low_30d, 0.5)

        # Volatility 30d (for silence_score)
        vol_30d = 0.0
        if len(klines_daily) >= 30:
            daily_returns = []
            for j in range(1, min(31, len(klines_daily))):
                c_j = float(klines_daily[-j][4])
                c_prev = float(klines_daily[-j-1][4]) if j+1 <= len(klines_daily) else c_j
                if c_prev > 0:
                    daily_returns.append(abs(c_j / c_prev - 1))
            vol_30d = stdev(daily_returns) if daily_returns else 0

        coin_signals[sym] = {
            "V": V, "D": D, "P": P, "M": M, "T": T,
            "v_raw": v_raw, "d_raw": d_raw, "p_raw": p_raw, "m_raw": {}, "t_raw": t_raw,
            "price": price,
            "return_1h": m_raw.get("return_1h", 0),
            "return_4h": m_raw.get("return_4h", 0),
            "return_24h": m_raw.get("return_24h", 0),
            "return_7d": return_7d,
            "return_30d": return_30d,
            "price_position_30d": price_position_30d,
            "vol_30d": vol_30d,
            "turnover24h": float(ticker.get("turnover24h", 0)),
            "klines_daily": klines_daily,
            "klines_1h": klines_1h,
        }

    print(f"  Computed signals for {len(coin_signals)} coins")

    # ---------------------------------------------------------------
    # Step 4: Cross-sectional percentile ranks
    # ---------------------------------------------------------------
    print("[5/7] Cross-sectional ranking & volume filter...")
    symbols = list(coin_signals.keys())
    V_values = [coin_signals[s]["V"] for s in symbols]
    D_values = [coin_signals[s]["D"] for s in symbols]
    P_values = [coin_signals[s]["P"] for s in symbols]
    M_values = [coin_signals[s]["M"] for s in symbols]
    ret30_values = [coin_signals[s]["return_30d"] for s in symbols]
    vol30_values = [coin_signals[s]["vol_30d"] for s in symbols]

    V_pcts = percentile_rank(V_values)
    D_pcts = percentile_rank(D_values)
    P_pcts = percentile_rank(P_values)
    M_pcts = percentile_rank(M_values)
    ret30_pcts = percentile_rank(ret30_values)
    vol30_pcts = percentile_rank(vol30_values)

    for i, sym in enumerate(symbols):
        coin_signals[sym]["V_xs_pct"] = V_pcts[i]
        coin_signals[sym]["D_xs_pct"] = D_pcts[i]
        coin_signals[sym]["P_xs_pct"] = P_pcts[i]
        coin_signals[sym]["M_xs_pct"] = M_pcts[i]
        coin_signals[sym]["return_30d_pct"] = ret30_pcts[i]
        coin_signals[sym]["vol_30d_pct"] = vol30_pcts[i]

        # Silence score
        coin_signals[sym]["silence_score"] = (
            1 if (ret30_pcts[i] < 0.5 and vol30_pcts[i] < 0.5) else 0
        )

    # Quick volume filter: pass if v_zscore_1h > 1.5 OR v_zscore_15m > 2
    r8_watchlist_coins = set(r8_watchlist.get("coins", []))
    passed_symbols = []
    for sym in symbols:
        sig = coin_signals[sym]
        v_z1h = sig["v_raw"].get("v_zscore_1h", 0)
        v_z15m = sig["v_raw"].get("v_zscore_15m", 0)
        if v_z1h > 1.5 or v_z15m > 2:
            passed_symbols.append(sym)
        elif sym in r8_watchlist_coins:
            passed_symbols.append(sym)

    print(f"  Volume filter: {len(passed_symbols)} passed (+ R8 watchlist)")

    # ---------------------------------------------------------------
    # Step 5: Fetch derivatives data + R8 for passed coins
    # ---------------------------------------------------------------
    print(f"[6/7] Computing full ignition score for {len(passed_symbols)} candidates...")
    candidates = []

    for i, sym in enumerate(passed_symbols):
        sig = coin_signals[sym]
        price = sig["price"]

        if (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{len(passed_symbols)}")

        # Fetch derivatives data + taker data now (heavy API calls)
        try:
            linear_ticker = fetch_linear_ticker(sym)
            time.sleep(0.02)
            funding_history = fetch_funding_history(sym)
            time.sleep(0.02)
            oi_history = fetch_open_interest_history(sym, "1h", 25)
            time.sleep(0.02)

            D, d_raw = compute_D(
                linear_ticker, funding_history, oi_history,
                sig.get("return_24h", 0),
            )
            sig["D"] = D
            sig["d_raw"] = d_raw
        except Exception:
            pass

        # Taker buy ratio (trade flow)
        try:
            taker_data = fetch_taker_buy_ratio(sym)
            time.sleep(0.02)
            V_new, v_raw_new = compute_V(
                coin_data[sym]["klines_1h"],
                coin_data[sym]["klines_15m"],
                taker_data=taker_data,
            )
            sig["V"] = V_new
            sig["v_raw"].update(v_raw_new)
        except Exception:
            pass

        # Recompute D percentile for passed coins
        # (We keep the original cross-sectional D_xs_pct since D was placeholder)

        # R8 model probability
        r8_proba = compute_r8_proba(r8_model, sig["klines_daily"])
        sig["R8_proba"] = r8_proba

        # ---------------------------------------------------------------
        # Ignition score
        # ---------------------------------------------------------------
        V_xs_pct = sig["V_xs_pct"]
        D_score = sig["D"]
        R8_proba_val = sig["R8_proba"]
        P_score = sig["P"]
        M_score = sig["M"]
        T_score = sig.get("T", 0.5)

        ignition_score = (
            0.25 * R8_proba_val   # 매집 패턴 (핵심 — 급등 전 사전 신호)
            + 0.20 * P_score      # 가격 패턴 (변동성 압축 = 곧 터짐)
            + 0.20 * T_score      # Multi-TF alignment (추세 정합)
            + 0.15 * D_score      # 파생상품 (숏 과열, OI 매집)
            + 0.10 * V_xs_pct     # 거래량 (너무 터졌으면 이미 늦음)
            + 0.10 * M_score      # 모멘텀 (보조)
        )

        silence_gate = 0.75 + 0.25 * sig["silence_score"]  # 조용하면 1.0, 시끄러워도 0.75 (약한 페널티)
        return_1h = sig["return_1h"]
        momentum_gate = 0.5 if return_1h > 0.03 else 1.0

        final_score = ignition_score * silence_gate * momentum_gate

        sig["ignition_score"] = ignition_score
        sig["silence_gate"] = silence_gate
        sig["momentum_gate"] = momentum_gate
        sig["final_score"] = final_score
        sig["direction"] = "LONG"

        # === SHORT signal (급락 전조) ===
        # 롱 과열 + 고점 근처 + 하락 추세 정합
        fr_raw = sig["d_raw"].get("funding_rate", 0)
        fr_z = sig["d_raw"].get("fr_zscore", 0)
        oi_change = sig["d_raw"].get("oi_change_1h", 0)
        price_pos_30d = sig["price_position_30d"]
        price_pos_7d = sig["p_raw"].get("price_position_7d", 0.5)
        ret_7d = sig["return_7d"]

        # Short sub-signals
        # 1. 롱 과열: FR 높고 양수 = 롱 포지션 과밀
        long_overheated = float(sigmoid((fr_raw - 0.0005) * 5000))  # FR > 0.05%면 과열
        # 2. 고점 근처: 30d/7d 고점 근접
        near_high = float(sigmoid((price_pos_30d - 0.7) * 5))
        # 3. 하락 추세: T 반전 (bearish alignment)
        T_score_raw = sig.get("T", 0.5)
        bearish_trend = 1.0 - T_score_raw  # T 낮으면 bearish
        # 4. OI 감소 = 자금 이탈
        oi_exit = float(sigmoid((-oi_change - 0.02) / 0.03))
        # 5. 최근 급등 후 = 되돌림 가능성
        overextended = float(sigmoid((ret_7d - 0.15) * 10))

        short_score = (
            0.30 * long_overheated     # FR 높음 = 롱 과밀 (핵심)
            + 0.30 * near_high          # 고점 근처 (핵심)
            + 0.05 * bearish_trend      # 하락추세 (보조 — 시장 전체 노이즈 줄임)
            + 0.10 * oi_exit            # OI 감소
            + 0.25 * overextended       # 최근 급등 후 되돌림 (핵심)
        )

        # Short도 silence/momentum gate 적용 (반대 방향)
        short_momentum_gate = 0.5 if return_1h < -0.03 else 1.0  # 이미 급락 중이면 늦음
        final_short_score = short_score * short_momentum_gate

        sig["short_score"] = final_short_score
        sig["short_raw"] = {
            "long_overheated": round(float(long_overheated), 3),
            "near_high": round(float(near_high), 3),
            "bearish_trend": round(float(bearish_trend), 3),
            "oi_exit": round(float(oi_exit), 3),
            "overextended": round(float(overextended), 3),
        }

        candidates.append(sym)

    # ---------------------------------------------------------------
    # Step 6: Filtering + ranking
    # ---------------------------------------------------------------
    # === EMA Smoothing + 연속 등장 체크 ===
    score_history = load_json(SCORE_HISTORY_FILE, {})
    EMA_ALPHA = 0.3  # 현재 30%, 이전 70%

    for sym in candidates:
        sig = coin_signals[sym]
        coin = sym.replace("USDT", "")
        prev = score_history.get(coin, {})
        prev_score = prev.get("ema_score", sig["final_score"])
        prev_short = prev.get("ema_short", sig.get("short_score", 0))
        prev_count = prev.get("appear_count", 0)

        # EMA: 이전 70% + 현재 30%
        ema_score = EMA_ALPHA * sig["final_score"] + (1 - EMA_ALPHA) * prev_score
        ema_short = EMA_ALPHA * sig.get("short_score", 0) + (1 - EMA_ALPHA) * prev_short

        sig["final_score_raw"] = sig["final_score"]
        sig["final_score"] = ema_score
        sig["short_score_raw"] = sig.get("short_score", 0)
        sig["short_score"] = ema_short
        sig["appear_count"] = prev_count + 1

        score_history[coin] = {
            "ema_score": round(ema_score, 4),
            "ema_short": round(ema_short, 4),
            "appear_count": sig["appear_count"],
            "last_seen": now_str,
        }

    # 3시간 이상 안 나타난 코인 제거
    stale_coins = [c for c, v in score_history.items()
                   if v.get("last_seen", "") < (now - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%S")]
    for c in stale_coins:
        del score_history[c]

    save_json(SCORE_HISTORY_FILE, score_history)

    print("[7/7] Filtering and ranking...")
    eligible_long = []
    eligible_short = []

    for sym in candidates:
        sig = coin_signals[sym]
        coin_name = sym.replace("USDT", "")

        if sig["turnover24h"] < 200_000:
            continue
        if is_in_cooldown(coin_name, alert_history):
            continue

        # LONG candidates
        if sig["final_score"] >= 0.20:
            if sig["return_7d"] <= 0.30 and sig["return_30d"] <= 1.00 and sig["price_position_30d"] <= 0.85:
                eligible_long.append(sym)

        # SHORT candidates
        if sig["short_score"] >= 0.20:
            if sig["return_7d"] >= -0.30 and sig["price_position_30d"] >= 0.15:
                eligible_short.append(sym)

    eligible_long.sort(key=lambda s: coin_signals[s]["final_score"], reverse=True)
    eligible_short.sort(key=lambda s: coin_signals[s]["short_score"], reverse=True)
    eligible_long = eligible_long[:15]
    eligible_short = eligible_short[:10]

    # 중복 제거: 양쪽에 있으면 점수 높은 쪽만 남기기
    long_set = set(eligible_long)
    short_set = set(eligible_short)
    overlap = long_set & short_set
    for sym in overlap:
        long_score = coin_signals[sym]["final_score"]
        short_score = coin_signals[sym]["short_score"]
        if long_score >= short_score:
            eligible_short.remove(sym)
        else:
            eligible_long.remove(sym)

    eligible = eligible_long  # 기존 호환용
    eligible = eligible[:20]

    # Assign grades
    for sym in eligible:
        score = coin_signals[sym]["final_score"]
        if score > 0.40:
            coin_signals[sym]["grade"] = "HIGH"
        elif score >= 0.30:
            coin_signals[sym]["grade"] = "MED"
        else:
            coin_signals[sym]["grade"] = "LOW"

    elapsed = time.time() - t0
    print(f"\n  Scan complete: {len(filtered_symbols)} -> {len(passed_symbols)} -> {len(eligible)} candidates | {elapsed:.0f}s")

    # ---------------------------------------------------------------
    # Logging
    # ---------------------------------------------------------------
    log_file = LOG_DIR / f"{today_str}.jsonl"
    logged_count = 0

    # Log all candidates (not just eligible) for data collection
    all_to_log = set(candidates) | set(eligible)
    for sym in all_to_log:
        sig = coin_signals[sym]
        coin_name = sym.replace("USDT", "")
        is_alerted = sym in eligible

        log_entry = {
            "timestamp": now_str,
            "coin": coin_name,
            "final_score": round(sig.get("final_score", 0), 4),
            "V": round(sig["V"], 4),
            "V_xs_pct": round(sig.get("V_xs_pct", 0), 4),
            "D": round(sig["D"], 4),
            "P": round(sig["P"], 4),
            "M": round(sig["M"], 4),
            "T": round(sig.get("T", 0), 4),
            "R8_proba": round(sig.get("R8_proba", 0), 4),
            "silence_score": sig.get("silence_score", 0),
            "silence_gate": round(sig.get("silence_gate", 1.0), 2),
            "momentum_gate": round(sig.get("momentum_gate", 1.0), 2),
            "raw": {
                **sig.get("v_raw", {}),
                **sig.get("d_raw", {}),
                **sig.get("p_raw", {}),
                "return_1h": round(sig.get("return_1h", 0), 5),
                "return_4h": round(sig.get("return_4h", 0), 5),
                "return_24h": round(sig.get("return_24h", 0), 5),
                "return_7d": round(sig.get("return_7d", 0), 5),
                "return_30d": round(sig.get("return_30d", 0), 5),
                "price": sig["price"],
            },
            "alerted": is_alerted,
            "grade": sig.get("grade", ""),
            "price_at_scan": sig["price"],
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        logged_count += 1

    print(f"  Logged {logged_count} entries to {log_file}")

    # ---------------------------------------------------------------
    # Update alert history
    # ---------------------------------------------------------------
    for sym in eligible:
        coin_name = sym.replace("USDT", "")
        alert_history[coin_name] = now_str
    save_alert_history(alert_history)

    # ---------------------------------------------------------------
    # R8 watchlist update
    # ---------------------------------------------------------------
    if update_watchlist and r8_model:
        print("\n  Updating R8 watchlist...")
        wl_coins = []
        for sym in coin_data:
            cd = coin_data[sym]
            proba = compute_r8_proba(r8_model, cd["klines_daily"])
            if proba > 0.15:
                wl_coins.append(sym)
        wl_coins = wl_coins[:50]
        save_json(R8_WATCHLIST_FILE, {
            "coins": wl_coins,
            "updated": now_str,
            "count": len(wl_coins),
        })
        print(f"  R8 watchlist: {len(wl_coins)} coins saved")

    # ---------------------------------------------------------------
    # Slack alert
    # ---------------------------------------------------------------
    if eligible:
        lines = [f":fire: *IGNITION ALERT* ({now.strftime('%Y-%m-%d %H:%M')})"]
        lines.append(f"`V=거래량 D=파생상품 R8=매집 P=패턴 T=추세 M=모멘텀`")
        lines.append("```")
        lines.append(f" # {'코인':<8} {'점수':>5} {'등급':<4} {'가격':<10} {'V':>4} {'D':>4} {'R8':>4} {'P':>4} {'T':>4} {'M':>4} {'7d':>7} {'FR':>10} {'비고'}")

        for i, sym in enumerate(eligible):
            sig = coin_signals[sym]
            coin_name = sym.replace("USDT", "")
            grade = sig["grade"]
            score = sig["final_score"]

            # Format price
            price = sig["price"]
            if price >= 100:
                price_str = f"${price:,.0f}"
            elif price >= 1:
                price_str = f"${price:.2f}"
            elif price >= 0.001:
                price_str = f"${price:.4f}"
            else:
                price_str = f"${price:.6f}"

            fr = sig["d_raw"].get("funding_rate", 0)
            fr_str = f"{fr*100:+.4f}%" if fr != 0 else "n/a"
            ret7 = sig["return_7d"] * 100

            # Notes
            notes = []
            v_z = sig["v_raw"].get("v_zscore_1h", 0)
            if v_z > 2:
                notes.append(f"{v_z:.0f}σ")
            if sig["d_raw"].get("funding_flip", 0):
                notes.append("flip↗")
            if sig.get("silence_score", 0):
                notes.append("조용")
            if sig["p_raw"].get("breakout", 0):
                notes.append("돌파")
            note_str = " ".join(notes)

            lines.append(
                f"{i+1:>2} {coin_name:<8} {score:.2f} {grade:<4} {price_str:<10} "
                f"{sig['V']:.2f} {sig['D']:.2f} {sig.get('R8_proba',0):.2f} "
                f"{sig['P']:.2f} {sig.get('T',0):.2f} {sig['M']:.2f} "
                f"{ret7:>+6.1f}% {fr_str:>10} {note_str}"
            )

        lines.append("```")

        # SHORT candidates
        if eligible_short:
            lines.append("")
            lines.append(f":chart_with_downwards_trend: *SHORT 후보 ({len(eligible_short)}개)*")
            lines.append("```")
            lines.append(f" # {'코인':<8} {'점수':>5} {'가격':<10} {'롱과열':>5} {'고점':>4} {'하락':>4} {'OI탈':>4} {'과열':>4} {'7d':>7} {'FR':>10}")
            for i, sym in enumerate(eligible_short):
                sig = coin_signals[sym]
                coin_name = sym.replace("USDT", "")
                sr = sig["short_raw"]
                price = sig["price"]
                if price >= 100: price_str = f"${price:,.0f}"
                elif price >= 1: price_str = f"${price:.2f}"
                elif price >= 0.001: price_str = f"${price:.4f}"
                else: price_str = f"${price:.6f}"
                fr = sig["d_raw"].get("funding_rate", 0)
                fr_str = f"{fr*100:+.4f}%" if fr != 0 else "n/a"
                ret7 = sig["return_7d"] * 100
                lines.append(
                    f"{i+1:>2} {coin_name:<8} {sig['short_score']:.2f} {price_str:<10} "
                    f"{sr['long_overheated']:.2f} {sr['near_high']:.2f} "
                    f"{sr['bearish_trend']:.2f} {sr['oi_exit']:.2f} {sr['overextended']:.2f} "
                    f"{ret7:>+6.1f}% {fr_str:>10}"
                )
            lines.append("```")

        lines.append(
            f"_스캔: {len(filtered_symbols)}개 → 필터 {len(passed_symbols)}개 → "
            f"LONG {len(eligible_long)}개 + SHORT {len(eligible_short)}개 | 실행 {elapsed:.0f}초_"
        )
        msg = "\n".join(lines)
        send_slack(msg)
        print(msg)
    elif eligible_short:
        # Short만 있는 경우
        lines = [f":chart_with_downwards_trend: *SHORT ALERT* ({now.strftime('%Y-%m-%d %H:%M')})"]
        lines.append("```")
        lines.append(f" # {'코인':<8} {'점수':>5} {'가격':<10} {'롱과열':>5} {'고점':>4} {'하락':>4} {'OI탈':>4} {'과열':>4} {'7d':>7} {'FR':>10}")
        for i, sym in enumerate(eligible_short):
            sig = coin_signals[sym]
            coin_name = sym.replace("USDT", "")
            sr = sig["short_raw"]
            price = sig["price"]
            if price >= 100: price_str = f"${price:,.0f}"
            elif price >= 1: price_str = f"${price:.2f}"
            elif price >= 0.001: price_str = f"${price:.4f}"
            else: price_str = f"${price:.6f}"
            fr = sig["d_raw"].get("funding_rate", 0)
            fr_str = f"{fr*100:+.4f}%" if fr != 0 else "n/a"
            ret7 = sig["return_7d"] * 100
            lines.append(
                f"{i+1:>2} {coin_name:<8} {sig['short_score']:.2f} {price_str:<10} "
                f"{sr['long_overheated']:.2f} {sr['near_high']:.2f} "
                f"{sr['bearish_trend']:.2f} {sr['oi_exit']:.2f} {sr['overextended']:.2f} "
                f"{ret7:>+6.1f}% {fr_str:>10}"
            )
        lines.append("```")
        lines.append(f"_스캔: {len(filtered_symbols)}개 → SHORT {len(eligible_short)}개 | {elapsed:.0f}초_")
        msg = "\n".join(lines)
        send_slack(msg)
        print(msg)
    else:
        summary = (
            f":mag: *Ignition Scan* ({now.strftime('%Y-%m-%d %H:%M')}) — 후보 없음\n"
            f"_스캔: {len(filtered_symbols)}개 → 0개 | {elapsed:.0f}초_"
        )
        print(summary)

    # ---------------------------------------------------------------
    # Track results
    # ---------------------------------------------------------------
    track_results()

    return eligible


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

def track_results():
    """
    Track performance of past alerts.
    For each alerted coin, fetch current price and compute returns.
    """
    print("\n[Track] Checking past alert performance...")
    now = datetime.now()

    # Collect all alerted entries from log files
    alerts = []
    if not LOG_DIR.exists():
        print("  No log directory found")
        return

    for log_file in sorted(LOG_DIR.glob("*.jsonl")):
        try:
            with open(log_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if entry.get("alerted", False):
                        alerts.append(entry)
        except (json.JSONDecodeError, IOError):
            continue

    if not alerts:
        print("  No past alerts found")
        return

    # Load existing results to avoid re-tracking completed ones
    existing_results = {}
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    key = f"{r.get('coin')}_{r.get('alert_ts')}"
                    existing_results[key] = r
        except (json.JSONDecodeError, IOError):
            pass

    # Fetch current spot tickers for price lookup
    tickers = fetch_all_spot_tickers()
    price_map = {}
    if tickers:
        for t in tickers:
            price_map[t["symbol"]] = float(t.get("lastPrice", 0))

    updated = 0
    for alert in alerts:
        coin = alert["coin"]
        alert_ts_str = alert["timestamp"]
        price_at_alert = alert.get("price_at_scan", 0)
        if price_at_alert <= 0:
            continue

        try:
            alert_ts = datetime.fromisoformat(alert_ts_str)
        except (ValueError, TypeError):
            continue

        hours_since = (now - alert_ts).total_seconds() / 3600
        if hours_since < 1:
            continue  # Too recent

        key = f"{coin}_{alert_ts_str}"
        result = existing_results.get(key, {
            "alert_ts": alert_ts_str,
            "coin": coin,
            "price_at_alert": price_at_alert,
            "grade": alert.get("grade", ""),
            "final_score": alert.get("final_score", 0),
        })

        sym = coin + "USDT"
        current_price = price_map.get(sym, 0)
        if current_price <= 0:
            continue

        # Fill in time-based returns
        if hours_since >= 1 and "price_1h" not in result:
            result["price_1h"] = current_price
        if hours_since >= 6 and "price_6h" not in result:
            result["price_6h"] = current_price
        if hours_since >= 24 and "price_24h" not in result:
            result["price_24h"] = current_price
        if hours_since >= 72 and "price_72h" not in result:
            result["price_72h"] = current_price

        # Max returns
        if "max_price" not in result:
            result["max_price"] = current_price
        else:
            result["max_price"] = max(result["max_price"], current_price)

        if price_at_alert > 0:
            result["current_return"] = round((current_price / price_at_alert - 1), 5)
            result["max_return"] = round((result["max_price"] / price_at_alert - 1), 5)

            if "price_24h" in result:
                result["max_return_24h"] = round(
                    max(
                        safe_div(result.get("price_1h", price_at_alert) - price_at_alert, price_at_alert),
                        safe_div(result.get("price_6h", price_at_alert) - price_at_alert, price_at_alert),
                        safe_div(result["price_24h"] - price_at_alert, price_at_alert),
                    ), 5
                )
            if "price_72h" in result:
                result["max_return_72h"] = round(result["max_return"], 5)

        result["last_checked"] = now.strftime("%Y-%m-%dT%H:%M:%S")
        existing_results[key] = result
        updated += 1

    # Write all results back
    if existing_results:
        with open(RESULTS_FILE, "w") as f:
            for r in sorted(existing_results.values(), key=lambda x: x.get("alert_ts", "")):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  Tracked {updated} alerts, total {len(existing_results)} results in file")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ignition Scanner")
    parser.add_argument("--update-watchlist", action="store_true",
                        help="Also update R8 watchlist")
    parser.add_argument("--track-only", action="store_true",
                        help="Only run result tracking, no scan")
    args = parser.parse_args()

    print("=" * 60)
    print("  IGNITION SCANNER")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if args.track_only:
        track_results()
    else:
        run_scan(update_watchlist=args.update_watchlist)

    print("\nDone.")


if __name__ == "__main__":
    raise SystemExit("Legacy notifier retired. Use node momentum-alert/scan.mjs; see momentum-alert/USAGE.txt")
    main()
