#!/usr/bin/env python3
"""
ML 기반 급등 전조 스캐너 v2 — Round 6 2-Stage Meta-labeling 모델
매 30분 실행, Slack DM으로 리포트
"""

import json, math, os, sys, time, datetime, pickle, warnings
import urllib.request
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "ml_data"
STATE_FILE = BASE_DIR / "ml_scanner_v2_state.json"
MIN_VOL = 1_000_000
MAX_VOL = 10_000_000

WEBHOOK_URL = ""
_env = os.path.expanduser("~/.openclaw/.env.slack-surge")
if os.path.exists(_env):
    with open(_env) as f:
        for line in f:
            if line.startswith("SLACK_WEBHOOK_URL="):
                WEBHOOK_URL = line.strip().split("=", 1)[1]


def send_slack(msg):
    if not WEBHOOK_URL:
        print(msg)
        return
    for chunk in [msg[i:i + 3500] for i in range(0, len(msg), 3500)]:
        payload = json.dumps({"text": chunk})
        req = urllib.request.Request(
            WEBHOOK_URL, data=payload.encode(),
            headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            urllib.request.urlopen(req, timeout=30)
        except Exception as e:
            print(f"Slack err: {e}")
        time.sleep(0.5)


# Reuse feature extraction from round4
sys.path.insert(0, str(BASE_DIR))
from round4_metalabel import extract_features, has_recent_surge, mean, compute_atr

# Import predict_two_stage from round6_v2
from round6_v2 import predict_two_stage

SKIP = {
    "USDCUSDT", "UUSDT", "XUSDUSDT", "PAXGUSDT", "DAIUSDT", "FDUSDUSDT",
    "TUSDUSDT", "BUSDUSDT", "EURUSDT", "USTCUSDT", "USDTUSDT", "USDDUSDT",
    "RLUSDUSDT", "USD1USDT", "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT",
    "SOLUSDT", "ADAUSDT", "DOGEUSDT", "TRXUSDT", "DOTUSDT", "MATICUSDT",
    "LTCUSDT", "BCHUSDT", "LINKUSDT", "AVAXUSDT", "UNIUSDT", "AAVEUSDT",
    "ATOMUSDT",
}


def bybit_get(ep, params=None):
    url = "https://api.bybit.com" + ep
    if params:
        url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
    req = urllib.request.Request(url, headers={"User-Agent": "ml-scanner-v2/1.0"})
    for i in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read())
        except:
            if i == 2:
                raise
            time.sleep(1 + i)


def fetch_derivatives(coin):
    """Bybit 파생상품 데이터: funding rate, OI"""
    sym = coin + "USDT"
    result = {"funding_rate": 0.0, "funding_rate_zscore": 0.0, "oi_change_pct": 0.0}
    try:
        # Funding rate
        data = bybit_get("/v5/market/tickers", {"category": "linear", "symbol": sym})
        items = data.get("result", {}).get("list", [])
        if items:
            fr = float(items[0].get("fundingRate", 0))
            result["funding_rate"] = fr
            # z-score: rough estimate (avg FR ~0.01%, std ~0.03%)
            result["funding_rate_zscore"] = (fr - 0.0001) / 0.0003 if abs(fr) > 0 else 0
        # OI
        oi_data = bybit_get("/v5/market/open-interest", {
            "category": "linear", "symbol": sym, "intervalTime": "1d", "limit": "2"
        })
        oi_list = oi_data.get("result", {}).get("list", [])
        if len(oi_list) >= 2:
            oi_now = float(oi_list[0].get("openInterest", 0))
            oi_prev = float(oi_list[1].get("openInterest", 0))
            if oi_prev > 0:
                result["oi_change_pct"] = (oi_now - oi_prev) / oi_prev
    except:
        pass
    return result


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"prev_hits": [], "last_scan": ""}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def main():
    now = datetime.datetime.now()
    print(f"[{now.strftime('%H:%M')}] ML Scanner v2 시작")

    # Load v2 model
    pkl_path = DATA_DIR / "best_model_v2.pkl"
    if not pkl_path.exists():
        print("v2 모델 없음!")
        return

    with open(pkl_path, "rb") as f:
        pkg = pickle.load(f)

    s1_model = pkg["stage1_model"]
    s2_model = pkg["stage2_model"]
    scaler = pkg["scaler"]
    fcols = pkg["feature_cols"]
    s1_threshold = pkg["s1_threshold"]
    meta_threshold = pkg["meta_threshold"]

    state = load_state()
    prev_hits = set(state.get("prev_hits", []))

    # Get tickers
    tickers = bybit_get("/v5/market/tickers", {"category": "spot"}).get("result", {}).get("list", [])
    alts = [t["symbol"] for t in tickers
            if t["symbol"].endswith("USDT") and t["symbol"] not in SKIP
            and MIN_VOL <= float(t.get("turnover24h", 0)) <= MAX_VOL]

    # Also fetch BTC data for cross-sectional
    btc_data = None
    try:
        btc_kl = bybit_get("/v5/market/kline", {"category": "spot", "symbol": "BTCUSDT", "interval": "D", "limit": "100"})
        btc_klines = list(reversed(btc_kl.get("result", {}).get("list", [])))
        if len(btc_klines) >= 60:
            btc_data = {
                "c": [float(k[4]) for k in btc_klines],
                "v": [float(k[5]) for k in btc_klines],
            }
    except:
        pass

    # Collect all coin data for cross-sectional percentile computation
    all_coin_stats = []  # (coin, ret_30d, volatility_30d, vol_30d)
    coin_cache = {}

    for sym in alts:
        try:
            data = bybit_get("/v5/market/kline", {"category": "spot", "symbol": sym, "interval": "D", "limit": "100"})
            klines = list(reversed(data.get("result", {}).get("list", [])))
            if len(klines) < 60:
                continue
            o = [float(k[1]) for k in klines]
            h = [float(k[2]) for k in klines]
            l = [float(k[3]) for k in klines]
            c = [float(k[4]) for k in klines]
            v = [float(k[5]) for k in klines]
            coin_cache[sym] = {"o": o, "h": h, "l": l, "c": c, "v": v}

            idx = len(c) - 1
            ret_30d = c[idx] / c[idx - 30] - 1 if idx >= 30 and c[idx - 30] > 0 else 0
            from round4_metalabel import stdev
            vol_30d = stdev([(c[j] / c[j-1] - 1) for j in range(max(1, idx-29), idx+1)]) if idx >= 30 else 0
            vol_usd = mean([v[j] * c[j] for j in range(max(0, idx-29), idx+1)])
            all_coin_stats.append((sym, ret_30d, vol_30d, vol_usd))
            time.sleep(0.03)
        except:
            continue

    # Compute percentiles
    if all_coin_stats:
        rets = sorted([s[1] for s in all_coin_stats])
        vols = sorted([s[2] for s in all_coin_stats])
        vol_usds = sorted([s[3] for s in all_coin_stats])

    # Process each coin
    results = []
    deriv_count = 0

    for sym, ret_30d, vol_30d, vol_usd in all_coin_stats:
        try:
            cd = coin_cache[sym]
            o, h, l, c, v = cd["o"], cd["h"], cd["l"], cd["c"], cd["v"]
            idx = len(c) - 1

            if has_recent_surge(c, idx):
                continue

            feat = extract_features(o, h, l, c, v, idx)
            if not feat:
                continue

            # Cross-sectional features
            def percentile_rank(val, sorted_list):
                n = len(sorted_list)
                if n == 0: return 0.5
                rank = sum(1 for x in sorted_list if x <= val)
                return rank / n

            feat["return_30d_percentile"] = percentile_rank(ret_30d, rets)
            feat["volatility_percentile"] = percentile_rank(vol_30d, vols)
            feat["volume_percentile"] = percentile_rank(vol_usd, vol_usds)

            # Stealth accumulation
            vol_z = (vol_usd - mean([s[3] for s in all_coin_stats])) / max(stdev([s[3] for s in all_coin_stats]), 1e-10)
            price_z = (ret_30d - mean([s[1] for s in all_coin_stats])) / max(stdev([s[1] for s in all_coin_stats]), 1e-10)
            feat["stealth_accumulation"] = vol_z - price_z

            # Relative strength vs BTC
            btc_ret = 0
            if btc_data and len(btc_data["c"]) >= 31:
                btc_c = btc_data["c"]
                btc_ret = btc_c[-1] / btc_c[-8] - 1 if btc_c[-8] > 0 else 0
            coin_ret_7d = c[idx] / c[idx - 7] - 1 if idx >= 7 and c[idx - 7] > 0 else 0
            feat["relative_strength_vs_btc"] = coin_ret_7d - btc_ret

            # Derivatives features
            coin_name = sym.replace("USDT", "")
            deriv = fetch_derivatives(coin_name)
            feat["funding_rate"] = deriv["funding_rate"]
            feat["funding_rate_zscore"] = deriv["funding_rate_zscore"]
            feat["oi_change_pct"] = deriv["oi_change_pct"]
            if deriv["funding_rate"] != 0:
                deriv_count += 1

            # Build feature vector
            fv = np.array([[feat.get(fc, 0) for fc in fcols]], dtype=np.float64)
            fv = np.nan_to_num(fv, nan=0, posinf=0, neginf=0)
            fvs = scaler.transform(fv)

            # Two-stage prediction
            final_probs, final_preds, s1_probs = predict_two_stage(
                fvs, s1_model, s2_model, s1_threshold, meta_threshold
            )

            prob = final_probs[0]
            s1_prob = s1_probs[0]
            hit = final_preds[0] == 1

            r7 = (c[-1] / c[-8] - 1) * 100 if len(c) >= 8 and c[-8] > 0 else 0
            r14 = (c[-1] / c[-15] - 1) * 100 if len(c) >= 15 and c[-15] > 0 else 0
            rsi = feat.get("rsi_14", 0)

            if s1_prob >= s1_threshold * 0.8:  # Show anything near Stage 1 threshold
                fr_str = f"{deriv['funding_rate']*100:+.4f}%" if deriv['funding_rate'] != 0 else "n/a"
                results.append({
                    "coin": coin_name, "symbol": sym, "prob": float(prob),
                    "s1_prob": float(s1_prob),
                    "price": c[-1], "ret_7d": r7, "ret_14d": r14,
                    "rsi": rsi, "hit": hit,
                    "fr_str": fr_str,
                })
            time.sleep(0.03)
        except:
            continue

    results.sort(key=lambda x: x["prob"], reverse=True)
    hits = [r for r in results if r["hit"]]
    hit_coins = set(r["coin"] for r in hits)
    new_coins = hit_coins - prev_hits

    # Report
    lines = [f":chart_with_upwards_trend: *ML 전조 스캐너 v2* ({now.strftime('%m/%d %H:%M')})"]
    lines.append(f"`2-Stage Meta | R6v2 | s1={s1_threshold:.2f} meta={meta_threshold:.2f}`")

    if new_coins:
        lines.append(f"\n:rotating_light: *신규 후보 {len(new_coins)}개:* {', '.join(sorted(new_coins))}")

    if hits:
        lines.append(f"\n*전조 후보 ({len(hits)}개):*")
        lines.append("```")
        lines.append(f" # {'코인':<10} {'확률':>6} {'S1':>5} {'현재가':<11} {'7d':>7} {'14d':>7} {'RSI':>4} {'FR':>10}")
        for i, r in enumerate(hits):
            new = "NEW" if r["coin"] in new_coins else ""
            lines.append(f"{i+1:>2} {r['coin']:<10} {r['prob']:>5.1%} {r['s1_prob']:>4.1%} ${r['price']:<10.4f} {r['ret_7d']:>+6.1f}% {r['ret_14d']:>+6.1f}% {r['rsi']:>4.0f} {r['fr_str']:>10} {new}")
        lines.append("```")
    else:
        lines.append("\n전조 후보 없음")

    # Watchlist (Stage 1 통과했지만 Stage 2 미달)
    watchlist = [r for r in results if not r["hit"] and r["s1_prob"] >= s1_threshold][:5]
    if watchlist:
        lines.append(f"\n*주시 목록 (S1 통과, S2 미달):*")
        for r in watchlist:
            lines.append(f"  {r['coin']}: S1={r['s1_prob']:.1%} S2={r['prob']:.1%} (7d {r['ret_7d']:+.1f}%, RSI {r['rsi']:.0f})")

    lines.append(f"\n_스캔: {len(alts)}개 대상, {len(results)}개 분석, {len(hits)}개 후보, 파생상품 {deriv_count}개_")

    send_slack("\n".join(lines))
    print(f"  완료: {len(hits)}개 후보, {len(new_coins)}개 신규")

    # Save state
    state["prev_hits"] = list(hit_coins)
    state["last_scan"] = now.isoformat()
    save_state(state)


if __name__ == "__main__":
    main()
