#!/usr/bin/env python3
"""
4시간봉 전환 직전 모멘텀 스캐너
- 전환 20분 전에 실행
- 15분봉 16개(4시간) 분석
- Slack webhook으로 직접 전송
"""
import requests
import json
import os
from datetime import datetime
from urllib.request import Request, urlopen

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


def send_slack(text):
    if not SLACK_WEBHOOK_URLS:
        print(f"[NO SLACK]\n{text}\n")
        return
    payload = json.dumps({"text": text}).encode()
    for wh_url in SLACK_WEBHOOK_URLS:
        req = Request(wh_url, data=payload,
                      headers={"Content-Type": "application/json"})
        try:
            urlopen(req, timeout=10)
        except Exception as e:
            print(f"Slack 전송 실패 ({wh_url[:50]}...): {e}")


def scan():
    now = datetime.now()
    # 다음 4시간봉 시각 계산 (9, 13, 17, 21, 01, 05)
    h = now.hour
    candle_hours = [1, 5, 9, 13, 17, 21]
    next_4h = next((ch for ch in candle_hours if ch > h), candle_hours[0])

    # 1. 전체 USDT 선물 티커
    resp = requests.get("https://api.bybit.com/v5/market/tickers",
                         params={"category": "linear"}, timeout=10)
    tickers = resp.json()["result"]["list"]

    usdt_tickers = {}
    for t in tickers:
        sym = t["symbol"]
        if not sym.endswith("USDT"):
            continue
        turnover = float(t.get("turnover24h", 0))
        if turnover < 3_000_000:
            continue
        usdt_tickers[sym] = {
            "price": float(t["lastPrice"]),
            "chg24": float(t.get("price24hPcnt", 0)) * 100,
            "turnover_m": turnover / 1e6,
            "funding": float(t.get("fundingRate", 0)) * 100,
        }

    # 2. 각 코인 15분봉 분석 (최근 16개 = 4시간)
    results = []

    for sym, info in usdt_tickers.items():
        try:
            kl = requests.get("https://api.bybit.com/v5/market/kline",
                               params={"category": "linear", "symbol": sym,
                                        "interval": "15", "limit": 16},
                               timeout=10).json()
            klines = list(reversed(kl["result"]["list"]))
            if len(klines) < 16:
                continue

            closes = [float(k[4]) for k in klines]
            opens = [float(k[1]) for k in klines]
            turnovers = [float(k[6]) for k in klines]

            # A. 최근 1시간(4봉) vs 이전 3시간(12봉) 거래량 비율
            recent_vol = sum(turnovers[-4:])
            older_vol = sum(turnovers[:12]) / 3
            vol_surge = recent_vol / older_vol if older_vol > 0 else 0

            # B. 최근 1시간 변동률
            recent_chg = ((closes[-1] - closes[-4]) / closes[-4] * 100) if closes[-4] > 0 else 0

            # C. 최근 4봉 양봉 비율
            recent_bullish = sum(1 for i in range(-4, 0) if closes[i] > opens[i])

            # D. 하단 꼬리 비율
            tail_scores = []
            for k in klines[-4:]:
                o, h, l, c = float(k[1]), float(k[2]), float(k[3]), float(k[4])
                total = h - l
                lower_tail = min(o, c) - l
                if total > 0:
                    tail_scores.append(lower_tail / total)
            avg_tail = sum(tail_scores) / len(tail_scores) if tail_scores else 0

            # E. 전체 4시간 추세
            full_chg = ((closes[-1] - closes[0]) / closes[0] * 100) if closes[0] > 0 else 0

            # --- 점수 ---
            score = 0
            if vol_surge > 2.0: score += 3
            elif vol_surge > 1.5: score += 2
            elif vol_surge > 1.2: score += 1

            if 0.3 <= recent_chg <= 8: score += 2
            elif recent_chg > 8: score += 1

            if recent_bullish >= 3: score += 2
            elif recent_bullish >= 2: score += 1

            if -0.05 <= info["funding"] <= 0.03: score += 1
            elif info["funding"] < -0.05: score += 2

            if avg_tail > 0.4: score += 1

            if score >= 4:
                results.append({
                    "sym": sym.replace("USDT", ""),
                    "price": info["price"],
                    "chg24": info["chg24"],
                    "turnover_m": info["turnover_m"],
                    "funding": info["funding"],
                    "vol_surge": vol_surge,
                    "recent_chg": recent_chg,
                    "bullish": recent_bullish,
                    "full_chg": full_chg,
                    "score": score,
                })

        except Exception:
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    top5 = results[:5]

    # Slack 메시지
    lines = [f"*[4H 스캔] {next_4h:02d}:00 봉 전환 직전 | {now.strftime('%m/%d %H:%M')}*"]
    lines.append(f"분석: {len(usdt_tickers)}개 | 기준충족: {len(results)}개\n")

    if not top5:
        lines.append("기준 충족 코인 없음")
    else:
        for i, r in enumerate(top5, 1):
            tags = []
            if r["vol_surge"] > 2: tags.append("거래량폭증")
            if r["funding"] < -0.05: tags.append("숏과밀")
            if r["bullish"] >= 3: tags.append("연속양봉")
            if 0.5 <= r["recent_chg"] <= 5: tags.append("초기가속")
            tag_str = " | ".join(tags) if tags else ""
            lines.append(
                f"{i}. *{r['sym']}* {r['price']:.4f} | "
                f"1h {r['recent_chg']:+.1f}% | vol {r['vol_surge']:.1f}x | "
                f"FR {r['funding']:+.3f}% | {r['score']}/11"
            )
            if tag_str:
                lines.append(f"   _{tag_str}_")

    msg = "\n".join(lines)
    send_slack(msg)
    print(msg)

    # JSON 저장
    output = {
        "scan_time": now.isoformat(),
        "timeframe": "4H",
        "next_candle": f"{next_4h:02d}:00",
        "total_scanned": len(usdt_tickers),
        "passed": len(results),
        "results": results[:10]
    }
    with open(os.path.join(os.path.dirname(__file__), "last_scan.json"), "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    scan()
