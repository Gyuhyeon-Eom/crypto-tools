#!/usr/bin/env python3
"""
4시간봉 전환 직전 스캔 → Slack 포맷 출력
속도 최적화: 1차 필터 후 상세 분석
"""
import requests
import json
from datetime import datetime

def scan():
    now = datetime.now()
    
    hour = now.hour
    next_candles = [9, 13, 17, 21, 1, 5]
    next_c = min((c for c in next_candles if c > hour), default=next_candles[0])
    
    # 1. 전체 티커 (1회 호출)
    resp = requests.get("https://api.bybit.com/v5/market/tickers",
                         params={"category": "linear"}, timeout=15)
    tickers = resp.json()["result"]["list"]
    
    # 1차 필터: USDT + 거래대금 $3M+
    usdt = {}
    for t in tickers:
        sym = t["symbol"]
        if not sym.endswith("USDT"): continue
        turnover = float(t.get("turnover24h", 0))
        if turnover < 3_000_000: continue
        usdt[sym] = {
            "price": float(t["lastPrice"]),
            "chg24": float(t.get("price24hPcnt", 0)) * 100,
            "turnover_m": turnover / 1e6,
            "funding": float(t.get("fundingRate", 0)) * 100,
        }
    
    # 2. 15분봉 분석 (개별 호출)
    results = []
    for sym, info in usdt.items():
        try:
            kl = requests.get("https://api.bybit.com/v5/market/kline",
                               params={"category":"linear","symbol":sym,"interval":"15","limit":16},
                               timeout=8).json()
            klines = list(reversed(kl["result"]["list"]))
            if len(klines) < 16: continue
            
            closes = [float(k[4]) for k in klines]
            opens = [float(k[1]) for k in klines]
            turnovers = [float(k[6]) for k in klines]
            
            recent_vol = sum(turnovers[-4:])
            older_vol = sum(turnovers[:12]) / 3
            vol_surge = recent_vol / older_vol if older_vol > 0 else 0
            
            recent_chg = ((closes[-1] - closes[-4]) / closes[-4] * 100) if closes[-4] > 0 else 0
            recent_bullish = sum(1 for i in range(-4, 0) if closes[i] > opens[i])
            
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
            
            tail_scores = []
            for k in klines[-4:]:
                o, h, l, c = float(k[1]), float(k[2]), float(k[3]), float(k[4])
                total = h - l
                lower_tail = min(o, c) - l
                if total > 0: tail_scores.append(lower_tail / total)
            if tail_scores and sum(tail_scores)/len(tail_scores) > 0.4: score += 1
            
            if score >= 5:
                results.append({
                    "sym": sym, "price": info["price"], "chg24": info["chg24"],
                    "turnover_m": info["turnover_m"], "funding": info["funding"],
                    "vol_surge": vol_surge, "recent_chg": recent_chg,
                    "bullish": recent_bullish, "score": score,
                })
        except:
            continue
    
    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[:5]
    
    # Slack 메시지 포맷
    lines = [f"*4H 전환 스캔* | {now.strftime('%H:%M')} (다음 봉: {next_c}시)"]
    lines.append(f"분석: {len(usdt)}개 | 기준 충족: {len(results)}개\n")
    
    if not top:
        lines.append("기준 충족 코인 없음 - 쉬어가는 타이밍")
    else:
        for i, r in enumerate(top, 1):
            tags = []
            if r["vol_surge"] > 2: tags.append("거래량폭증")
            if r["funding"] < -0.05: tags.append("숏과밀")
            if r["bullish"] >= 3: tags.append("연속양봉")
            if 1 <= r["recent_chg"] <= 5: tags.append("초기가속")
            tag_str = " ".join(f"`{t}`" for t in tags) if tags else ""
            
            sym_clean = r["sym"].replace("USDT", "")
            lines.append(
                f"*{i}. {sym_clean}* `{r['price']:.4f}` | "
                f"1h `{r['recent_chg']:+.1f}%` | "
                f"vol `{r['vol_surge']:.1f}x` | "
                f"fund `{r['funding']:+.3f}%` | "
                f"`{r['score']}/11`"
            )
            if tag_str:
                lines.append(f"  {tag_str}")
    
    msg = "\n".join(lines)
    
    with open("/Users/eomgyuhyeon/.openclaw/workspace/4h-scanner/slack_msg.txt", "w") as f:
        f.write(msg)
    
    with open("/Users/eomgyuhyeon/.openclaw/workspace/4h-scanner/last_scan.json", "w") as f:
        json.dump({"scan_time": now.isoformat(), "results": results[:10]}, f, indent=2, ensure_ascii=False)
    
    print(msg)
    return msg

if __name__ == "__main__":
    scan()
