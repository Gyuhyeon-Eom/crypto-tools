#!/usr/bin/env python3
"""Bybit에서 1000일치 일봉 데이터 수집"""
import json, time, urllib.request, sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "ml_data"
MIN_VOL = 500_000  # 최소 거래량 (현재 기준)

SKIP = {"USDCUSDT","UUSDT","XUSDUSDT","PAXGUSDT","DAIUSDT","FDUSDUSDT",
        "TUSDUSDT","BUSDUSDT","EURUSDT","USTCUSDT","USDTUSDT","USDDUSDT",
        "RLUSDUSDT","USD1USDT"}

def bybit_get(ep, params=None):
    url = "https://api.bybit.com" + ep
    if params: url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
    req = urllib.request.Request(url, headers={"User-Agent": "data-fetch/1"})
    for i in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read())
        except:
            if i == 2: raise
            time.sleep(1 + i)

# 1. 티커 목록
print("티커 목록 가져오는 중...")
tickers = bybit_get("/v5/market/tickers", {"category": "spot"}).get("result", {}).get("list", [])
alts = [t["symbol"] for t in tickers 
        if t["symbol"].endswith("USDT") and t["symbol"] not in SKIP
        and float(t.get("turnover24h", 0)) >= MIN_VOL]
print(f"  대상: {len(alts)}개 코인")

# 2. 각 코인별 1000일치 일봉
all_data = {}
for i, sym in enumerate(alts):
    try:
        data = bybit_get("/v5/market/kline", {
            "category": "spot", "symbol": sym, "interval": "D", "limit": "1000"
        })
        klines = list(reversed(data.get("result", {}).get("list", [])))
        if len(klines) < 60:
            continue
        all_data[sym] = {
            "o": [float(k[1]) for k in klines],
            "h": [float(k[2]) for k in klines],
            "l": [float(k[3]) for k in klines],
            "c": [float(k[4]) for k in klines],
            "v": [float(k[5]) for k in klines],
            "ts": [int(k[0]) for k in klines],
        }
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(alts)}] {sym}: {len(klines)}일치")
        time.sleep(0.05)
    except Exception as e:
        print(f"  [ERR] {sym}: {e}")
        continue

print(f"\n완료: {len(all_data)}개 코인")

# 데이터 길이 통계
lengths = [len(d["c"]) for d in all_data.values()]
print(f"  일수: min={min(lengths)} median={sorted(lengths)[len(lengths)//2]} max={max(lengths)}")

# 저장
out_path = DATA_DIR / "raw_data_extended.json"
with open(out_path, "w") as f:
    json.dump(all_data, f)
print(f"  저장: {out_path} ({out_path.stat().st_size / 1e6:.1f}MB)")
