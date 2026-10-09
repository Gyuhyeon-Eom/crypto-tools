#!/usr/bin/env python3
"""BTC 현재가 + 24h 변동률 조회"""
import urllib.request
import json
import sys

def get_btc_price():
    url = "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
    
    price = float(data["lastPrice"])
    change_pct = float(data["priceChangePercent"])
    high = float(data["highPrice"])
    low = float(data["lowPrice"])
    volume = float(data["quoteVolume"]) / 1e9  # billions
    
    arrow = "+" if change_pct >= 0 else ""
    print(f"BTC ${price:,.0f} ({arrow}{change_pct:.1f}%) | 24h H ${high:,.0f} L ${low:,.0f} | Vol ${volume:.1f}B")

if __name__ == "__main__":
    get_btc_price()
