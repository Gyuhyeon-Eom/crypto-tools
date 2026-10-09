#!/usr/bin/env python3
"""
Stage 2 — 실시간 거래량 스파이크 감시
======================================
Stage 1에서 뽑은 후보(stage1_watchlist.json)를 대상으로
15분~1시간봉 거래량 스파이크를 실시간 모니터링.

스파이크 감지 시 즉시 Slack 알림.

실행: 5분마다 (cron 또는 launchd)
"""

import json
import math
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
WATCHLIST_FILE = os.path.join(WORKSPACE, "stage1_watchlist.json")
ALERT_HISTORY_FILE = os.path.join(WORKSPACE, "stage2_alert_history.json")
STATE_FILE = os.path.join(WORKSPACE, "stage2_state.json")

# 알림 설정
SPIKE_THRESHOLD_15M = 5.0    # 15분봉 거래량이 평균의 5배 이상
SPIKE_THRESHOLD_1H = 4.0     # 1시간봉 거래량이 평균의 4배 이상
SPIKE_THRESHOLD_5M = 8.0     # 5분봉 거래량이 평균의 8배 이상
MIN_SCORE_FOR_WATCH = 8      # stage1 score 최소
COOLDOWN_MINUTES = 60        # 같은 코인 재알림 쿨다운 (분)
MAX_ALERTS_PER_RUN = 5       # 한 번에 최대 알림 수

# Slack webhook
SLACK_WEBHOOK_URL = ""
_env_path = os.path.expanduser("~/.openclaw/.env.slack-crypto")
if os.path.exists(_env_path):
    with open(_env_path) as f:
        for line in f:
            if line.startswith("SLACK_WEBHOOK_URL="):
                SLACK_WEBHOOK_URL = line.strip().split("=", 1)[1]


def api_get(url, retries=2):
    for i in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except:
            if i < retries:
                time.sleep(0.5)
    return None


def get_klines(symbol, interval, limit=50, category="linear"):
    url = f"https://api.bybit.com/v5/market/kline?category={category}&symbol={symbol}&interval={interval}&limit={limit}"
    data = api_get(url)
    if not data or data.get('retCode') != 0 or not data['result']['list']:
        return None
    raw = data['result']['list'][::-1]
    return [{
        'open': float(d[1]), 'high': float(d[2]), 'low': float(d[3]),
        'close': float(d[4]), 'volume': float(d[5]),
        'quote_volume': float(d[6]) if len(d) > 6 else float(d[5]) * float(d[4]),
        'timestamp': int(d[0])
    } for d in raw]


def get_current_ticker(symbol, category="linear"):
    url = f"https://api.bybit.com/v5/market/tickers?category={category}&symbol={symbol}"
    data = api_get(url)
    if not data or data.get('retCode') != 0 or not data['result']['list']:
        return None
    d = data['result']['list'][0]
    return {
        'price': float(d.get('lastPrice', 0)),
        'change_24h': float(d.get('price24hPcnt', 0)) * 100,
        'volume_24h': float(d.get('turnover24h', 0)),
        'high_24h': float(d.get('highPrice24h', 0)),
        'low_24h': float(d.get('lowPrice24h', 0)),
    }


def load_watchlist():
    try:
        with open(WATCHLIST_FILE) as f:
            data = json.load(f)
        return data.get('watchlist', [])
    except:
        return []


def load_alert_history():
    try:
        with open(ALERT_HISTORY_FILE) as f:
            return json.load(f)
    except:
        return {}


def save_alert_history(h):
    with open(ALERT_HISTORY_FILE, 'w') as f:
        json.dump(h, f, indent=2)


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except:
        return {}


def save_state(s):
    with open(STATE_FILE, 'w') as f:
        json.dump(s, f, indent=2, default=str)


def calc_obv(candles):
    if len(candles) < 2:
        return []
    o = [0]
    for i in range(1, len(candles)):
        if candles[i]['close'] > candles[i-1]['close']:
            o.append(o[-1] + candles[i]['volume'])
        elif candles[i]['close'] < candles[i-1]['close']:
            o.append(o[-1] - candles[i]['volume'])
        else:
            o.append(o[-1])
    return o


def analyze_realtime(symbol, stage1_data):
    """
    실시간 분석. 여러 시간프레임에서 거래량 스파이크 감지.
    
    Returns: dict with alert info, or None
    """
    has_futures = stage1_data.get('has_futures', True)
    cat = "linear" if has_futures else "spot"
    
    alerts = []
    spike_score = 0
    details = {}
    
    # ═══ 5분봉 분석 (초단기 스파이크) ═══
    c5m = get_klines(symbol, "5", 50, cat)
    if c5m and len(c5m) >= 20:
        vols_5m = [c['quote_volume'] for c in c5m]
        
        # 최근 3개 5분봉 vs 이전 평균
        avg_5m = sum(vols_5m[:-3]) / max(len(vols_5m) - 3, 1)
        recent_3 = sum(vols_5m[-3:]) / 3
        latest = vols_5m[-1]
        
        if avg_5m > 0:
            ratio_latest = latest / avg_5m
            ratio_recent = recent_3 / avg_5m
            details['vol_spike_5m'] = round(ratio_latest, 1)
            details['vol_spike_5m_avg3'] = round(ratio_recent, 1)
            
            if ratio_latest >= SPIKE_THRESHOLD_5M:
                spike_score += 5
                alerts.append(f"5분봉 거래량 {ratio_latest:.1f}x 폭발!")
            elif ratio_recent >= SPIKE_THRESHOLD_5M * 0.6:
                spike_score += 3
                alerts.append(f"5분봉 최근3개 평균 {ratio_recent:.1f}x")
        
        # 연속 양봉 + 거래량 증가 (모멘텀 시작)
        consec_up = 0
        vol_increasing = True
        for i in range(-1, max(-6, -len(c5m)), -1):
            if c5m[i]['close'] > c5m[i]['open']:
                consec_up += 1
                if i < -1 and c5m[i]['volume'] < c5m[i+1]['volume'] * 0.8:
                    vol_increasing = False
            else:
                break
        if consec_up >= 4 and vol_increasing:
            spike_score += 3
            alerts.append(f"5분 연속양봉{consec_up}개+거래량증가")
    
    time.sleep(0.1)
    
    # ═══ 15분봉 분석 ═══
    c15m = get_klines(symbol, "15", 50, cat)
    if c15m and len(c15m) >= 15:
        vols_15m = [c['quote_volume'] for c in c15m]
        
        avg_15m = sum(vols_15m[:-2]) / max(len(vols_15m) - 2, 1)
        latest_15m = vols_15m[-1]
        recent_2 = sum(vols_15m[-2:]) / 2
        
        if avg_15m > 0:
            ratio_15m = latest_15m / avg_15m
            ratio_15m_2 = recent_2 / avg_15m
            details['vol_spike_15m'] = round(ratio_15m, 1)
            
            if ratio_15m >= SPIKE_THRESHOLD_15M:
                spike_score += 5
                alerts.append(f"15분봉 거래량 {ratio_15m:.1f}x 폭발!")
            elif ratio_15m_2 >= SPIKE_THRESHOLD_15M * 0.7:
                spike_score += 3
                alerts.append(f"15분봉 최근2개 평균 {ratio_15m_2:.1f}x")
        
        # 15분봉 OBV 급등 (최근 6개 vs 이전 6개)
        obv_15m = calc_obv(c15m)
        if len(obv_15m) >= 12:
            obv_recent = obv_15m[-1] - obv_15m[-6]
            obv_prev = obv_15m[-7] - obv_15m[-12]
            if obv_recent > 0 and (obv_prev <= 0 or obv_recent > abs(obv_prev) * 3):
                spike_score += 2
                alerts.append("15분 OBV 급반전")
    
    time.sleep(0.1)
    
    # ═══ 1시간봉 분석 ═══
    c1h = get_klines(symbol, "60", 48, cat)
    if c1h and len(c1h) >= 12:
        vols_1h = [c['quote_volume'] for c in c1h]
        
        avg_1h = sum(vols_1h[:-2]) / max(len(vols_1h) - 2, 1)
        latest_1h = vols_1h[-1]
        
        if avg_1h > 0:
            ratio_1h = latest_1h / avg_1h
            details['vol_spike_1h'] = round(ratio_1h, 1)
            
            if ratio_1h >= SPIKE_THRESHOLD_1H:
                spike_score += 4
                alerts.append(f"1시간봉 거래량 {ratio_1h:.1f}x!")
            elif ratio_1h >= SPIKE_THRESHOLD_1H * 0.7:
                spike_score += 2
                alerts.append(f"1시간봉 거래량 {ratio_1h:.1f}x 상승")
        
        # 1시간봉 가격 액션
        if c1h[-1]['close'] > c1h[-1]['open']:
            pct = (c1h[-1]['close'] - c1h[-1]['open']) / c1h[-1]['open'] * 100
            if pct > 3 and ratio_1h >= 2:
                spike_score += 2
                alerts.append(f"1h 양봉 +{pct:.1f}% + 거래량")
    
    # ═══ 가격 브레이크아웃 감지 ═══
    if c15m and len(c15m) >= 30:
        closes_15m = [c['close'] for c in c15m]
        # 최근 20개 15분봉의 고점 대비 현재가
        recent_high = max(c['high'] for c in c15m[-30:-3])
        current = c15m[-1]['close']
        if current > recent_high:
            pct_above = (current - recent_high) / recent_high * 100
            spike_score += 3
            alerts.append(f"7.5h 고점 돌파! (+{pct_above:.1f}%)")
            details['breakout_pct'] = round(pct_above, 2)
    
    # ═══ 현재 호가/가격 정보 ═══
    ticker = get_current_ticker(symbol, cat)
    if ticker:
        details['current_price'] = ticker['price']
        details['change_24h'] = round(ticker['change_24h'], 2)
        details['volume_24h_m'] = round(ticker['volume_24h'] / 1e6, 2)
    
    if spike_score < 5:
        return None
    
    return {
        'symbol': symbol,
        'spike_score': spike_score,
        'stage1_score': stage1_data.get('score', 0),
        'combined_score': spike_score + stage1_data.get('score', 0),
        'alerts': alerts,
        'details': details,
        'stage1_signals': stage1_data.get('signals', []),
        'price': details.get('current_price', stage1_data.get('price', 0)),
    }


def send_slack_alert(alert_data):
    """Slack webhook으로 즉시 알림"""
    if not SLACK_WEBHOOK_URL:
        print("  [!] Slack webhook 미설정")
        return False
    
    sym = alert_data['symbol'].replace('USDT', '')
    combined = alert_data['combined_score']
    spike = alert_data['spike_score']
    s1 = alert_data['stage1_score']
    price = alert_data['price']
    
    # 긴급도 이모지
    if combined >= 25:
        urgency = ":rotating_light::rotating_light::rotating_light:"
    elif combined >= 20:
        urgency = ":rotating_light::rotating_light:"
    elif combined >= 15:
        urgency = ":rotating_light:"
    else:
        urgency = ":warning:"
    
    # 실시간 스파이크 정보
    spike_lines = "\n".join(f"  > {a}" for a in alert_data['alerts'])
    
    # Stage1 전조 요약 (상위 3개)
    s1_sigs = alert_data.get('stage1_signals', [])[:3]
    s1_lines = "\n".join(f"  > {s}" for s in s1_sigs)
    
    details = alert_data.get('details', {})
    chg_24h = details.get('change_24h', 0)
    vol_24h = details.get('volume_24h_m', 0)
    
    text = f"""{urgency} *{sym}* 급등 전조 감지!

*종합 점수: {combined}* (전조 {s1} + 실시간 {spike})
*현재가:* {price} ({chg_24h:+.1f}% 24h)
*24h 거래량:* ${vol_24h:.1f}M

*실시간 스파이크:*
{spike_lines}

*Stage1 전조 시그널:*
{s1_lines}

_Bybit: https://www.bybit.com/trade/usdt/{sym}_"""
    
    payload = json.dumps({"text": text}).encode()
    try:
        req = urllib.request.Request(
            SLACK_WEBHOOK_URL,
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"  [!] Slack 알림 실패: {e}")
        return False


def main():
    now = datetime.now(KST)
    now_ts = now.timestamp()
    
    print(f"{'='*60}")
    print(f"  STAGE 2 — 실시간 거래량 스파이크 감시")
    print(f"  {now.strftime('%Y-%m-%d %H:%M:%S KST')}")
    print(f"{'='*60}")
    
    # Watchlist 로드
    watchlist = load_watchlist()
    if not watchlist:
        print("\n  [!] watchlist 비어있음. Stage 1 먼저 실행 필요.")
        return
    
    # score 기준 필터
    candidates = [w for w in watchlist if w.get('score', 0) >= MIN_SCORE_FOR_WATCH]
    print(f"\n  watchlist: {len(watchlist)}개 중 {len(candidates)}개 감시 대상 (score >= {MIN_SCORE_FOR_WATCH})")
    
    # 알림 히스토리 (쿨다운)
    history = load_alert_history()
    state = load_state()
    
    # 스캔
    results = []
    for i, cand in enumerate(candidates):
        sym = cand['symbol']
        
        # 쿨다운 체크
        last_alert_ts = history.get(sym, 0)
        if (now_ts - last_alert_ts) < COOLDOWN_MINUTES * 60:
            remaining = int((COOLDOWN_MINUTES * 60 - (now_ts - last_alert_ts)) / 60)
            print(f"  [{i+1}/{len(candidates)}] {sym:<12} — 쿨다운 {remaining}분 남음")
            continue
        
        print(f"  [{i+1}/{len(candidates)}] {sym:<12} (stage1={cand['score']}) ... ", end="", flush=True)
        
        try:
            result = analyze_realtime(sym, cand)
            if result:
                results.append(result)
                print(f"SPIKE! (spike={result['spike_score']}, combined={result['combined_score']})")
            else:
                print("정상")
        except Exception as e:
            print(f"에러: {e}")
        
        time.sleep(0.2)
    
    # 결과 정렬
    results.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # 알림 전송
    alerts_sent = 0
    if results:
        print(f"\n{'='*60}")
        print(f"  감지된 스파이크: {len(results)}개")
        print(f"{'='*60}")
        
        for r in results:
            sym = r['symbol'].replace('USDT', '')
            print(f"\n  {sym} — 종합 {r['combined_score']} (전조 {r['stage1_score']} + 실시간 {r['spike_score']})")
            for a in r['alerts']:
                print(f"    > {a}")
            
            # 알림 전송 (최대 제한)
            if alerts_sent < MAX_ALERTS_PER_RUN:
                if send_slack_alert(r):
                    print(f"    -> Slack 알림 전송 완료!")
                    history[r['symbol']] = now_ts
                    alerts_sent += 1
                else:
                    print(f"    -> Slack 알림 실패")
    else:
        print("\n  스파이크 없음. 정상 감시 중.")
    
    # 히스토리/상태 저장
    # 24시간 지난 히스토리 정리
    history = {k: v for k, v in history.items() if (now_ts - v) < 86400}
    save_alert_history(history)
    
    state['last_run'] = now.isoformat()
    state['last_results'] = len(results)
    state['candidates_watched'] = len(candidates)
    save_state(state)
    
    print(f"\n  완료. 알림 {alerts_sent}건 전송.")


if __name__ == "__main__":
    main()
