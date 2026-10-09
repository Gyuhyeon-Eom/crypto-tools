#!/usr/bin/env python3
"""
급등 전조 자동 알림 - 1시간마다 실행
조건 충족 시에만 Slack 알림. 억지 추천 X.

필터:
1. 소형캡만 (OI $300K~$30M)
2. 이미 피뢰침 있던 건 제외 (24h +15% 이상 이미 오른 것)
3. 전조점수 12점 이상만 알림
4. 이전에 알림 보낸 건 중복 알림 X (쿨다운 6시간)
"""

import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone, timedelta

# pump_scanner_v2의 분석 함수 재사용
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pump_scanner_v2 import get_small_cap_tickers, analyze_pump_signals

KST = timezone(timedelta(hours=9))
ALERT_THRESHOLD = 12        # 이 점수 이상만 알림
ALREADY_PUMPED_PCT = 15.0   # 24h +15% 이상은 이미 피뢰침
COOLDOWN_HOURS = 6          # 같은 코인 재알림 쿨다운
ALERT_HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pump_alert_history.json")

SLACK_WEBHOOK_URL = ""
_env_path = os.path.expanduser("~/.openclaw/.env.slack-crypto")
if os.path.exists(_env_path):
    with open(_env_path) as f:
        for line in f:
            if line.startswith("SLACK_WEBHOOK_URL="):
                SLACK_WEBHOOK_URL = line.strip().split("=", 1)[1]


def load_history():
    try:
        with open(ALERT_HISTORY_FILE) as f:
            return json.load(f)
    except:
        return {}


def save_history(h):
    with open(ALERT_HISTORY_FILE, "w") as f:
        json.dump(h, f)


def send_slack(msg):
    if not SLACK_WEBHOOK_URL:
        print(f"[NO SLACK]\n{msg}")
        return
    payload = json.dumps({"text": msg}).encode("utf-8")
    req = urllib.request.Request(SLACK_WEBHOOK_URL, data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        urllib.request.urlopen(req, timeout=10)
        print("[OK] Slack 알림 전송")
    except Exception as e:
        print(f"[ERR] Slack: {e}")


def main():
    now = datetime.now(KST)
    now_ts = time.time()
    print(f"[{now.strftime('%H:%M')}] 급등 전조 스캔 시작...")

    tickers = get_small_cap_tickers()
    if not tickers:
        print("  티커 조회 실패")
        return

    print(f"  {len(tickers)}개 소형캡 스캔 중...")

    history = load_history()
    alerts = []

    for i, t in enumerate(tickers):
        # 필터1: 이미 피뢰침
        if t['change_24h'] > ALREADY_PUMPED_PCT:
            continue

        result = analyze_pump_signals(t['symbol'], t)
        if not result:
            continue

        # 필터2: 점수 미달
        if result['score'] < ALERT_THRESHOLD:
            continue

        # 필터3: 쿨다운
        sym = result['symbol']
        last_alert = history.get(sym, 0)
        if now_ts - last_alert < COOLDOWN_HOURS * 3600:
            print(f"  {sym} 점수 {result['score']} - 쿨다운 중 (스킵)")
            continue

        alerts.append(result)
        time.sleep(0.15)

    if not alerts:
        print(f"  알림 조건 충족 종목 없음 (점수 {ALERT_THRESHOLD}+ & 피뢰침 제외)")
        return

    alerts.sort(key=lambda x: x['score'], reverse=True)

    # 상위 3개만 알림
    top = alerts[:3]

    lines = []
    lines.append(f"*급등 전조 감지* ({now.strftime('%m/%d %H:%M')})")
    lines.append("")

    for s in top:
        sym = s['symbol'].replace('USDT', '')
        lines.append(f"*{sym}* (전조점수: {s['score']})")
        lines.append(f"  가격: {s['price']} | 24h: {s['change_24h']:+.1f}% | 볼륨: ${s['vol_24h_m']:.1f}M | OI: ${s['oi_m']:.1f}M")
        lines.append(f"  RSI: 1d={s['rsi_1d']:.0f} 4h={s['rsi_4h']:.0f} | 펀딩: {s['funding_rate']:+.4f}%")

        # 핵심 시그널만 (최대 4개)
        for sig in s['signals'][:4]:
            lines.append(f"  - {sig}")
        lines.append("")

        # 히스토리 업데이트
        history[s['symbol']] = now_ts

    lines.append(f"_점수 {ALERT_THRESHOLD}+ / 24h +{ALREADY_PUMPED_PCT:.0f}% 이상 제외 / 소형캡(OI<$30M)_")

    msg = "\n".join(lines)
    send_slack(msg)
    save_history(history)

    # 결과 저장
    output = {
        'scan_time': now.isoformat(),
        'alerts_sent': len(top),
        'total_qualified': len(alerts),
        'alerts': [a for a in top],
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pump_alert_latest.json"), 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
