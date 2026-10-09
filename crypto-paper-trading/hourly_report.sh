#!/bin/bash
# 매시간 페이퍼 트레이딩 리포트를 파일에 기록
# cron에서 호출 -> heartbeat에서 읽어서 Slack 전송

cd /Users/eomgyuhyeon/.openclaw/workspace/crypto-paper-trading

# DONE 파일 있으면 종료
if [ -f DONE ]; then
    echo "DONE" > latest_report.txt
    exit 0
fi

# 봇 살아있는지 확인
BOT_PID=$(pgrep -f "python3 bot.py")
if [ -z "$BOT_PID" ]; then
    echo "BOT_DEAD" > latest_report.txt
    exit 1
fi

# state.json에서 리포트 생성
python3 -c "
import json, urllib.request
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))
now = datetime.now(KST)

with open('state.json') as f:
    s = json.load(f)

def get_price(sym):
    try:
        req = urllib.request.Request(f'https://fapi.binance.com/fapi/v1/ticker/price?symbol={sym}', headers={'User-Agent':'M'})
        with urllib.request.urlopen(req, timeout=5) as r:
            return float(json.loads(r.read())['price'])
    except: return None

bal = s['balance']
unrealized = 0
lines = []
for p in s['positions']:
    cp = get_price(p['symbol'])
    if not cp: continue
    if p['side']=='LONG': pnl = (cp-p['entry_price'])/p['entry_price']*100
    else: pnl = (p['entry_price']-cp)/p['entry_price']*100
    pu = p['size_usdt']*p['leverage']*pnl/100
    unrealized += pu
    e = '+' if pnl > 0 else ''
    lines.append(f'{p[\"side\"]} {p[\"symbol\"].replace(\"USDT\",\"\")} | {p[\"entry_price\"]} -> {cp} | {e}{pnl:.2f}% ({e}{pu:.0f} USDT)')

tv = bal + sum(p['size_usdt'] for p in s['positions']) + unrealized
tp = tv - s['initial_balance']
tpp = tp/s['initial_balance']*100

wins = [t for t in s['closed_trades'] if t['net_pnl']>0]
losses = [t for t in s['closed_trades'] if t['net_pnl']<=0]
tt = len(s['closed_trades'])
wr = len(wins)/tt*100 if tt else 0

report = f'[{now.strftime(\"%m/%d %H:%M\")} 리포트]\\n'
report += f'평가금: {tv:.0f} USDT ({tv*1390:,.0f}원)\\n'
report += f'손익: {tp:+.0f} USDT ({tpp:+.1f}%) | {tp*1390:+,.0f}원\\n'
report += f'거래: {tt}건 (승{len(wins)}/패{len(losses)}) 승률 {wr:.0f}%\\n'
report += f'\\n보유 ({len(s[\"positions\"])}개):\\n'
for l in lines: report += f'  {l}\\n'
if not lines: report += '  (없음)\\n'

recent = s['closed_trades'][-3:]
if recent:
    report += '\\n최근 거래:\\n'
    for t in recent:
        e = 'W' if t['net_pnl']>0 else 'L'
        report += f'  [{e}] {t[\"side\"]} {t[\"symbol\"].replace(\"USDT\",\"\")} | {t[\"net_pnl\"]:+.0f} USDT ({t[\"pnl_pct\"]:+.1f}%) | {t[\"close_reason\"]}\\n'

with open('latest_report.txt','w') as f: f.write(report)
print(report)
" 2>/dev/null

echo "$(date '+%Y-%m-%d %H:%M:%S')" > last_report_time.txt
