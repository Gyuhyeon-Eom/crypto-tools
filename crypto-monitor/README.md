# 크립토 시황 모니터

바이낸스 공개 API로 시장 데이터 수집 → Slack으로 알림 전송

## 기능
- 4시간마다 정기 시황 리포트 (BTC/ETH + 알트코인 상승/하락/거래량)
- 5분마다 급등/급락 감지 (24h +-5% 이상, 5분내 +-3% 이상)
- Fear & Greed Index 포함
- 외부 라이브러리 없이 순수 Python3 (urllib만 사용)

## 사용법

```bash
# 테스트 (1회 실행)
python3 market_monitor.py --once

# 실행 (Slack 없이 stdout)
python3 market_monitor.py

# Slack Webhook 연결해서 실행
SLACK_WEBHOOK_URL="https://hooks.slack.com/services/XXX/YYY/ZZZ" python3 market_monitor.py

# 백그라운드 실행
nohup python3 market_monitor.py > monitor.log 2>&1 &
```

## 설정 변경
`market_monitor.py` 상단의 설정 섹션에서:
- `WATCHLIST` - 모니터링 코인 추가/제거
- `ALERT_PUMP_PCT` / `ALERT_DUMP_PCT` - 급등/급락 기준 (기본 +-5%)
- `REPORT_INTERVAL` - 정기 리포트 간격 (기본 4시간)
- `CHECK_INTERVAL` - 체크 간격 (기본 5분)

## LaunchAgent (맥에서 자동 실행)
`com.crypto.monitor.plist` 파일을 `~/Library/LaunchAgents/`에 복사 후:
```bash
launchctl load ~/Library/LaunchAgents/com.crypto.monitor.plist
```
