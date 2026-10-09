#!/bin/bash
# v4 봇 알림 체크 -> openclaw로 Slack 전송
cd /Users/eomgyuhyeon/.openclaw/workspace/crypto-paper-trading

OUTPUT=$(python3 slack_notifier.py 2>/dev/null)

if [ -n "$OUTPUT" ]; then
    # openclaw send로 Slack에 전달
    openclaw send "$OUTPUT" 2>/dev/null
fi
