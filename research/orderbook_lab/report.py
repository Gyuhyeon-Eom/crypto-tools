"""Build the Korean smoke-test report from actual capture and replay audit files."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import gzip
import html
import json
import statistics
import sys

run = Path(sys.argv[1]).resolve()
summary = json.loads((run / 'summary.json').read_text())
audit = json.loads((run / 'audit.json').read_text())
rows = [json.loads(line) for line in gzip.open(run / 'features.jsonl.gz', 'rt')]
validation = json.loads((run.parent / 'validation.json').read_text())
esc = html.escape

def table(headers, data):
    return '<table><thead><tr>' + ''.join(f'<th>{esc(str(x))}</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{esc(str(x))}</td>' for x in row) + '</tr>' for row in data) + '</tbody></table>'

def stamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S KST')

def median(symbol, field, percent=False):
    values = [r[field] for r in rows if r['symbol'] == symbol and r.get(field) is not None]
    if not values:
        return '산출 불가'
    value = statistics.median(values)
    return f'{value * 100:+.2f}%' if percent else f'{value:,.2f}'

health = []
metrics = []
for symbol in summary['symbols']:
    own = [r for r in rows if r['symbol'] == symbol]
    count = lambda status: sum(r['status'] == status for r in own)
    health.append([symbol, summary['raw_messages'].get(f'orderbook.1000.{symbol}', 0), summary['raw_messages'].get(f'publicTrade.{symbol}', 0), count('ready'), count('warmup'), count('partial_depth'), count('no_snapshot') + count('stale_book_or_sample_gap')])
    metrics.append([symbol, median(symbol, 'spread_bps'), median(symbol, 'ask_depth_change', True), median(symbol, 'bid_recovery', True), median(symbol, 'ask_cost_change', True), median(symbol, 'trade_count_60s')])

checks = [
    ['수집 종료 상태', summary['status']],
    ['연결 / 재연결 횟수', f"{summary['connections']} / {summary['reconnects']}"],
    ['원본 청크 / 원본 레코드', f"{audit['raw_chunks']} / {audit['raw_records']:,}"],
    ['중복 제거 후 수신 체결 / 제거한 중복', f"{summary['trade_records']:,} / {summary['duplicate_trades']:,}"],
    ['재생 검증', '통과' if audit['passed'] else '실패'],
    ['검증한 지표 행 / 불일치', f"{audit['replayed_rows']:,} / {audit['mismatched_rows']:,}"],
    ['압축 파일 합계', f"{summary['total_disk_bytes'] / 1024:,.1f} KiB"],
    ['수집 오류 기록', json.dumps(summary['errors'], ensure_ascii=False)],
]
observations = []
for symbol in summary['symbols']:
    own = [r for r in rows if r['symbol'] == symbol]
    stale = sum(r['status'] == 'stale_book_or_sample_gap' for r in own)
    if stale:
        age = max(r.get('book_age_ms', 0) for r in own) / 1000
        observations.append(f'{symbol}: 최대 호가 갱신 대기 {age:.2f}초, 신선도·간격 규칙으로 {stale}개 행 제외. 실제 누락인지 단순한 무변동인지는 이 결과로 판별할 수 없습니다.')
    if not summary['raw_messages'].get(f'publicTrade.{symbol}', 0):
        observations.append(f'{symbol}: 이 실행에서 체결 메시지를 받지 못했습니다. 체결 이후의 호가 회복 가설을 검증할 사례가 없습니다.')
output = run.parent / 'orderbook_report.html'
content = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>급등 전조 연구 — 호가 수집 검증</title>
<style>body{{font:16px/1.75 system-ui,sans-serif;color:#182238;background:#f4f6fa;margin:0}}main{{max-width:1100px;margin:36px auto;padding:30px;background:white;border-radius:16px}}h1{{font-size:30px;line-height:1.35}}h2{{font-size:22px;margin-top:34px}}.callout{{background:#ecf3ff;border-left:5px solid #2464c6;padding:18px}}.muted{{color:#566477}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{text-align:left;border-bottom:1px solid #dce2ea;padding:10px}}th{{background:#f1f5f9}}code,pre{{background:#f1f5f9;padding:4px;overflow-wrap:anywhere;white-space:pre-wrap}}a{{color:#1658ba}}.scroll{{overflow:auto}}</style><main>
<p class="muted">급등 전조 연구 · 측정 파이프라인 검증 · {stamp(summary['started_at'])}</p>
<h1>가격·거래량 외에, 매수·매도 호가의 변화를 측정한다</h1>
<div class="callout"><strong>이번 단계의 결과는 데이터 수집·복원 검증입니다.</strong> 적중률 개선이나 80% 달성은 아직 검증하지 않았습니다. 짧은 실시간 표본에는 급등/비급등 정답을 붙이지 않았고, 예측 알림도 만들지 않았습니다.</div>
<p>이전 연구에서 사전에 선택한 가격·거래량 모델은 평가 기간에 755건 중 23건을 맞혔습니다(3.05%). 이번에는 같은 정보의 임계값을 다시 조정하는 대신, 호가와 체결 흐름을 함께 보며 새 정보가 있는지 검증할 준비를 했습니다.</p>
<h2>검증할 세 가지 가설</h2>
{table(['가설', '이번에 구현한 측정', '해석할 때 함께 볼 것'], [
['매수 체결 뒤 매도 물량 감소가 지속된다', '60초 매도 호가 금액 변화 + 마지막 10초 최대 잔량 비율', '일반 매수 체결량, 중간가격 변화, 고정 가격 구간 이탈'],
['매도 체결에도 매수 대기 물량이 회복된다', '(현재 매수 호가 금액 − 창 안 최소 금액) / 시작 금액', '일반 매도 체결량, 가격 하락 정도; 1초 사이의 최저점은 놓칠 수 있음'],
['같은 목표 가격까지 표시된 매도 물량이 줄어든다', '창 시작 중간가격 + 0.25%까지의 매도 호가 금액 변화', '이미 목표를 넘겼거나 깊이가 부족하면 결측; 실행 비용 예측이 아님']])}
<p>세 지표 모두 60초 창 시작 시점의 중간가격을 기준으로 계산합니다. 호가의 추가·취소·체결 원인을 이 자료만으로 분리할 수는 없습니다. 큰 변화만으로 매집이나 급등 전조라고 해석하지 않습니다.</p>
<h2>실시간 시험 결과</h2>
<p>{stamp(summary['started_at'])}부터 {stamp(summary['finished_at'])}까지 약 {summary['elapsed_seconds']:.1f}초. Bybit 현물 4종목의 공개 호가 1,000단계와 체결 스트림을 수신했습니다. BTC는 비교용이며, 이 종목 선택은 시장 전체를 대표하지 않습니다.</p>
<div class="scroll">{table(['항목', '결과'], checks)}</div>
<p>아래 ready는 양쪽 고정 가격 구간을 모두 관측하고 과거 60초의 신선도 검사를 통과했다는 뜻입니다. <strong>급등 신호 또는 예측 성공을 뜻하지 않습니다.</strong> 체결 메시지 하나에는 여러 체결이 들어갈 수 있습니다.</p>
<div class="scroll">{table(['종목', '호가 메시지', '체결 메시지', 'ready 행', '준비 행', '깊이 부족 행', '미수신·신선도 탈락 행'], health)}</div>
<ul>{''.join('<li>' + esc(x) + '</li>' for x in observations)}</ul>
<h2>측정값 확인</h2>
<p>다음은 유효하게 계산된 값의 중앙값입니다. 같은 60초 창을 1초씩 옮겨 측정했으므로 행끼리 강하게 겹칩니다. 행 수를 독립 사건 수로 세거나, 이 표로 가설의 예측력을 판단하면 안 됩니다.</p>
<div class="scroll">{table(['종목', '스프레드(bp)', '매도 잔량 변화', '매수 잔량 회복', '목표까지 매도 금액 변화', '창 안 체결 수'], metrics)}</div>
<h2>검증 범위와 남은 한계</h2>
<p>테스트 결과: 통과 {validation['observed_test_result']['passed']}개, 실패 {validation['observed_test_result']['failed']}개. 단위 테스트는 스냅샷 교체, 호가 삽입·수정·삭제, 역순 업데이트 거부, 동일 체결 ID 중복 제거, RPI 분리, 깊이 부족, 준비 시간, 수집 간격 및 상태 초기화를 확인합니다. 원본 재생 검증은 저장된 메시지로 계산한 모든 지표 행과 실제 저장 행을 대조하며, 실행 당시 소스 해시와 메시지 집계도 확인합니다. 같은 엔진을 재사용하는 재생 검증은 수식의 독립 검증이나 거래소 데이터 완전성 보증은 아닙니다.</p>
<p>연결 장애 또는 잘못된 호가를 발견하면 상태를 버리고 재연결한 뒤 다시 준비 시간을 둡니다. 이번 실시간 시험에서 장애가 실제로 발생하지 않았다면 재연결 경로의 실환경 동작을 검증했다고 볼 수 없습니다. 업데이트 번호의 점프만으로 메시지 누락을 단정하지 않습니다. 30초 이상 호가 갱신이 없으면 보수적으로 제외하며, 실제로 호가가 변하지 않았던 구간도 제외될 수 있습니다.</p>
<p>가격 구간을 1,000단계 호가가 포함하지 못하면 관련 지표를 결측으로 남깁니다. 공개 호가는 RPI 주문을 제외하므로 RPI 체결량은 별도로 저장합니다. 지표는 수신 시간 기준이며, 체결소 타임스탬프도 원본에 보관합니다. 숨겨진 유동성이나 순간적인 주문 재충전까지 측정한 것은 아닙니다.</p>
<h2>다음 예측력 검증의 조건</h2>
<ol><li>수집 종목과 기간, 학습·보정·최종 평가 경계를 먼저 고정합니다. 성공 사례만 저장하지 않고 연속된 비급등 구간까지 보관합니다. 현재 3분 시험은 이 자료를 대체하지 못합니다.</li>
<li>기존 급등 정의(다음 1시간 종가 +5%, 이후 24시간 고가 +15%)와 1~6시간 사전 예고 조건을 유지합니다. 지표 시점 이후 최대 30시간의 정답 데이터가 성숙한 뒤 연결합니다.</li>
<li>60초 측정치를 과거 1·5·15·60분 문맥으로 요약하고, 기존 모델과 새 지표를 더한 모델을 같은 조건에서 비교합니다. 알림 수, 포착률, 선행 시간과 주별 신뢰구간을 함께 보고 80% 여부를 판단합니다.</li>
<li>최종 평가를 열기 전에 필요한 표본 수와 최소 알림 수를 고정합니다. 몇 건만 맞힌 결과나 학습에 쓴 기간의 성능으로 80% 달성을 선언하지 않습니다.</li></ol>
<p>이번 실행은 종료되었습니다. 상시 수집·자동 실행·실거래·알림 전송은 설정하지 않았으며 운영 스캐너도 변경하지 않았습니다.</p>
<h2>재현 및 파일</h2>
<pre>node --test research/orderbook_lab/engine.test.mjs research/orderbook_lab/audit.test.mjs
node research/orderbook_lab/capture.mjs 180 JTOUSDT,TREEUSDT,WLDUSDT,BTCUSDT
node research/orderbook_lab/audit.mjs {esc(str(run))}
python3 research/orderbook_lab/report.py {esc(str(run))}</pre>
<p><a href="{esc(run.name)}/summary.json">실행 요약</a> · <a href="{esc(run.name)}/audit.json">재생 검증 결과</a> · <a href="../protocol.json">측정 규약</a> · <a href="../../precision_frontier/output/precision_report.html">이전 정확도 검증</a></p>
<p class="muted">공식 데이터 규격: <a href="https://bybit-exchange.github.io/docs/v5/websocket/public/orderbook">Bybit Orderbook</a> · <a href="https://bybit-exchange.github.io/docs/v5/websocket/public/trade">Bybit Public Trade</a> · <a href="https://bybit-exchange.github.io/docs/v5/ws/connect">연결 및 heartbeat</a></p>
</main></html>'''
output.write_text(content)
print(output)
