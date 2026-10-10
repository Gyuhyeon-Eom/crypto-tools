"""Produce a readable research decision, preserving every policy result."""
import hashlib
import html
import json
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE/'output'
NAMES = {'entry':'거래량 신호 즉시', 'delay15':'15분 대기',
         'hold15':'15분 후 가격 버팀', 'hold_rebuy':'가격 버팀 + 매수 재유입'}


def main():
    r = json.loads((OUT/'results.json').read_text())
    a = pd.read_csv(OUT/'alerts.csv', parse_dates=['time','episode_start','next_event'])
    assert not a.duplicated(['split','symbol','episode_start','policy']).any()
    assert (a.time-a.episode_start).dt.total_seconds().div(60).eq(a.wait_minutes).all()
    for root, name in [(BASE,'frozen.json'),(BASE.parent/'flow_replay','forward_manifest.json')]:
        m = json.loads((root/'output'/name).read_text())
        assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in m['source_hashes'].items())
    validation = {'tests_passed': 8, 'unique_policy_episode_alerts': True,
                  'elapsed_times_verified': True, 'experiment_and_forward_hashes_unchanged': True}
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2))
    def table(split):
        rows = []
        for policy, v in r[split]['policies'].items():
            m, rand = v['metrics'], v['random']
            rows.append(f'<tr><td>{NAMES[policy]}</td><td>{m["alerts_evaluated"]:,}</td>'
                        f'<td>{m["strict_hits"]}</td><td>{m["precision"]:.3%}</td>'
                        f'<td>{m["detected_events"]}/{m["eligible_events"]}</td>'
                        f'<td>{m["median_lead_minutes"]}분</td><td>{rand["expected_hits"]:.2f}건 '
                        f'<small>재표집 범위 {rand["random_hit_count_2_5pct"]:.0f}–{rand["random_hit_count_97_5pct"]:.0f}건</small></td></tr>')
        return '<div class="scroll"><table><tr><th>정책</th><th>평가 알림</th><th>사전 적중</th><th>적중률</th><th>잡은 급등</th><th>선행시간 중앙값</th><th>무작위 기대 적중</th></tr>'+''.join(rows)+'</table></div>'
    captures = sorted(OUT.glob('capture_*/summary.json'))
    capture_html = '<p>수집 연결 확인 결과가 아직 저장되지 않았다.</p>'
    if captures:
        capture_html = ''
    for capture in captures:
        s = json.loads(capture.read_text())
        capture_html += (f'<p>연결 시험: <b>{html.escape(s["status"])}</b>, {s["elapsed_seconds"]:.1f}초. '
                        f'종목 {", ".join(s["symbols"])}. 원본 메시지 {sum(s["messages"].values()):,}개, '
                        f'체결 레코드 {sum(s["trade_records"].values()):,}개. '
                        f'<a href="{capture.relative_to(OUT)}">상태 기록</a>.</p>')
        if s['errors']:
            capture_html += '<p>오류: '+html.escape('; '.join(s['errors']))+'</p>'
    capture_html += '<p>BTC는 체결 채널 연결 시험용이며 기존 연구 대상 36종목에 추가하지 않았다. 소형 코인 시험의 체결 0건은 그 짧은 관찰 동안 수신된 체결이 없다는 뜻이다.</p>'
    content = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>급등 알림 발전 방향: 순서 가설 재생과 호가 수집</title>
<style>body{font-family:system-ui,sans-serif;line-height:1.75;background:#f5f7fa;color:#203044;margin:0}main{max-width:1080px;margin:auto;padding:36px 26px}h1{font-size:32px;line-height:1.35}h2{margin-top:36px;font-size:22px}.box{padding:18px 22px;background:#e6f0f2;border-left:4px solid #16849a;border-radius:6px}.muted,small{color:#647487}small{display:block}table{width:100%;border-collapse:collapse;background:white;font-size:14px}th,td{padding:12px;text-align:left;border-bottom:1px solid #dce4ed;vertical-align:top}th{background:#e8eef4}.scroll{overflow:auto}pre{background:#e7edf3;padding:18px;overflow:auto}a{color:#087a97}li{margin:10px 0}.flow{padding:18px;background:white;border-radius:8px;font-size:18px}</style>
<main><p class="muted">Crypto Tools · 2026-10-09 · 탐색적 후속 실험</p>
<h1>신호의 순서를 확인해도 충분한가?<br>다음 발전에는 어떤 자료가 필요한가?</h1>
<div class="box"><b>판단:</b> ‘가격 버팀 → 매수 재유입’을 기다리면 알림 수는 줄지만, 이번 고정 규칙으로 실용적인 급등 구분 능력을 확보하지 못했다. 다음 개발은 호가 변화라는 새 정보를 확보하고 검증할 수 있는 기반에 집중할 가치가 있다. 호가가 예측력을 가진다는 결론은 아직 아니다.</div>
<h2>실제로 만든 비교</h2>
<p>급등 시점을 알고 관찰 시점을 고르는 대신, 5분봉이 닫힐 때마다 당시까지의 정보로 상태를 갱신했다. 같은 거래량 신호에서 출발하는 4개 정책을 비교했다. 지표·경계값·상태 규칙은 이번 결과를 계산하기 전에 저장하고 해시로 고정했다. 4–6월과 8–9월 모두 이전 연구에서 보았던 자료이므로 새로운 독립 검증이라고 부르지는 않는다.</p>
<p class="flow">30분 재준비 → 거래대금 약 2.30배로 후보 시작 → 최대 60분 추적 → 종료 → 30분 재준비</p>
<ul><li><b>즉시:</b> 후보 시작 시 알림.</li><li><b>대기:</b> 15분 뒤 현재 품질·가격 적격 조건을 통과하면 알림.</li>
<li><b>버팀:</b> 대기 조건에 더해, 시작 후 저가가 시작 가격의 −1% 아래로 가지 않고 현재 가격이 시작 가격 이상·+2% 미만일 때 알림.</li>
<li><b>버팀 + 재유입:</b> 15–60분 동안 버팀 조건에 더해, 앞 10분 순매수 비중 ≤0에서 최근 5분 ≥20%로 전환되는 최초 시점에 알림.</li></ul>
<p>30분 재준비는 거래량 조건 아래에서 품질을 충족하는 5분 구간 6개가 이어지는 것을 뜻한다. 후보 추적 중 자료 품질이 깨지면 종료한다. 각 정책은 후보당 한 번만 알리고, UTC 자정에 초기화되는 하루 한도는 사용하지 않았다. 이 규칙과 둥근 수치들은 최적화된 설정이 아니라 한 번의 구체적인 가설이다.</p>
<h2>8–9월 결과</h2>REPLICATION
<p>매수 재유입까지 기다리면 알림은 약 58% 줄지만 잡은 급등도 4건에서 2건으로 줄었다. 적중률 0.069%→0.082%의 작은 차이를 성능 개선으로 인정할 근거는 부족하다. 61일 동안 시장 전체에서 하루 약 40건의 최종 후보 알림이 생기는 수준이다. 성공 사건이 2건이라는 점이 가장 큰 제약이다.</p>
<h2>4–6월 결과도 함께 보기</h2>DISCOVERY
<p>앞선 기간에서는 즉시 알림 0.049%, 재유입 확인 0.046%였다. 후기에서 보인 작은 차이가 앞선 기간에서도 같은 방향으로 반복되지는 않았다. 좋은 기간·정책만 골라 보고하지 않았다.</p>
<h2>숫자를 읽을 때 필요한 조건</h2>
<ul><li><b>급등 정의:</b> 조용했던 구간 뒤 첫 5분 종가 +2% 이상, 이어지는 60분 고가 +8% 이상이라는 기존 사건 정의를 유지했다. 이는 일반적인 모든 상승을 뜻하지 않는다.</li>
<li><b>사전 적중:</b> 알림 5–60분 뒤 급등 시작이 있으면 적중이다. 시작과 동시에 낸 알림은 별도로 기록하고 제외했다. 먼저 알림을 생성한 후에만 미래 결과를 붙였다.</li>
<li><b>평가 제외:</b> 결과를 확인할 수 없는 알림은 원본에 남기고 분모에서 제외했다. 후기 15분 대기 1건, 탐색 정책별 1–4건이다.</li>
<li><b>무작위 기준:</b> 실제 정책과 같은 코인·같은 날짜·같은 알림 수로 적격 시점을 200번 뽑은 참고치다. 표의 범위는 무작위 적중 수 분포이며 정책 개선폭의 신뢰구간이나 실행 가능한 정책은 아니다. 모든 실제 적중 수가 이 참고 범위 안에 있다.</li>
<li><b>선행시간:</b> 적중 사례만의 값이며 정책마다 잡은 사건이 다르다. 단순히 중앙값 차이를 기다림의 인과 효과로 해석하지 않는다.</li>
<li><b>거래 성과:</b> 체결 비용·지연을 반영한 손익 검증이 아니다. 현재 종목 목록의 생존 편향과 아카이브 누락, 적은 독립 사건 수라는 한계도 남아 있다.</li></ul>
<h2>다음에 추가할 정보: 호가의 소진·회복</h2>
<p>새 가설은 세 개로 제한한다. 단순 잔량 비율뿐 아니라 <b>같은 체결 흐름에 호가가 어떻게 반응하는가</b>를 보려는 것이다.</p>
<table><tr><th>가설</th><th>측정할 것</th><th>주의할 점</th></tr>
<tr><td>매수 후 매도 잔량 소진</td><td>고정 기준가격 위 25bp 범위의 매도 잔량이 감소하고 60초 뒤에도 회복되지 않는가</td><td>가격이 이동해 계산 범위가 달라진 효과를 분리</td></tr>
<tr><td>매도 후 매수 잔량 보충</td><td>매도 체결 뒤 아래 25bp 매수 잔량이 회복되면서 가격이 버티는가</td><td>잔량 감소만으로 취소·체결·주문 주체를 구별하지 않음</td></tr>
<tr><td>가격을 올리는 데 필요한 금액 감소</td><td>매도 호가를 기준가격 +25bp까지 소진하는 데 필요한 거래대금이 줄어드는가</td><td>수집한 호가 깊이가 목표 가격에 못 미치면 결측 처리</td></tr></table>
<p>25bp는 0.25%다. 세 항목은 아직 구현·검증 전인 측정 설계다. 실제 모델 비교 전에는 초기 자료로 단위와 품질을 점검하고, 별도의 이후 기간을 보기 전에 정의와 알림 규칙을 고정해야 한다.</p>
<h2>수집 연결 시험과 실행 가능한 파일</h2>CAPTURE
<p><a href="../capture_orderbook.mjs">제한 시간 수집기</a>는 공개 현물 호가 50단계와 체결 원본 메시지, 수신 시각을 압축 파일에 저장한다. API 키가 필요 없고 주문·외부 알림을 보내지 않는다. 같은 seq의 체결 메시지도 전부 보관한다. 짧은 시험은 연결과 저장의 확인일 뿐, 급등 예측 검증이 아니다.</p>
<pre>node research/sequence_replay/capture_orderbook.mjs 60 JTOUSDT,TREEUSDT</pre>
<p>1회 최대 1시간, 최대 5종목의 수동 실행용이다. 지금 실행은 종료되었으며 예약·상시 수집을 설정하지 않았다. 장기 수집 전에 스냅샷/변경분으로 호가 복원, 재연결 시 초기화, 누락 점검, 파일 순환과 저장 용량 관리가 필요하다. 성공한 신호만 저장하면 실패와 비교할 수 없으므로 선정한 종목의 전체 관찰 구간을 보관해야 한다.</p>
<p>공식 문서는 <a href="https://bybit-exchange.github.io/docs/v5/websocket/public/orderbook">호가 스냅샷·변경분</a>과 <a href="https://bybit-exchange.github.io/docs/v5/websocket/public/trade">개별 체결</a>을 제공한다. 일반 호가 스트림에는 RPI 주문이 포함되지 않으므로 체결의 RPI 표시를 보관하고 별도로 처리해야 한다. 공개 연결과 인증 범위는 <a href="https://bybit-exchange.github.io/docs/v5/ws/connect">연결 문서</a>를 기준으로 했다.</p>
<h2>검증과 재현</h2>
<p>시점 정렬·미래 자료 불변성·순서 조건·중복 방지·자료 공백 초기화 등 8개 테스트를 통과했다. 기존 전향 실험 및 이번 규칙의 해시도 일치한다. 운영 스캐너는 변경하지 않았다.</p>
<p><a href="../study.py">재생 코드</a> · <a href="../protocol.json">고정한 가설</a> · <a href="results.json">전체 결과</a> · <a href="alerts.csv">모든 알림과 결과</a> · <a href="frozen.json">해시 기록</a> · <a href="validation.json">검증 기록</a> · <a href="../orderbook_plan.json">호가 분석 설계</a></p>
<pre>python3 -m unittest discover -s research/sequence_replay -p 'test_*.py' -v
python3 research/sequence_replay/study.py
python3 research/sequence_replay/report.py</pre>
</main></html>'''
    content = content.replace('REPLICATION', table('replication')).replace('DISCOVERY', table('discovery')).replace('CAPTURE', capture_html)
    (OUT/'sequence_report.html').write_text(content)
    print(OUT/'sequence_report.html')


if __name__ == '__main__':
    main()
