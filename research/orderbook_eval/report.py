"""Render readiness and locked comparison design; never invent accuracy."""
from pathlib import Path
from html import escape
import json

BASE=Path(__file__).resolve().parent
OUT=BASE/'output'
r=json.loads((OUT/'readiness.json').read_text())
p=json.loads((OUT/'prices_manifest.json').read_text())
v=json.loads((OUT/'implementation_manifest.json').read_text())

def table(head,rows):
    return '<table><thead><tr>'+''.join(f'<th>{escape(str(x))}</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{escape(str(x))}</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'

checks=[]
for x in r['capture_end_window_check']:
    checks.append([x['symbol'],f"{x.get('coverage_1m',0):.1%}",f"{x['coverage_5m']:.1%}" if 'coverage_5m' in x else '앞 단계에서 탈락',x['reason']])
html=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>호가 지표 추가 효과 검증 준비</title>
<style>body{{font:16px/1.75 system-ui,sans-serif;color:#172337;background:#f4f6fa;margin:0}}main{{max-width:1060px;padding:32px;margin:28px auto;background:white;border-radius:16px}}h1{{font-size:30px;line-height:1.4}}h2{{font-size:22px;margin-top:32px}}.notice{{padding:18px;background:#eef3fc;border-left:5px solid #2c65b8}}.muted{{color:#59687a}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{border-bottom:1px solid #dae1e9;text-align:left;padding:10px}}th{{background:#f3f6fa}}a{{color:#1a5bb2}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;padding:16px;background:#f4f6fa}}.scroll{{overflow:auto}}</style><main>
<p class="muted">급등 전조 연구 · 2026-10-10 · 3종목 입력 시험 / 본 검증은 기존 114종목</p>
<h1>새 호가 지표가 기존 모델에 정보를 더하는가?</h1>
<div class="notice"><strong>검증 코드를 구현하고 입력 시험을 마쳤습니다. 실제 예측력 비교는 아직 시작하지 않았습니다.</strong><br>현재 비교 가능한 표본은 {r['mature_paired_candidates']}건입니다. 적중률은 0%가 아니라 미산출입니다. 연속 호가 데이터와 30시간 이후 정답이 필요합니다.</div>
<h2>이번에 구현한 것</h2>
<ul><li>초 단위 호가 지표를 과거 1·5·15·60분으로 요약해 정시 가격·거래량 문맥과 연결합니다. 수신 이후 자료는 과거 시점의 지표에 포함하지 않습니다.</li>
<li>기존 급등 정의와 1~6시간 선행 조건을 그대로 사용합니다. 정답 미성숙과 가격 누락은 실패로 바꾸지 않고 별도 상태로 남깁니다.</li>
<li>학습, 보정, 최종 평가를 분리하고 평가 시작 전에 모델과 임계값을 저장합니다. 소스 변경과 평가 후 학습을 거부합니다.</li>
<li>기존 모델과 호가 추가 모델을 같은 종목·시간에서 비교하고, 호가 누락으로 잃은 후보 수를 별도로 셉니다.</li></ul>
<h2>실제 입력 검사</h2>
<p>기존 호가 시험은 11:42:50~11:45:50 KST의 약 3분입니다. 새로 닫힌 시간봉 {p['new_rows']}개를 받아 기존 이력과 합쳤으며, 기존 원본 파일은 보존했습니다. 아래 검사는 {escape(r['asof'])} 기준, 3종목의 27개 정시 시점에 대한 입력 점검입니다.</p>
{table(['항목','결과'],[['정시 관측 행',r['hourly_observations']],['기존 후보 조건 충족',r['base_candidates']],['호가 요건 충족',r['micro_eligible']],['정답까지 확정된 비교 후보',r['mature_paired_candidates']],['정답 대기 행',r['label_status'].get('pending_time',0)],['예측 적중률','미산출']])}
<p>점검 시점의 마지막 닫힌 시간봉은 11:00 KST입니다. 호가는 그 이후부터 수집했으므로, 이전 정시 관측에 소급해서 붙이지 않았습니다. 별도로 수집 종료 시각에서 창 길이를 검사해도 다음처럼 부족합니다.</p>
{table(['종목','1분 유효 수집률','5분 유효 수집률','제외 사유'],checks)}
<p class="muted">유효 수집률은 1초 구간 중 지표가 유효한 비율입니다. 모든 창에서 95% 이상, 관측 간격 2.5초 이하를 요구합니다. TREE의 낮은 비율은 이전 시험의 보수적인 호가 신선도 규칙이 영향을 줍니다. 체결이 없던 구간을 매수·매도 우위로 간주하지 않습니다.</p>
<h2>고정한 비교 설계</h2>
<div class="scroll">{table(['항목','조건'],[
['대상','기존 114종목 유지. 현재 3종목 시험으로 전체 시장 성능을 주장하지 않음'],
['성공 정의','급등 앵커 이후 1시간 종가 +5%, 24시간 고가 +15%; 알림은 앵커 1~6시간 전'],
['기존 모델','최근 24시간 상승률, 거래대금 비율, 과거 고가 대비 위치의 3개 변수'],
['새 모델','같은 3개 변수 + 호가·체결 요약 변수 24개'],
['새 요약 변수','매수 동반 매도 잔량 감소 / 매도 동반 매수 잔량 회복 / 목표까지 매도 금액 감소 / 가격 변화 / 스프레드 / 일반 체결 부재 비율; 각각 1·5·15·60분'],
['학습 방법','표준화 + 동일한 로지스틱 회귀. 추가 탐색·특징 선택·음성 표본 축소 없음'],
['기간','시작일 사전 등록 → 학습 28일 → 48시간 간격 → 보정 14일 → 48시간 간격 → 평가 56일. 이후 정답 확정 대기 30시간'],
['알림 조건','보정 점수의 80번째 백분위 임계값, 종목별 24시간 대기. 평가 데이터로 임계값 재선택 금지'],
['표본 요건','학습 후보 1,000개·서로 다른 정답 사건 20개, 보정 후보 200개·서로 다른 사건 10개, 각 기간 두 정답 클래스 존재'],
['추가 효과 판단','두 모델 알림 수 비율 0.8~1.25, 각 100건 이상·8개 활성 주 이상, 주별 짝지은 재표집에서 적중률 차이의 95% 하한 >0'],
['80% 근거 요건','적중률 80% 이상뿐 아니라 100개 이상 알림·사건 포착률 10% 이상·8개 활성 주 이상 및 기존 두 신뢰구간 하한 80% 이상']])}</div>
<p>이 기간과 표본 요건은 효과를 보장하지 않습니다. 자료가 부족하면 미결론으로 남깁니다. 같은 보정 백분위를 써도 실제 알림 수가 같아지지는 않으므로, 평가 알림 수가 크게 다르면 새 정보의 효과라고 결론 내리지 않습니다. 호가가 없었던 종목의 급등도 전체 포착률 분모에 포함합니다.</p>
<h2>검증 결과와 한계</h2>
<p>새 파이프라인 테스트 {v['tests_passed']}개 통과, {v['tests_failed']}개 실패. 기존 정의와 정답 일치, 미래 정보 차단, 결측 정답 처리, 단절된 창 제외, 미래 날짜 사전 등록, 평가 이후 학습 거부, 평가 정답을 바꿔도 학습 모델이 바뀌지 않는지 확인했습니다. 모델 학습·평가 동작 시험에는 합성 자료를 사용했으며 성능 증거로 세지 않습니다.</p>
<p>호가 감소·회복이 발생한 이유를 주문 취소와 체결로 분리한 것은 아닙니다. 지표를 수식으로 만들었다는 사실도 예측력이 있다는 증거가 아닙니다. 오프라인 가격 문맥은 닫힌 시간봉을 정시 경계에서 알 수 있다는 가정을 사용하므로, 실운영에서는 확정 캔들의 실제 수신 지연까지 검증해야 합니다.</p>
<h2>아직 필요한 실제 작업</h2>
<p><strong>연속 수집을 실행할 서버 또는 계속 켜둘 컴퓨터가 필요합니다.</strong> 실행 환경에 대한 답변은 아직 받지 않았습니다. 현재는 장기 프로세스·예약 실행을 설치하거나 시작하지 않았으며, 수집 시작일도 등록하지 않았습니다. 환경이 정해지면 기존 수집기의 연결당 최대 5종목, 저장 공간, 실행 중단과 재시작 시 데이터 손실을 고려해 114종목 수집을 배치해야 합니다.</p>
<p>완성된 것은 정답 연결과 비교 코드입니다. 장기 수집 배치와 실제 통계 검증은 남아 있습니다. 검증 전 운영 알림 기준은 변경하지 않았습니다.</p>
<h2>파일과 재현</h2>
<p><a href="readiness.json">실제 입력 검사</a> · <a href="implementation_manifest.json">테스트·소스 해시</a> · <a href="../protocol.json">비교 규약</a> · <a href="../universe.json">114종목 목록</a> · <a href="../dataset.py">지표·정답 연결 코드</a> · <a href="../study.py">학습·평가 코드</a> · <a href="../../orderbook_lab/output/orderbook_report.html">이전 수집 시험</a></p>
<pre>python3 -m unittest discover -s research/orderbook_eval -p 'test_*.py' -v
python3 research/orderbook_eval/fetch_prices.py --help
python3 research/orderbook_eval/dataset.py --help
python3 research/orderbook_eval/study.py register --help
python3 research/orderbook_eval/study.py fit --help
python3 research/orderbook_eval/study.py evaluate --help</pre>
<p class="muted">데이터 규격: 미완성 캔들의 close는 마지막 체결 가격이므로 닫힌 캔들만 사용합니다. <a href="https://bybit-exchange.github.io/docs/v5/market/kline">Bybit 공식 Kline 문서</a></p>
</main></html>'''
(OUT/'evaluation_report.html').write_text(html)
print(OUT/'evaluation_report.html')
