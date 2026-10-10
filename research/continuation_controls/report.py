"""Transparent hard-negative results; no fabricated alert precision."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parent
OUT=BASE/'output'
NAMES={'volume_acceleration':'거래대금 가속','price_efficiency':'상승 경로의 효율성','close_retention_change':'봉 고점 대비 되밀림의 감소'}


def main():
 r=json.loads((OUT/'results.json').read_text())
 m=pd.read_csv(OUT/'matched.csv',parse_dates=['time','anchor'])
 for root,file,key in [(BASE,'frozen.json','hashes'),(BASE.parent/'flow_replay','forward_manifest.json','source_hashes'),(BASE.parent/'sequence_replay','frozen.json','source_hashes')]:
  frozen=json.loads((root/'output'/file).read_text())
  assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in frozen[key].items())
 for _,g in m.groupby(['split','group']):
  c=g[g.role.eq('case')].iloc[0];z=g[g.role.eq('control')]
  assert len(z) in [1,2,3] and z.symbol.eq(c.symbol).all()
  assert z.time.dt.floor('D').nunique()==len(z)
  assert (z.time-c.time).abs().between(pd.Timedelta(hours=48),pd.Timedelta(days=28)).all()
  hour=(z.time.dt.hour-c.time.hour).abs()
  assert np.minimum(hour,24-hour).le(4).all()
  assert z.prior_runup.ge(.05).all()
  assert (z.prior_runup-c.prior_runup).abs().le(max(.03,.25*c.prior_runup)+1e-10).all()
  assert np.log(z.volume_build/c.volume_build).abs().le(np.log(1.5)+1e-10).all()
  assert (z.near_high-c.near_high).abs().le(.05+1e-10).all()
  assert z.baseline_distance.le(3).all() and z.max24.lt(.15).all() and ~z.event_nearby.any()
 (OUT/'validation.json').write_text(json.dumps({'tests_passed':7,'all_real_control_calipers_verified':True,
   'same_coin_and_distinct_dates':True,'current_and_previous_frozen_hashes_unchanged':True},indent=2))
 rows=[]
 for split,v in r['periods'].items():
  for key,s in v['features'].items():rows.append({'split':split,'feature':key,**s})
 pd.DataFrame(rows).to_csv(OUT/'comparison_summary.csv',index=False)
 fig,ax=plt.subplots(figsize=(10,5.3),constrained_layout=True)
 labels=['Volume acceleration','Upward path efficiency','Improved close retention']
 for split,color,offset,label in [('discovery','#8995a8',.13,'Apr-Jun (28 events)'),('replication','#147e94',-.13,'Aug-Sep (27 events)')]:
  s=[r['periods'][split]['features'][key] for key in NAMES]
  x=np.array([v['concordance'] for v in s])*100
  lo=np.array([v['lower'] for v in s])*100;hi=np.array([v['upper'] for v in s])*100
  ax.errorbar(x,np.arange(3)+offset,xerr=[x-lo,hi-x],fmt='o',capsize=4,color=color,label=label)
 ax.axvline(50,color='#a7b0bc',linestyle='--')
 ax.set(yticks=range(3),yticklabels=labels,ylim=(2.5,-.5),xlim=(0,100),xlabel='Matched rank concordance (%) — NOT alert precision',
        title='After matching prior rise, activity and distance to high\nFixed positive directions; 98.33% weekly-block intervals')
 ax.grid(axis='x',alpha=.15)
 ax.legend(loc='lower left')
 fig.savefig(OUT/'hard_controls.png',dpi=160);fig.savefig(OUT/'hard_controls.svg');plt.close(fig)

 body=[]
 for key,label in NAMES.items():
  cells=[]
  for split in ['discovery','replication']:
   s=r['periods'][split]['features'][key]
   cells.append(f'<td>{s["concordance"]:.1%}<small>구간 {s["lower"]:.1%}–{s["upper"]:.1%}</small></td>')
  body.append(f'<tr><td>{label}</td>'+''.join(cells)+'</tr>')
 table='<table><tr><th>같은 수준의 상승·거래량 안에서 비교</th><th>4–6월 28건</th><th>8–9월 27건</th></tr>'+''.join(body)+'</table>'
 balance=[]
 for split,label in [('discovery','4–6월'),('replication','8–9월')]:
  a=r['periods'][split]['audit'];b=a['balance'];drops=pd.DataFrame(a['dropped'])
  balance.append(f'<tr><td>{label}</td><td>{a["matched_cases"]}/{a["original_cases"]}건</td>'
   f'<td>{b["prior_runup"]["case_median"]:.1%} / {b["prior_runup"]["control_median"]:.1%}</td>'
   f'<td>{b["volume_build"]["case_median"]:.2f} / {b["volume_build"]["control_median"]:.2f}배</td>'
   f'<td>{drops.prior_runup.median():.1%} · 거래대금 {drops.volume_build.median():.2f}배</td></tr>')
 balance_table='<div class="scroll"><table><tr><th>기간</th><th>짝 확보 / 전체 급등</th><th>직전24h 상승 중앙값<br>급등 / 실패</th><th>직전6h 거래대금 비율 중앙값<br>급등 / 실패</th><th>짝을 못 찾은 급등의 중앙값</th></tr>'+''.join(balance)+'</table></div>'
 content='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>이미 오른 종목 중 추가 급등을 가르는 차이</title><style>body{font-family:system-ui,sans-serif;background:#f5f7fa;color:#203348;line-height:1.75;margin:0}main{max-width:1060px;margin:auto;padding:36px 28px}h1{font-size:32px;line-height:1.3}h2{font-size:22px;margin-top:36px}.box{padding:18px 22px;border-left:4px solid #188399;background:#e7f0f4;border-radius:7px}table{width:100%;border-collapse:collapse;background:white;font-size:14px}th,td{text-align:left;padding:13px;vertical-align:top;border-bottom:1px solid #dce5ed}th{background:#e6edf4}.scroll{overflow:auto}small,.muted{color:#66788c}small{display:block}img{width:100%;border-radius:8px}a{color:#0a7794}pre{padding:16px;background:#e7edf3;overflow:auto}li{margin:10px 0}</style><main>
<p class="muted">Crypto Tools · 2026-10-10 · 상승 지속형의 비슷한 실패 사례 비교</p>
<h1>이미 올랐고 거래도 활발했다.<br>그다음 급등을 구별하는 특징은?</h1>
<div class="box"><b>결과:</b> 최근 6시간의 상승 경로가 더 일관된 정도가 가장 눈에 띄었다. 후기 비교 점수는 75.3%였지만, 앞선 기간은 60.1%에 불확실성 구간이 넓어 두 기간 공통 채택 기준을 통과하지 못했다. <b>이 점수는 알림 적중률이 아니다.</b> 현재 새 운영 모델을 채택할 근거는 부족하다.</div>
<h2>이번에는 비교 상대를 더 어렵게 만들었다</h2>
<p>이전 급등 목록에서 직전 24시간에 이미 5% 이상 오른 사건 136건(73건·63건)을 사용했다. 급등 정의는 다음 1시간 종가 +5% 이상 및 다음 24시간 고가 +15% 이상으로 그대로 유지했다. 같은 코인에서 역시 5% 이상 올랐지만 추가 급등하지 않은 구간을 찾았다.</p>
<ul><li>같은 기간, 같은 코인, ±28일 안이면서 최소 48시간 떨어진 날짜, UTC 시간대 차이 4시간 이내.</li>
<li>직전 24시간 상승률 차이: 3%p 또는 급등 사례 상승률의 25% 중 큰 값 이내.</li>
<li>직전 6시간 거래대금 증가 비율: 서로 1.5배 범위 안. 예를 들어 2배 활성화 사건이면 비교 구간은 약 1.33–3배 이내.</li>
<li>7일 고점 대비 현재 가격 비율 차이 5%p 이내, 사건 24시간 전까지의 유동성과 변동성도 비교.</li>
<li>전후 24시간 안에 다른 급등 시작이 없고, 다음 24시간 고가가 15%에 도달하지 않은 구간을 실패 사례로 사용. ‘실패’가 반드시 가격 하락이나 투자 손실을 뜻하지는 않는다.</li></ul>
<p>사건당 서로 다른 날짜의 비교 구간을 1–3개 선택했다. 검사할 세 지표는 짝 선택에 사용하지 않았고 결과를 본 뒤 짝이나 기준을 바꾸지 않았다. 비교 구간 수에 관계없이 급등 사건마다 같은 가중치를 주었다.</p>
<h2>비교 가능한 사건은 55건이었다</h2>BALANCE
<p>136건 중 55건(40.4%)만 비슷한 실패 사례를 확보했다. 나머지 81건을 임의로 느슨한 기준에 끼워 맞추지 않았다. 따라서 이번 결과는 모든 상승 지속형 급등을 대표하지 않는다. 짝 확보 여부는 비슷한 사례가 자료에 있는지에 달려 있으며, 그 자체를 새 신호로 사용하면 안 된다.</p>
<p>탐색은 대조 관측 51개 모두 서로 다르고, 후기는 53개 중 52개가 서로 다르다. 탐색 28건은 23종목·11개 주간 블록, 후기 27건은 24종목·8개 주간 블록이다. 상승률 등은 허용 범위 안에서 맞췄지만 완벽히 동일하지 않으므로 잔여 차이도 남는다.</p>
<h2>고정한 세 가지 가설의 결과</h2>COMPARISON
<img src="hard_controls.png" alt="급등과 비슷한 실패 사례 간 세 특징의 짝 비교 점수 및 불확실성 구간">
<p>점수는 각 급등 사건의 지표가 짝지은 실패 사례보다 높은 비율을 사건별로 평균한 값이다. 동점은 절반으로 센다. 예를 들어 75.3%는 ‘같은 짝에서 상승 경로 지표가 높은 편이었다’는 뜻이며 ‘알림 100건 중 75건이 급등한다’는 뜻이 아니다. 50%를 차이 없는 순위 기준으로 사용했다.</p>
<ol><li><b>거래대금 가속:</b> 마지막 2시간 거래대금 평균을 그 앞 4시간 평균으로 나눈 로그값. 전체 6시간 거래량 수준을 비슷하게 맞춘 뒤, 뒤쪽에 거래가 더 집중되는지 봤다. 58.3%→63.0%였지만 두 기간 모두 구간이 50%를 포함했다.</li>
<li><b>상승 경로의 효율성:</b> 최근 6시간 로그수익률 합 ÷ 각 시간 로그수익률 절댓값의 합. 최종 상승이 같아도 오르내림이 심하면 낮아진다. 후기 급등 중앙값 0.677, 실패 0.310으로 차이가 보였다. 탐색은 0.644 대 0.541로 차이가 작고 불확실했다. 실제 매수 거래대금당 가격 반응이나 분 단위 되밀림을 직접 측정한 것은 아니다.</li>
<li><b>봉 고점 대비 되밀림 감소:</b> 앞 3시간 대비 최근 3시간의 ‘고가−종가’ 평균이 얼마나 작아졌는지를 과거 변동폭으로 정규화했다. 33.3%→50.0%로 기대한 양의 방향이 반복되지 않았다. 봉 내부의 가격 이동 순서나 실제 최대 낙폭을 복원한 지표는 아니다.</li></ol>
<p class="muted">불확실성 구간은 3개 고정 가설을 고려한 명목 98.33% 주간 블록 부트스트랩이다. 방향은 세 지표 모두 ‘높을수록 유리’로 사전 고정했다. 결과에 맞춰 반대로 뒤집지 않았다. 지표 중앙값은 관측을 모은 값이고 비교 점수는 각 짝을 같은 비중으로 계산하므로 두 값의 방향이 항상 일치하지는 않는다.</p>
<h2>채택 판단과 다음에 필요한 검증</h2>
<p>두 기간 모두 사건 20건 이상·주간 블록 6개 이상·비교 점수 60% 이상·구간 하한 50% 초과를 충족해야 후보로 채택하기로 고정했다. 이 기준을 통과한 지표는 <b>0개</b>다. 상승 경로 효율성은 후기만 통과했으므로, 유망해 보이는 관찰로 남기되 모델의 검증된 추가 필터로 채택하지 않았다.</p>
<p>다음 자료를 확보할 때 우선 추적할 것은 ‘상승 경로 효율성이 앞선 기간의 불확실성을 넘어 다른 기간에서도 차이를 보이는가’다. 그 단계에서도 같은 정도로 오른 실패 사례와 비교해야 한다. 이후 실제 알림 효과는 전체 후보 빈도를 유지한 시간순 재생에서 별도로 검증해야 한다. 실패 사례를 인위적으로 골라 만든 이번 표본의 급등 비율을 시장 확률로 쓸 수 없다.</p>
<p>기존 자료의 결과를 보고 임계값을 더 맞추거나 새 조합을 골라 같은 기간에서 확인하는 방식은 피했다. 현재 두 기간은 여러 번 살펴본 과거 자료라 독립된 최종 검증 구간이 아니다. 자료 확대 없이 작은 차이를 높은 적중률로 바꾸었다고 주장할 수 없다.</p>
<h2>검증과 재현 자료</h2>
<p>미래 자료 불변성, 겹치지 않는 거래량 창, 가격 경로 계산, 고점 대비 되밀림 계산, 매칭 조건, 지표와 독립적인 짝 선택, 사건별 동일 가중치와 결측 처리 등 7개 테스트를 통과했다. 실제 선택한 모든 짝의 허용 범위도 재검사했다. 이전 전향 실험과 운영 스캐너는 변경하지 않았다.</p>
<p><a href="matched.csv">전체 성공·실패 비교 자료</a> · <a href="paired_scores.csv">사건별 점수</a> · <a href="comparison_summary.csv">비교표</a> · <a href="results.json">탈락 사건을 포함한 전체 결과</a> · <a href="../protocol.json">고정한 조건</a> · <a href="../study.py">분석 코드</a> · <a href="validation.json">검증 기록</a></p>
<pre>python3 -m unittest discover -s research/continuation_controls -p 'test_*.py' -v
python3 research/continuation_controls/study.py
python3 research/continuation_controls/report.py</pre></main></html>'''
 content=content.replace('BALANCE',balance_table).replace('COMPARISON',table)
 (OUT/'continuation_report.html').write_text(content)
 print(OUT/'continuation_report.html')


if __name__=='__main__':main()
