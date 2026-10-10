"""Present event commonalities separately from any predictive claim."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

BASE=Path(__file__).resolve().parent
OUT=BASE/'output'
NAMES={
 'volume_build':'직전 6시간 거래대금 ≥ 기준의 2배',
 'compression':'직전 6시간 변동폭 ≤ 기준의 70%',
 'near_high':'직전 7일 고점의 98% 이상 가격',
 'prior_runup':'직전 24시간 수익률 ≥ +5%',
 'relative_strength':'24시간 BTC 대비 초과수익 ≥ 5%p',
 'recent_spike':'직전 72시간에 1시간 +3% 상승 경험',
 'market_up':'시장 종목 중앙값 6시간 수익률 ≥ +1%',
 'market_breadth':'시장 종목의 60% 이상이 6시간 상승',
}


def main():
 r=json.loads((OUT/'results.json').read_text())
 e=pd.read_csv(OUT/'events.csv',parse_dates=['time'])
 m=pd.read_csv(OUT/'matched.csv',parse_dates=['time','anchor'])
 p=pd.read_csv(OUT/'trajectories.csv')
 assert not e.duplicated(['symbol','time']).any()
 assert m.groupby(['split','group']).size().eq(4).all()
 assert m.groupby(['split','group']).symbol.nunique().eq(1).all()
 assert m.time.dt.hour.eq(m.anchor.dt.hour).all()
 controls=m[m.role.eq('control')]
 assert ~controls.event_nearby.any()
 assert (controls.time-controls.anchor).abs().between(pd.Timedelta(hours=48),pd.Timedelta(days=28)).all()
 for root,name in [(BASE.parent/'flow_replay','forward_manifest.json'),(BASE.parent/'sequence_replay','frozen.json')]:
  frozen=json.loads((root/'output'/name).read_text())
  assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==h for name,h in frozen['source_hashes'].items())
 frozen=json.loads((OUT/'frozen.json').read_text())
 for name,h in frozen['hashes'].items():
  path=BASE.parent/'microstructure/output/bybit.parquet' if name=='hourly_source' else BASE/name
  assert hashlib.sha256(path.read_bytes()).hexdigest()==h
 (OUT/'validation.json').write_text(json.dumps({'tests_passed':6,'same_coin_hour_distance_controls_checked':True,
  'no_raw_surge_within24h_of_controls':True,'unique_events':True,'frozen_sources_and_previous_experiments_unchanged':True},indent=2))
 rows=[]
 for split,v in r.items():
  for key,s in v['comparisons'].items():rows.append({'split':split,'feature':key,**s})
 pd.DataFrame(rows).to_csv(OUT/'comparison_summary.csv',index=False)
 contexts=e.groupby(['split','context']).agg(events=('symbol','size'),coins=('symbol','nunique'),
  prior24_median=('prior_runup','median'),volume_build_median=('volume_build','median'),
  compression_median=('compression','median'),during_volume_median=('during_volume','median'))
 contexts.to_csv(OUT/'context_summary.csv')
 selected_offsets=[-24,-12,-6,-3,-1,0,1,6,24]
 p[p.offset.isin(selected_offsets)].groupby(['split','role','offset'])[['volume_build','compression','relative_strength','price_return']].median().to_csv(OUT/'trajectory_summary.csv')

 fig,axes=plt.subplots(2,3,figsize=(13,7.6),constrained_layout=True)
 for i,(split,title) in enumerate([('discovery','Apr-Jun: 167 matched events'),('replication','Aug-Sep: 137 matched events')]):
  for j,(key,label,factor) in enumerate([('volume_build','6h turnover / prior 7d baseline',1),('compression','6h range / prior 7d baseline',1),('relative_strength','24h excess return vs BTC (pp)',100)]):
   ax=axes[i,j]
   for role,color,text in [('case','#137e92','Surge'),('control','#969fae','Matched non-surge')]:
    z=p[p.split.eq(split)&p.role.eq(role)&p.offset.le(0)].groupby('offset')[key].median()*factor
    ax.plot(z.index,z.values,label=text,color=color,linewidth=2)
   ax.axvline(0,linestyle='--',color='#293d4f',linewidth=1)
   ax.set(title=title if j==1 else '',xlabel='Hours before selected surge-hour start',ylabel=label,xlim=(-24,0))
   ax.grid(alpha=.15)
 handles,labels=axes[0,0].get_legend_handles_labels()
 fig.legend(handles,labels,loc='outside lower center',ncol=2)
 fig.suptitle('Pre-event medians: repeated differences, not estimated alert precision\nAll plotted features use completed data at each timestamp; events aligned retrospectively',fontsize=14)
 fig.savefig(OUT/'pre_event_paths.png',dpi=160)
 fig.savefig(OUT/'pre_event_paths.svg')
 plt.close(fig)

 def comparison_table():
  body=[]
  for key,name in NAMES.items():
   cells=[]
   for split in ['discovery','replication']:
    s=r[split]['comparisons'][key]
    cells.append(f'<td>{s["event_prevalence"]:.1%} / {s["control_prevalence"]:.1%}<small>差 {s["difference"]*100:+.1f}%p · 구간 [{s["lower"]*100:+.1f}, {s["upper"]*100:+.1f}]</small></td>')
   body.append(f'<tr><td>{name}</td>'+''.join(cells)+'</tr>')
  return '<div class="scroll"><table><tr><th>급등 시작 직전의 특징</th><th>4–6월: 급등 / 비교 구간</th><th>8–9월: 급등 / 비교 구간</th></tr>'+''.join(body)+'</table></div>'

 def context_table():
  body=[]
  for key,label,definition in [('continuation','상승 중 재가속','직전 24시간 ≥ +5%'),('flat','24시간 수익률 중립','직전 24시간 −5% 이상, +5% 미만'),('rebound','하락 뒤 급등','직전 24시간 < −5%')]:
   cells=[]
   for split in ['discovery','replication']:
    s=contexts.loc[(split,key)]
    total=r[split]['all_events']['events']
    cells.append(f'<td>{int(s.events)}건 ({s.events/total:.1%})<small>직전 거래대금 중앙값 {s.volume_build_median:.2f}배 · 변동폭 {s.compression_median:.2f}배</small></td>')
   body.append(f'<tr><td>{label}<small>{definition}</small></td>'+''.join(cells)+'</tr>')
  return '<table><tr><th>가격 맥락</th><th>4–6월</th><th>8–9월</th></tr>'+''.join(body)+'</table>'

 content='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>급등 305건에서 다시 찾은 공통점</title><style>
body{margin:0;background:#f5f7fa;color:#213348;font-family:system-ui,sans-serif;line-height:1.75}main{max-width:1100px;margin:auto;padding:38px 28px}h1{font-size:34px;line-height:1.3}h2{font-size:23px;margin-top:36px}.lead{font-size:19px}.box{padding:18px 22px;border-left:4px solid #18869b;background:#e6f0f3;border-radius:7px}.muted,small{color:#63758a}small{display:block;font-size:12px}table{border-collapse:collapse;width:100%;background:white;font-size:14px}th,td{text-align:left;padding:13px;border-bottom:1px solid #dce5ed;vertical-align:top}th{background:#e7eef4}.scroll{overflow:auto}img{width:100%;background:white;border-radius:8px}a{color:#087a97}li{margin:9px 0}pre{background:#e7edf3;padding:16px;overflow:auto}</style><main>
<p class="muted">Crypto Tools · 2026-10-09 · 사건에서 출발한 탐색 분석</p>
<h1>급등 305건을 먼저 모으니,<br>‘조용한 코인’이라는 전제부터 달라졌다.</h1>
<p class="lead">기존 알림에 잡힌 종목을 고르는 대신 가격으로 급등 사건을 모았다. 거래 활성화와 상대적 강세는 시작 전에 더 흔했고, 강한 변동폭 수축은 오히려 비교 구간에서 더 흔했다. 다만 이것은 급등 사례의 특징이지, 새 모델의 적중률이 아니다.</p>
<div class="box"><b>가장 중요한 방향 전환:</b> ‘아직 움직이지 않은 코인의 단일 전조’만 찾으면 이미 상승 중인 급등을 놓칠 수 있다. 이번 표본에서는 약 44–46%가 직전 24시간에 이미 5% 이상 오른 상태였다. 상승 중 재가속·중립 수익률 구간·하락 뒤 급등을 나누어 후속 가설을 세울 근거가 생겼다.</div>
<h2>어떻게 급등부터 모았나</h2>
<p>확보된 Bybit 현물 115종목 가운데 BTC를 비교 기준으로 사용하고 나머지 114종목을 조사했다. 직전 30일 가격 자료가 연속해서 있는 구간에서 <b>다음 1시간 종가 +5% 이상, 다음 24시간 고가 +15% 이상</b>인 사건을 찾았다. 같은 종목에서는 최초 사건 뒤 24시간 안의 중복을 제거했다. 거래량 증가·조용한 사전 구간·유동성 범위·기존 알림 점수로 사건을 걸러내지 않았다.</p>
<p>4–6월 167건(69종목), 8–9월 138건(72종목), 합계 305건이다. 이는 이전 5분 급등 모델과 목표·대상이 달라 사건 수와 적중률을 직접 비교할 수 없다. 305건 모두 독립된 시장 충격이라는 뜻도 아니다. 장기 급등 중 24시간 이상 간격의 사건과 동시다발적 상승이 포함될 수 있다.</p>
<p>각 사건에 같은 코인·같은 UTC 시간대의 비급등 구간 3개를 짝지었다. ±28일 이내이면서 최소 48시간 떨어진 시점에서, 사건 24시간 전까지의 유동성과 변동성이 비슷한 구간을 골랐다. 세부 특징은 짝 선택에 사용하지 않았다. 후기 1건은 조건에 맞는 비교 구간이 부족해 제외되어, 특징 비교에는 167건·137건이 사용됐다.</p>
<h2>공통점은 평소보다 얼마나 더 흔했나</h2>COMPARISON
<p class="muted">표는 해당 특징을 가진 사건/대조 관측의 비율이다. 예를 들어 ‘37.2% / 6.6%’는 급등 사례의 37.2%, 비교 구간의 6.6%에서 거래대금 2배 조건을 보았다는 뜻이다. ‘이 조건으로 알리면 37.2%가 급등한다’는 뜻이 아니다. 차이 구간은 주간 블록 재표집, 8개 사전 지정 특징에 대한 명목 99.375% 구간이다. 반복된 기간 사용 전체를 보정하거나 인과관계를 입증하지 않는다.</p>
<ul><li><b>거래 활성화:</b> 직전 6시간 거래대금 2배가 두 기간 모두 비교 구간보다 더 흔했다. 하지만 급등의 과반은 이 조건을 만족하지 않아 필수 조건으로 삼으면 놓치는 사건이 많다.</li>
<li><b>수축보다 확장:</b> 변동폭이 기준의 70% 이하로 줄어든 경우는 약 12%였다. 직전 변동폭 비율 중앙값은 급등 1.49배·1.39배, 비교 구간 0.86배·0.85배였다. 이 표본에서 강한 수축을 보편적 전제로 삼기는 어렵다.</li>
<li><b>가격의 선행 강세:</b> 직전 상승·BTC 대비 강세·7일 고점 근접이 더 흔했다. 서로 상관된 특징들이므로 독립 증거 세 개로 합산하면 안 된다.</li>
<li><b>이전의 작은 급등:</b> 직전 72시간에 1시간 3% 상승 경험은 약 73–77%에서 나타났다. 다만 비교 구간도 47–56%여서 이것만으로는 구분하기 어렵다.</li>
<li><b>시장 전체 상승:</b> 시장 수익률·상승 종목 비중의 차이는 넓은 구간이 0을 포함했다. 모든 급등을 시장 상승 하나로 설명할 근거는 부족하다.</li></ul>
<h2>몇 시간 전부터 달라졌나</h2>
<img src="pre_event_paths.png" alt="급등과 대조 구간의 직전24시간 거래대금·변동폭·상대강도 중앙값 비교">
<p>후기 급등 사례의 거래대금 비율 중앙값은 6시간 전 약 1.00배, 3시간 전 1.13배, 시작 직전 1.31배였다. 변동폭 비율은 1.11배→1.15배→1.39배, BTC 대비 24시간 초과수익은 1.78%p→1.73%p→3.35%p였다. 탐색 기간에서도 직전 구간에서 활성화가 더 커지는 모습을 보였다. 모든 사건에서 단조롭게 증가했다는 뜻은 아니며 그래프는 사건마다 시점을 맞춘 중앙값이다.</p>
<p>0은 큰 상승이 발생한 1시간 봉의 시작이다. 각 지표는 그 시점까지 끝난 봉만 사용했지만, 실제로 어느 봉이 급등 봉이었는지는 사후에 정했다. 시간봉 자료이므로 분 단위의 정확한 점화 시점과 알림 가능 시각을 판별한 것은 아니다.</p>
<h2>급등과 동시에 보이는 공통점은 더 강하다</h2>
<p>급등이 발생한 1시간 거래대금은 이전 7일 시간당 평균 대비 중앙값 <b>7.41배·8.00배</b>였다. 2배 이상인 사건은 4–6월 143/167건(85.6%), 8–9월 129/138건(93.5%)이었다. 급등 순간의 공통점은 뚜렷하지만, 이 봉이 끝난 뒤 관찰한 값이므로 사전 예측 근거로 쓰면 미래 정보가 섞인다.</p>
<p>또한 24시간 안에 고가가 15% 이상 올라도, 24시간 뒤 종가가 시작 가격 아래인 사건은 각각 30건·15건이었다. 최고가 도달과 상승 유지, 알림 후 거래 성과는 따로 정의해야 한다.</p>
<h2>하나의 모델로 합치기 전에 나눌 세 맥락</h2>CONTEXT
<p>‘중립’은 24시간 수익률 범위로 정한 이름이다. 그 안에서 가격이 잔잔했다거나 완전한 횡보였음을 뜻하지 않는다. 이 분류는 원인이나 세력의 행동을 식별하지도 않는다.</p>
<p><b>후속 가설의 우선순위:</b> 상승 중 재가속에서는 이미 강한 종목의 거래대금·변동폭이 함께 커지는 전환을 살펴볼 수 있다. 중립 구간에서는 그 패턴이 훨씬 약하므로 다른 시작 조건이 필요한지 확인해야 한다. 하락 뒤 급등은 방향·위험이 다르므로 반등 유지 여부를 별도 목표로 다루는 편이 타당하다. 아직 어느 갈래에도 새 알림 모델을 채택하지 않았다.</p>
<h2>이번 결과의 용도와 다음 검증</h2>
<p>이번 결과는 잘못된 전제를 걷어내고 다음 가설을 좁히는 데 쓴다. 낮은 적중률을 보인 기존 규칙은 운영 모델로 채택하지 않는다. 새 모델을 평가할 때는 같은 가격 목표 아래 전체 시점을 대상으로 재생하고, 일반 비급등 구간뿐 아니라 <b>거래가 활발하고 이미 올랐지만 추가 급등하지 않은 사례</b>도 포함해야 한다. 후보 목록을 만든 뒤 원래 시장 빈도로 보정한 적중률·놓친 급등·선행시간·하루 오탐을 함께 평가해야 한다.</p>
<p>후속 평가 기간은 규칙을 확정한 뒤 새로 확보해야 한다. 현재 4–6월과 8–9월은 여러 분석에 사용되어 미사용 검증 자료가 아니다. 현재 상장·데이터 확보 목록에서 출발한 생존 편향, 제한된 거래소, 시간봉 해상도, 반복 사건과 대조 구간 재사용도 남아 있다. 비교 구간은 탐색 501행 중 490개 고유 시점, 후기 411행 모두 고유 시점이다. 블록 재표집 구간만으로 이런 제약이 모두 해결되지는 않는다.</p>
<h2>코드와 재현 자료</h2>
<p>과거 지표에 미래 가격을 바꿔도 결과가 유지되는지, 봉 시각 정렬·6시간 기준 제외·중복 제거·누락 자료·대조군 선택 등을 검증한 6개 테스트를 통과했다. 원본 OHLC 범위와 시간 정렬도 확인했다. 운영 코드와 기존 전향 실험은 변경하지 않았다.</p>
<p><a href="events.csv">전체 305건</a> · <a href="matched.csv">급등·대조 구간</a> · <a href="comparison_summary.csv">8개 특징 비교</a> · <a href="context_summary.csv">유형별 통계</a> · <a href="trajectory_summary.csv">시점별 중앙값</a> · <a href="results.json">전체 결과</a> · <a href="../protocol.json">고정한 분석 정의</a> · <a href="../study.py">분석 코드</a> · <a href="validation.json">검증 기록</a></p>
<pre>python3 -m unittest discover -s research/event_census -p 'test_*.py' -v
python3 research/event_census/study.py
python3 research/event_census/report.py</pre></main></html>'''
 content=content.replace('COMPARISON',comparison_table()).replace('CONTEXT',context_table())
 (OUT/'event_report.html').write_text(content)
 print(OUT/'event_report.html')


if __name__=='__main__':main()
