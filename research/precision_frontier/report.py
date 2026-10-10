"""Report all fixed policies and the calibration-selected result."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

BASE=Path(__file__).resolve().parent
OUT=BASE/'output'
NAMES={'volume':'거래대금만','efficiency':'상승 경로만','context3':'기본 3개 특징','full5':'기본 + 경로·가속'}


def main():
 r=json.loads((OUT/'results.json').read_text())
 lock=json.loads((OUT/'model_lock.json').read_text())
 d=pd.read_csv(OUT/'frontier.csv');a=pd.read_csv(OUT/'alerts.csv',parse_dates=['time','target_event'])
 assert not a.duplicated(['split','policy_id','symbol','time']).any()
 assert a.sort_values('time').groupby(['split','policy_id','symbol']).time.diff().dropna().ge(pd.Timedelta(hours=24)).all()
 h=a[a.hit]
 assert (h.target_event-h.time).dt.total_seconds().div(3600).between(1,6).all()
 assert not h.duplicated(['split','policy_id','symbol','target_event']).any()
 for root,name,key in [(BASE,'source_manifest.json','hashes'),(BASE.parent/'flow_replay','forward_manifest.json','source_hashes'),(BASE.parent/'continuation_controls','frozen.json','hashes')]:
  manifest=json.loads((root/'output'/name).read_text())
  assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==v for p,v in manifest[key].items())
 for policy in lock['policies']:
  z=d[d.policy_id.eq(policy['policy_id'])]
  assert len(z)==2 and (z.threshold-policy['threshold']).abs().le(1e-12).all()
 (OUT/'validation.json').write_text(json.dumps({'tests_passed':7,'cooldown_and_strict_lead_verified':True,
   'unique_detected_events_per_policy':True,'locked_thresholds_match_both_periods':True,
   'previous_and_current_source_hashes_unchanged':True},indent=2))
 e=d[d.split.eq('evaluation')]
 selected=e[e.policy_id.eq(r['selected_policy'])].iloc[0]
 fig,axes=plt.subplots(1,2,figsize=(12,5.5),constrained_layout=True)
 colors={'volume':'#aeb3bf','efficiency':'#b7843b','context3':'#168297','full5':'#8b72ba'}
 names={'volume':'Turnover only','efficiency':'Path only','context3':'Context (3)','full5':'Context + path (5)'}
 for ax in axes:
  for model,g in e.groupby('model'):
   ax.scatter(g.all_recall*100,g.precision*100,s=28+g.alerts/45,color=colors[model],alpha=.8,label=names[model])
  ax.scatter([selected.all_recall*100],[selected.precision*100],marker='*',s=230,color='#0e3347',zorder=5,label='Chosen using June')
  ax.set(xlabel='Share of all surge events detected (%)',ylabel='Actual alert precision (%)',xlim=(-1,23))
  ax.grid(alpha=.15)
 axes[0].set(ylim=(-2,85),title='All tested policies vs 80% target')
 axes[0].axhline(80,color='#be5050',linestyle='--',label='80% target')
 axes[1].set(ylim=(-.3,7),title='Same points, zoomed to 0–7%')
 handles,labels=axes[0].get_legend_handles_labels()
 fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=9)
 fig.suptitle('August–September chronological replay | 32 thresholds fixed using June\nOne alert per coin per24h; surge size unchanged; lead1–6h',fontsize=13)
 fig.savefig(OUT/'precision_frontier.png',dpi=160);fig.savefig(OUT/'precision_frontier.svg');plt.close(fig)

 def rows_table(rows):
  body=[]
  for _,z in rows.iterrows():
   body.append(f'<tr><td>{NAMES[z.model]}<small>{z.policy_id}</small></td><td>{int(z.alerts):,}</td><td>{int(z.hits)}</td>'
    f'<td>{z.precision:.2%}</td><td>{z.all_recall:.1%}</td><td>{z.alerts_per_day:.2f}</td></tr>')
  return '<div class="scroll"><table><tr><th>モデル</th><th>알림</th><th>적중</th><th>적중률</th><th>전체 급등 포착률</th><th>하루 알림</th></tr>'+''.join(body)+'</table></div>'
 # Apply the same prespecified June criterion separately for each model, without using later results.
 best=[]
 for model,g in d[d.split.eq('calibration')].groupby('model'):
  valid=g[g.alerts.ge(30)&g.all_recall.ge(.1)].sort_values(['precision','all_recall','alerts','model','threshold'],ascending=[False,False,True,True,True])
  if len(valid):best.append(valid.iloc[0].policy_id)
 primary=rows_table(e[e.policy_id.isin(best)])
 full=rows_table(e)
 content='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>적중률 80% 목표의 전체 후보 재생 결과</title><style>
body{margin:0;background:#f5f7fa;color:#203348;font-family:system-ui,sans-serif;line-height:1.75}main{max-width:1100px;margin:auto;padding:36px 28px}h1{font-size:33px;line-height:1.3}h2{font-size:22px;margin-top:36px}.box{padding:18px 22px;background:#e5eff3;border-left:4px solid #18859b;border-radius:7px}.warn{border-color:#bb8050;background:#f7eee4}table{width:100%;border-collapse:collapse;background:white;font-size:14px}th,td{text-align:left;padding:12px;border-bottom:1px solid #dce5ed;vertical-align:top}th{background:#e7eef4}small,.muted{color:#66788a}small{display:block}.scroll{overflow:auto}img{width:100%;border-radius:8px}a{color:#0a7a96}li{margin:9px 0}pre{padding:16px;background:#e7edf3;overflow:auto}summary{cursor:pointer;font-weight:600}</style><main>
<p class="muted">Crypto Tools · 2026-10-10 · 전체 후보 빈도를 유지한 시간순 재생</p>
<h1>적중률 80%에 도달하는 구간은<br>이번 모델들에서 확인되지 않았다.</h1>
<div class="box"><b>6월 자료로 선택한 설정:</b> 8–9월 알림 755건 중 23건 적중, <b>3.05%</b>. 전체 급등 138건 중 23건(16.7%)을 잡았고, 적중 사례의 선행시간 중앙값은 3시간이었다. 사전에 고정한 32개 설정 모두 80%에 도달하지 못했다.</div>
<h2>이번에 고정한 목표와 방법</h2>
<p>가격으로 정한 급등의 크기는 유지했다. 급등 시작 후 첫 1시간 종가 +5% 이상, 이후 24시간 고가 +15% 이상이다. 이번 실제 알림 평가는 <b>그 시작 1–6시간 전</b>에 알렸을 때만 적중으로 센다. 이전 성공·실패 짝 비교는 급등 시작 직전 시점의 설명력이었으므로, 더 이른 알림 성능과 같은 숫자가 아니다. 이전 5분 급등 실험과도 목표가 달라 ‘0.1%에서3%로 개선’이라고 비교할 수 없다.</p>
<p>현재 확보한 114개 비BTC 현물 코인의 시간별 관측에서, 직전 24시간 5% 이상 상승하고 과거 자료가 완전한 후보를 모두 사용했다. 성공·실패 개수를 인위적으로 맞추지 않았다. 학습 후보 16,783개, 6월 기준 설정 후보 7,951개, 8–9월 평가 후보 23,939개다. 평가 후보 행의 급등 발생 비율은 약 1.31%였으며, 이는 중복 시점이 포함된 행 비율이므로 알림 적중률과 동일하지 않다.</p>
<ul><li><b>학습:</b> 4–5월. 이후 30시간 결과가 6월에 걸치는 행은 제외.</li>
<li><b>기준 설정:</b> 6월. 이후 30시간 결과가 7월에 걸치는 행은 제외.</li>
<li><b>후기 재생:</b> 8–9월. 수치 임계값과 모델 계수, 선택한 설정을 저장한 뒤 평가.</li>
<li><b>알림:</b> 점수가 기준 이상인 최초 시점에 발생. 종목별 24시간 재알림 제한. 하루 한도를 자정에 소진하는 정책은 사용하지 않음.</li>
<li><b>정책 선택:</b> 6월 알림 30건 이상·전체 급등 포착률 10% 이상인 설정 중 적중률이 가장 높은 하나를 선택. 조건 미달이면 선택하지 않기로 사전 고정.</li></ul>
<h2>비교한 네 가지 점수</h2>
<p>거래대금 증가 비율만 사용한 점수, 상승 경로 효율성만 사용한 점수, 기본 특징 3개를 사용한 로지스틱 모델, 특징 5개 모델을 비교했다. 기본 3개는 직전 상승률·거래대금 증가·7일 고점까지의 거리다. 5개 모델은 여기에 상승 경로 효율성과 거래대금 가속을 추가했다. 표준화는 학습 자료에서만 계산했고, 복잡한 모델 검색이나 성공 사례 과다 샘플링은 하지 않았다.</p>
<p>각 점수의 6월 분포에서 고정한 8개 기준을 사용했다. 분위수는 기준을 정하는 방법이지 예측 확률이 아니다. 예컨대 80백분위 기준이 ‘급등 확률80%’라는 뜻은 아니다. 로지스틱 출력도 이번에는 순위 점수로 취급했으며 확률 보정을 검증하지 않았다.</p>
<h2>모델별로 6월에서 선택한 설정의 후기 결과</h2>PRIMARY
<p>전체 정책 중 선택된 것은 기본 3개 특징 모델이었다. 추가 특징이 있는 모델은 이번 기준 설정과 후기 재생에서 더 높은 적중률을 보여주지 못했다. 각 모델의 알림 수가 다르므로 표만으로 통계적으로 확정된 우열을 주장하지는 않는다. 같은 알림 빈도 부근도 아래 전체 설정에서 확인할 수 있다.</p>
<img src="precision_frontier.png" alt="32개 고정 정책의 실제 적중률과 급등 포착률. 80% 목표 전체 축과0–7% 확대 축 비교.">
<h2>알림 기준을 더 높이면 해결되나</h2>
<p>검사한 기준 중 후기 적중률이 가장 높아 보인 값도 <b>4/67건,5.97%</b>였다. 이는 후기 결과를 보고 찾은 기술적 최대값이며 선택된 운영 설정이 아니다. 전체 급등 포착률도 2.9%에 불과했다. 가장 엄격한 기준에서는 각 모델이 8–40건을 알렸고 적중은 모두 0건이었다. 기준을 높이면 적중률이 계속 오르는 관계를 관측하지 못했다.</p>
<p>종목별 재알림 제한이 있어 기준이 바뀌면 알림 시점도 달라진다. 점을 매끄러운 곡선으로 연결하거나 검사하지 않은 기준의 성능을 추정하지 않았다. 32개 설정의 사후 최대값은 이 모델군의 이론적 한계도, 다른 자료를 쓴 모델의 한계도 아니다.</p>
<details><summary>32개 설정의 후기 결과 전체 보기</summary>FULL</details>
<h2>표본 수와 불확실성</h2>
<p>선택 설정의 적중률 3.05%에 대한 95% 주간 블록 구간은 약 <b>1.72–5.11%</b>, 독립 시행을 가정한 정확 이항 구간은 약 <b>1.94–4.54%</b>다. 반복 시도와 사건 상관을 완전히 해결하는 구간은 아니지만, 이번 결과가 80% 근처라고 볼 근거는 없다.</p>
<p>사전에 정한 80% 근거 점검 조건은 적중률80% 이상·알림100건 이상·전체 급등 포착률10% 이상·활성 주간8개 이상·두 명목 구간의 하한도80% 이상이었다. 이는 이번 실험의 명시적 증거 기준이며 보편적인 표본 수 정리가 아니다. 이번에는 점추정 자체가80%인 설정이 하나도 없어서 통과하지 못했다.</p>
<p>현재 ‘이미5% 오른 종목’ 후보 조건과1–6시간 선행 창 안에서 접근 가능한 급등은 138건 중73건이었다. 나머지65건은 점수를 아무리 잘 매겨도 이 고정 후보 목록으로는 잡을 수 없다. 선택 설정은 접근 가능한73건 중23건(31.5%)을 포착했다. 후보 범위의 한계와 후보 안에서의 분류 성능은 따로 관리해야 한다.</p>
<h2>판단과 다음 개발 방향</h2>
<div class="box warn"><b>이 모델들은80% 목표에 미달했다.</b> 상승 경로에 대한 회고적 차이가 실제 사전 알림의 높은 적중률로 이어지지 않았다. 기존 가격·거래량의 임계값을 계속 맞추는 방식으로 목표에 도달했다고 주장하지 않는다.</div>
<p>다음 개발은 같은 목표·선행시간을 고정한 채, 호가 소진·보충 또는 거래소 간 선행 흐름 같은 <b>새 정보가 주는 추가 효과</b>를 비교하는 쪽이 타당하다. 그 자료가80%를 가능하게 한다는 뜻은 아니다. 데이터 품질과 실패 사례를 함께 확보하고, 새 기간에서 기존 기본 모델보다 오탐을 줄이는지부터 검증해야 한다. 개선이 없으면80% 미달이라는 판단을 유지해야 한다.</p>
<p>평가 기간은 이전 분석에서 여러 차례 사용되어 미사용 최종 검증 구간이 아니다. 현재 상장·자료 확보 종목의 편향, 드문 급등, 종목 간 동시 상승, 시간봉 해상도도 한계다. 수수료·슬리피지·체결 지연을 반영한 손익 시험은 하지 않았다. 24시간 뒤 수익률과 하락폭은 별도 기록했으며 매매 성과로 표현하지 않는다.</p>
<h2>검증과 파일</h2>
<p>7개 테스트가 통과했다. 미래 결과와 무관한 후보·알림 생성,1–6시간 선행,30시간 결과 창 분리,종목별 재알림 제한,표본 부족 설정 제외 등을 확인했다. 실제 알림의 중복·선행시간·동결 임계값도 검증했다. 기존 운영 스캐너와 전향 실험은 변경하지 않았다.</p>
<p><a href="frontier.csv">모든 설정과 통계</a> · <a href="alerts.csv">모든 알림·결과</a> · <a href="results.json">선택 설정 결과</a> · <a href="model_lock.json">평가 전 고정한 계수·기준</a> · <a href="../protocol.json">실험 정의</a> · <a href="../study.py">코드</a> · <a href="validation.json">검증 기록</a></p>
<pre>python3 -m unittest discover -s research/precision_frontier -p 'test_*.py' -v
python3 research/precision_frontier/study.py
python3 research/precision_frontier/report.py</pre></main></html>'''
 content=content.replace('PRIMARY',primary).replace('FULL',full).replace('モデル','모델')
 (OUT/'precision_report.html').write_text(content)
 print(OUT/'precision_report.html')


if __name__=='__main__':main()
