"""Render the actual analysis and its support limitations; never invent model metrics."""
import json
from pathlib import Path
import html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study import census, HOUR

OUT = Path(__file__).resolve().parent / 'output'
r = json.loads((OUT / 'report.json').read_text())
f = pd.read_parquet(OUT / 'features.parquet')
names = {'ret1':'직전1시간 수익률', 'ret4':'직전4시간 수익률', 'ret24':'직전24시간 수익률',
         'volume_build':'거래대금 배율', 'compression':'변동성 비율', 'range_position':'24시간 가격 범위 내 위치',
         'btc_excess1':'BTC 대비1시간 수익률', 'eth_excess1':'ETH 대비1시간 수익률',
         'oi_ret1':'미결제약정1시간 변화', 'oi_ret4':'미결제약정4시간 변화', 'funding':'직전 정산 펀딩비'}
percent = {'ret1','ret4','ret24','btc_excess1','eth_excess1','oi_ret1','oi_ret4','funding'}
def fmt(key, value):
    return f'{value*100:.3f}%' if key in percent else f'{value:.3f}'

start, stop = map(pd.Timestamp, r['windows']['train'])
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for direction, color in [('long','#148578'),('short','#c85454')]:
    events, _ = census(f, direction, .05, start)
    events = events[(events >= start) & (events+24*HOUR < stop)]
    events = [t for t in events if t-HOUR in f.index and abs(f.loc[t-HOUR,'ret1']) < .01]
    offsets = np.array([-6.,-4.,-3.,-2.,-1.,-.5,-.25])
    for ax, key, scale in [(axes[0],'volume_build',1),(axes[1],'btc_excess1',100)]:
        values = [[f.loc[t+pd.Timedelta(hours=float(h)),key] for t in events if t+pd.Timedelta(hours=float(h)) in f.index] for h in offsets]
        med = [np.nanmedian(v)*scale for v in values]
        ax.plot(offsets,med,marker='o',color=color,label=f'{direction} ({len(events)} cases)')
        ax.set_xlabel('Hours before operational onset');ax.grid(alpha=.2);ax.legend()
axes[0].set_title('Pre-onset turnover / earlier24h baseline')
axes[1].set_title('Pre-onset1h return minus BTC (percentage points)')
axes[0].axhline(1,color='gray',linestyle='--',alpha=.5)
axes[1].axhline(0,color='gray',linestyle='--',alpha=.5)
fig.suptitle('NEAR futures | training cases only | descriptive, not predictive validation')
fig.savefig(OUT/'antecedents.png',dpi=170);plt.close(fig)

body=[]
body.append('<h1>NEAR 선물: 오르기 전·내리기 전 분석</h1><p>자료: 2026-04-17~2026-10-17 00:00 UTC. NEAR/BTC/ETH 각17,568개 확정15분봉, NEAR 미결제약정17,568개, 정산 펀딩비549개.</p>')
body.append('<p class="notice">현재 알리미는 추세 관찰이며 전조 예측 모델이 아니다. 이번 연구에서도 기준설정 구간의 독립 사건 수가 부족해 모델 학습·실전 적용을 보류했다. 적중률을 산출할 검증된 모델은 아직 없다.</p>')
body.append('<h2>무엇을 예측하는가</h2><p>시점t 뒤1시간 종가가 해당 방향2% 이상 움직이고24시간 내 가격이5%/10%에 도달하는 후보 중 첫 시점을 선택한 뒤24시간을 띄웠다. 이는 운영상 시작점 정의로, 모든5%/10% 움직임이나 정확한 경제적 시작점은 아니다. 판단은 그15~60분 전이며 직전1시간에 이미1% 이상 움직인 후보는 제외했다.</p>')
body.append('<h2>사건 수와 학습 가능 여부</h2><table><tr><th>방향·목표</th><th>학습 사건 / 접근 가능</th><th>기준설정 사건 / 접근 가능</th><th>검증 사건 / 접근 가능</th><th>상태</th></tr>')
for s in r['studies']:
    cells=[f'{s["direction"]} {s["target"]:.0%}']+[f'{s["distinct_events"][k]} / {s["reachable_events"][k]}' for k in ('train','calibration','holdout')]+['표본 부족' if not s['models'] else '후향 검증만 완료']
    body.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in cells)+'</tr>')
body.append('</table><p>훈련은4/20~7/19, 기준설정은7/21~8/18, 검증은8/20~10/17(UTC). 각 끝에서25시간 미래 정답을 제거했다. 최소 학습20건·기준설정5건의 접근 가능한 독립 사건을 요구했다. 사건 수가 적다고 결과를 본 뒤 구간을 옮기지 않았다.</p>')
body.append('<h2>직전 지표의 실제 변화</h2><img src="antecedents.png" alt="Training-only pre-onset turnover and relative strength"/><p>학습구간의 시작점1시간 전과 같은 시간대·비슷한 과거 변동성의 비사건 구간을 짝지었다. 아래는 발견용 비교다. 미래 사건을 골라 그린 그래프이므로 실시간 예측 성공의 증거가 아니다.</p>')
for s in r['studies']:
    body.append(f'<h3>{s["direction"]} {s["target"]:.0%} · 비교 {s["training_antecedents"]["pairs"]}쌍</h3><table><tr><th>지표</th><th>시작1시간 전 중앙값</th><th>비사건 중앙값</th><th>짝별 차이 중앙값</th><th>높음 / 같음 / 낮음</th></tr>')
    for x in s['training_antecedents']['comparisons']:
        key=x['feature']
        cells=[names[key],fmt(key,x['before_onset_median']),fmt(key,x['control_median']),fmt(key,x['median_paired_difference']),
               f'{x["case_greater_fraction"]:.0%} / {x["tie_fraction"]:.0%} / {x["case_less_fraction"]:.0%}']
        body.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in cells)+'</tr>')
    body.append('</table>')
body.append('<h2>해석과 다음 단계</h2><p>거래대금 증가·상승 추세를 먼저 요구하는 기존 알림을 전조 모델로 부를 근거는 없다. 작은 표본의 지표 중앙값 차이를 확률이나 적중률로 바꾸지 않는다. OI/펀딩은15분 공개 지연을 가정한 과거자료이며, 실제 수신 지연·호가·체결방향은 별도 연속 수집으로 검증해야 한다. 롱·숏,5%·10%는 별도 평가하며 선정한 유사 종목에 NEAR 결과를 그대로 일반화하지 않는다.</p>')
body.append('<p>판정 임계값을 정할 사건 수가 부족하므로, 지금은 후보 가설과 실제 수신 자료를 쌓을 단계다. 별도 실험을 설계한다면 새 기간·조건을 미리 기록하고 새 데이터로 검증한다. 본 결과를 보고 조건을 바꾼 뒤 같은 검증 기간에서 성공했다고 주장하지 않는다.</p>')
page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>NEAR 선물 전조 분석</title><style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.65 system-ui;color:#173047;background:#f7fafc}table{border-collapse:collapse;width:100%;background:white;font-size:14px}td,th{padding:10px;border-bottom:1px solid #dce5eb;text-align:left}th{background:#e9f1f6}h2{margin-top:36px}img{width:100%}.notice{padding:20px;background:#fff1dc;border-left:5px solid #ba7624}</style><body>'+''.join(body)+'</body></html>'
(OUT/'report.html').write_text(page)
print(OUT/'report.html')
