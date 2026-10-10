"""Describe changes around a price-defined onset without claiming predictive success."""
import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parent
OUT=BASE/'output'
NAMES={'quote_accel':'거래금액','count_accel':'체결 횟수','size_accel':'평균 체결금액','net_change':'순매수 비율 변화','range_accel':'가격 고저폭','price_accel':'가격 상승 가속도'}
ENGLISH={'quote_accel':'Hourly traded notional','count_accel':'Hourly trade count','size_accel':'Average trade notional','net_change':'Net-buy fraction change','range_accel':'Hourly price range','price_accel':'Price-return acceleration'}


def table(rows):
    return pd.DataFrame(rows).to_html(index=False,border=0,escape=True)


def main():
    f=json.loads((OUT/'frozen_hypotheses.json').read_text());r=json.loads((OUT/'results.json').read_text())
    discovery=pd.read_csv(OUT/'discovery_trajectories.csv',parse_dates=['anchor'])
    replication=pd.read_csv(OUT/'replication_trajectories.csv',parse_dates=['anchor'])
    summaries=[]
    for key,name in NAMES.items():
        a=f['discovery'][key];b=r['replication'][key]
        direction='증가' if a['direction']>0 else '감소'
        summaries.append({'발견 기간에서 정한 방향':name+' '+direction,'4~6월 일치 비율':f"{a['concordance']:.1%}",
                          '8~9월 일치 비율':f"{b['concordance']:.1%}",'후기 99.17% 구간':f"{b['lower']:.1%}–{b['upper']:.1%}",
                          '판정':'반복 확인 못함' if b['lower']<=.5 else '탐색적 반복 관찰'})
    pd.DataFrame(summaries).to_csv(OUT/'hypothesis_summary.csv',index=False)
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(13,7.4))
    for ax,(key,name) in zip(axes.flat,NAMES.items()):
        for data,period,color in [(discovery,'Apr–Jun','#2563eb'),(replication,'Aug–Sep','#d97706')]:
            for role,style in [('case','-'),('control','--')]:
                q=data[(data.role==role)&(data.offset>=-6)].groupby('offset')[key].median()
                transformed=np.exp(q) if key not in ['net_change','price_accel'] else q*100
                ax.plot(q.index,transformed,style,color=color,marker='o',ms=3,lw=1.5,label=period+' '+role)
        ax.axhline(1 if key not in ['net_change','price_accel'] else 0,color='#94a3b8',lw=.7,alpha=.6)
        ax.set_title(ENGLISH[key],fontsize=10)
        ax.set_xticks([-6,-3,-2,-1,0]);ax.set_xlabel('Hours before breakout-hour start',fontsize=9)
        ax.set_ylabel('Ratio to prior baseline' if key not in ['net_change','price_accel'] else 'Percentage points',fontsize=9)
        ax.grid(alpha=.12)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,fontsize=9)
    fig.suptitle('Pre-onset changes: median event paths do not establish predictive signals',fontsize=14)
    fig.tight_layout(rect=(0,.045,1,.95))
    fig.savefig(OUT/'trajectories.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/'trajectories.svg',bbox_inches='tight');plt.close(fig)
    image=base64.b64encode((OUT/'trajectories.png').read_bytes()).decode()
    timing=[]
    for offset in [-24,-12,-6,-3,-2,-1,0]:
        row={'급등 시간봉 시작 전':f'{-offset}시간'}
        for label,data in [('초기',discovery),('후기',replication)]:
            a=data[(data.role=='case')&(data.offset==offset)].size_accel.median()
            b=data[(data.role=='control')&(data.offset==offset)].size_accel.median()
            row[label+' 평균 체결금액 배율(급등/대조)']=f'{np.exp(a):.2f} / {np.exp(b):.2f}'
        timing.append(row)
    coverage=[]
    for label,z in [('4~6월 발견',f['matching']),('8~9월 반복 검사',r['matching'])]:
        coverage.append({'구간':label,'가격 정의 사건':z['price_defined_events'],'최종 비교 사건':z['matched_events'],
                         '체결 궤적 부족 제외':z['drops']['bad_case_trajectory'],'대조군 부족 제외':z['drops']['insufficient_controls']})
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>급등 직전 지표 변화와 가설 발견</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif;max-width:1140px;margin:44px auto;padding:0 24px;color:#172339;background:#fbfcfe;line-height:1.75}}h1{{font-size:34px;line-height:1.3}}h2{{font-size:23px;margin-top:40px}}.tag,.small{{font-size:13px;color:#52627a}}.lead{{font-size:20px}}.callout{{background:#edf4ff;border-left:4px solid #2563eb;padding:18px 22px}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #dce5ef}}th{{background:#edf2f8;white-space:nowrap}}.scroll{{overflow:auto}}img{{width:100%}}a{{color:#185bc4}}pre{{background:#edf1f6;padding:18px;overflow:auto}}</style><body>
<p class="tag">2026-10-09 · 사건 정렬 분석 → 가설 후보 → 다른 기간 반복 검사</p>
<h1>급등하기 직전,<br>실제로 무엇이 바뀌었나?</h1>
<p class="lead">직전 평균 체결금액 증가가 초기 후보로 보였지만, 후기에는 반복 확인되지 않았습니다. 이번 자료에서 안정적인 공통 전조는 찾지 못했습니다.</p>
<div class="callout"><b>분석 질문을 바꿨습니다.</b> 임의로 지표를 조합해 7일 후 급등을 맞히는 대신, 급등 시간봉의 시작 직전으로 사례를 정렬해 관측된 변화를 먼저 조사했습니다. 사례에서 발견한 방향은 고정한 뒤 후기 기간에서 다시 검사했습니다. 아래 비교 비율은 알림 적중률이 아닙니다.</div>
<h2>급등의 시작을 어떻게 정했나</h2>
<p>t 시점 마감가 대비 다음 1시간 종가가 <b>+5% 이상</b>이고, 다음 24시간 고가가 <b>+15% 이상</b>인 시간봉의 시작을 t로 정의했습니다. t 이전 6시간은 각 시간봉 상승률 +3% 미만, 누적 상승률 +5% 미만이어야 합니다. 같은 코인의 사건은 처음 발견한 시점부터 24시간 간격을 두어 중복을 줄였습니다. 이미 끝난 급등의 최저점을 찾아 시작점으로 삼지는 않았습니다.</p>
<p>t는 시간봉 단위의 시작 근사치입니다. 실제 급등이 그 시간봉의 어느 분에 시작됐는지는 이 정의로 알 수 없습니다. t에서 계산한 ‘최근 1시간’은 <b>t−1시간~t 이전</b>이고, +5% 오른 시간봉 자체의 체결은 지표에 넣지 않았습니다. 이 실험의 +15%·24시간 기준은 앞선 +30%·7일 기준과 달라 두 실험의 적중률을 직접 비교할 수 없습니다.</p>
<h2>조사한 변화</h2>
<p>각 관측 시점마다 직전 1시간을, 그보다 앞선 18시간(t−24~−6시간)의 평소 수준과 비교했습니다. 거래금액·체결 횟수·평균 체결금액·가격 고저폭은 배율, 순매수는 비율의 차이로 측정했습니다. 가격 가속도는 최근 1시간 수익률에서 앞선 3시간 평균 시간당 수익률을 뺀 값입니다.</p>
<p>사건별 −24·−12·−6·−3·−2·−1·0시간의 궤적을 저장했습니다. 모든 계산은 그때까지 마감된 가격·체결만 사용합니다. 거래금액은 체결 횟수×평균 체결금액이므로 세 지표가 독립적인 증거는 아닙니다. 평균 체결금액 증가만으로 ‘고래 매집’이라고 해석할 수도 없습니다.</p>
<h2>실패 사례와 비교</h2>
<p>이전 실험에서 고정한 36종목을 유지했습니다. 같은 시각의 코인 중 향후 24시간 +15% 급등이 없고 동일한 유동성·최근 가격 조건을 만족하는 대조군 3개를 짝지었습니다. 매칭에는 t−6시간 시점의 거래대금·7일 변동성·24시간 수익률을 사용했습니다. 직전 6시간의 변화를 매칭으로 지워버리지 않기 위해서입니다. 표준화 척도는 4~6월 자료에서만 계산했고 거리 3 이하를 요구했습니다.</p>
<p>사례와 세 대조군 모두 일곱 관측 시점에서 체결 파일 완전성, 5분 구간 80% 이상 체결, 시작 구간 체결, 마지막 체결 5분 이내, 시간봉 대비 거래대금 오차 1% 이내를 만족해야 합니다. 하나라도 부족하면 짝 전체를 제외했습니다. 이는 자료가 충분한 사례로 분석 범위를 제한합니다.</p>
<div class="scroll">{table(coverage)}</div>
<h2>급등 직전 변화의 궤적</h2>
<p>아래는 마지막 6시간의 중앙값입니다. 실선은 급등 사례, 점선은 대조군입니다. 파랑은 4~6월, 주황은 8~9월입니다. 시점 0도 급등 시간봉 직전까지의 자료입니다. 두 선의 집계 중앙값 차이가 모든 짝에서 일관된 차이를 뜻하지는 않습니다. 그림 자체는 탐색용이며 신뢰구간이 아닙니다.</p>
<img src="data:image/png;base64,{image}" alt="급등 전 여섯 지표의 초기와 후기 사건 및 대조군 궤적">
<h2>발견한 방향을 가설로 고정하면 반복되는가</h2>
<p>발견 기간에서 급등 사례가 대조보다 높았는지 낮았는지로 가설 방향을 정했습니다. 그 방향을 유지한 채 후기 사례와 대조를 비교했습니다. 가설 방향과 일치하면 1, 동점은 0.5, 반대면 0으로 계산해 평균했습니다. 차이가 없으면 약 50%입니다. 같은 시각의 대조군을 비교하는 비율로, 알림 적중률이나 인과 효과가 아닙니다.</p>
<div class="scroll">{table(summaries)}</div>
<p>후기 구간은 여섯 지표 비교를 고려해 99.17% 구간을 표시했습니다. 일곱 날 단위 달력 블록을 10,000회 재표집했습니다. 모든 구간이 50%를 포함합니다. 후기의 독립 시간 블록은 5개뿐이므로 이 구간도 정확한 통계 보장으로 해석하면 안 됩니다.</p>
<p><b>가설 후보: ‘평소보다 한 번의 체결금액이 커지는 변화가 급등에 선행한다.’</b> 초기 일치 비율은 64.8%였지만 95% 하한은 47.4%로 불확실했습니다. 후기 일치 비율은 47.6%로 반복되지 않았습니다. 급등 사례의 직전 평균 체결금액 배율 중앙값은 초기 1.09배, 후기 1.08배였지만, 평소에도 비슷한 변화가 나타나 대조군과 안정적으로 구분하지 못했습니다.</p>
<div class="scroll">{table(timing)}</div>
<p>거래금액·체결 횟수는 초기에는 오히려 감소 방향이었고 후기에는 그 방향이 뒤집혔습니다. ‘거래량이 늘면 전조’라는 단순한 공통 규칙도 이번 사례에서 확인되지 않았습니다. 순매수 증가와 고저폭 확대는 후기 비교 비율이 50%를 넘었지만 발견 기간의 차이가 약했고 불확실성도 컸습니다.</p>
<h2>알림 가설로 채택한 것은 있는가</h2>
<p>사전에 정한 채택 조건은 <b>발견 사례 10개 이상, 방향 일치 60% 이상, 주간 블록 95% 하한 50% 초과</b>였습니다. 조건을 통과하면 최대 두 개 지표의 방향을 고정하고, 전체 발견 기간 유효 시점의 상위 10% 경계를 알림 임계값으로 삼도록 구현했습니다. 거래금액·체결 횟수는 중복 선택하지 않습니다.</p>
<p><b>통과한 지표가 0개여서 새로운 알림 규칙을 만들지 않았습니다.</b> 평균 체결 크기가 가장 그럴듯해 보였다는 이유로 기준을 완화하거나, 후기 결과를 보고 방향을 바꾸지 않았습니다. 따라서 이번 결과에는 새 규칙의 알림 적중률이 없습니다. ‘예측할 수 없다’는 증명이 아니라 ‘이 정의와 표본에서는 채택할 근거가 부족하다’는 판정입니다.</p>
<h2>다음 연구에서 분리해야 할 문제</h2>
<p>시간봉으로 급등 시작을 잡으면 분 단위 전조가 평균에 묻힐 수 있습니다. 다음 실험은 더 많은 독립 사건을 확보하고, 시작점 자체를 5분봉으로 정한 뒤 −60·−30·−15·−5분의 변화를 보는 별도 프로토콜이 적합합니다. 이번 후기 그림에서 눈에 띄는 시점을 골라 임계값을 맞추면 새로 과적합되므로, 시간 창·시작점·성공 기준을 먼저 고정해야 합니다.</p>
<p>이번 자료는 이미 다른 연구에서 가격 결과를 살펴본 기간이고, 현재 남아 있는 코인과 확보 가능한 파일 위주의 표본입니다. 순수한 신규 보류 검증은 아닙니다. 기존에 고정한 10~11월 미래 검증 코드는 수정하지 않았습니다. 신규 가설을 그 미래 실험에 조용히 섞지 않았습니다.</p>
<h2>재현과 근거 파일</h2>
<pre>python3 research/onset_study/study.py prepare
python3 -m unittest discover -s research/onset_study -p 'test_*.py' -v
python3 research/onset_study/study.py discover
python3 research/onset_study/study.py replicate
python3 research/onset_study/report.py</pre>
<p>discover 단계는 가설 파일이 이미 있으면 재적합을 거부합니다. replicate는 고정된 분석 코드·프로토콜 해시가 다르면 중단합니다. 기존 연구의 시간봉과 체결 캐시가 필요합니다. 가격·체결의 미래 불변성, 급등 기준 시점, 후속 상승 없는 급등 시간봉 제외, 시간 순서에 따른 알림 제한, 발견 구간 경계에 대한 테스트 5개가 통과했습니다.</p>
<p><a href="../protocol.json">분석 프로토콜</a> · <a href="frozen_hypotheses.json">후기 검사 전 고정한 가설 후보</a> · <a href="hypothesis_summary.csv">결과 표</a> · <a href="discovery_trajectories.csv">초기 전체 궤적</a> · <a href="replication_trajectories.csv">후기 전체 궤적</a> · <a href="../study.py">분석 코드</a></p>
<p class="small">데이터: <a href="https://public.bybit.com/spot/">Bybit 현물 체결 아카이브</a> · <a href="https://bybit-exchange.github.io/docs/v5/market/kline">시간봉 API</a>. 체결 side는 Bybit 공개 API의 taker 방향 관례로 해석했으며 아카이브 전용 명세 확인은 제한적입니다. RPI 포함, 호가·주문 취소 자료 없음.</p></body></html>'''
    (OUT/'onset_report.html').write_text(page,encoding='utf-8')
    print(OUT/'onset_report.html')


if __name__=='__main__':
    main()
