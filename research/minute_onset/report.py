"""Minute-scale evidence, fixed-rule replay and explicitly posthoc saturation diagnosis."""
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
NAMES={'quote_accel':'거래금액 증가','count_accel':'체결 횟수 증가','size_accel':'평균 체결금액 증가','net_change':'순매수 비율 상승','range_accel':'가격 고저폭 확대','price_accel':'가격 상승 가속'}
ENGLISH=['Traded notional','Trade count','Average trade size','Net-buy fraction change','Price range','Price acceleration']


def table(rows):
    return pd.DataFrame(rows).to_html(index=False,border=0,escape=True)


def main():
    f=json.loads((OUT/'frozen.json').read_text());r=json.loads((OUT/'results.json').read_text());diag=json.loads((OUT/'diagnostics.json').read_text())
    discovery=pd.read_csv(OUT/'discovery_trajectories.csv');later=pd.read_csv(OUT/'replication_trajectories.csv')
    rows=[]
    for key,name in NAMES.items():
        a=f['discovery'][key];b=r['replication'][key]
        rows.append({'가설 방향':name,'4~6월 비교 비율':f"{a['concordance']:.1%}",'8~9월 비교 비율':f"{b['concordance']:.1%}",
                     '후기 99.17% 구간':f"{b['lower']:.1%}–{b['upper']:.1%}",
                     '해석':'제한된 표본에서 같은 방향 관찰' if b['lower']>.5 else '뚜렷한 차이 미확인'})
    pd.DataFrame(rows).to_csv(OUT/'comparison_summary.csv',index=False)
    fig,axes=plt.subplots(2,3,figsize=(13,7.4))
    for ax,(key,name),english in zip(axes.flat,NAMES.items(),ENGLISH):
        for data,color,period in [(discovery,'#2563eb','Apr–Jun'),(later,'#d97706','Aug–Sep')]:
            for role,style in [('case','-'),('control','--')]:
                q=data[data.role.eq(role)].groupby('offset')[key].median()
                y=np.exp(q) if key not in ['net_change','price_accel'] else q*100
                ax.plot(q.index,y,style,color=color,lw=1.5,marker='o',ms=3,label=period+' '+role)
        ax.axhline(0 if key in ['net_change','price_accel'] else 1,color='#94a3b8',lw=.6)
        ax.set_title(english,fontsize=11);ax.set_xticks([-60,-30,-15,-5,0]);ax.tick_params(axis='x',labelsize=8)
        ax.set_xlabel('Minutes before breakout5min bar starts',fontsize=9)
        ax.set_ylabel('Ratio to prior baseline' if key not in ['net_change','price_accel'] else 'Percentage points',fontsize=9)
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.12)
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,fontsize=9)
    fig.suptitle('5-minute onset study: activity differs, usable alert precision remains low',fontsize=14)
    fig.tight_layout(rect=(0,.045,1,.95));fig.savefig(OUT/'minute_paths.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/'minute_paths.svg',bbox_inches='tight');plt.close(fig)
    image=base64.b64encode((OUT/'minute_paths.png').read_bytes()).decode()
    cutoff=float(np.exp(f['rules'][0]['cutoff']));main=r['replay']['quote_accel']['metrics'];z=diag['quote_accel'];uncapped=z['remove_daily_cap_keep24h_cooldown']
    replay=[{'구분':'사전 고정: 하루3건+24시간 재알림 제한','신호 수':main['alerts'],'적중':main['hits'],'비율':f"{main['precision']:.2%}",'포착 사건':f"{main['detected_events']}/{main['eligible_events']}"},
            {'구분':'사후 진단: 하루 상한만 제거','신호 수':uncapped['alerts'],'적중':uncapped['hits'],'비율':f"{uncapped['precision']:.2%}",'포착 사건':f"{uncapped['detected_events']}/{uncapped['eligible_events']}"},
            {'구분':'모든 조건 충족 관측(반복 포함)','신호 수':z['raw_rows'],'적중':z['raw_hit_rows'],'비율':f"{z['raw_precision']:.3%}",'포착 사건':f"{z['raw_detected_events']}/{main['eligible_events']}"}]
    coverage=[]
    for name,a in [('발견 4~6월',f['matching']),('반복 확인 8~9월',r['matching'])]:
        coverage.append({'기간':name,'가격 기준 급등':a['price_events'],'최종 비교 사건':a['matched_events'],'사례 품질 제외':a['drops']['bad_case'],'대조군 부족 제외':a['drops']['missing_controls']})
    candidate_events=pd.read_csv(OUT/'raw_event_candidates_quote_accel.csv')
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>5분봉 급등 직전 변화 검증</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif;max-width:1160px;margin:44px auto;padding:0 24px;background:#fbfcfe;color:#172338;line-height:1.75}}h1{{font-size:34px;line-height:1.3}}h2{{font-size:23px;margin-top:42px}}.lead{{font-size:21px}}.tag,.small{{font-size:13px;color:#52637a}}.callout{{padding:18px 22px;border-left:4px solid #2563eb;background:#edf4ff}}table{{width:100%;border-collapse:collapse;font-size:14px;margin:20px 0}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #dae4ef}}th{{background:#edf2f8;white-space:nowrap}}.scroll{{overflow:auto}}img{{width:100%}}a{{color:#185bc4}}pre{{padding:18px;background:#edf1f6;overflow:auto}}</style><body>
<p class="tag">2026-10-09 · 5분봉 시작점 · 거래금액·체결 빈도 변화 · 고정 규칙 반복 검사</p>
<h1>급등 직전 거래가 활발해지는 모습은 관찰됐습니다.<br>알림으로 쓰기에는 오탐이 너무 많았습니다.</h1>
<p class="lead">직전 15분의 거래금액과 체결 횟수는 후기 사례에서도 대조군보다 상대적으로 높았습니다. 한 건의 체결금액이나 순매수 우위가 커진다는 가설은 반복 확인되지 않았습니다.</p>
<div class="callout"><b>76.7%·80.0%는 적중률이 아닙니다.</b> 후기 급등 사례와 비슷한 비급등 대조군을 비교했을 때, 거래금액·체결 횟수의 평소 대비 비율이 급등 사례에서 더 높았던 비율입니다. 고정 알림 규칙은 183건 중 0건 적중했습니다. 패턴 관찰과 알림 성능을 분리해서 봐야 합니다.</div>
<h2>기준과 순서</h2>
<p>다음 5분봉 종가가 현재 마감가보다 <b>+2% 이상</b> 오르고, 다음 60분 고가가 <b>+8% 이상</b> 도달한 5분봉의 시작을 t로 정의했습니다. 직전 30분의 각 5분 상승률은 +1% 미만, 누적 상승률은 +2% 미만이어야 합니다. 같은 코인에서는 첫 사건 이후 24시간을 건너뛰었습니다. 시작점의 실제 오차는 최대 5분 수준이며 정확한 첫 급등 체결 시점을 식별한 것은 아닙니다.</p>
<p>−60·−30·−15·−5·0분 시점을 정렬했습니다. 각 시점의 <b>직전 15분</b>을, 그보다 앞선 <b>5시간(6시간 전~1시간 전)</b>의 평소 수준과 비교했습니다. 0분도 급등한 5분봉이 시작되기 전 자료만 사용합니다. 가격 가속도는 최근 15분 수익률에서 그 이전 45분 수익률의 3분의1을 뺀 값입니다.</p>
<p>이전에 고정한 36종목과 수집한 체결 자료를 재사용했습니다. 마지막 마감 시간봉 기준 30일 평균 일 거래대금 20만~2,000만 달러 및 연속 30일 자료를 요구했습니다. 직전 6시간 파일이 모두 있고 5분 구간의 80% 이상에 체결이 있으며 마지막 구간 체결이 5분 이내여야 합니다. 최근 마감한 6시간의 체결 합계와 거래소 시간봉 금액도 1% 이내로 일치해야 합니다. 미완성 시간봉의 최종 거래대금은 사용하지 않았습니다.</p>
<p>각 급등 사건에 같은 시각의 비급등 코인 3개를 짝지었습니다. 대조군은 다음 60분 +8% 상승이 없고 동일한 조건을 통과해야 합니다. 매칭은 t−60분에 알고 있던 거래대금·7일 변동성·24시간 수익률을 사용했습니다. 사례와 대조의 모든 관측 지점에 자료가 있어야 하며, 부족하면 전체 짝을 제외했습니다.</p>
<div class="scroll">{table(coverage)}</div>
<h2>반복 관찰된 변화</h2>
<p>4~6월에서 방향을 정한 뒤 8~9월에서는 바꾸지 않았습니다. 아래 수치는 사례 점수가 대조 점수보다 높으면 1, 같으면 0.5, 낮으면 0으로 계산한 평균입니다. 무차별 기준은 약 50%입니다. 지표 6개를 함께 비교해 후기에는 99.17% 구간을 표시했습니다.</p>
<div class="scroll">{table(rows)}</div>
<p>거래금액과 체결 횟수는 초기·후기에 같은 방향의 차이가 나타났습니다. 두 값은 서로 연관되므로 독립적인 두 가지 증거로 합산하면 안 됩니다. <b>현재의 후보 해석은 ‘체결 빈도와 총 거래 활동이 평소에 비해 상대적으로 높아진다’</b>입니다. 큰손의 매집이나 강한 매수 방향성을 확인한 결과는 아닙니다.</p>
<p>후기 급등 사례의 직전 거래금액 배율 중앙값은 약 {np.exp(r['replication']['quote_accel']['case_median']):.2f}배, 대조군은 {np.exp(r['replication']['quote_accel']['control_median']):.2f}배였습니다. 모든 급등 직전에 거래량이 폭발한다는 뜻은 아닙니다. 평소 수준을 유지한 사례가 같은 시각 거래가 줄어든 대조군보다 높게 나온 경우도 포함합니다.</p>
<h2>급등 전 60분의 변화</h2>
<p>실선은 급등 사례, 점선은 대조군의 중앙값입니다. 파랑은 초기, 주황은 후기입니다. 인접 시점의 직전 15분 창은 겹칠 수 있습니다. 이 선은 신뢰구간이 아니며, 선택된 사건의 사후 묘사입니다.</p>
<img src="data:image/png;base64,{image}" alt="5분봉 급등 시작 전 60분부터 직전까지의 지표 변화">
<h2>그 변화를 고정 알림으로 만들면</h2>
<p>발견 표본 10개 이상·방향 일치 60% 이상·주간 블록 95% 하한 50% 초과를 사전 채택 조건으로 정했습니다. 거래금액과 체결 횟수는 같은 활동 계열이어서 하나만 선택했습니다. 선택된 거래금액 지표의 임계값은 급등 사례에 맞추지 않고, 발견 기간의 모든 유효 5분 관측 {f['eligible_rows']:,}개에서 상위 10% 경계로 정했습니다.</p>
<p>고정한 규칙은 <b>직전 15분 거래금액 ≥ 평소 15분 환산 거래금액의 {cutoff:.2f}배</b>입니다. 8~9월 유효 관측 {r['eligible_rows']:,}개를 시간순으로 재생했습니다. 하루 최대 3건, 동일 코인 24시간 재알림 제한을 적용했고 늦은 신호로 앞선 알림을 바꾸지 않았습니다. 성공은 알림 후 0~60분 안에 위에서 정의한 급등 시작점이 오는 것입니다.</p>
<div class="scroll">{table(replay)}</div>
<p>사전 고정 규칙의 무작위 비교도 같은 시각·개수에 맞춘 1,000회 재생에서 평균 0건 적중했습니다. 그 시각들에는 유효 급등 사건이 없어 비율이나 우위 구간을 계산할 수 없었습니다. 0건은 시장 전체에서 성공 확률이 0이라는 뜻이 아닙니다. 기본 규칙에서는 최소 5분 선행·5분 처리 지연 조건도 모두 0건이었습니다.</p>
<h2>왜 패턴은 보이는데 알림은 실패했나</h2>
<p>실패 원인을 찾기 위해 추가한 다음 진단은 <b>결과를 본 뒤 실시한 사후 분석</b>입니다. 독립 확인 실험이나 새로운 최적 규칙으로 취급하지 않았습니다.</p>
<p>같은 조건이 61일 동안 <b>{z['raw_rows']:,}번</b> 충족됐고, 하루 중앙값은 <b>{z['raw_daily_median']:.0f}번</b>이었습니다. 알림 시각의 중앙값은 UTC 00:05, 가장 늦은 알림도 UTC 01:00이었습니다. 한도가 너무 일찍 소진되어 나중 신호가 잘렸습니다. 한국시간으로는 각각 오전 9시 5분, 오전 10시입니다.</p>
<p>하지만 하루 제한만 제거해도 <b>{uncapped['alerts']:,}건 중 {uncapped['hits']}건({uncapped['precision']:.2%})</b>에 그쳤습니다. 알림 정책만 바꾸면 해결된다고 볼 수 없습니다. 조건 자체가 평소에도 너무 흔한 것이 핵심 문제입니다. 같은 코인 24시간 제한은 유지한 진단이며, 그 영향까지 독립적으로 제거한 실험은 아닙니다.</p>
<p>반복 관측을 모두 포함하면 유효 급등 {main['eligible_events']}건 중 {z['raw_detected_events']}건에서 최소 한 번 이 조건이 보였습니다. 모두 5분 이상 앞선 관측도 있었습니다. 그러나 전체 조건 충족 중 60분 내 급등을 앞둔 관측은 {z['raw_hit_rows']}개, {z['raw_precision']:.3%}에 불과했습니다. ‘급등 전에 자주 보인다’와 ‘보이면 급등한다’가 다른 이유입니다. 이 반복 관측 비율을 독립 알림의 적중률로 제시하지 않습니다.</p>
<div class="scroll">{candidate_events.rename(columns={'symbol':'종목','onset_utc':'급등 시작 UTC','earliest_candidate_utc':'최초 후보 UTC','maximum_lead_minutes':'최대 선행 분','candidate_rows':'후보 관측 수','max_quote_ratio':'최대 거래금액 배율'}).to_html(index=False,border=0,float_format=lambda x:f'{x:.2f}')}</div>
<h2>이번에 얻은 판단</h2>
<p><b>거래 활동 증가를 후보 특징으로 남길 근거는 생겼지만, 단독 알림에는 채택하지 않는 것이 타당합니다.</b> 다음 모델에서는 이 특징이 다른 정보에 추가로 기여하는지 확인해야 합니다. 단순히 매수 비율·체결 크기 조건을 더하는 것은 이번 후기 결과가 뒷받침하지 않습니다. 사후에 관측된 성공 사례에 맞춰 조건을 붙이면 과적합될 수 있습니다.</p>
<p>후기 비교 사건은 10개, 시간 블록은 7개뿐입니다. 전체 과거 연구에서 같은 시장 기간을 반복해 보았고, 이번 6개 비교 보정이 모든 이전 탐색까지 보정하는 것은 아닙니다. 수집 가능한 현존 코인 중심의 생존 편향과 원자료 누락도 있습니다. 독립적인 미래 검증이나 수수료·슬리피지를 반영한 수익성 검증을 마친 결과가 아닙니다. 기존에 고정한 미래 실험과 실시간 스캐너는 수정하지 않았습니다.</p>
<h2>재현과 파일</h2>
<pre>python3 research/minute_onset/study.py prepare
python3 -m unittest discover -s research/minute_onset -p 'test_*.py' -v
python3 research/minute_onset/study.py discover
python3 research/minute_onset/study.py replicate
python3 research/minute_onset/diagnose.py
python3 research/minute_onset/report.py</pre>
<p>가설이 이미 고정돼 있으면 discover 재적합은 거부합니다. replicate는 코드·프로토콜·의존 코드 해시를 확인합니다. 테스트 6개: 시작점과 선행시간, 미래 파일 누락, 정규화와 마감 시간봉만 사용, 미래 체결의 과거 특징 불변, 과거 파일 누락, 발견 구간의 결과 경계. 모두 통과했습니다.</p>
<p><a href="../protocol.json">고정 프로토콜</a> · <a href="frozen.json">후기 보기 전 고정한 기준</a> · <a href="comparison_summary.csv">지표 비교</a> · <a href="alerts_quote_accel.csv">모의 알림 전체</a> · <a href="diagnostics.json">사후 실패 원인 진단</a> · <a href="discovery_trajectories.csv">초기 변화</a> · <a href="replication_trajectories.csv">후기 변화</a></p>
<p class="small">자료 출처: <a href="https://public.bybit.com/spot/">Bybit 공개 현물 체결 아카이브</a> · <a href="https://bybit-exchange.github.io/docs/v5/market/kline">시간봉 API</a>. RPI 체결 포함. buy/sell은 공개 API의 taker 관례로 해석했으며 아카이브 전용 정의의 확인에는 한계가 있습니다. 호가 잔량·주문 취소는 관측하지 않았습니다.</p></body></html>'''
    (OUT/'minute_report.html').write_text(page,encoding='utf-8');print(OUT/'minute_report.html')


if __name__=='__main__':
    main()
