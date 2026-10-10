"""Readable audit report for the fixed microstructure hypotheses."""
import base64
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import block_bounds

BASE=Path(__file__).resolve().parent
OUT=BASE/'output'
NAMES={'H1_buying_flat':'가격 정체 + 매수 흐름 증가',
       'H2_oi_funding':'OI 증가 + 펀딩 하락',
       'H3_absorption':'압축 후 매도 압력 흡수',
       'H4_venue_lead':'Binance 가격·매수 선행',
       'H3b_absorption_breakout':'흡수 이후 초기 돌파 (보조)'}
ENGLISH=['Buy flow + flat price','OI growth + falling funding','Compression → absorption','Binance leads Bybit']


def table(df):
    return df.to_html(index=False,escape=True,border=0,float_format=lambda x:f'{x:.3f}')


def main():
    results=json.loads((OUT/'results.json').read_text())
    audit=json.loads((OUT/'audit.json').read_text())
    samples=pd.read_parquet(OUT/'samples.parquet')
    test=samples[samples.split.eq('test')]
    primary=list(NAMES)[:4]
    rows=[]
    for rule in NAMES:
        r=results['rules'][rule]
        a=r['test']['alerts'];v=r['validation']['alerts']
        comparator=r['simple_comparators'][r['selected_simple_comparator']]['test']
        rows.append({'가설':NAMES[rule],'7월 적중/신호':f"{v['hits']}/{v['alerts']}",
                     '최종 신호':a['alerts'],'최종 적중':a['hits'],'적중률':f"{a['precision']:.1%}" if a['precision'] is not None else '—',
                     '같은 시점 무작위 기대':round(a['random_expected_hits'],2),
                     '무작위 대비':round(a['random_comparison']['lift'],2) if a['random_comparison']['lift'] is not None else None,
                     '동일 예산 단순 기준 적중':comparator['hits'],
                     '판정':'표본 부족' if r['status']=='insufficient_alert_sample' else '예측 우위 미확인'})
    pd.DataFrame(rows).to_csv(OUT/'hypothesis_summary.csv',index=False)

    # Secondary target stays descriptive and uses exactly the original alert policy.
    secondary=[]
    for rule in NAMES:
        alerts=pd.read_csv(OUT/f'{rule}_test_alerts.csv',parse_dates=['time'])
        expected=alerts.time.map(test.groupby('time').y_short.mean())
        bounds=block_bounds(pd.DataFrame({'time':alerts.time,'observed':alerts.y_short,'expected':expected}))
        secondary.append({'가설':NAMES[rule],'신호':len(alerts),'24h +15% 적중':int(alerts.y_short.sum()),
                          '동일 시점 무작위 기대':float(expected.sum()),'리프트':bounds['lift'],
                          '단측 98.75% 하한':bounds['lower_98_75']})
    pd.DataFrame(secondary).to_csv(OUT/'secondary_target.csv',index=False)
    # Data availability and source identity, not just successful API return counts.
    missing_binance=sorted(set(json.loads((OUT/'universe.json').read_text())['symbols'])-
                           set(pd.read_parquet(OUT/'binance.parquet',columns=['symbol']).symbol.unique()))
    validation_event_count=results['overview']['validation']['events']
    intervals=[]
    for rule in primary:
        r=results['rules'][rule]['test']['conditional']['risk_matched']
        intervals.append({'가설':NAMES[rule],'보정 리프트':r['lift'],
                          '단측 98.75% 하한':r['lower_98_75'],'상위 경계':r['upper_98_75'],
                          '비교군 확보 비율':f"{r['coverage']:.1%}"})

    fig,axes=plt.subplots(1,2,figsize=(12.5,5.3),gridspec_kw={'width_ratios':[1.15,1]})
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False})
    x=np.arange(4)
    hit=np.array([results['rules'][k]['test']['alerts']['hits'] for k in primary])
    expected=np.array([results['rules'][k]['test']['alerts']['random_expected_hits'] for k in primary])
    axes[0].barh(x+.17,hit,.31,color='#2463eb',label='Observed hits')
    axes[0].barh(x-.17,expected,.31,color='#b5c2d6',label='Matched-time random expectation')
    axes[0].set_yticks(x,ENGLISH,fontsize=10)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Hits: +30% within 7 days, quiet first 24h')
    axes[0].set_title('No clear advantage in final period',fontsize=12)
    axes[0].set_xlim(0,22)
    for i,k in enumerate(primary):
        a=results['rules'][k]['test']['alerts']
        axes[0].text(hit[i]+.35,i+.17,f"{a['hits']}/{a['alerts']}",va='center',fontsize=9)
    axes[0].legend(frameon=False,fontsize=8,loc='lower right')
    for i,k in enumerate(primary):
        b=results['rules'][k]['test']['conditional']['risk_matched']
        axes[1].plot([b['lower_98_75'],b['upper_98_75']],[i,i],color='#8b9bb2',lw=3)
        axes[1].scatter([b['lift']],[i],color='#2463eb',s=45,zorder=4)
    axes[1].axvline(1,color='#ce5c42',ls='--',lw=1.3)
    axes[1].set_yticks(x,['H1','H2','H3','H4'])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0,3.1)
    axes[1].set_xlabel('Lift vs time / liquidity / volatility controls')
    axes[1].set_title('All exploratory intervals include 1×',fontsize=12)
    fig.suptitle('Frozen microstructure hypotheses · Aug 1–Oct 2, 2026',fontsize=15,y=1.02)
    fig.tight_layout(pad=2)
    fig.savefig(OUT/'hypotheses.png',dpi=170,bbox_inches='tight')
    fig.savefig(OUT/'hypotheses.svg',bbox_inches='tight')
    plt.close(fig)
    embedded=base64.b64encode((OUT/'hypotheses.png').read_bytes()).decode()
    model_rows=[]
    for name,m in results['models'].items():
        t=m['test']
        model_rows.append({'모델':name,'최종 PR-AUC':t['pr_auc'],'최종 ROC-AUC':t['roc_auc'],
                           '신호':t['alerts'],'적중':t['hits'],'무작위 기대':t['random_expected_hits']})
    coverage=results['coverage']
    dropped=coverage['price_eligible']-coverage['complete_case']
    protocol_digest=hashlib.sha256((BASE/'protocol.json').read_bytes()).hexdigest()
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>급등 전조 가설 검증 · 체결/OI/펀딩/거래소 비교</title><style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif;color:#172033;background:#edf1f7;line-height:1.8;margin:0}}
main{{max-width:1120px;margin:36px auto;padding:44px 48px;background:white;border-radius:18px}}h1{{font-size:32px;line-height:1.35;letter-spacing:-1px}}h2{{font-size:22px;margin-top:38px}}
.eyebrow{{font-size:12px;color:#607089;letter-spacing:2px}}.summary{{background:#edf4ff;padding:20px 24px;border-left:4px solid #2463eb;margin:24px 0}}.note{{background:#fff4e4;padding:18px 24px;border-radius:8px}}
img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}}th,td{{border-bottom:1px solid #dbe2eb;text-align:left;padding:10px 9px}}th{{background:#f3f6fa}}
.scroll{{overflow-x:auto}}code,pre{{background:#f2f5fa}}pre{{padding:16px;font-size:13px;overflow:auto}}a{{color:#1756bc}}footer{{border-top:1px solid #dbe2eb;margin-top:36px;padding-top:20px;font-size:12px;color:#607089;overflow-wrap:anywhere}}
</style><main><div class="eyebrow">CRYPTO TOOLS · HYPOTHESIS STUDY 02 · 2026-10-09</div>
<h1>체결·포지션·거래소 간 변화는 급등보다 먼저 나타나는가?</h1>
<div class="summary"><b>고정한 네 가설에서, 이번 목표를 예측하는 확실한 우위를 확인하지 못했습니다.</b><br>
매수 흐름 증가: 121개 신호 중 17개 적중, 같은 시점 무작위 기대 16.58개.<br>
OI·펀딩: 93개 중 11개, 무작위 기대 12.09개. 흡수 패턴은 23개 신호로 표본이 부족합니다.
이는 모든 미시구조 신호의 부재가 아니라, 아래에 명시한 조건·시간 해상도·종목군에 대한 결과입니다.</div>
<h2>검증 대상을 먼저 고정</h2>
<p>새 특징과 결과의 관계를 보기 전에 <a href="../protocol.json">분석 프로토콜</a>에 가설·임계값·평가 방법을 기록했습니다.
기존 목표를 유지해, 신호 시점 이후 첫 24시간은 +10% 고가를 넘지 않고 그 뒤 7일 안에 Bybit 고가가 +30%에 도달해야 성공입니다.
신호 당시 최근 24시간 상승률이 +5% 이상이거나 최근 7일 상승률이 +10% 이상이면 제외했습니다.</p>
<p>보조 목표는 첫 3시간 +3% 미만을 유지한 뒤 24시간 안에 +15% 도달입니다. 보조 목표의 결과로 주 목표의 실패를 성공으로 바꾸지 않았습니다.
고가 도달은 실제 체결 가능한 매매 수익을 보장하지 않으며, 수수료·호가 깊이·슬리피지를 포함하지 않습니다.</p>
<ul><li><b>H1 매수 흐름:</b> Bybit 최근 6시간 수익률 ±2% 이내 + Binance 최근 6시간 매수 주도 체결 비율 ≥55% + 이전 18시간 대비 ≥5%p 증가 + 거래량 비율 ≥1.25배.</li>
<li><b>H2 포지션 변화:</b> 가격 정체 + OI 24시간 증가율 ≥5% + 최근 24시간 정산 펀딩 합계 ≤0이며 직전 24시간보다 하락.</li>
<li><b>H3 흡수 순서:</b> 먼저 최근 24~6시간 가격 범위가 이전 기준의 75% 이하로 압축 → 이후 6시간 매수 체결 비율 ≤45%로 매도 우세지만 가격 하락은 0.5% 이내, 거래량은 ≥1.25배.</li>
<li><b>H4 거래소 선행:</b> Binance/Bybit 가격비가 과거 168시간 중앙값보다 0.2% 이상 높고, Binance 1시간 수익률이 더 높으며 매수 비율 ≥55%.</li></ul>
<p>H3에서 과거 저항도 돌파하는 변형을 별도로 기록했지만, 표본이 적다고 조건을 사후 완화하지 않았습니다.
다른 테마 종목의 선행 여부나 내부자 매집 자체는 이번 가설에 포함하지 않았습니다.</p>
<h2>데이터·정보 시점·표본</h2>
<p>2026년 3월 31일까지 기록된 90개 이상 일봉과 당시 최근 30일 거래대금으로 종목을 고르고,
현재 조회 가능한 Binance 현물·Bybit 무기한 계약과 교집합을 취해 114개를 선정했습니다.
이후 매 시점 최근 30일 Bybit 현물 거래대금이 $200K~$20M인 경우만 분석합니다. BTC는 시장 비교에만 사용합니다.</p>
<div class="scroll">{table(pd.DataFrame([{'자료':k,**v} for k,v in audit['counts'].items()]))}</div>
<p>Binance 과거 데이터가 비어 있는 종목: {html.escape(', '.join(missing_binance) or '없음')}.
가격 조건에 해당한 {coverage['price_eligible']:,}개 관측 중 필요한 자료가 모두 있는 {coverage['complete_case']:,}개를 사용했습니다.
자료 부족 {dropped:,}개는 제외했고, 누락 항목별 개수는 결과 JSON에 보존했습니다.</p>
<p>시간봉은 마감 시점부터 사용합니다. OI는 4시간 간격 관측에 <b>1시간 공개 지연</b>을 적용하고 오래된 값을 무제한 채우지 않습니다.
펀딩은 실제 정산 시각의 수치를 1시간 뒤부터 사용하며, 종목별 정산 간격을 가정하지 않고 24시간 합계를 계산합니다.
매수·매도 흐름은 <b>Binance 현물의 실제 taker quote volume</b>이며, Bybit 현물 체결 흐름을 복원한 값이 아닙니다.
원본 기록의 실제 최초 공개 시각은 별도로 확보하지 못했으므로, 1시간 지연은 보수적으로 정한 이용 가능 시점의 가정입니다.</p>
<div class="scroll">{table(pd.DataFrame([{'기간':s,**v} for s,v in results['overview'].items()]))}</div>
<p>4~6월 학습, 7월 검증, 8~10월 최종 비교로 나눴습니다. 결과 기간이 다음 구간과 겹치는 마지막 7일의 학습·검증 관측은 제거했습니다.
관측은 4시간마다 하고, 신호는 시간순으로 처리해 하루 최대 3개·동일 종목 7일 재알림 제한을 적용했습니다.
그날 나중에 나온 좋은 신호로 앞선 신호를 교체하는 식의 미래 정보 사용은 하지 않습니다.</p>
<div class="note"><b>최종 기간의 의미:</b> 8~10월 가격 결과는 앞선 일봉 연구에서 이미 살펴봤습니다.
이번에는 새로운 미시구조 특징과 고정 가설을 추가 검증했으며, 완전히 미관측한 미래 기간의 확증 실험이라고 부를 수 없습니다.
7월 검증 구간의 관찰 가능한 급등 사건은 {validation_event_count}개뿐이라 모델 임계값 선택도 불안정합니다.</div>
<img src="data:image/png;base64,{embedded}" alt="가설별 실제 적중과 같은 시점 무작위 기대, 비교군 보정 리프트">
<h2>신호로 재생했을 때의 결과</h2>
<div class="scroll">{table(pd.DataFrame(rows))}</div>
<p>‘무작위 기대’는 각 신호와 <b>같은 시각에 조건을 충족한 종목군</b>의 성공률을 합한 값입니다.
실제로 무작위 매매를 실행한 결과는 아닙니다. ‘동일 예산 단순 기준’은 같은 시각·같은 개수의 후보를 가격·거래량·변동성으로 고르는 비교군이며,
세 기준 중 어떤 것을 사용할지는 7월 결과로 선택했습니다. 서로 다른 가설의 신호를 합산하면 중복 사건이 생길 수 있습니다.</p>
<h2>시장·유동성·변동성을 맞춰도 차이가 있는가?</h2>
<p>전체 관측 중 해당 가설이 나타난 시점과, 같은 시각·비슷한 거래대금·비슷한 변동성에서 가설이 나타나지 않은 종목을 비교했습니다.
비교군이 3개 미만이면 그 관측은 보정 비교에서 제외했습니다. 1배는 차이가 없음을 뜻합니다.</p>
<div class="scroll">{table(pd.DataFrame(intervals))}</div>
<p>구간은 7일 달력 블록을 재추출한 탐색적 부트스트랩 결과입니다. 네 주 가설을 고려해 단측 98.75% 하한을 사용했습니다.
최종 기간의 블록 수가 적고 이웃 블록 간 의존성이 남을 수 있어 정밀한 유의성 검정으로 해석하지 않습니다.
H3·H3b처럼 희소한 가설에서 계산된 비율은 특히 불안정합니다. 표본 부족은 가설이 틀렸다는 증거가 아닙니다.</p>
<h2>더 짧은 급등 목표에서는?</h2>
<div class="scroll">{table(pd.DataFrame(secondary))}</div>
<p>동일한 신호를 ‘첫 3시간 조용함 + 24시간 내 +15%’ 기준으로 다시 읽은 보조 분석입니다.
적중 사건이 매우 적어, 이 기간에서는 수시간 선행 효과도 확정하기 어렵습니다. 목표와 임계값을 바꿔가며 가장 좋은 숫자만 고르지 않았습니다.</p>
<h2>모델에 추가하면 개선되는가?</h2>
<div class="scroll">{table(pd.DataFrame(model_rows))}</div>
<p>같은 관측·같은 로지스틱 회귀 설정으로 가격·거래량·시장 특징만 사용한 모델과, 체결·OI·펀딩·가격차 특징을 추가한 모델을 비교했습니다.
최종 PR-AUC는 약 0.136에서 0.135로, ROC-AUC는 약 0.509에서 0.507로 바뀌었습니다. 이번 설정에서는 추가 정보의 예측 개선이 확인되지 않았습니다.
각 모델의 임계값은 7월에서만 정했고, 최종 숫자로 모델이나 조건을 재선택하지 않았습니다. 확률 보정과 실제 매매 성과 검증은 수행하지 않았습니다.</p>
<h2>이 결과가 다음 결정을 어떻게 바꾸는가</h2>
<p><b>H1·H2·H4를 현재 조건 그대로 급등 알림의 핵심 근거로 넣을 증거는 부족합니다.</b>
H3의 과정 가설은 거래소가 다른 체결 자료와 1시간 단위 집계의 한계가 있고, 독립 신호 수도 적습니다.
Bybit 자체 체결·호가를 1~5분 단위로 관측해 흡수 과정의 정확한 순서를 기록하는 후속 실험이 더 적절합니다.
공개 Bybit 현물 체결 아카이브의 존재와 예시 파일 형식은 확인했지만, 이번 전체 패널에는 해당 원시 틱을 통합하지 않았습니다.</p>
<p>검증할 후속 조건은 새 자료를 보기 전에 다시 고정하고, 이번에 이미 본 기간은 탐색 자료로 취급해야 합니다.
새 미래 기간에서 희소 신호를 충분히 모으기 전에는 신호를 늘리기 위한 사후 임계값 완화를 하지 않는 편이 좋습니다.</p>
<div class="note"><b>범위 제한:</b> 현재 조회 가능한 공통 상장 종목으로 제한돼 생존·상장 선택 편향이 있습니다.
신규 상장·Binance 미상장 소형 코인의 급등으로 일반화할 수 없습니다. OI 증가는 롱 매집 자체를 뜻하지 않으며, 체결 방향도 주문자 신원이나 의도를 보여주지 않습니다.
H4는 같은 코인의 두 거래소 관계를 시간봉으로 검사한 것이며, 초·밀리초 수준의 가격 전파나 테마 종목 간 선행 효과를 검사한 것이 아닙니다.</div>
<h2>재현·증거 파일</h2>
<pre>cd /Users/eomgyuhyeon/projects/crypto-tools
python3 research/microstructure/fetch.py
python3 -m unittest discover -s research/microstructure -p 'test_*.py' -v
python3 research/microstructure/analyze.py
python3 research/microstructure/report.py</pre>
<p>수집기는 저장된 원본 캐시를 재사용합니다. 6개 테스트로 미래 데이터 추가 시 과거 특징 불변, OI 공개 지연과 노후화,
실제 시간 기준 펀딩 합산, 선행 시간·같은 봉 손절 처리, 결측 시간, 시간순 알림 제한을 확인했습니다.
연구 코드에서는 거래·알림을 실행하지 않습니다.</p>
<ul><li><a href="hypothesis_summary.csv">가설별 결과 표</a></li><li><a href="secondary_target.csv">24시간 보조 목표 결과</a></li>
<li><a href="results.json">모든 기간·가설·비교군·모델 결과</a></li><li><a href="audit.json">API 수집·원본 해시·오류 기록</a></li>
<li><a href="universe.json">종목군 선정과 제외 목록</a></li><li><a href="hypotheses.svg">차트 SVG</a></li></ul>
<footer>프로토콜 SHA-256: {protocol_digest}<br>
자료 정의: <a href="https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/market">Binance 현물 Kline / taker quote volume</a> ·
<a href="https://bybit-exchange.github.io/docs/v5/market/open-interest">Bybit OI</a> ·
<a href="https://bybit-exchange.github.io/docs/v5/market/history-fund-rate">Bybit 정산 펀딩</a> ·
<a href="https://bybit-exchange.github.io/docs/v5/market/kline">Bybit Kline</a> ·
<a href="https://public.bybit.com/spot/">Bybit 현물 체결 아카이브</a></footer></main></html>'''
    (OUT/'hypothesis_report.html').write_text(page)
    print('Report:',OUT/'hypothesis_report.html')
    print(pd.DataFrame(secondary).to_string(index=False))


if __name__=='__main__':main()
