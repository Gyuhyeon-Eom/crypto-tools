"""Self-contained evidence report; every tested variant remains visible."""
from pathlib import Path
import base64,html,json,hashlib,importlib.metadata
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate import OUT


def pct(x):return '—' if x is None else f'{x*100:.1f}%'
def number(x):return '—' if x is None else f'{x:.2f}%'
def ci(x):return '—' if not x or x[0] is None else f'{pct(x[0])} ~ {pct(x[1])}'
def table(headers,rows):
    return '<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def image(name):return '<img src="data:image/png;base64,'+base64.b64encode((OUT/name).read_bytes()).decode()+'">'
NAMES={'price_logistic':'가격·거래량 / 로지스틱','price_boosting':'가격·거래량 / 부스팅','derivatives_logistic':'OI·펀딩·베이시스 추가 / 로지스틱','derivatives_boosting':'OI·펀딩·베이시스 추가 / 부스팅','scheduled':'6시간 간격 기준선','momentum':'4시간 모멘텀 기준선'}
POLICIES={'prior_4h':'직전4시간 고저점','prior_12h':'직전12시간 고저점','prior_24h':'직전24시간 고저점','confirmed_pivot':'확인된15분봉 스윙'}


def main():
    r=json.loads((OUT/'report.json').read_text());rob=json.loads((OUT/'robustness.json').read_text());ant=json.loads((OUT/'antecedents.json').read_text());cur=json.loads((OUT/'current_rules_metrics.json').read_text());null=json.loads((OUT/'null_audit.json').read_text())
    tasks={x['key']:x for x in r['tasks']};nulls={(x['task'],x['model']):x for x in null['comparisons']}
    models=list(NAMES)[:4];colors=['#237987','#4e9c92','#b56d3f','#ce9867']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,d in zip(axes,['long','short']):
        t=tasks[d+'_5_72_touch'];rows={x['name']:x for x in t['aggregate']}
        values=[rows[m]['precision']*100 for m in models]
        sl=[next(x for x in rob['stops'] if x['task']==t['key'] and x['model']==m and x['policy']=='prior_4h')['target_before_stop']*100 for m in models]
        pos=np.arange(4);ax.bar(pos-.18,values,.36,label='Target touched within72h',color='#288b97');ax.bar(pos+.18,sl,.36,label='Target before prior4h stop',color='#d68b47')
        ax.set_xticks(pos,['Price LR','Price GB','+Deriv LR','+Deriv GB']);ax.set_ylim(0,85);ax.set_title(d.upper()+' | unchanged5% target');ax.set_ylabel('Percent of alerts');ax.axhline(80,color='gray',linestyle='--',linewidth=1);ax.grid(axis='y',alpha=.15);ax.legend(fontsize=8,loc='upper left')
        for i,(a,b) in enumerate(zip(values,sl)):ax.text(i-.18,a+1,f'{a:.1f}',ha='center',fontsize=8);ax.text(i+.18,b+1,f'{b:.1f}',ha='center',fontsize=8)
    fig.suptitle('NEAR walk-forward historical tests | stop placement changes the conclusion')
    fig.savefig(OUT/'target_vs_stop.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,d in zip(axes,['long','short']):
        t=tasks[d+'_5_72_onset'];rows={x['name']:x for x in t['aggregate']}
        v=np.array([rows[m]['precision']*100 for m in models]);bounds=np.array([rows[m]['block95'] for m in models])*100
        ax.bar(np.arange(4),v,color=colors);ax.errorbar(np.arange(4),v,yerr=[v-bounds[:,0],bounds[:,1]-v],fmt='none',ecolor='#26384a',capsize=4)
        ax.set_xticks(np.arange(4),['Price LR','Price GB','+Deriv LR','+Deriv GB']);ax.set_ylim(0,11);ax.set_ylabel('Pre-onset precision (%)');ax.set_title(d.upper()+' | 15–60min before hindsight onset');ax.grid(axis='y',alpha=.15)
        for i,m in enumerate(models):ax.text(i,v[i]+.15,f"{rows[m]['hits']}/{rows[m]['alerts']}",ha='center',fontsize=9)
    fig.suptitle('Pre-onset detection is a different outcome |95% weekly block intervals')
    fig.savefig(OUT/'precursor_precision.png',dpi=170);plt.close(fig)
    body=['<h1>NEAR 선물 전조 모델: 2년 자료 심층 검증</h1>',
          '<p class="meta">분석일2026-10-17 · Bybit USDT 무기한 · 롱/숏 별도 · 코인 가격5%/10% 목표 유지</p>',
          '<div class="notice"><strong>결론: 현재 모델을 적중률80%의 전조 알리미로 사용할 근거는 없다.</strong> 목표 가격을 언젠가 찍는 비율, 움직임 전에 맞추는 비율, 전저점·전고점 손절보다 먼저 목표에 도달하는 비율을 분리했다. 기존 알리미도 검증된 전조 예측기가 아니다. 아래는 실패 결과와 유망해 보이지만 아직 확정할 수 없는 결과를 모두 포함한다.</div>',
          '<h2>자료와 검증 설계</h2><p>2024-10-17~2026-10-17 UTC의 완성된730일. NEAR/BTC/ETH/SUI/APT/SEI 각각70,080개15분봉, NEAR OI70,080개·펀딩2,190개·마크가격70,080개를 확보했다. 가격 시간축의 중복·누락·비정상OHLC를 검사했다. 가격 전용/파생지표 모델은 동일한10,488개 적격 후보로 비교했다.</p>',
          '<p>판단은 완성된 봉을 사용한 정시마다 한다. 직전1시간 절대수익률&lt;1%,4시간&lt;2%인 구간만 선택했다. 진입은 다음15분봉 시가 대용값이다. 각 모델은 과거360일 학습, 다음90일 임계값 설정, 이어지는90일 검증 순서로3회 이동했다. 경계의73시간 정답 제거와72시간 공백으로 미래 결과가 학습에 섞이지 않게 했다. 모델과 임계값은 각 검증 결과를 계산하기 전에 파일·해시로 저장했다.</p>']
    body.append(table(['구간','학습','임계값 설정','검증'],[[i+1]+[' ~ '.join(f[k]) for k in ('train','calibration','test')] for i,f in enumerate(r['folds'])]))
    body.extend(['<p>검증 기간은2026-01~10의3구간이다. 마지막 구간 끝의73시간은 미성숙 정답이라 제외했다. 6시간 알림 간격은 방향별로 적용했다. 동시 보유가 가능하므로 건별 수익률을 자본곡선이나 실제 계좌수익률로 해석하지 않는다. 기간·임계값을 결과에 맞춰 다시 고르지 않았다. 첫6개월 연구는 표본 부족 결과 그대로 보존했다.</p>',
                 '<h2>기존 알림이 전조 모델인가</h2><p>실제 assess/takeSignal 코드를 과거 데이터에 적용한 진단도 별도로 실행했다. 다만 과거 실시간 스프레드가 없어0으로 놓았고, 현재가 대신 봉 종가·5분 대신15분 간격을 사용했다. 따라서 실제 Slack 알림 기록 재현이나 실거래 백테스트로 주장하지 않는다. 아래 손절은 진입 전4시간 고저점이다.</p>'])
    body.append(table(['방향·목표','알림','72시간 목표 접촉','시작15~60분 전 적중','목표 먼저 / 손절'],[[f"{x['direction']} {x['target']:.0%}",x['alerts'],pct(x['target_touch_rate']),pct(x['precision']),pct(x['stops'][0]['target_before_stop'])] for x in cur['metrics'] if x['hours']==72]))
    body.extend(['<h2>목표 접촉률과 손절 반영 결과</h2>',image('target_vs_stop.png'),'<p>초록은 손절 없이72시간 동안 목표를 한 번이라도 찍는 비율이다. 주황은 직전4시간 저점(롱)·고점(숏)에 먼저 손절되지 않고 목표를 달성한 비율이다. 같은15분봉에서 둘 다 닿으면 손절로 처리했다. 손절 갭은 불리한 시가를 적용했다. 80% 목표는 달성되지 않았다.</p>'])
    for target in (5,10):
        body.append(f'<h3>{target}% 목표 ·72시간 · 모든 모델</h3>')
        rows=[]
        for d in ('long','short'):
            task=tasks[f'{d}_{target}_72_touch']
            for m in task['aggregate']:
                s=next(x for x in rob['stops'] if x['task']==task['key'] and x['model']==m['name'] and x['policy']=='prior_4h')
                rows.append([d,NAMES[m['name']],m['alerts'],pct(m['precision']),ci(m['block95']),pct(s.get('target_before_stop')),number(s.get('mean_net20bps_pct'))])
        body.append(table(['방향','모델','알림','목표 접촉','95% 주간 구간','4h손절 전 목표','비용 차감 평균/건'],rows))
    body.append('<p>비용은 왕복10/20bp 두 가정이며 표는20bp다. 실제 체결 비용이 아니다.24/72시간은 평가 창이며 사용자의 실제 보유기간을 뜻하지 않는다. 수익 계산은 창 끝까지 목표/손절이 없으면 그때 청산하는 가정을 추가했다. 정산 펀딩은 진입 명목금액을 고정한 근사치이고 봉 안의 정산/청산 순서는 알 수 없다. 레버리지·강제청산·자금배분은 계산하지 않았다. 평균수익률이 양수라도 아래 모든 손절 조합을 살펴본 탐색 결과이므로 우위가 입증된 것은 아니다.</p>')
    body.extend(['<h2>정말 움직이기 전에 잡았는가</h2>',image('precursor_precision.png'),'<p>시작점은 종가가5%/10% 반전한 경우를 교대로 찾고 이전 극값으로 거슬러 올라간 지점이다. 사후에 정한 정답이며 입력 지표로 넣지 않았다. 정답은 시작15~60분 전이고, 시작~확인 시간이72시간 이하인 사건만 포함한다. 이 정의가 모든 급등·급락을 대표하지 않는다.</p>'])
    rows=[]
    for d in ('long','short'):
        for m in tasks[d+'_5_72_onset']['aggregate']:
            rows.append([d,NAMES[m['name']],f"{m.get('hits',0)}/{m['alerts']}",pct(m.get('precision')),ci(m.get('block95')),f"{m.get('events_caught',0)}/{m.get('events',0)}",pct(m.get('pre_onset_recall'))])
    body.append(table(['방향','모델','적중/알림','전조 적중률','95% 주간 구간','찾은 사건/전체','사건 재현율'],rows))
    body.append('<p>10% 전조 모델은3구간 모두 최소 학습20건·임계값 설정5건의 접근 가능한 독립 사건을 충족하지 못해 학습하지 않았다. 10% 목표 접촉 모델과 혼동하면 안 된다.5% 전조 모델은 최고 관측값도 롱5/154=3.2%, 숏7/124=5.6%다. 이는 결과를 본 뒤 가장 높은 값을 고른 설명용 수치이며 배포할 모델 선정이 아니다.</p>')
    cover=[]
    for d in ('long','short'):
        for target in (.05,.1):
            items=[x for x in rob['coverage'] if x['direction']==d and x['target']==target];total=sum(x['events'] for x in items);reachable=sum(x['reachable'] for x in items)
            cover.append([d,pct(target),total,reachable,pct(reachable/total)])
    body.append(table(['방향','목표','사건 수','입력조건으로 접근 가능','접근 가능 비율'],cover))
    body.append('<p>조용한 구간만 보는 필터 자체가 많은 시작점을 놓쳤다. 따라서 이번 결과는 “어떤 전조도 없다”는 증거가 아니라 이 정의·입력·필터·모델의 성능 한계다. 필터를 바꾸어 연구할 경우 기존 검증 구간을 다시 독립 검증이라고 부를 수 없다.</p>')
    body.append('<h2>상승 전·하락 전 공통 지표를 따로 비교</h2><p>앞선360일에서 특징의 방향을 발견하고 마지막270일에서 같은 방향이 반복되는지 확인했다. 시작 직전 정시와 같은 시간대·28일 이내·과거 변동성0.5~2배인 비사건 시점을 일대일로 짝지었다. 비교 구간은 시작점 전후6시간을 제외했고 같은 대조 시점은 재사용하지 않았다. 이는 사후 사례 비교이며 예측 확률이 아니다. 주간 단위 부호 재배열과96개 가설 Holm 보정을 적용했다.</p>')
    body.append(table(['방향·목표','발견 쌍','후속 확인 쌍','보정 후 p<0.05'],[[f"{x['direction']} {x['target']:.0%}",x['pairs']['discovery'],x['pairs']['replication'],sum(y['replication']['holm_p']<.05 for y in x['features'])] for x in ant['studies']]))
    body.append('<p>5% 상승 사례는 시작 전에24시간 범위 내 위치가 낮고 BTC 대비4시간 상대수익률이 낮은 모습, 하락 사례는 가격 위치가 높은 모습이 반복됐다. 그러나96개 비교 보정 뒤 유의한 특징은0개다. 사후 저점·고점으로 시작점을 정의했기 때문에 낮은/높은 가격 위치는 정의 자체에서 생길 수 있다. 거래대금 폭증·OI 상승만으로 방향을 확인했다고 말할 수 없다. 특히10% 하락 확인 표본은2쌍뿐이다.</p>')
    body.append('<details><summary>24개 지표 ×4방향·목표: 모든 비교값</summary>')
    for s in ant['studies']:
        body.append(f"<h3>{s['direction']} {s['target']:.0%}</h3>")
        body.append(table(['지표','발견 짝차이 중앙값','확인 짝차이 중앙값','확인 높음/동일/낮음','주간 p','Holm p'],[[x['feature'],f"{x['discovery']['paired_difference_median']:.6g}",f"{x['replication']['paired_difference_median']:.6g}",'/'.join(pct(x['replication'][k]) for k in ('higher_fraction','equal_fraction','lower_fraction')),f"{x['replication']['weekly_one_sided_p']:.4f}",f"{x['replication']['holm_p']:.4f}"] for x in s['features']]))
    body.append('</details><h2>우연·시장 국면·다른 코인에서도 확인</h2><p>각 모델과 같은 주별 알림 수·6시간 간격을 가진 무작위 시점1,999회 생성 시도를 비교했다. 일부 촘촘한 주는 같은 수의 시점을 채우지 못해 제외했으므로 유효 횟수도 공개한다. 탐욕적 시점 선택이라 모든 가능한 조합을 균등하게 추출한 정식 무작위 실험은 아니다.40개 모델 비교 보정값은 탐색용이다. 임계값이나 모델은 바꾸지 않았다.</p>')
    body.append(table(['모델 비교 수','Holm<0.05 탐색 비교 수','80% 달성 모델'],[[len(null['comparisons']),sum(x['holm_p']<.05 for x in null['comparisons']),0]]))
    body.append('<p>숏의 일부 조합은 보정 후에도 같은 주 무작위 시점보다 나은 탐색 결과를 보였다. 이를 신호가 전혀 없다는 뜻으로 해석하지 않는다. 하지만 낮은 전조 적중률·낮은 사건 재현율·손절 후 성과를 해결하지 못했으므로 실전 우위나80% 근거로 바꿀 수 없다.</p>')
    body.append('<details><summary>모든 무작위 비교 · 원 p/보정 p/유효횟수</summary>'+table(['과제','모델','무작위 중앙값','원 p','Holm p','유효/1999'],[[x['task'],NAMES[x['model']],pct(x.get('null_precision_median')),f"{x.get('one_sided_p',1):.4f}",f"{x['holm_p']:.4f}",x['valid_repeats']] for x in null['comparisons']])+'</details>')
    body.append('<p>SUI/APT/SEI는 최근 자료를 보고 선정한 종목이므로 독립 검증용 종목이 아니다. NEAR에서 고정한 가격 모델·임계값을 그대로 적용한 탐색만 했다.5%/72시간 접촉률은 롱 약36~47%, 숏48~61% 범위였다. 손절·펀딩 반영 성과가 아니며 NEAR 결과를 종목들에 그대로 보장하지 않는다.</p>')
    body.append('<details><summary>유사 코인 모든 결과</summary>'+table(['종목','과제','모델','알림','목표 접촉','95% 주간 구간'],[[x['symbol'],x['task'],NAMES[x['model']],x['alerts'],pct(x['target_touch']),ci(x['block95'])] for x in rob['peer_transfer']])+'</details>')
    body.append('<h2>전저점·전고점 손절 전체 민감도</h2><p>직전4/12/24시간의 완성된 봉 고저점 및 좌우2개15분봉으로 확인된 최근 스윙을 사용했다. 스윙은 오른쪽2개 봉이 닫힌 뒤에만 사용하며24시간이 지나면 만료한다. 진입가보다 이미 불리하게 넘어간 손절선은 무효로 제외하고 건수를 표시한다. 고정 초기 손절이며 추적손절은 아니다. 거래가격 기준 트리거 가정이다. 사용자 차트에서 직접 선택한 손절과 완전히 같다는 뜻은 아니다.</p>')
    body.append('<details><summary>모든240개 손절·모델 조합</summary>'+table(['과제','모델','손절','유효/알림','목표 먼저','손절거리 중앙값','수익/건20bp','평균수익95%구간'],[[x['task'],NAMES[x['model']],POLICIES[x['policy']],f"{x['valid_stops']}/{x['alerts']}",pct(x.get('target_before_stop')),number(x.get('median_stop_distance_pct')),number(x.get('mean_net20bps_pct')),' ~ '.join(number(v) for v in x.get('net20bps_block95_pct',[]))] for x in rob['stops']])+'</details>')
    body.append('<p>일부 롱 전조 조합의 건별 평균 수익률과 개별95% 구간이 양수였다. 이는240개 조합을 비교한 뒤 보이는 후보이고 해당 구간은 다중검정 보정 전이다. 같은 검증 기간에서 좋은 손절만 골라 실전 수익성이 증명됐다고 주장하지 않는다. 손절별3구간 결과는 robustness.json에 보존했다.</p>')
    body.append('<details><summary>24/72시간 모든 모델·구간·시장 국면·극단일 제거 결과</summary>')
    for task in r['tasks']:
        body.append('<h3>'+task['key']+'</h3>')
        if not task['aggregate']:body.append('<p>독립 사건 표본 부족으로 학습 보류.</p>');continue
        body.append(table(['모델','알림','과제 적중률','전조 사건 재현율','최대 역행','극단3일 제거 적중률'],[[NAMES[x['name']],x['alerts'],pct(x.get('precision')),pct(x.get('pre_onset_recall')),number(x.get('worst_adverse_pct')),pct(x['excluding_top3_market_days']['precision'])] for x in task['aggregate']]))
        for fold in task['fold_results']:
            body.append(f"<h4>검증구간{fold['fold']+1} · 기본 발생률{pct(fold['test_base_rate'])}</h4>")
            body.append(table(['모델','알림','적중률','95% 주간 구간','AUC'],[[NAMES[x['name']],x.get('alerts',0),pct(x.get('precision')),ci(x.get('block95')),f"{x['auc']:.3f}" if x.get('auc') is not None else '—'] for x in fold['models']]))
        body.append(table(['모델','BTC7일 국면','알림','적중률'],[[NAMES[x['name']],reg,g['alerts'],pct(g['precision'])] for x in task['aggregate'] for reg,g in x['regimes'].items()]))
    body.append('</details><h2>다음 단계와 배포 판단</h2><p>기존 알림은 추세 관찰로 명시하고 전조 적중률을 붙이지 않는다. 다음 연구의 후보는 방향별 위치·상대강도·회복/이탈 확인을 구분하고, 호가·체결흐름을 추가해 사후 극값으로 정한 모양의 한계를 벗어나는 것이다. 아직 확정된 매매 규칙이 아니다. 새 조건을 고른 뒤 같은 과거 구간에서 재성공을 주장하지 말고 미래 구간에 고정해서 관찰한다.</p><p>가격5%/10% 목표와 손절 정의를 유지한 채 목표-before-stop, 시작 전 선행시간, 사건 재현율, 알림 수, 비용을 동시에 기록해야 한다.80% 주장은 최소100개 알림·8주 관찰·사건 재현율10% 이상과 주간95% 하한80%를 충족할 때만 검토한다. 이는 필요한 확인 기준이며 달성을 보장하지 않는다. 실시간 NEAR 선물 호가/체결 자료는 이번 역사적 캔들 자료에 없고, 첨부 ZIP 역시 NEAR 장기 기록이 아니었다. 새 예측 모델의 운영 전환·원격 Mac mini 배포는 하지 않았다.</p>')
    body.append('<h2>검증 가능한 자료와 한계</h2><p>미래 가격·OI·펀딩·마크가격을 바꾸어도 과거 입력이 바뀌지 않는 테스트, 확인 전 피벗 금지, 봉 내 목표/손절 모호성, 가격 갭, 숏 펀딩 방향, 미성숙 정답 제외, 구간 경계, 알림 간격을 테스트했다. 원자료·설계·모델의 해시와 전체 결과를 로컬 output에 저장했다. raw API 응답도 함께 보존했다. OI15분 공개 지연은 실제 수신 시각을 보장하지 않는 가정이며 주문 실행 지연도 재현하지 못한다.</p>')
    body.append('<p>자료 규격: <a href="https://bybit-exchange.github.io/docs/v5/market/kline">Bybit Kline</a> · <a href="https://bybit-exchange.github.io/docs/v5/market/open-interest">Open Interest</a> · <a href="https://bybit-exchange.github.io/docs/v5/market/history-fund-rate">Funding history</a>. OI는 롱 유입을 직접 뜻하지 않는다. 이번 결과는 과거 OHLC/OI/펀딩에서 얻은 후향 분석으로 실제 미래 성능과 구분해야 한다.</p>')
    body.append('<pre>data SHA256: '+r['manifest']['data_sha256']+'\nprotocol SHA256: '+r['manifest']['protocol_sha256']+'</pre>')
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>NEAR 전조 모델 심층 검증</title><style>body{max-width:1200px;margin:36px auto;padding:0 24px;font:16px/1.7 system-ui;color:#173047;background:#f7fafc}h1{font-size:30px;line-height:1.3}h2{margin-top:44px}h3{margin-top:28px}.meta{color:#587184}.notice{background:#fff0db;border-left:5px solid #bd7a23;padding:20px}.notice strong{display:block}.table{overflow:auto}table{border-collapse:collapse;width:100%;background:white;font-size:13px}th,td{padding:9px;border-bottom:1px solid #d9e4ec;text-align:left;vertical-align:top}th{background:#e4eef5}details{background:#edf3f7;padding:16px;margin:20px 0}summary{cursor:pointer;font-weight:600}img{width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}a{color:#116f85}</style><body>'+''.join(body)+'</body></html>'
    (OUT/'report.html').write_text(page)
    artifact_files=['report.json','robustness.json','antecedents.json','null_audit.json','current_rules_metrics.json','report.html']
    sources=[p for p in OUT.parent.iterdir() if p.suffix in ('.py','.mjs','.json')]
    audit={'artifacts':{n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in artifact_files}, 'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, 'dependencies':{str(p.relative_to(OUT.parent.parent.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT.parent.parent/'near_precursor/study.py',OUT.parent.parent.parent/'momentum-alert/core.mjs',OUT.parent.parent.parent/'momentum-alert/policy.json']}, 'versions':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','pyarrow','matplotlib','joblib']}}
    (OUT/'artifact_manifest.json').write_text(json.dumps(audit,indent=2))
    print(OUT/'report.html')

if __name__=='__main__':main()
