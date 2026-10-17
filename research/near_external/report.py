"""Render all external-feature results, including the acquisition amendment and failures."""
from pathlib import Path
import base64,hashlib,html,json,importlib.metadata
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from features import BASE,OUT

GROUPS={'baseline':'기존 가격·OI·펀딩','sector':'+ L1·AI 섹터','spot':'+ 현물 체결 수급','chain48':'+ 체인 지표48h','account':'+ 롱·숏 계좌 비율','sector_spot':'+ 섹터·현물 수급','all48':'전체 추가48h','all24':'전체 추가24h'}
POLICIES={'prior_4h':'직전4h고저점','prior_12h':'직전12h고저점','prior_24h':'직전24h고저점','confirmed_pivot':'확인된15분봉스윙'}

def pct(x):return '—' if x is None else f'{x*100:.1f}%'
def num(x):return '—' if x is None else f'{x:.2f}%'
def ci(x):return '—' if not x or x[0] is None else ' ~ '.join(pct(v) for v in x)
def label(name):
    if name=='scheduled':return '6시간 간격 기준선'
    g,k=name.rsplit('_',1);return GROUPS[g]+' / '+('로지스틱' if k=='logistic' else '부스팅')
def table(headers,rows):
    return '<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def img(name):return '<img src="data:image/png;base64,'+base64.b64encode((OUT/name).read_bytes()).decode()+'">'


def main():
    r=json.loads((OUT/'results.json').read_text());q=json.loads((OUT/'quality.json').read_text());d=json.loads((OUT/'diagnostics.json').read_text());m=json.loads((OUT/'manifest.json').read_text())
    tasks={t['key']:t for t in r['tasks']};diags={(x['task'],x['name']):x for x in d['comparisons']}
    group_order=list(GROUPS)
    fig,axes=plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
    for ax,direction in zip(axes,['long','short']):
        rows={x['name']:x for x in tasks[direction+'_5_72']['aggregate']}
        y=np.arange(8);a=np.array([rows[g+'_boosting']['precision']*100 for g in group_order])
        bounds=np.array([rows[g+'_boosting']['block95'] for g in group_order])*100
        ax.barh(y,a,color=['#287f91']+['#70a99f']*6+['#c7a27e']);ax.errorbar(a,y,xerr=[a-bounds[:,0],bounds[:,1]-a],fmt='none',ecolor='#284557',capsize=3)
        ax.set_yticks(y,['Baseline','+ Sector','+ Spot flow','+ Chain48h','+ Accounts','+ Sector & spot','All48h','All24h']);ax.invert_yaxis();ax.set_xlim(0,85)
        ax.set_title(direction.upper()+' | target5% before prior4h stop');ax.set_xlabel('Precision (%) | weekly95% interval');ax.axvline(80,color='gray',linestyle='--');ax.grid(axis='x',alpha=.15)
        for i,g in enumerate(group_order):ax.text(62,i,f"{a[i]:.1f}% / n={rows[g+'_boosting']['alerts']}",va='center',fontsize=8)
    fig.suptitle('NEAR | external-feature ablation |72h evaluation window | frozen boosting model family')
    fig.savefig(OUT/'ablation.png',dpi=170);plt.close(fig)
    features=['l1_btc_residual1','ai_btc_residual1','spot_delta1','chain48_stable7','chain48_dex_build','accounts_change4']
    heat=np.array([[next(x['correlation'] for x in d['lead_lag'] if x['feature']==f and x['future_hours']==h) for h in (0,1,4,24)] for f in features])
    fig,ax=plt.subplots(figsize=(10,5.3),layout='constrained');v=max(.1,float(np.max(abs(heat))));im=ax.imshow(heat,cmap='RdBu_r',vmin=-v,vmax=v,aspect='auto')
    ax.set_xticks(range(4),['Concurrent\n1h','Next 1h','Next 4h','Next 24h']);ax.set_yticks(range(6),['L1 residual','AI residual','Spot taker delta 1h','Stablecoin change 7d','DEX volume build','Account change 4h'])
    for i in range(6):
        for j in range(4):ax.text(j,i,f'{heat[i,j]:+.3f}',ha='center',va='center',color='white' if abs(heat[i,j])>.6*v else '#163148')
    ax.set_title('NEAR external indicators: concurrent vs future returns\nBTC beta adjusted | exploratory quiet-candidate sample',fontsize=12,pad=12);fig.colorbar(im,ax=ax,label='Pearson correlation');fig.savefig(OUT/'lead_lag.png',dpi=170);plt.close(fig)
    body=['<h1>NEAR 외부 지표 적용·검증 결과</h1><p class="meta">2026-10-17 ·730일 자료 · 가격±5%/±10% 목표 유지 · 롱/숏 별도</p>',
          '<div class="notice"><strong>외부 지표를 추가했지만, 이번 검증에서 안정적인 성능 개선은 확인하지 못했다.</strong> 섹터·현물 체결 수급을 추가한 일부 조합의 관측 적중률은 조금 높아졌지만, 전체 지표 결합은 오히려 나빠진 경우가 많았다.112개 추가 지표 비교의 확률 예측오차 개선을 주간 검정·Holm 보정한 결과 유의한 조합은0개다.80% 적중률의 전조 모델은 확보하지 못했다.</div>',
          '<h2>무엇을 실제로 적용했나</h2><p>이전 연구의 가격·OI·펀딩·베이시스24개 지표에 현 시점에서 계산한 손절거리와 외부 지표31개를 연결했다. 최종 공통 후보 10,192개를 모든 모델에서 동일하게 사용했다. 원래 조용한 구간 후보 10,488개 중 296개는 주로 30일 시장 베타 준비기간 등으로 빠졌다. 이전 연구와 다른 정답을 사용하므로 이전 수치와 바로 비교하지 않고 기존 지표 모델도 다시 학습했다.</p>']
    body.append(table(['자료군','입력','실제 확보','주의점'],[
        ['현물 수급','NEAR 매수/매도 주도 체결 불균형15m/1h/4h·가격괴리·현물비중','Binance NEAR15분봉70,080개','외부현금 순입금이 아니라 공격적인 체결 우위'],
        ['L1 비교','SOL/AVAX/ATOM/SUI/APT/SEI 동일가중','종목별 Bybit선물70,080개','NEAR제외,현재생존종목·기존선정편향'],
        ['AI 비교','FET/RENDER/TAO 동일가중','종목별 Binance현물70,080개','FET Bybit계약종료로 결과계산전 출처수정'],
        ['체인','USD페그스테이블코인 잔액1/7일 변화·DEX7일/28일·TVL7일','원자료1561/1883/1885개 일별행','가격효과·수정이력·실제공개시각불명'],
        ['포지션','롱계좌 비중·1/4시간 변화','Bybit1시간17,520개','포지션금액 비중이 아니라 계좌수 비중']]))
    body.append('<p>FETUSDT의 Bybit 계약이 종료돼 분석 구간의 캔들을 받을 수 없었다. 종목을 성과에 맞춰 교체하지 않고 AI 3종목 전체를 Binance 현물로 맞췄다. 이 변경은 모델 평가 전에 acquisition_amendment.json에 기록했다. L1 선물과 AI 현물의 차이, 과거 당시 섹터 구성이 아닌 현재의 고정 묶음이라는 한계가 남는다.</p>')
    body.append('<h2>검증 기준</h2><p>정답은 다음 15분봉 시가를 진입 대용 가격으로 삼고, 롱은 직전 4시간 저점·숏은 고점의 손절보다 먼저 가격 5%/10% 목표를 달성했는가다. 24/72시간 평가 창은 사용자의 실제 보유기한이 아니다. 같은 봉에서 목표와 손절에 모두 닿으면 손절로, 손절 갭은 불리한 시가로 계산했다. 입력에는 완성된 봉만 사용한다.</p><p>과거 360일 학습 → 90일 임계값 설정 → 다음 90일 평가를 3번 반복했다. 경계의 73시간 정답 제거와 72시간 공백, 6시간 알림 간격을 유지했다. 로지스틱/부스팅 2종과 입력 8종을 고정했고 모델·임계값을 각 평가 전에 저장했다. 총 128개 방향/목표/기간/입력/모델 조합, 384개 학습 파일을 남겼다.</p>')
    body.append(table(['구간','학습','임계값 설정','검증'],[[i+1]+[' ~ '.join(f[k]) for k in ('train','calibration','test')] for i,f in enumerate(r['folds'])]))
    body.append('<p>기본 후보는 정시·직전 1시간 절대변화 &lt;1%·4시간 &lt;2%이다. 이는 이전 연구에서 많은 시작점을 놓친 필터이며 이번 외부 지표 비교에서는 동일하게 유지했다. 방향에 따라 손절거리가 0인 시점을 제외했다. 다음 봉 시가가 이미 손절선을 넘은 진입 취소도 별도로 기록했다. 통계표의 알림 수는 기준설정 구간에서 고른 임계값에 따라 달라질 수 있다.</p>')
    body.extend(['<h2>5% 목표·72시간: 외부지표 추가 전후</h2>',img('ablation.png')])
    for kind in ('boosting','logistic'):
        body.append('<h3>'+('부스팅' if kind=='boosting' else '로지스틱')+' · 모든 입력 조합</h3>')
        rows=[]
        for direction in ('long','short'):
            t=tasks[direction+'_5_72']
            for g in group_order:
                name=g+'_'+kind;x=next(m for m in t['aggregate'] if m['name']==name);delta=diags[(t['key'],name)]['precision_delta_vs_baseline']
                rows.append([direction,GROUPS[g],x['alerts'],pct(x.get('precision')),ci(x.get('block95')),num(x.get('mean_net20_pct')),f"{delta['difference']*100:+.1f}%p" if delta else '—',' ~ '.join(f'{v*100:+.1f}%p' for v in delta['paired_week95']) if delta else '—'])
        body.append(table(['방향','입력','알림','손절 전 목표','95%주간 구간','비용후 평균/건','기준선 대비','차이95%구간'],rows))
    body.append('<p>부스팅에서 롱은 기존 31.1% → 섹터 32.5% → 현물 33.1%, 전체48h 29.7%였다. 숏은 기존 24.1% → 섹터 26.3% → 현물 23.9% → 체인 32.2% → 전체 25.2%였다. 숏 체인의 개선은 세 구간 모두 관찰돼 후보로 남길 만하다. 다만 알림은 390건에서 171건으로 줄었고, 전체 후보의 확률 예측오차는 나빠졌다. 로지스틱에서는 같은 체인 추가가 35.4% → 31.7%로 악화됐다.</p>')
    body.append('<p>예를 들어 롱 전체48h 로지스틱의 비용 후 평균은 건당 약 +0.16%였지만 주간 95% 구간은 약 −0.36%~+0.67%로 0을 포함한다. 숏 체인 부스팅의 평균은 약 −0.19%였다. 비용은 왕복 20bp 가정이며 10bp도 함께 계산했다. 타임아웃은 평가 창 끝 청산 가정이고 펀딩은 진입 명목금액 고정 근사다. 동시 보유 가능한 건별 수익을 실제 계좌 수익률로 해석하지 않는다.</p>')
    body.append('<h2>가격보다 먼저 움직였는가</h2>'+img('lead_lag.png')+'<p>0시간 열은 직전 1시간의 동행 관계이고 나머지는 앞으로 1/4/24시간이다. NEAR 수익률에서 과거 30일 자료로 추정한 BTC 베타 효과를 뺐다. L1·AI·현물 수급의 동행 상관은 각각 약 0.44·0.37·0.21이었지만 다음 1시간 상관은 거의 0이었다. 이는 동일 검증 구간의 추가 탐색이며 24개 상관관계를 선택 보정한 확정 증거가 아니다. 같은 시간의 상관관계가 높아도 미래 수익 예측을 의미하지 않는다.</p>')
    body.append('<details><summary>모든 선행·동행 상관계수와 주간 구간</summary>'+table(['지표','미래시간(0=동행)','표본','상관계수','95%주간 구간'],[[x['feature'],x['future_hours'],x['n'],f"{x['correlation']:+.4f}",' ~ '.join(f'{v:+.4f}' for v in x['block95'])] for x in d['lead_lag']])+'</details>')
    body.append('<h2>통계적으로 개선을 확인한 비교는?</h2><p>확률 예측의 제곱오차(Brier)가 기존 모델보다 줄었는지를 같은 후보에서 짝지어 비교했다. 양수 개선값이 좋다. 주 단위 9,999회 부호 재배열과 주간 부트스트랩을 사용하고, 112개 추가 조합을 Holm 보정했다. 보정 후 p&lt;0.05인 개선은 0개다. 이것은 시험한 모델·지표·후보 조건에서 개선을 입증하지 못했다는 뜻이며 모든 외부 정보가 무의미하다는 증명은 아니다.</p>')
    body.append('<p>적중률 자체의 차이도 결과 확인 후 보조 검정했다. 좋은 조합만 골라 검사하지 않고 추가 지표 112개 비교 전체에 주별 모델 결과 교환과 Holm 보정을 적용했다. 여기서도 보정 후 유의한 개선은 0개였다. 숏 체인 부스팅의 차이는 +8.1%p, 개별 주간95% 구간은 +2.8~+13.6%p였지만, 보조 검정의 원 p=0.004가 전체 비교 보정 후 0.448이 됐다. 이 검정은 사후 탐색이며 미래 독립 검증을 대체하지 않는다.</p>')
    body.append('<details><summary>112개 개선 검정 전체</summary>'+table(['과제','입력/모델','Brier개선','95%주간 구간','구간별개선','원p','Holm p'],[[t['key'],label(x['name']),f"{x['incremental']['mean_brier_improvement']:+.6f}",' ~ '.join(f'{v:+.6f}' for v in x['incremental']['paired_week95']),', '.join(f'{v:+.5f}' for v in x['incremental']['fold_brier_improvements']),f"{x['incremental']['week_sign_p']:.4f}",f"{x['incremental']['holm_p']:.4f}"] for t in r['tasks'] for x in t['aggregate'] if 'incremental' in x])+'</details>')
    body.append('<p>기존/전체48h 모델에는 같은 주별 알림 수·6시간 간격의 무작위 시점 499회 생성 시도도 비교했다. 이 추가 진단은 탐욕적 추출이라 가능한 시점 조합을 균등 추출한 정식 무작위 실험이 아니다. 비교별 유효 횟수를 보존했다. 외부 지표의 증분 평가는 위의 동일 후보 기준선 비교가 중심이다.</p>')
    body.append('<details><summary>같은 주 무작위 시점 대조</summary>'+table(['과제','모델','무작위 적중률 중앙값','원p(탐색)','유효/499'],[[x['task'],label(x['name']),pct(x['matched_random'].get('null_precision_median')),f"{x['matched_random'].get('one_sided_p',1):.4f}",x['matched_random']['valid_repeats']] for x in d['comparisons'] if x.get('matched_random')])+'</details>')
    body.append('<h2>모든 기간·목표·손절 결과</h2><p>5% 목표를 낮추거나 10% 실패를 숨기지 않았다. 기본 학습 정답은 4시간 손절이고 12/24시간 및 확인 스윙은 같은 알림에 적용한 민감도다. 다른 손절에 맞춰 별도 최적화한 모델이 아니다. 각각의 개별 구간을 보고 최고 조합만 선택하면 다중 선택 편향이 생긴다.</p>')
    for t in r['tasks']:
        body.append('<details><summary>'+t['key']+' · 모든 모델/구간/손절</summary>')
        body.append(table(['모델','알림','손절전목표','95%주간구간','평균/건20bp','전조적중률','찾은시작점/전체','Brier'],[[label(x['name']),x['alerts'],pct(x.get('precision')),ci(x.get('block95')),num(x.get('mean_net20_pct')),pct(x.get('pre_onset_precision')),f"{x.get('events_caught',0)}/{x.get('events',0)}",f"{x['brier']:.5f}" if 'brier' in x else '—'] for x in t['aggregate']]))
        for fold in t['folds']:
            body.append(f"<h4>검증구간{fold['fold']+1} · 공통후보{fold['counts']['test']}개 · 기본발생률{pct(fold['base_rate'])}</h4>")
            body.append(table(['모델','알림','손절전목표','95%주간구간','평균/건20bp'],[[label(x['name']),x['alerts'],pct(x.get('precision')),ci(x.get('block95')),num(x.get('mean_net20_pct'))] for x in fold['models']]))
        body.append(table(['모델','손절','유효알림/원알림','목표먼저','평균/건20bp','평균95%구간'],[[label(x['name']),POLICIES[s['policy']],f"{s['alerts']}/{s['original_alerts']}",pct(s.get('precision')),num(s.get('mean_net20_pct')),' ~ '.join(num(v) for v in s.get('net20_block95_pct',[]))] for x in t['aggregate'] for s in x['stop_sensitivity']]))
        body.append(table(['모델','BTC7일국면','알림','손절전목표','평균/건20bp'],[[label(x['name']),reg,g['alerts'],pct(g['precision']),num(g['mean_net20_pct'])] for x in t['aggregate'] for reg,g in x['regimes'].items()]))
        body.append('</details>')
    body.append('<h2>자료 해석에서 남는 한계</h2><ul><li>이번에 얻은 것은 스테이블코인 잔액·TVL·DEX 거래량이며, NEAR 실제 거래소 순입금이나 브리지 순유입은 아니다. 브리지 API는 402 응답으로 확보하지 못했다.</li><li>체인 자료는 현재 조회한 수정 가능한 역사 자료다. 24/48시간 지연을 넣어도 당시 최초 공개본을 재현하지 못한다. 분석한 730일 안에서는 세 일별 자료에 누락된 날짜가 없었다.</li><li>현재 고정한 섹터 묶음은 과거 당시 구성의 완전한 재현이 아니다. NEAR·BTC·ETH는 바스켓에서 제외했지만 종목 선정 편향은 남는다.</li><li>현물 흐름은 Binance 한 거래소 자료다. 매수 주도 체결−매도 주도 체결은 새 현금 유입량이 아니다. 롱 계좌 비율은 계좌 수 비율이지 포지션 금액 비율이 아니다.</li><li>다음 봉 시가 진입·거래 가격 손절·평가 창 끝 청산은 가정이며 실제 체결·레버리지 강제 청산·포트폴리오를 재현하지 않는다.</li><li>이 730일은 앞선 연구에서도 이미 살펴봤다. 이번 시간순 검증도 후향 탐색이며 최종 독립 검증은 미래 고정 기간이어야 한다.</li></ul>')
    body.append('<h2>판단과 다음 연구</h2><p>외부 지표 전체를 운영 알림에 바로 추가할 근거는 부족하다. 섹터·현물 체결의 가벼운 조합과 숏 체인 필터를 후보로 남기되, 지금 관측된 수 %p 차이를 확정 개선으로 부르지 않는다. 체인 잔액/TVL의 큰 변화가 곧 NEAR 매수라는 가정도 채택하지 않는다.</p><p>다음에 검증할 정보는 현재 봉 집계에 없는 실제 체결 순서·호가 흡수/소진·가격 회복/이탈과의 상호작용이다. 실수신 시각과 원자료를 함께 수집하고 조건을 미리 정한 뒤 미래 구간에서 판정해야 한다. 근거 없이 유료 데이터를 구매하거나 80%를 보장하지 않는다. 이번 연구로 모델을 승격하거나 Slack 매매 신호를 바꾸지 않았다.</p>')
    body.append('<p>자료정의: <a href="https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints">Binance 현물봉</a> · <a href="https://bybit-exchange.github.io/docs/v5/market/long-short-ratio">Bybit 계좌비율</a> · <a href="https://docs.llama.fi/analysts/data-definitions">DefiLlama 지표정의</a> · <a href="https://www.coingecko.com/en/coins/near">NEAR 범주</a>.</p>')
    body.append('<pre>Base data SHA256: '+m['base_sha256']+'\nProtocol SHA256: '+m['protocol_sha256']+'\nAcquisition amendment SHA256: '+m['amendment_sha256']+'</pre>')
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NEAR 외부지표 적용 검증</title><style>body{max-width:1250px;margin:36px auto;padding:0 24px;font:16px/1.75 system-ui;color:#183349;background:#f7fafc}h1{font-size:30px;line-height:1.4}h2{margin-top:40px}.meta{color:#597183}.notice{padding:20px;background:#fff1da;border-left:5px solid #b27b29}.notice strong{display:block}.table{overflow:auto}table{border-collapse:collapse;width:100%;background:white;font-size:13px}th,td{padding:9px;border-bottom:1px solid #dce7ef;text-align:left;vertical-align:top}th{background:#e6eff5}details{background:#edf3f7;padding:15px;margin:18px 0}summary{cursor:pointer;font-weight:600}img{width:100%;height:auto}pre{font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere}a{color:#146e85}li{margin:10px 0}</style><body>'+''.join(body)+'</body></html>'
    (OUT/'report.html').write_text(page)
    artifacts=['precision_audit.json','manifest.json','quality.json','features.parquet','results.json','diagnostics.json','report.html','ablation.png','lead_lag.png']
    sources=list(BASE.glob('*.py'))+list(BASE.glob('*.mjs'))+list(BASE.glob('*.json'))+list(BASE.glob('*.txt'))
    audit={'artifacts':{n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in artifacts},'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
           'reused_source_hashes':{str(p.relative_to(BASE.parent.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [BASE.parent/'near_deep/data.py',BASE.parent/'near_deep/evaluate.py',BASE.parent/'near_deep/stops.py',BASE.parent/'near_precursor/study.py']},
           'versions':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','pyarrow','matplotlib','joblib']}}
    (OUT/'artifact_manifest.json').write_text(json.dumps(audit,indent=2))
    print(OUT/'report.html')

if __name__=='__main__':main()
