"""Full-grid replay report; distinguish historical replay from future confirmation."""
import base64
import collections
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parent
OUT=BASE/'output'


def pct(x):
    return '—' if x is None else f'{x:.1%}'


def table(rows):
    return pd.DataFrame(rows).to_html(index=False,border=0,escape=True)


def main():
    r=json.loads((OUT/'results.json').read_text())
    audit=json.loads((OUT/'download_audit.json').read_text())
    u=json.loads((OUT/'universe.json').read_text())
    a=pd.read_csv(OUT/'alerts.csv',parse_dates=['time'])
    monthly=pd.read_csv(OUT/'monthly.csv')
    p=pd.read_parquet(OUT/'panel.parquet')
    random={x['period']:x for x in r['random']}
    labels={'early':'4~6월','july':'7월','late':'8~10월','all':'전체'}
    rows=[]
    for name,label in labels.items():
        s=r['periods'][name];f=s['flow'];v=s['volume'];z=random[name];b=s['vs_random']
        rows.append({'기간':label,'알림':f['alerts'],'급등 적중':f['hits'],'적중률':pct(f['precision']),
                     '거래량 기준 적중/알림':f"{v['hits']}/{v['alerts']}",
                     '무작위 평균 적중':f"{z['mean_hits']:.2f}",'무작위 평균 알림':f"{z['mean_alerts']:.2f}",
                     '무작위 대비':('0회 적중 · 구간 해석 보류' if f['hits']==0 else f"{b['lift']:.2f}배 ({b['lower95']:.2f}~{b['upper95']:.2f})") if b else '계산 불가'})
    pd.DataFrame(rows).to_csv(OUT/'comparison_summary.csv',index=False)
    all_result=r['periods']['all']['flow'];late=r['periods']['late']['flow'];b=r['periods']['late']['vs_random']
    supported=b and b['lower95']>1 and late['precision']>r['periods']['late']['volume']['precision']
    conclusion='후기 재생에서 개선 후보가 관찰됐지만, 독립 확인은 남아 있습니다.' if supported else '이번 고정 규칙에서 기존 알림을 개선할 예측 우위는 확인되지 않았습니다.'
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),gridspec_kw={'width_ratios':[1,1.2]})
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False})
    x=np.arange(3);names=['early','july','late']
    flow=[r['periods'][k]['flow']['hits'] for k in names]
    volume=[r['periods'][k]['volume']['hits'] for k in names]
    chance=[random[k]['mean_hits'] for k in names]
    for offset,data,color,label in [(-.24,flow,'#2563eb','Flow rule'),(0,volume,'#64748b','Volume baseline'),(.24,chance,'#cbd5e1','Random replay mean')]:
        axes[0].bar(x+offset,data,.23,color=color,label=label)
    axes[0].set_xticks(x,['Apr–Jun','July','Aug–Oct']);axes[0].set_ylabel('Hits (+30%, quiet first 24h)');axes[0].legend(frameon=False,fontsize=8)
    axes[0].set_title('Same signal-time comparison')
    for i,k in enumerate(names):
        q=r['periods'][k]['vs_random']
        if q:
            axes[1].plot([q['lower95'],q['upper95']],[i,i],color='#93b4f4',lw=4)
            axes[1].scatter(q['lift'],i,color='#2563eb',s=45,zorder=3)
            if r['periods'][k]['flow']['hits']==0:
                axes[1].text(.07,i,'0 observed hits; interval uninformative',va='center',fontsize=8,color='#64748b')
    axes[1].axvline(1,ls='--',color='#d97706');axes[1].set_yticks(x,['Apr–Jun','July','Aug–Oct']);axes[1].invert_yaxis()
    axes[1].set_xlabel('Hit count / expected random hits (95% interval)');axes[1].set_title('7-day calendar-block uncertainty')
    fig.suptitle('Fixed buy-flow shift rule · retrospective full-grid replay',fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT/'replay.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/'replay.svg',bbox_inches='tight');plt.close(fig)
    encoded=base64.b64encode((OUT/'replay.png').read_bytes()).decode()
    risk=[]
    for name,label in labels.items():
        f=r['periods'][name]['flow']
        risk.append({'기간':label,'포착 사건/유효 사건':f"{f['detected_episodes']}/{f['eligible_episodes']}",
                     '사건 포착률':pct(f['episode_recall']),'성공 알림 중앙 선행시간':f"{f['median_lead_hours']:.1f}시간" if f['median_lead_hours'] is not None else '—',
                     '7일 최저수익률 중앙값':pct(f['mae7_median']),'불리한 10% 경계':pct(f['mae7_worst10pct']),
                     '−10% 경험 알림':f['down10_count'],'−20% 경험 알림':f['down20_count']})
    sensitivity=[]
    for name,label in labels.items():
        f=r['periods'][name]['flow']
        sensitivity.append({'기간':label,'기본 +30% 적중':f['hits'],'+30% 전 −10% 미경험':f['clean_hits'],
                            '다음 시간봉 시가 기준 적중':f['next_open_hits'],'시가 기준 손절 미경험 적중':f['next_open_clean_hits'],
                            '보조 24시간 +15% 적중':f['secondary_hits'],'7일 종가 수익률 중앙값':pct(f['median_close_return7'])})
    quality=[]
    for symbol,g in p.groupby('symbol'):
        price=g[g.price_eligible]
        quality.append({'종목':symbol,'가격 조건 관측':len(price),'품질·결과 확인 관측':int((price.quality&price.mature).sum())})
    pd.DataFrame(quality).to_csv(OUT/'coverage_by_symbol.csv',index=False)
    errors=collections.Counter((x['symbol'],x['error']) for x in audit if 'error' in x)
    error_text='; '.join(f'{s}: {n}일 ({error})' for (s,error),n in errors.items()) or '없음'
    total_trades=sum(x.get('trades',0) for x in audit)
    total_bytes=sum(x.get('bytes',0) for x in audit)
    c=r['coverage']
    complete_error=float(p.loc[p.files_complete.fillna(False).astype(bool),'turnover_error'].max())
    late_random=random['late']
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>매수 흐름 고정 규칙 전체 시계열 검증</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif;max-width:1180px;margin:45px auto;padding:0 24px;color:#172338;line-height:1.75;background:#fbfcfe}}h1{{font-size:33px;line-height:1.35}}h2{{font-size:23px;margin-top:42px}}.tag,.small{{font-size:13px;color:#516178}}.lead{{font-size:21px}}.callout{{padding:18px 22px;border-left:4px solid #2563eb;background:#edf4ff}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #dae3ef}}th{{background:#edf2f8;white-space:nowrap}}.scroll{{overflow:auto}}img{{width:100%}}a{{color:#1d58ba}}pre{{background:#edf1f6;padding:18px;overflow:auto}}code{{background:#edf1f6}}</style><body>
<p class="tag">2026-10-09 · 36종목 · 4시간 간격 전수 재생 · 규칙·표본 사전 고정</p>
<h1>매수 우위 변화를 알림으로 만들면<br>실제로 급등을 더 잘 잡는가?</h1>
<p class="lead">{conclusion}</p>
<div class="callout">전체 {all_result['alerts']}개 알림 중 {all_result['hits']}개 적중, 적중률 {pct(all_result['precision'])}. 후기 8~10월은 {late['alerts']}개 중 {late['hits']}개({pct(late['precision'])})였고, 같은 시각 무작위 재생의 평균 적중은 {late_random['mean_hits']:.2f}개였습니다. 이전 사례 비교의 61.5%와 달리, 이 페이지의 적중률은 전체 시계열에서 발생시킨 모의 알림의 결과입니다.</div>
<h2>알림 규칙</h2>
<p>순매수 비율 = (매수 체결금액−매도 체결금액)/전체 체결금액. <b>최근 2시간 순매수 비율 ≥10%, 앞선 4시간보다 ≥15%p 개선</b>일 때 후보로 잡습니다. 두 임계값은 이전 순서 가설의 매수 전환 조건을 그대로 사용했습니다. 압축·매도 흡수는 필수 조건에서 제외했고, 결과를 보고 임계값을 다시 맞추지 않았습니다.</p>
<p>매 UTC 00·04·08·12·16·20시에 당시 마감된 자료만 계산합니다. 같은 시각의 후보는 순매수 개선폭 순으로 정렬하며 하루 최대 3개, 같은 코인은 168시간 재알림 금지를 적용합니다. 늦게 나온 강한 신호가 이미 보낸 알림을 대체하지 않습니다. 월 경계에서도 재알림 상태를 초기화하지 않았습니다.</p>
<h2>분석 범위와 데이터 품질</h2>
<p>기존 114개 코인을 3월 말 거래대금 기준 3등분하고 구간별 12개를 난수 시드 20261009로 추출했습니다. 종목 선정에는 급등 여부나 신규 체결 특징을 쓰지 않았습니다. 36개 종목의 4월 1일~10월 2일 00시 전체 {c['all_grid_rows']:,}개 관측에서, 당시 30일 평균 일 거래대금 20만~2,000만 달러·최근 24시간 수익률 5% 미만·최근 7일 수익률 10% 미만·연속 720시간 자료가 있는 {c['price_eligible']:,}개를 추렸습니다.</p>
<p>직전 24시간 체결 파일이 모두 있고, 5분봉의 80% 이상과 시작 5분봉에 체결이 있으며, 마지막 체결이 5분 이내이고, 시간봉 거래대금과 체결 합계 차이가 1% 이내인 관측만 사용했습니다. 품질과 향후 7일 결과를 모두 확인한 관측은 {c['quality_and_mature']:,}개입니다. 누락은 0 거래량으로 해석하지 않았습니다.</p>
<p>공식 일별 원자료 {len(audit):,}건 요청, 수집 체결 {total_trades:,}건, 압축 자료 {total_bytes/1e9:.2f} GB. 다운로드 실패: {error_text}. 실패 종목을 다른 종목으로 교체하지 않았습니다. 필요한 파일이 모두 있는 관측에서 체결·시간봉 거래대금 최대 상대 오차는 {complete_error:.2e}였습니다. 표본별 누락은 <a href="coverage_by_symbol.csv">종목별 품질 표</a>에 있습니다.</p>
<h2>같은 시각의 비교 기준</h2>
<p>기본 성공 정의는 <b>알림 후 첫 24시간 고가 +10% 미만, 25~168시간 내 첫 +30% 터치</b>입니다. 비교 기준은 동일하게 유효한 코인 중 ①최근 6시간 거래량/이전 168시간 거래량 비율이 큰 코인, ②균등 무작위 코인입니다. 실제 규칙이 알림을 발생시킨 시각과 개수에 맞추고 각 비교 기준에도 7일 재알림 제한을 독립적으로 적용했습니다.</p>
<p>무작위 기준은 시드 고정 1,000회 전체 재생입니다. 재알림 제한으로 슬롯을 채울 수 없는 경우 선택을 꾸며내지 않으며, 아래 평균 알림 수가 그 차이를 보여줍니다. 비교는 ‘이 시각에 어느 코인을 고르는가’에 대한 것이며, 신호가 없는 시각과 비교한 알림 타이밍 자체의 우위는 측정하지 않습니다.</p>
<div class="scroll">{table(rows)}</div><img src="data:image/png;base64,{encoded}" alt="고정 매수 전환 규칙과 거래량 및 무작위 비교 결과">
<p>괄호는 7일 달력 블록을 10,000회 재표집한 무작위 대비 적중 개수 비율의 95% 구간입니다. 무작위 적중이 없는 기간은 비율을 계산하지 않습니다. 7월처럼 관측 적중이 0이면 재표집 구간도 0으로 퇴화하므로 구간 해석을 보류합니다. 이는 실제 성공 확률이 0이라는 증거가 아닙니다. 날짜 간 의존성과 블록 수의 제약이 있으므로 근사 구간으로 봐야 합니다.</p>
<h2>얼마나 일찍 잡고, 얼마나 밀리는가</h2>
<div class="scroll">{table(risk)}</div>
<p>사건 포착률은 이 36종목의 <b>가격·체결 품질 조건을 통과한 전조 사건</b> 중 알림으로 잡은 비율입니다. 같은 코인에서 +30% 도달 시점이 첫 사건 기준 168시간 안에 모인 양성 관측을 하나로 묶었습니다. 거래소 전체 급등 포착률이 아니며 자료가 부족한 사건은 분모에서 빠집니다. 가격 조건만으로는 {c['price_eligible_episodes']}개 사건이 있었고, 최종 유효 사건은 {all_result['eligible_episodes']}개입니다. 기간별 사건은 경계를 넘을 수 있어 합계와 일치하지 않을 수 있습니다.</p>
<p>‘7일 최저수익률’은 알림 가격 대비 이후 7일 저가의 최솟값입니다. 예를 들어 −10%는 그 정도의 하락을 경험했다는 뜻이며, 계좌 최대 낙폭이나 전략 손익이 아닙니다. 선행시간은 성공한 알림에 대해서만 계산했습니다.</p>
<h2>체결 시점·손절·급등 정의 민감도</h2>
<div class="scroll">{table(sensitivity)}</div>
<p>기본 진입 가격은 신호 시점 마감가입니다. 다음 시간봉 시가를 진입 기준으로 바꾼 결과도 함께 표시했습니다. +30%와 −10%가 같은 시간봉 안에서 모두 닿으면 보수적으로 손절이 먼저였다고 간주했습니다. 보조 타깃은 첫 3시간 +3% 미만 및 24시간 안 +15%입니다. 결과에 맞춰 타깃이나 규칙을 다시 고르지 않았습니다. 수수료·슬리피지·포지션 배분을 적용한 거래 전략은 아니므로 고가 터치 적중을 실현 수익으로 해석하면 안 됩니다.</p>
<h2>월별 안정성</h2>
<div class="scroll">{table([{'월':x.month,'알림':int(x.alerts),'적중':int(x.hits),'적중률':pct(x.precision),'유효 관측':int(x.eligible_observations)} for x in monthly.itertuples()])}</div>
<h2>독립 검증은 어디까지 준비됐나</h2>
<p>이 과거 기간의 가격 결과는 앞선 실험에 이미 사용됐습니다. 이번 체결 전수 재생과 고정 규칙은 신규 분석이지만 완전히 독립적인 보류 검증이라고 부르지 않습니다. 현재 남아 있는 공통 상장 코인 중심의 생존 편향, 36개 종목 표본, 오래된 공개 파일의 사후 이용 가능성도 제약입니다. 아카이브의 buy/sell을 taker 방향으로 해석한 가정과 RPI 포함은 이전 연구와 같습니다.</p>
<p>독립 미래 평가를 위해 <b>2026-10-10 00:00 UTC~11-10 00:00 UTC 미만</b>의 동일 36종목·공식·임계값·알림 제한을 고정했습니다. 마지막 7일 결과까지 확인 가능한 시각은 <b>2026-11-17 00:00 UTC, 한국시간 오전 9시</b>입니다. 소스와 종목 목록 해시를 저장했고 변경되면 평가 스크립트가 중단합니다. 아직 미래 성과는 존재하지 않습니다.</p>
<p><b>현재 상태는 평가 코드와 프로토콜 준비 완료입니다.</b> 자동 실행 일정이나 실시간 알림 수집기는 등록하지 않았습니다. 공식 일별 파일을 이용하는 미래 기간의 사후 평가이며, 실제 당시 기록한 실시간 알림 로그를 대신하지 않습니다. 미래 날짜 전 실행하면 결과 대기 상태만 반환합니다.</p>
<pre>python3 research/flow_replay/forward.py status
# 2026-11-17 09:00 KST 이후 미래 기간 평가
python3 research/flow_replay/forward.py run</pre>
<h2>재현과 파일</h2>
<pre>python3 research/flow_replay/fetch.py
python3 -m unittest discover -s research/flow_replay -p 'test_*.py' -v
python3 research/flow_replay/study.py
python3 research/flow_replay/report.py</pre>
<p>기존 microstructure 시간봉과 precursor 일봉 파일을 사용하며, 원체결·5분봉 캐시는 tradeflow와 공유합니다. numpy·pandas·pyarrow·matplotlib·certifi가 필요합니다. 기존 실시간 스캐너의 점수는 수정하지 않았습니다.</p>
<p><a href="../protocol.json">고정 프로토콜</a> · <a href="universe.json">36종목 선정 기록</a> · <a href="alerts.csv">모든 모의 알림과 결과</a> · <a href="results.json">전체 지표</a> · <a href="monthly.csv">월별 결과</a> · <a href="forward_manifest.json">미래 검증 고정 기록</a> · <a href="../test_study.py">테스트</a></p>
<p class="small">검증 테스트 9개 통과: 정확한 시간창, 미래 체결 제외, 원자료 누락 처리, 가격 특징의 미래 불변성, 결과 성숙도·동일 봉 손절 순서, 일별 상한·달을 넘는 재알림 제한, 무작위 슬롯 부족과 무신호 기간 처리, 미래 평가 경계·중복 제거, 고정 코드 변경 거부.<br>공식 출처: <a href="https://public.bybit.com/spot/">Bybit 현물 체결</a> · <a href="https://bybit-exchange.github.io/docs/v5/market/kline">시간봉 API</a>.</p></body></html>'''
    (OUT/'replay_report.html').write_text(page,encoding='utf-8')
    print(OUT/'replay_report.html')


if __name__=='__main__':
    main()
