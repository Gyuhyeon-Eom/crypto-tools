"""Korean, self-contained report with explicit case-control limitations."""
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

BASE=Path(__file__).resolve().parent
OUT=BASE/'output'
NAMES={'buy_shift':'매수 우위 전환','sell_resilience':'매도 대비 하락 반응 감소',
       'compression':'가격 범위 압축','quiet_activity':'가격 이동 대비 거래 활성화','sequence':'흡수 → 매수 전환 순서'}
ENGLISH=['Buy-flow shift','Lower downside impact','Range compression','Activity / price movement']


def table(records):
    return pd.DataFrame(records).to_html(index=False,border=0,escape=True)


def main():
    r=json.loads((OUT/'results.json').read_text())
    audit=json.loads((OUT/'download_audit.json').read_text())
    selection=json.loads((OUT/'selection_audit.json').read_text())
    d=pd.read_parquet(OUT/'features.parquet')
    good=d[d.group_valid]
    scores=list(NAMES)[:4]
    rows=[]
    for key in scores:
        train=r['splits']['train']['scores'][key]
        val=r['splits']['validation']['scores'][key]
        test=r['splits']['test']['scores'][key]
        rows.append({'지표':NAMES[key],'4~6월':f"{train['concordance']:.1%}",'7월':f"{val['concordance']:.1%}",
                     '8~10월':f"{test['concordance']:.1%}",'후기 95% 구간':f"{test['lower95']:.1%}–{test['upper95']:.1%}",
                     '후기 99% 구간':f"{test['lower99']:.1%}–{test['upper99']:.1%}"})
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    for ax,split,title,color in zip(axes,['train','test'],['Apr–Jun: 44 matched sets','Aug–Oct: 45 matched sets'],['#64748b','#2563eb']):
        for i,key in enumerate(scores):
            s=r['splits'][split]['scores'][key]
            ax.plot([s['lower99'],s['upper99']],[i,i],color=color,alpha=.30,lw=6,solid_capstyle='round')
            ax.plot([s['lower95'],s['upper95']],[i,i],color=color,lw=2)
            ax.scatter(s['concordance'],i,color=color,s=45,zorder=3)
        ax.axvline(.5,color='#d97706',ls='--',lw=1.5)
        ax.set_xlim(.18,.82);ax.set_xticks([.2,.3,.4,.5,.6,.7,.8],['20%','30%','40%','50%','60%','70%','80%'])
        ax.set_title(title);ax.set_xlabel('Case score > matched control score (ties: half)')
        ax.grid(axis='x',alpha=.12)
    axes[0].set_yticks(range(4),ENGLISH);axes[0].invert_yaxis()
    fig.suptitle('Bybit spot trades: candidate indicators, no confirmed predictor',fontsize=14)
    fig.text(.5,.015,'Thin line: 95% interval · wide pale line: 99% interval · 7-day block bootstrap · NOT alert precision',ha='center',fontsize=9,color='#475569')
    fig.tight_layout(rect=(0,.045,1,.96))
    fig.savefig(OUT/'evidence.png',dpi=180,bbox_inches='tight')
    fig.savefig(OUT/'evidence.svg',bbox_inches='tight')
    plt.close(fig)
    image=base64.b64encode((OUT/'evidence.png').read_bytes()).decode()
    coverage=[]
    for split,label in [('train','4~6월'),('validation','7월'),('test','8~10월')]:
        s=r['splits'][split]
        coverage.append({'기간':label,'선정 사건':selection['sampling'][split]['selected_events'],'품질 통과 사건':s['groups'],
                         '대조 관측':s['groups']*3,'중앙 선행시간':f"{s['median_lead_hours']:.1f}시간",'선행시간 범위':f"{s['min_lead_hours']:.0f}–{s['max_lead_hours']:.0f}시간"})
    seq=[]
    for split,label in [('train','4~6월'),('validation','7월'),('test','8~10월')]:
        for key in ['absorption_rule','sequence']:
            s=r['splits'][split]['scores'][key]
            seq.append({'기간':label,'조건':'매도 흡수' if key=='absorption_rule' else '매도 흡수 → 매수 전환',
                        '급등 사례에서 발생':f"{s['case_exposed']}/{s['n']}",'대조군에서 발생':f"{s['control_exposed']}/{s['controls']}"})
    actual_hash=hashlib.sha256((BASE/'protocol.json').read_bytes()).hexdigest()
    assert actual_hash==selection['protocol_sha256']
    total_trades=sum(a.get('trades',0) for a in audit)
    size=sum(a.get('bytes',0) for a in audit)/1e6
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bybit 체결 기반 급등 전조 가설 검증</title><style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif;max-width:1080px;margin:48px auto;padding:0 24px;color:#182235;line-height:1.7;background:#fbfcfe}}
h1{{font-size:32px;letter-spacing:-1px;line-height:1.3}}h2{{margin-top:42px;font-size:23px}}p{{max-width:970px}}.tag{{color:#52647c;font-size:14px}}.lead{{font-size:20px}}.callout{{padding:18px 22px;border-left:4px solid #2563eb;background:#edf4ff;border-radius:5px}}
table{{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}}th,td{{border-bottom:1px solid #dce4ef;padding:11px 9px;text-align:left}}th{{background:#eef3f9;white-space:nowrap}}img{{width:100%;height:auto}}code,pre{{background:#edf1f6;border-radius:5px}}pre{{padding:18px;overflow:auto}}a{{color:#185ac4}}.small{{font-size:13px;color:#52647c}}.scroll{{overflow:auto}}
</style><body><p class="tag">2026-10-09 · 공개 현물 체결 · 사후 사례–대조 비교 · 연구용</p>
<h1>급등 전에 매도 압력을 버티고<br>매수 흐름이 돌아서는가?</h1>
<p class="lead">매수 우위 전환은 추가 검증할 후보입니다. 이번 결과에서 실전 알림에 넣을 만큼 안정적으로 확인된 지표는 없습니다.</p>
<div class="callout"><b>61.5%는 알림 적중률이 아닙니다.</b> 8~10월 급등 사례와 비슷한 비급등 코인을 짝지어 비교했을 때, 매수 전환 점수가 급등 사례 쪽에서 더 높았던 비율입니다. 무차별 기준은 50%이며, 초기 기간에는 49.2%였습니다.</div>
<h2>무엇을 수치화했나</h2>
<p>관측 시점을 t로 놓고, 직전 24시간을 기준 구간(t−24~−6시간), 매도 흡수 구간(t−6~−2시간), 전환 구간(t−2~0시간)으로 나눴습니다. 모든 계산에는 t 이전 체결만 사용했습니다. 5분봉을 만들 때 경계의 체결을 다음 구간으로 보내고, 미래 가격으로 결측치를 채우지 않았습니다.</p>
<ol><li><b>매수 우위 전환:</b> 순매수 비율 = (매수 체결금액−매도 체결금액)/전체 체결금액. 최근 2시간 비율에서 앞선 4시간 비율을 뺍니다. 매수 비율이 높기만 한 상태와 매수세가 강화되는 변화를 구분합니다.</li>
<li><b>매도 대비 하락 반응 감소:</b> 매도 우위가 20% 이상인 5분 구간에서 하락률 합계/매도 불균형 합계를 계산합니다. 최근 4시간의 값이 앞선 18시간보다 줄었는지 봅니다. 기준 8개·최근 3개 이상의 매도 우위 구간이 필요합니다. 호가에서 실제 물량을 받아냈는지를 직접 관측한 지표는 아닙니다.</li>
<li><b>가격 범위 압축:</b> 기준 18시간의 평균 5분 고저폭/종가를, 그 이전 7일의 평균 시간봉 고저폭/종가와 비교합니다. 시간 단위 보정 상수 12를 고정했습니다. 서로 다른 봉 주기의 근사 비교이며 상수는 순위에 영향을 주지 않습니다.</li>
<li><b>가격 이동 대비 거래 활성화:</b> 최근 4시간의 시간당 체결금액이 기준 18시간 대비 늘어나는 정도에서 가격 변동 페널티를 뺍니다. 정확한 점수는 log(거래 활성화 비율)−|4시간 수익률|/1%입니다.</li>
<li><b>순서 조건:</b> 매도 우위 10% 이상, 시간당 매도금액 1.25배 이상, 가격 하락 0.5% 이내가 먼저 나타난 뒤, 최근 2시간에 매수 우위 10% 이상·순매수 비율 15%p 이상 개선·가격 상승이 이어지는지 확인합니다.</li></ol>
<h2>어떻게 비교했나</h2>
<p>급등은 기존 연구와 동일하게 <b>관측 후 첫 24시간에는 고가 +10% 미만을 유지하고, 25~168시간 사이 처음 +30%를 터치</b>하는 사건입니다. 미리 선정된 3월 말 종목군과 4시간 간격 관측에서 사건별 +30% 도달 48시간 전과 가장 가까운 유효 관측을 골랐습니다. 실제 후기 표본의 중앙 선행시간은 64시간, 범위는 45~165시간입니다. 목표에 맞는 48시간 관측이 없는 사건도 포함되어 있습니다.</p>
<p>각 사건마다 같은 시각·같은 기간에서 7일 내 +30% 터치가 없었던 서로 다른 코인 3개를 골랐습니다. 거래대금 로그·주간 변동성 로그·최근 24시간 수익률의 거리가 가장 가까운 종목이며, 표준화 척도는 4~6월 자료에서 계산했습니다. 신규 체결 지표로 대조군이나 사건을 고르지 않았습니다. 비교 대상 코인과 관측은 다른 사건에서 재사용될 수 있습니다.</p>
<div class="scroll">{table(coverage)}</div>
<p>급등 사건을 먼저 알고 그 앞을 살펴본 <b>사후 사례 연구</b>입니다. 이 방식은 후보 지표의 차이를 조사하는 데 쓰이며, 실제 운용 가능한 진입 시점이나 전체 시장의 발생 확률을 검증하지 않습니다. 대조군은 +30% 급등이 전혀 없는 코인으로 제한했으므로, 이미 빠르게 급등한 코인과의 구분도 시험하지 않았습니다.</p>
<h2>결과: 기간이 바뀌면 차이가 유지되는가</h2>
<p>각 사례의 점수를 3개 대조 점수와 비교하고, 더 크면 1·같으면 0.5·작으면 0을 주어 평균했습니다. 모든 지표는 값이 클수록 전조에 가깝다는 가설로 시작했습니다. 다섯 가지 가설을 함께 보았기 때문에 95%뿐 아니라 보수적인 99% 구간도 표시했습니다. 7일 달력 블록을 10,000회 재표집했습니다.</p>
<div class="scroll">{table(rows)}</div><img src="data:image/png;base64,{image}" alt="초기와 후기의 지표별 사례 대조 비교 및 불확실성 구간">
<p><b>매수 우위 전환:</b> 후기 61.5%, 95% 구간 50.7~73.3%로 약한 후보 신호가 있습니다. 그러나 다섯 가설을 고려한 99% 구간은 47.5~76.7%로 50%를 포함하고, 초기에는 49.2%여서 반복 확인되지 않았습니다.</p>
<p><b>매도 대비 하락 반응 감소:</b> 후기 46.7%로 기대한 방향의 차이가 없었습니다. <b>가격 범위 압축:</b> 초기와 후기 모두 오히려 급등 사례의 가격 범위가 상대적으로 넓었습니다. 이것만으로 변동성 확대가 유효한 신호라고 뒤집어 결론 내릴 수는 없습니다. <b>거래 활성화:</b> 후기 60.0%였으나 초기 43.2%, 후기 99% 하한 50.0%로 확정하기 어렵습니다.</p>
<h2>순서 조건은 너무 드물었나</h2><div class="scroll">{table(seq)}</div>
<p>엄격한 ‘흡수 → 매수 전환’은 후기 급등 45건 중 1건에서만 나타났습니다. 대조군 135건에서 0건이었다고 100% 적중이라고 말할 수 없습니다. 표본을 급등·비급등 1:3으로 구성했기 때문이며, 운용 시 전체 신호 개수와 기본 급등 확률을 알 수 없습니다. 이 조건을 필수 관문으로 삼으면 이번 급등 사례 44/45개를 놓칩니다. 임계값을 결과에 맞춰 낮추는 작업은 하지 않았습니다.</p>
<h2>자료 품질과 해석 한계</h2>
<p>공식 Bybit 현물 일별 파일 {len(audit):,}개, 체결 {total_trades:,}건, 압축 자료 {size:,.1f} MB를 수집했습니다. 다운로드 오류는 {sum('error' in a for a in audit)}건입니다. 관측별 24시간 체결금액 합계와 기존 Bybit 시간봉 합계의 최대 상대 오차는 {d.turnover_relative_error.max():.2e}였습니다.</p>
<p>전체 173개 짝 중 77개를 품질 기준으로 제외해 96개만 남았습니다. 각 관측의 5분봉 활성 비율 80% 이상, 마지막 체결 경과 5분 이하, 시간봉과 체결금액 차이 1% 이하를 요구했습니다. 구현상 시작 5분봉에도 체결이 있어야 합니다. 이 시작 구간 조건은 별도 사전 프로토콜에 명시하지 않은 보수적 구현 조건이며, 미래 가격 보간을 피하기 위해 적용했습니다. 자료가 드문 코인이 많이 빠졌으므로 결과를 초저유동성 종목으로 일반화할 수 없습니다.</p>
<p>후기 비교군의 거래대금·변동성·24시간 수익률 잔여 표준화 평균 차이는 각각 0.159, 0.140, −0.003입니다. 매칭은 완벽하지 않고 최대 거리가 큰 관측도 있습니다. 후기 9개 시간 블록만 있어 신뢰구간도 근사치입니다. 99% 구간은 다중 비교의 보조 기준이지 엄밀한 독립 확인을 보장하지 않습니다.</p>
<p>기존에 결과를 본 2026년 기간을 재사용했고 현재 다른 거래소에도 남아 있는 코인 중심의 표본이라는 제약을 물려받았습니다. 분봉 수준의 체결은 있지만 호가 잔량·주문 취소·숨은 주문은 없습니다. Bybit API 문서는 side를 taker 방향으로 정의합니다. 이번 아카이브의 buy/sell도 그 관례로 해석했지만 아카이브 전용 명세에서 이를 별도 확인하지는 못했습니다. RPI 체결은 포함했습니다. 다른 분·시간 단위나 단기 급등 정의에서는 결과가 달라질 수 있습니다.</p>
<h2>모델에 반영할 판단</h2>
<p><b>매수 우위 변화는 후보 특징으로 남기고, 단독 알림 조건으로 승격하지 않습니다.</b> 절대 거래량이나 압축 여부만으로 점수를 더하는 방식보다, 체결 방향과 같은 코인의 가격 반응을 함께 저장하는 편이 다음 검증에 유용합니다. 매도 흡수·압축·순서 조건은 현재 증거로 필수 조건이 될 이유가 없습니다.</p>
<p>다음 확인 실험은 아직 보지 않은 기간의 모든 유효 시점에서 이 공식을 고정해 계산하고, 알림 수·정밀도·재현율·급등까지 남은 시간·선행 하락폭을 함께 측정해야 합니다. 이번 코드에는 데이터 다운로드·지표 계산·사례 비교·검증 테스트를 구현했습니다. 실시간 알림 점수는 변경하지 않았습니다.</p>
<h2>재현과 원자료 추적</h2>
<pre>python3 research/tradeflow/fetch.py
python3 -m unittest discover -s research/tradeflow -p 'test_*.py' -v
python3 research/tradeflow/analyze.py
python3 research/tradeflow/report.py</pre>
<p>기존 <code>research/microstructure/output/samples.parquet</code>와 <code>bybit.parquet</code>가 필요합니다. pandas·numpy·matplotlib·pyarrow·certifi를 사용합니다. 프로토콜과 표본 선정은 신규 체결 지표 결과를 보기 전에 저장했습니다. 재실행은 원자료 캐시를 재사용합니다.</p>
<p><a href="../protocol.json">고정 프로토콜</a> · <a href="matched_samples.csv">선정 사례·대조군</a> · <a href="features.csv">관측별 계산 지표·품질 판정</a> · <a href="summary.csv">전체 결과</a> · <a href="download_audit.json">다운로드 URL·해시 감사</a> · <a href="../test_tradeflow.py">검증 테스트</a></p>
<p>검증 테스트 5개: 체결 구간 경계와 거래금액 보존, 미래 체결 추가 시 특징 불변, 희소 자료 제외, 흡수 다음 실제 가격 상승 확인, 동점·결측 매칭 계산. 모두 통과했습니다.</p>
<p class="small">공식 자료: <a href="https://public.bybit.com/spot/">Bybit 현물 아카이브</a>, <a href="https://bybit-exchange.github.io/docs/v5/market/recent-trade">공개 체결 API 방향 정의</a>.<br>프로토콜 SHA-256: {html.escape(actual_hash)}</p></body></html>'''
    (OUT/'tradeflow_report.html').write_text(page,encoding='utf-8')
    (OUT/'versions.json').write_text(json.dumps({'pandas':pd.__version__,'numpy':np.__version__,'matplotlib':matplotlib.__version__},indent=2))
    print(OUT/'tradeflow_report.html')


if __name__=='__main__':
    main()
