"""Render a self-contained, source-linked research report and exportable charts."""
import base64
import html
import importlib.metadata
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / 'output'


def table(frame):
    return frame.to_html(index=False, escape=True, border=0, float_format=lambda x: f'{x:.3f}')


def image(path):
    return '<img src="data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode() + '">'


def main():
    r = json.loads((OUT / 'results.json').read_text())
    audit = json.loads((OUT / 'data_audit.json').read_text())
    evidence = pd.read_csv(OUT / 'signal_evidence.csv')
    samples = pd.read_parquet(OUT / 'samples.parquet')
    test = samples[samples.split.eq('test')]
    alerts = pd.read_csv(OUT / 'test_alerts.csv', parse_dates=['date'])
    # Paired block bootstrap of policy hits versus date-matched random expectation.
    daily = pd.DataFrame(index=pd.date_range(test.date.min(), test.date.max()))
    daily['hits'] = alerts.groupby('date').y.sum()
    daily['n'] = alerts.groupby('date').size()
    daily = daily.fillna(0)
    daily['expected'] = daily.n * test.groupby('date').y.mean().reindex(daily.index).fillna(0)
    daily['block'] = np.arange(len(daily)) // 14
    a = daily.groupby('block')[['hits', 'n', 'expected']].sum().to_numpy()
    rng = np.random.default_rng(42)
    boot = a[rng.integers(0, len(a), (2000, len(a)))].sum(axis=1)
    ok = boot[:, 2] > 0
    ci = np.quantile(boot[ok, 0]/boot[ok, 2], [.025, .975])
    extra = {'test_policy_date_matched_lift_block95': ci.tolist(),
             'calendar_block_days': 14, 'blocks': len(a),
             'caution': 'Only five calendar blocks in final test; interval is exploratory, not conclusive.'}
    (OUT / 'uncertainty.json').write_text(json.dumps(extra, indent=2))
    versions = {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'scipy', 'matplotlib', 'pyarrow', 'joblib', 'certifi']}
    (OUT / 'versions.json').write_text(json.dumps(versions, indent=2))

    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={'width_ratios': [1, 1.35]})
    labels = ['Validation\nMay–Jul', 'Final test\nAug–Oct']
    primary = [r['primary']['validation'], r['primary']['test']]
    precision = np.array([m['precision'] for m in primary])*100
    random = np.array([m['same_date_random_expected_hits']/m['alerts'] for m in primary])*100
    x = np.arange(2)
    axes[0].bar(x-.18, precision, .34, label='Selected model', color='#2463eb')
    axes[0].bar(x+.18, random, .34, label='Date-matched random', color='#bac6d7')
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel('Target hit rate (%)')
    axes[0].set_ylim(0, 16)
    axes[0].set_title('Validation advantage did not persist')
    axes[0].legend(frameon=False, fontsize=9)
    for i, m in enumerate(primary):
        axes[0].text(i-.18, precision[i]+.4, f"{m['hits']}/{m['alerts']}", ha='center', fontsize=10)
    feature_names = {'drawdown_30': 'Deep 30-day drawdown', 'distance_high_60': 'Far below 60-day high',
                     'volatility_20': 'High recent volatility', 'volume_ratio_3_20': 'Low relative volume',
                     'close_location_3': 'Closes near daily lows', 'green_days_7': 'Few up days in prior week'}
    subset = evidence.set_index('feature').loc[list(feature_names)]
    matrix = subset[['train_lift', 'validation_lift', 'test_lift']].to_numpy()
    im = axes[1].imshow(matrix, cmap='RdBu', vmin=.5, vmax=1.5, aspect='auto')
    axes[1].set_xticks(range(3), ['Discovery', 'Validation', 'Final test'])
    axes[1].set_yticks(range(len(subset)), list(feature_names.values()), fontsize=10)
    for i in range(len(subset)):
        for j in range(3):
            axes[1].text(j, i, f'{matrix[i,j]:.2f}×', ha='center', va='center', color='white' if abs(matrix[i,j]-1)>.3 else '#172033')
    axes[1].set_title('Rule lift vs eligible-universe base rate')
    fig.colorbar(im, ax=axes[1], fraction=.03, pad=.03)
    fig.tight_layout(pad=2)
    fig.savefig(OUT / 'evidence.png', dpi=170, bbox_inches='tight')
    fig.savefig(OUT / 'evidence.svg', bbox_inches='tight')
    plt.close(fig)

    metrics_rows = []
    for split, m in r['primary'].items():
        metrics_rows.append({'기간': split, '신호': m['alerts'], '목표 충족': m['hits'],
                            '적중률': f"{m['precision']:.1%}",
                            '같은 날짜 무작위 기대 적중': f"{m['same_date_random_expected_hits']:.1f}",
                            '무작위 대비': f"{m['same_date_lift']:.2f}배",
                            '종가로도 +30% 충족': m['close_confirmed_hits'],
                            '-10% 선행 하락 없이 충족': m['clean_hits_before_10pct_drop']})
    rule_rows = []
    for feature, name in feature_names.items():
        x = evidence.set_index('feature').loc[feature]
        rule_rows.append({'가설': name, '방향': x.direction, '학습 리프트': x.train_lift,
                         '검증 리프트': x.validation_lift, '최종 리프트': x.test_lift,
                         '최종 조건 적중률': f'{x.test_precision:.1%}',
                         '날짜·유동성·변동성 보정 리프트': x.test_matched_lift})
    baselines = []
    for name, m in r['test_baselines_fixed_budget'].items():
        baselines.append({'조건': name, '신호': m['alerts'], '적중': m['hits'],
                          '적중률': f"{m['precision']:.1%}", '같은 날짜 무작위 대비': f"{m['same_date_lift']:.2f}배"})
    # Rule results are exploratory; report the selection direction learned in train.
    weak = evidence.set_index('feature').loc['close_location_3']
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>급등 선행 신호 탐색 · 2026-10-09</title><style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif;color:#172033;background:#eef2f7;margin:0;line-height:1.75}}
main{{max-width:1100px;margin:36px auto;background:white;padding:46px 52px;border-radius:18px}}h1{{font-size:32px;line-height:1.35;letter-spacing:-1px}}h2{{font-size:22px;margin-top:38px}}p{{max-width:970px}}
.eyebrow{{color:#51647d;font-size:13px;letter-spacing:2px}}.summary{{background:#edf4ff;border-left:4px solid #2463eb;padding:18px 24px;margin:24px 0}}.caution{{background:#fff5e8;padding:16px 22px;border-radius:8px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}}th,td{{border-bottom:1px solid #dbe2eb;text-align:left;padding:10px 9px}}th{{background:#f3f6fa}}img{{width:100%;height:auto;margin:16px 0}}code,pre{{background:#f2f5fa;border-radius:4px}}pre{{padding:16px;overflow-x:auto;font-size:13px}}a{{color:#1756bc}}.scroll{{overflow-x:auto}}footer{{margin-top:36px;border-top:1px solid #dbe2eb;padding-top:20px;font-size:13px;color:#607089}}
</style><main><div class="eyebrow">CRYPTO TOOLS · PRECURSOR RESEARCH · 2026-10-09</div>
<h1>급등하기 전에 구분 가능한 신호가 있는가?</h1>
<div class="summary"><b>현재 결론: 일봉 기반 특징 35개를 검사했으나, 최종 기간에서 안정적인 예측 우위는 확인하지 못했습니다.</b><br>
기초 모델의 최종 신호 132건 중 15건이 목표를 충족했습니다. 같은 날짜에 무작위로 선택했을 때의 기대 적중은 13.94건입니다.
특정 약세 패턴은 약한 연구 후보로 남지만, 매집이나 임박한 급등의 증거로 해석할 수준은 아닙니다.</div>
<h2>1. 이번에 실제로 검증한 질문</h2>
<p>종목이 아직 조용할 때 일봉 마감 정보를 보고 신호를 냅니다. 이후 첫 24시간 고가가 기준가 대비 +10% 미만이고,
그 다음부터 7일 안에 고가가 +{r['protocol']['target']:.0%}에 도달하면 성공으로 정의했습니다. 최근 하루 상승률이 +5% 이상이거나
최근 7일 상승률이 +10% 이상이면 애초에 후보에서 제외했습니다. 목표 크기와 선행 시간에 대한 사용자 답변 전 적용한 탐색용 기본값입니다.</p>
<p>이는 <b>목표 도달보다 최소 하루 앞선 조용한 후보</b>를 찾는 실험입니다. 실제 상승이 시작되는 정확한 순간보다 하루 앞섰다는 뜻은 아닙니다.
일봉만으로는 장중의 정확한 순서를 알 수 없습니다. 고가 목표를 스쳤다가 내려온 경우도 기본 성공에 포함하되, 종가 확인과 선행 하락 조건을 별도로 보고했습니다.</p>
<h2>2. 데이터와 미래 정보 차단</h2>
<p>저장소 원본과 Bybit 공개 일봉을 합쳐 {audit['symbols']}개 종목, {audit['rows']:,}개 봉을 사용했습니다.
가격 데이터 범위는 {audit['date_start'][:10]}~{audit['date_end'][:10]}입니다. 과거 시점의 30일 평균 거래대금 근사값이
$200K~$20M인 알트코인을 대상으로 했고, 스테이블코인·주요 대형 종목을 제외했습니다. 실제 분석에 남는 종목 수는 기간별로 다릅니다.</p>
<div class="scroll">{table(pd.DataFrame([{'기간': s, **v} for s,v in r['overview'].items()]))}</div>
<p>4월까지 학습하고 5~7월에 모델·임계값을 선택했습니다. 8월 이후의 결과로 모델을 재선택하지 않았습니다.
7일 뒤 결과가 다음 기간에 걸치는 학습·검증 행은 제거했고, 같은 날짜의 종목들을 서로 다른 기간으로 나누지 않았습니다.
마감 전 봉 {audit['dropped_open_candles']}개를 제외했고, 결측일을 건너뛰어 연결하지 않았습니다. 평가 날짜는 UTC 일봉 마감 시각 기준입니다.</p>
<p>양성 행은 종목·관찰일 단위로 중복될 수 있어, 같은 종목의 목표 도달일이 7일 범위에 모이는 경우를 한 사건으로 묶었습니다.
모의 알림은 하루 최대 3건, 같은 종목은 7일간 재알림 금지입니다. 이 사건 수는 실제 시장의 모든 급등 수가 아니라 <b>이번 후보 조건으로 관찰 가능한 사건 수</b>입니다.</p>
{image(OUT / 'evidence.png')}
<h2>3. 어떤 패턴이 남았는가</h2>
<p>각 특징의 상·하위 25% 중 학습 기간에 더 유리했던 방향을 선택하고, 그 수치 기준을 검증·최종 기간에 고정 적용했습니다.
리프트 1배는 해당 기간 후보 전체의 평균 수준입니다. 오른쪽 마지막 열은 같은 날짜·비슷한 유동성·변동성의 비해당 종목과 비교한 값입니다.</p>
<div class="scroll">{table(pd.DataFrame(rule_rows))}</div>
<p><b>낙폭이 큰 종목:</b> 학습·검증에서는 강했지만 최종 기간에서 효과가 뒤집혔습니다. 과거 급등 사례에서 관찰된 낙폭을 보편적인 전조로 고정하면 실패할 수 있습니다.</p>
<p><b>거래량·압축:</b> 단독 순위 신호로는 뚜렷한 우위가 없었습니다. 따라서 ‘조용해졌다’ 또는 ‘거래량이 늘었다’만으로 급등 알림을 보내는 근거는 부족합니다.</p>
<p><b>약한 후속 연구 후보:</b> 최근 3일 종가가 각 일봉 범위의 아래쪽에 머문 조건은 최종 적중률 {weak.test_precision:.1%},
평균 대비 {weak.test_lift:.2f}배, 날짜·유동성·변동성 보정 후 {weak.test_matched_lift:.2f}배였습니다.
그러나 35개 후보를 탐색했고 다중비교 보정 없는 구간추정입니다. 매집이나 매도압력 소진을 확인한 것은 아니며,
별도의 미래 데이터에서 재검증해야 합니다. ‘최근 7일 상승일이 적음’도 약한 후보이나 불확실성이 큽니다.</p>
<h2>4. 기초 모델과 단순 기준의 성능</h2>
<p>고정 설정의 로지스틱 회귀와 작은 부스팅 모델을 학습했습니다. 검증 기간 PR-AUC로 선택된 모델은 <code>{r['selected_model']}</code>이며,
검증에서만 임계값 <code>{r['threshold']:.5f}</code>를 선택했습니다. 점수는 보정된 급등 확률로 해석하지 않습니다.</p>
<div class="scroll">{table(pd.DataFrame(metrics_rows))}</div>
<p>최종 132건 중 15건은 고가 기준 성공, 4건만 종가 기준으로도 +30%를 달성했습니다. 12건은 목표 도달 전에 -10% 저가를 터치하지 않았습니다.
21건은 이후 7일 중 -20% 저가를 터치했습니다. 상승·하락 수치는 경로에 따라 겹칠 수 있으며, 거래 수익률이나 손익비를 의미하지 않습니다.</p>
<p>같은 날짜 무작위 선택 대비 리프트는 {r['primary']['test']['same_date_lift']:.2f}배,
14일 단위 블록 부트스트랩 탐색 구간은 {ci[0]:.2f}~{ci[1]:.2f}배입니다. 최종 기간에 블록이 {len(a)}개뿐이므로
정밀한 통계적 확정으로 사용할 수 없습니다. 현재 결과로는 알림 운영에 충분한 우위가 입증되지 않았습니다.</p>
<p>다음 비교는 임계값 없이 하루 최대 3개를 고르는 동일한 알림 예산의 기준선입니다. 위의 임계값 적용 모델과 알림 수가 다르므로 적중률을 그대로 순위 비교하지 않습니다.</p>
<div class="scroll">{table(pd.DataFrame(baselines))}</div>
<h2>5. 다음 연구에서 추가할 관측</h2>
<p>다음 단계에서는 1시간·15분 단위로 선행 과정을 구분하고, 거래량 외에 누가 어떤 방향으로 체결했는지와 파생시장 포지션 변화를 확인하는 실험이 필요합니다.
아래 항목은 아직 이번 데이터로 검증하지 않은 가설입니다.</p>
<ul><li><b>가격 정체 + 매수 주도 체결 증가:</b> 같은 시간 길이의 체결 자료로 매수·매도 금액과 가격 반응을 비교합니다. 최근 체결 500건처럼 종목마다 시간 폭이 달라지는 표본은 피합니다.</li>
<li><b>OI 증가 + 펀딩 변화 + 가격 반응:</b> OI 증가만으로 롱 매집을 단정하지 않고, 방향별 가격 반응·체결 흐름을 함께 봅니다.</li>
<li><b>압축 → 매도 반응 약화 → 초기 돌파:</b> 상태의 순서와 지속시간을 특성으로 만듭니다. 압축 상태 자체의 빈번한 오탐을 비교군에 반드시 포함합니다.</li>
<li><b>시장·테마 대비 선행 움직임:</b> BTC·동일 테마·다른 거래소의 변화를 비교하되, 당시 이용 가능했던 정보만 사용합니다.</li></ul>
<p>기존 Ignition 로그의 OI·펀딩·체결 비율은 일부 필터 통과 종목에만 기록돼 있습니다. 이를 전 시장의 무편향 학습 표본처럼 쓰지 않았습니다.
전체 후보군을 같은 주기로 저장하거나, 공개 과거 API로 가능한 항목을 복원해야 합니다. 과거 호가·체결 자료는 보존 여부를 별도로 확인해야 합니다.</p>
<div class="caution"><b>해석의 한계</b><br>원본 종목 목록은 이전 수집기의 선정 결과입니다. 누락·상장폐지 종목이 있을 수 있어 생존편향을 완전히 제거하지 못했습니다.
과거 거래대금은 수량×종가의 근사값을 일관되게 사용했습니다. BTC 과거 자료가 없는 구간은 학습 기간의 중앙값으로 처리했으며 BTC 개별 신호 검사는 관측 구간만 사용했습니다.
가격 데이터만으로 내부자 매집·뉴스 선행 여부를 확인할 수 없습니다. 수수료·체결 가능성·슬리피지를 포함한 매매 성과 분석도 이번 범위에 포함하지 않았습니다.</div>
<h2>6. 재현과 파일</h2>
<pre>cd /Users/eomgyuhyeon/projects/crypto-tools
python3 research/precursor/prepare_data.py
python3 research/precursor/study.py --target 0.30
python3 research/precursor/report.py
python3 -m unittest discover -s research/precursor -p 'test_*.py' -v</pre>
<p>수집기는 저장된 <code>fresh_daily.json</code>이 있으면 같은 자료를 재사용합니다. 분석은 알림·주문을 실행하지 않습니다.
미래 봉 추가 시 과거 특징 불변, 조기 급등 제외, 같은 봉의 목표·하락 동시 충족 처리, 결측일, 날짜 기준 분리, 재알림 제한을 5개 테스트로 확인했습니다.</p>
<ul><li><a href="signal_evidence.csv">35개 신호의 전체 비교표</a></li><li><a href="results.json">평가 설정·모델·월별 결과</a></li>
<li><a href="test_alerts.csv">최종 평가의 132개 모의 신호</a></li><li><a href="data_audit.json">원본 해시와 수집 기록</a></li>
<li><a href="evidence.svg">차트 SVG</a></li><li><a href="versions.json">실행 라이브러리 버전</a></li></ul>
<p><code>research_model.joblib</code>은 이번에 직접 학습한 연구용 모델입니다. 기존 스캐너에 연결하지 않았고, 운영 모델로 승인된 결과도 아닙니다.</p>
<footer>원본 저장소: e6050d2 · 데이터 조회 {html.escape(audit['fetch_time'])}<br>
공개 API의 봉 구조와 미마감 봉 처리: <a href="https://bybit-exchange.github.io/docs/v5/market/kline">Bybit Kline 문서</a> ·
시간순 검증의 원칙: <a href="https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split">scikit-learn 문서</a><br>
이 결과는 일봉으로 구현한 탐색 실험의 결과이며, 보편적인 급등 전조의 존재 또는 부재를 증명하지 않습니다.</footer></main></html>'''
    (OUT / 'research_report.html').write_text(page)
    print('Report:', OUT / 'research_report.html')
    print('Block-bootstrap lift interval:', ci.tolist())


if __name__ == '__main__':
    main()
