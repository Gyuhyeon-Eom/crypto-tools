"""Render all raw-tape results, including unsuccessful and supplementary tests."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / 'output'
NAMES = {
    'buy_persistence': ('매수 우위의 지속성', 'Buy persistence'),
    'buy_acceleration': ('최근 5분 순매수의 가속', 'Buy acceleration'),
    'sell_resilience': ('매도 우위 구간의 가격 버팀', 'Sell resilience'),
    'price_efficiency': ('상승 경로의 효율성', 'Price efficiency'),
    'arrival_acceleration': ('체결 간격의 단축', 'Arrival acceleration'),
    'large_buy_change': ('큰 개별 매수 체결 비중의 변화', 'Large-buy change'),
}


def read(name):
    return json.loads((OUT / name).read_text())


def validate():
    d = pd.read_parquet(OUT / 'monitor_features.parquet')
    assert d.groupby(['group', 'checkpoint']).size().eq(4).all()
    assert (d.observed_at - d.time).dt.total_seconds().div(60).eq(d.elapsed_minutes).all()
    assert d.groupby(['group', 'checkpoint']).elapsed_minutes.nunique().eq(1).all()
    c = d[d.role.eq('case')]
    assert (c.case_onset - c.observed_at).dt.total_seconds().div(60).eq(
        c.checkpoint.map({'minus15': 15, 'minus5': 5})).all()
    controls = d[d.role.eq('control')]
    nearby = ((controls.next_event >= controls.observed_at) &
              (controls.next_event <= controls.observed_at + pd.Timedelta(minutes=60))).sum()
    assert nearby == 0
    for root, manifest in [(BASE, read('frozen.json')),
                           (BASE.parent / 'flow_replay', json.loads((BASE.parent / 'flow_replay/output/forward_manifest.json').read_text()))]:
        assert all(hashlib.sha256((root / p).read_bytes()).hexdigest() == h
                   for p, h in manifest['source_hashes'].items())
    result = {'equal_control_elapsed_time': True, 'case_checkpoints_before_onset': True,
              'control_onsets_in_shifted_horizon': int(nearby), 'frozen_hashes_unchanged': True,
              'max_raw_vs_cached_quote_error': float(d.quote_error.max()),
              'monitor_quality_errors': d.quality_error.value_counts().to_dict(),
              'supplementary_source_hashes': {p: hashlib.sha256((BASE / p).read_bytes()).hexdigest()
                                            for p in ['monitor.py', 'monitoring_protocol.json']}}
    (OUT / 'validation.json').write_text(json.dumps(result, indent=2))


def main():
    validate()
    frozen, result, monitor = read('frozen.json'), read('results.json'), read('monitor_results.json')
    rows = []
    for key in NAMES:
        direction = frozen['discovery'][key]['direction']
        for split in ['discovery', 'replication']:
            for stage in ['initial', 'minus15', 'minus5']:
                s = (frozen['discovery'][key] if split == 'discovery' else result['replication'][key]) if stage == 'initial' else monitor[split][stage]['scores'][key]
                rows.append({'feature': key, 'split': split, 'stage': stage, 'direction': direction,
                             **{k: s[k] for k in ['n', 'blocks', 'concordance', 'lower', 'upper']}})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'comparison_summary.csv', index=False)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for ax, (key, (_, english)) in zip(axes.flat, NAMES.items()):
        direction = frozen['discovery'][key]['direction']
        for split, color, offset in [('discovery', '#8a94a6', .12), ('replication', '#167d92', -.12)]:
            s = table[(table.feature == key) & (table.split == split)]
            x = s.concordance.to_numpy() * 100
            lo, hi = s.lower.to_numpy() * 100, s.upper.to_numpy() * 100
            y = np.arange(3) + offset
            ax.errorbar(x, y, xerr=np.array([x-lo, hi-x]), fmt='o', color=color,
                        capsize=3, label='Apr-Jun' if split == 'discovery' else 'Aug-Sep')
            for xx, yy, n in zip(x, y, s.n):
                ax.annotate(f'n={n}', (xx, yy), xytext=(3, 6), textcoords='offset points', fontsize=8, color=color)
        ax.axvline(50, color='#aeb8c4', linestyle='--', linewidth=1)
        ax.set(xlim=(-4, 104), ylim=(2.45, -.5), yticks=[0, 1, 2],
               yticklabels=['Initial trigger', '15 min before', '5 min before'],
               title=english + (' (higher)' if direction == 1 else ' (lower)'),
               xlabel='Matched rank concordance (%)')
        ax.grid(axis='x', alpha=.12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2, fontsize=10)
    fig.suptitle('Same-coin volume bursts: pump cases vs matched non-pump controls\nDirections fixed at initial discovery; later checkpoints are retrospective', fontsize=14)
    fig.savefig(OUT / 'tape_comparison.png', dpi=160)
    fig.savefig(OUT / 'tape_comparison.svg')
    plt.close(fig)

    def cell(row):
        return f"{row['concordance']:.1%} <small>[{row['lower']:.1%}, {row['upper']:.1%}] · n={row['n']}</small>"

    def html_table(stage):
        body = []
        for key, (korean, _) in NAMES.items():
            part = table[(table.feature == key) & (table.stage == stage)].set_index('split')
            direction = '높을수록' if frozen['discovery'][key]['direction'] == 1 else '낮을수록'
            body.append(f'<tr><td>{korean}</td><td>{direction}</td><td>{cell(part.loc["discovery"])}</td><td>{cell(part.loc["replication"])}</td></tr>')
        return '<div class="scroll"><table><thead><tr><th>지표</th><th>초기 탐색에서 정한 방향</th><th>4–6월</th><th>8–9월</th></tr></thead><tbody>' + ''.join(body) + '</tbody></table></div>'

    features = pd.read_parquet(OUT / 'features.parquet')
    sources = read('source_audit.json')
    html = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>급등 전 원체결 정밀 분석</title><style>
body{font-family:system-ui,-apple-system,sans-serif;background:#f4f7fa;color:#203044;margin:0;line-height:1.7}
main{max-width:1100px;margin:auto;padding:40px 28px}h1{font-size:32px;line-height:1.3}h2{margin-top:40px;font-size:22px}
.lead{font-size:19px}.box{background:white;border-left:4px solid #167d92;padding:16px 22px;border-radius:6px;margin:24px 0}
.warn{border-color:#bb8a35}small{display:block;color:#607083;font-size:12px}table{border-collapse:collapse;width:100%;font-size:14px;background:white}
th,td{text-align:left;padding:12px;border-bottom:1px solid #dde4ec;vertical-align:top}th{background:#e7eff4}.scroll{overflow:auto}
img{width:100%;background:white;border-radius:8px}a{color:#087a97}code,pre{background:#e7edf3;border-radius:4px}pre{padding:16px;overflow:auto}
.muted{color:#607083}li{margin:8px 0}footer{margin-top:40px;color:#607083;font-size:13px}
</style><main>
<p class="muted">Crypto Tools · 2026-10-09 · 원체결 후속 분석</p>
<h1>거래가 늘어난 뒤,<br>급등으로 이어지는 흐름은 따로 있는가?</h1>
<p class="lead">같은 코인의 거래량 증가 사례끼리 비교하고, 실제 체결을 1분 단위로 재구성했다. 이번에 조사한 6개 지표에서는 후속 알림 조건으로 채택할 만큼 일관된 근거를 찾지 못했다.</p>
<div class="box"><strong>해상도보다 독립 사건 수가 현재의 제약이다.</strong> 처음 거래량 신호 시점의 비교는 탐색 12건, 후기 7건이다. 같은 급등을 15분 전·5분 전에 다시 보아도 사건이 추가되는 것은 아니다. 결과는 전조의 부재를 입증하지 않으며, 이번 지표와 표본에서 확인되지 않았다는 뜻이다.</div>
<h2>이번에는 무엇을 더 세밀하게 봤나</h2>
<p>기존 거래대금 약 2.30배 조건을 통과한 시점만 사용했다. 급등 사례마다 <b>같은 코인</b>에서 거래는 늘었지만 이후 급등하지 않은 3개 시점을 골랐다. ±14일, 시간대 ±4시간 안에서 거래량 증가 정도·유동성·변동성·거래량 신호 지속시간을 맞췄다. 성공 사례와 대조 사례의 세부 체결 지표를 보고 짝을 바꾸지 않았다.</p>
<p>급등 정의는 기존과 동일하다. 시작 후 첫 5분 종가가 2% 이상 오르고 다음 60분 고가가 8% 이상 오른 사건 중, 직전 30분의 조용한 구간 조건을 만족하는 경우다. 사건마다 24시간 중복 제거를 적용했다. 첫 관찰 시점은 급등 5~60분 전에 나타난 가장 이른 거래량 신호다. 사건을 알고 시점을 정한 <b>회고적 비교</b>이며 실시간 알림 성공률이 아니다.</p>
<ol>
<li><b>매수 우위의 지속성:</b> 최근 15분 중 매수가 우세한 활성 1분 구간 비율이 이전 45분보다 늘었는가.</li>
<li><b>순매수 가속:</b> 최근 5분 순매수 거래대금 비중이 그 앞 10분보다 커졌는가.</li>
<li><b>매도 구간의 가격 버팀:</b> 매도 우위가 같은 정도일 때 가격 하락이 이전보다 작아졌는가. 매도 구간이 부족하면 계산하지 않는다.</li>
<li><b>상승 경로의 효율성:</b> 최근 15분의 순수익률이 1분 수익률 절댓값 합에 비해 큰가.</li>
<li><b>체결 간격의 단축:</b> 최근 5분의 체결 간격 중앙값이 이전 45분보다 짧아졌는가. 같은 타임스탬프의 일괄 체결은 간격 계산에서 합쳤다.</li>
<li><b>큰 개별 매수 체결:</b> 이전 45분 체결금액의 상위 5% 기준을 넘는 매수 체결 비중이 늘었는가. 개별 체결이므로 대형 주문이나 특정 주체를 식별하지는 않는다.</li>
</ol>
<h2>첫 거래량 신호 시점</h2>
<p>탐색 기간에서 방향을 정한 뒤 고정했다. 비교 가능한 사건 10건 이상, 짝 비교 점수 60% 이상, 주간 블록 재표집 95% 구간의 하한 50% 초과라는 선정 조건을 만족한 지표는 <b>0개</b>다. 후기 기간의 초기 시점에서는 6개 지표 모두 구간이 50%를 포함한다.</p>
INITIAL_TABLE
<p class="muted">점수는 고정한 방향으로 급등 사례가 같은 짝의 대조 사례보다 높게 평가된 비율이다. 50%는 같은 순위의 기준이며, 알림 적중률이나 급등 확률이 아니다. 동점은 0.5로 계산한다. 탐색의 방향은 이 자료에서 선택했으므로 탐색 점수 자체에도 선택 편향이 있다. 표의 구간은 탐색 95%, 후기 99.17% 주간 블록 부트스트랩이다.</p>
<h2>그 뒤 15분 전·5분 전까지 따라가면?</h2>
<p>첫 분석 결과를 본 뒤 추가한 보조 분석이다. 같은 짝을 유지하고, 급등 사례는 실제 시작 15분 전·5분 전에 다시 측정했다. 대조 사례도 자기 거래량 신호 이후 <b>동일한 시간</b>만큼 지나서 측정했다. 첫 거래량 신호보다 이른 체크포인트는 제외했고, 품질 조건을 통과하지 못한 짝 전체를 제외했다.</p>
<p>15분 전에는 탐색 10건·후기 4건, 5분 전에는 탐색 10건·후기 6건이 남았다. 매도 구간과 체결 간격의 최소 관측 조건을 적용하면 지표별 사건 수는 더 줄어든다. 거래량 신호 때 정한 방향을 계속 사용했으며 새 방향이나 임계값은 맞추지 않았다.</p>
<img src="tape_comparison.png" alt="6개 지표의 탐색·후기 짝 비교와 불확실성 구간. 시점별 표본 수 표시.">
<p class="muted">차트의 higher/lower는 초기 탐색에서 고정한 방향이다. 세 시점은 동일 간격의 시계열이 아니며 표본 구성도 일부 다르므로 연결선으로 표시하지 않았다.</p>
<h3>급등 15분 전</h3>MINUS15_TABLE
<h3>급등 5분 전</h3>MINUS5_TABLE
<div class="box warn"><strong>좋아 보이는 일부 숫자를 새 신호로 채택하지 않았다.</strong> 예를 들어 15분 전 순매수 가속의 후기 점수 83.3%는 “낮을수록”이라는 방향의 점수이고, 4건뿐이다. 같은 시점의 탐색 점수는 43.3%다. 5분 전 매수 지속성도 후기 19.4%로 처음 기대한 증가 방향과 반대다. 체결 간격의 후기 비교는 2건만 남는다. 작은 블록 표본의 부트스트랩 구간이 50%를 제외하더라도 확정적 근거로 해석할 수 없다.</div>
<p>보조 분석의 구간은 6개 지표 × 3시점을 고려한 명목 99.72% 구간이다. 이는 같은 기간에서 앞선 연구를 반복한 모든 선택을 보정하지 않으며, 2~6개 주간 블록에서의 구간 추정은 불안정하다. 실제 급등 시점을 알고 고른 체크포인트이므로 독립된 재검증도, 실행 가능한 알림 정책도 아니다.</p>
<h2>자료 검증과 설계 변경 기록</h2>
AUDIT
<p>초기 설계는 1분 구간의 80% 이상에 체결이 있어야 한다고 요구해 탐색 짝이 3건으로 줄었다. 실제로 거래가 없던 분을 자료 누락처럼 취급하는 문제가 있어, <b>후기 세부 지표 결과를 보기 전에</b> 파일 존재·거래대금 대조·최소 체결 수로 품질 조건을 수정했다. 변경 전 코드·프로토콜·탐색 결과도 보존했다. 탐색 자료를 본 뒤의 수정이므로 완전히 사전 등록된 실험으로 표현하지 않는다.</p>
<p>관찰 시점 이후 체결 제외, 창 구간별 순매수 계산, 무체결 분 처리, 체결 시간 일괄 기록, 결측 지표의 짝 전체 제외 등 7개 테스트를 통과했다. 보조 분석의 관찰 시점과 짝별 동일 경과시간도 검증했다. 기존 전향 실험의 해시는 변경되지 않았으며, 이 작업이 예약 실행이나 실시간 알림 기록을 시작한 것은 아니다.</p>
<h2>다음 연구 판단</h2>
<p>기존 거래량 신호는 후보를 넓게 찾는 역할에 머문다. 이번 분석은 그 후보 중 성공·실패를 가르는 추가 필터를 확보하지 못했다. 지금 같은 사건을 더 짧게 쪼개거나 여러 지표를 조합해 맞추면 우연한 패턴을 고를 가능성이 크다. 다음 검증에서는 사건 정의와 후보 조건을 고정한 채, 다른 기간의 독립 사건 수를 늘리는 것이 우선이다. 호가 잔량·취소·스프레드 가설은 별도의 호가 자료가 필요하며 이번 체결 분석으로 확인할 수 없다.</p>
<p>범위는 현재 목록에서 선정한 36개 Bybit 현물 코인과 확보된 자료다. 일부 종목의 아카이브 접근 실패 및 현재 상장 목록을 쓴 생존 편향이 있고, 전체 시장으로 일반화할 수 없다. 8–9월은 이전 연구에서도 사용한 기간이라 미사용 최종 검증 자료가 아니다.</p>
<h2>재현 자료</h2>
<p><a href="../study.py">분석 코드</a> · <a href="../protocol.json">v2 프로토콜</a> · <a href="../protocol_v1.json">변경 전 프로토콜</a> · <a href="frozen_v1.json">변경 전 탐색 결과</a> · <a href="selected.csv">비교 대상</a> · <a href="features.csv">초기 지표</a> · <a href="comparison_summary.csv">전체 비교표</a> · <a href="frozen.json">고정된 방향</a> · <a href="results.json">후기 결과</a> · <a href="../monitoring_protocol.json">보조 분석 계획</a> · <a href="monitor_features.csv">보조 관측값</a> · <a href="monitor_results.json">보조 결과</a> · <a href="validation.json">검증 기록</a></p>
<pre>python3 -m unittest discover -s research/tape_detail -p 'test_*.py' -v
python3 research/tape_detail/study.py replicate
python3 research/tape_detail/monitor.py
python3 research/tape_detail/report.py</pre>
<footer>공개 원체결: <a href="https://public.bybit.com/spot/">Bybit 현물 아카이브</a>. 매수·매도 측은 아카이브 side 값을 사용했다. <a href="https://bybit-exchange.github.io/docs/v5/market/recent-trade">공식 최근 체결 API 문서</a>는 side를 taker 방향으로 정의한다. 아카이브가 동일 규약을 쓴다는 해석을 적용했으며 별도의 아카이브 side 규격까지 확인한 것은 아니다. 호가 흡수나 주문 주체를 직접 관측한 것으로 해석하지 않는다.</footer>
</main></html>'''
    audit = (f'<p>최초 시점의 76개 관찰 창을 재구성하기 위해 원본 일별 파일 {len(sources)}개 '
             f'({sum(s["trades"] for s in sources.values()):,}개 체결)을 읽었다. 창 내부 체결 수의 합은 '
             f'{int(features.raw_trades.sum()):,}개다. 창이 겹칠 수 있으므로 고유 체결 수는 아니다. '
             '원체결 거래대금과 5분 집계의 상대 오차 최댓값은 약 4.44×10⁻¹⁶이었다. 이는 동일 자료 집계의 일관성 확인이며 거래소 누락 자체를 독립적으로 증명하는 검사는 아니다.</p>')
    html = html.replace('INITIAL_TABLE', html_table('initial')).replace('MINUS15_TABLE', html_table('minus15')).replace('MINUS5_TABLE', html_table('minus5')).replace('AUDIT', audit)
    (OUT / 'tape_report.html').write_text(html)
    print(OUT / 'tape_report.html')


if __name__ == '__main__':
    main()
