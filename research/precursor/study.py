"""Prospective precursor study: closed bars, calendar purging, held-out evaluation.

Run after prepare_data.py. A research experiment, not a live trading strategy.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent / 'output'
STABLE = {'USDC', 'U', 'XUSD', 'PAXG', 'DAI', 'FDUSD', 'TUSD', 'BUSD', 'EUR',
          'USTC', 'USDT', 'USDD', 'RLUSD', 'USD1', 'USDE', 'XAUT', 'USDS', 'PYUSD'}
LARGE = {'BTC', 'ETH', 'BNB', 'XRP', 'SOL', 'ADA', 'DOGE', 'TRX', 'DOT', 'MATIC',
         'LTC', 'BCH', 'LINK', 'AVAX', 'UNI', 'AAVE', 'ATOM'}
FEATURES = [
    'ret_1', 'ret_3', 'ret_7', 'ret_14', 'ret_30', 'volatility_20',
    'volatility_ratio_5_20', 'range_ratio_5_20', 'bb_width_20',
    'bb_relative_60', 'position_20', 'distance_high_60', 'drawdown_30',
    'volume_ratio_3_20', 'volume_ratio_1_20', 'volume_log_slope_7',
    'signed_volume_7', 'up_volume_share_7', 'close_location_3',
    'lower_wick_3', 'upper_wick_3', 'green_days_7', 'efficiency_7',
    'log_liquidity_30', 'illiquidity_20', 'volume_price_divergence',
    'btc_ret_7', 'btc_volatility_20', 'market_ret_7', 'market_breadth',
    'relative_market_7', 'relative_btc_7', 'volume_ratio_rank',
    'relative_strength_rank', 'volatility_rank',
]


def compute_features(bars):
    """All operations use information available by the close of each row only."""
    parts = []
    for symbol, raw in bars.groupby('symbol', sort=True):
        raw = raw.sort_values('date').set_index('date')
        # Missing calendar days must never become consecutive trading bars.
        d = raw.reindex(pd.date_range(raw.index.min(), raw.index.max(), freq='D'))
        c, o, h, low, v = (d[x] for x in ['close', 'open', 'high', 'low', 'usd_volume_proxy'])
        r = c.pct_change(fill_method=None)
        f = pd.DataFrame(index=d.index)
        f['symbol'], f['close'] = symbol, c
        for n in [1, 3, 7, 14, 30]:
            f[f'ret_{n}'] = c / c.shift(n) - 1
        f['volatility_20'] = r.rolling(20).std()
        f['volatility_ratio_5_20'] = r.rolling(5).std() / r.rolling(20).std().shift(5)
        day_range = (h - low) / c
        f['range_ratio_5_20'] = day_range.rolling(5).mean() / day_range.rolling(20).mean().shift(5)
        f['bb_width_20'] = 4 * c.rolling(20).std() / c.rolling(20).mean()
        f['bb_relative_60'] = f.bb_width_20 / f.bb_width_20.rolling(60).median().shift(1)
        f['position_20'] = (c - low.rolling(20).min()) / (h.rolling(20).max() - low.rolling(20).min())
        f['distance_high_60'] = c / h.rolling(60).max() - 1
        f['drawdown_30'] = c / h.rolling(30).max() - 1
        f['volume_ratio_3_20'] = v.rolling(3).mean() / v.rolling(20).mean().shift(3)
        f['volume_ratio_1_20'] = v / v.rolling(20).mean().shift(1)
        # Least-squares slope of log volume, scale-independent across symbols.
        f['volume_log_slope_7'] = np.log1p(v).rolling(7).apply(lambda a: np.dot(a, np.arange(7) - 3) / 28, raw=True)
        f['signed_volume_7'] = (np.sign(r) * v).rolling(7).sum() / v.rolling(7).sum()
        f['up_volume_share_7'] = v.where(r > 0, 0).rolling(7).sum() / v.rolling(7).sum()
        denom = (h - low).replace(0, np.nan)
        f['close_location_3'] = ((2 * c - h - low) / denom).rolling(3).mean()
        f['lower_wick_3'] = ((pd.concat([o, c], axis=1).min(axis=1) - low) / denom).rolling(3).mean()
        f['upper_wick_3'] = ((h - pd.concat([o, c], axis=1).max(axis=1)) / denom).rolling(3).mean()
        f['green_days_7'] = (r > 0).astype(float).rolling(7).mean()
        f['efficiency_7'] = (c - c.shift(7)).abs() / c.diff().abs().rolling(7).sum()
        f['liquidity_30'] = v.rolling(30).mean()
        f['log_liquidity_30'] = np.log1p(f.liquidity_30)
        f['illiquidity_20'] = (r.abs() / v.replace(0, np.nan) * 1e6).rolling(20).mean()
        f['volume_price_divergence'] = np.log(f.volume_ratio_3_20.clip(lower=1e-8)) - f.ret_3 * 10
        f['continuous_90'] = c.notna().rolling(90).sum().eq(90)
        f['high'], f['low'] = h, low
        f['date'] = f.index
        parts.append(f.reset_index(drop=True))
    panel = pd.concat(parts, ignore_index=True)
    btc = panel[panel.symbol.eq('BTCUSDT')][['date', 'ret_7', 'volatility_20']].rename(columns={
        'ret_7': 'btc_ret_7', 'volatility_20': 'btc_volatility_20'})
    panel = panel.merge(btc, on='date', how='left', validate='many_to_one')
    crypto = ~panel.symbol.str.removesuffix('USDT').isin(STABLE | LARGE)
    base = panel.loc[crypto & panel.continuous_90 & panel.liquidity_30.between(200_000, 20_000_000)].copy()
    market = base.groupby('date').agg(market_ret_7=('ret_7', 'median'),
                                      market_breadth=('ret_1', lambda x: float((x > 0).mean())))
    panel = panel.merge(market, on='date', how='left', validate='many_to_one')
    panel['relative_market_7'] = panel.ret_7 - panel.market_ret_7
    panel['relative_btc_7'] = panel.ret_7 - panel.btc_ret_7
    ranks = base[['date', 'symbol']].copy()
    for feature, name in [('volume_ratio_3_20', 'volume_ratio_rank'),
                          ('ret_7', 'relative_strength_rank'), ('volatility_20', 'volatility_rank')]:
        ranks[name] = base.groupby('date')[feature].rank(pct=True)
    panel = panel.merge(ranks, on=['date', 'symbol'], how='left', validate='one_to_one')
    return panel.replace([np.inf, -np.inf], np.nan)


def label_panel(panel, target=.30, horizon=7, quiet_next_day=.10):
    """Label only, never used to decide eligibility or construct a feature.

    Observe at the end of day t. Success requires a target touch on days 2..7,
    with the high on day 1 below +10%, hence a full quiet day after observation.
    This measures lead to a target, not an unobservable pump organizer's actions.
    """
    frames = []
    for symbol, g in panel.groupby('symbol', sort=True):
        g = g.sort_values('date').copy()
        highs = np.column_stack([g.high.shift(-k).to_numpy() for k in range(1, horizon + 1)])
        lows = np.column_stack([g.low.shift(-k).to_numpy() for k in range(1, horizon + 1)])
        closes = np.column_stack([g.close.shift(-k).to_numpy() for k in range(1, horizon + 1)])
        entry = g.close.to_numpy()[:, None]
        mature = np.isfinite(highs).all(axis=1) & np.isfinite(lows).all(axis=1)
        touch = highs >= entry * (1 + target)
        first = np.where(touch.any(axis=1), touch.argmax(axis=1) + 1, 0)
        quiet = highs[:, 0] < entry[:, 0] * (1 + quiet_next_day)
        y = mature & quiet & (first >= 2)
        g['y'] = y.astype(int)
        g['y_close'] = (mature & quiet & (closes[:, 1:] >= entry * (1 + target)).any(axis=1)).astype(int)
        g['y_down'] = (mature & (lows <= entry * .80).any(axis=1)).astype(int)
        g['first_hit_day'] = first
        g['mature'] = mature
        g['feature_at'] = g.date + pd.Timedelta(days=1)
        g['label_end'] = g.date + pd.Timedelta(days=horizon + 1)
        g['target_date'] = g.date + pd.to_timedelta(first, unit='D')
        g['forward_max'] = np.max(highs / entry - 1, axis=1)
        g['forward_min'] = np.min(lows / entry - 1, axis=1)
        # Conservative same-day ordering: any stop on/before target day fails.
        days = np.arange(1, horizon + 1)[None, :]
        stopped = ((lows <= entry * .90) & (days <= first[:, None])).any(axis=1)
        g['y_clean'] = (y & ~stopped).astype(int)
        frames.append(g)
    d = pd.concat(frames, ignore_index=True)
    known = (~d.symbol.str.removesuffix('USDT').isin(STABLE | LARGE)
             & d.continuous_90 & d.liquidity_30.between(200_000, 20_000_000)
             & (d.ret_1 < .05) & (d.ret_7 < .10))
    # Retain rows without BTC history: imputer fitted on train, plus no BTC-based
    # hypothesis is interpreted where that feature is missing.
    d = d.loc[known & d.mature].sort_values(['date', 'symbol']).reset_index(drop=True)
    d['split'] = 'purged'
    d.loc[d.label_end < pd.Timestamp('2026-05-01'), 'split'] = 'train'
    d.loc[(d.feature_at >= '2026-05-01') & (d.label_end < '2026-08-01'), 'split'] = 'validation'
    d.loc[d.feature_at >= '2026-08-01', 'split'] = 'test'
    # Same-symbol hits occurring within 7 days form one evaluable surge episode.
    # Negative forecasts have no event; this is an event-accounting convention.
    d['event_id'] = ''
    for symbol, g in d[d.y.eq(1)].groupby('symbol'):
        anchor, event = None, 0
        for idx, row in g.sort_values('target_date').iterrows():
            if anchor is None or (row.target_date - anchor).days > horizon:
                anchor, event = row.target_date, event + 1
            d.at[idx, 'event_id'] = f'{symbol}:{event}'
    return d


def wilson(hits, n):
    if not n:
        return [None, None]
    p, z = hits / n, 1.96
    mid = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [max(0, mid-half), min(1, mid+half)]


def alert_policy(frame, scores, threshold=-np.inf, daily_cap=3, cooldown=7):
    """Causal daily ranking; no forced alerts below threshold; per-coin cooldown."""
    d = frame.assign(score=np.asarray(scores)).sort_values(['date', 'score', 'symbol'], ascending=[True, False, True])
    selected, last = [], {}
    for date, group in d.groupby('date', sort=True):
        count = 0
        for idx, row in group.iterrows():
            if row.score < threshold:
                continue
            if row.symbol in last and (date - last[row.symbol]).days < cooldown:
                continue
            selected.append(idx)
            last[row.symbol] = date
            count += 1
            if count >= daily_cap:
                break
    return d.loc[selected].sort_values(['date', 'symbol'])


def metrics(frame, scores, selected):
    n, hits = len(selected), int(selected.y.sum())
    baseline = frame.y.mean()
    # Compare against random eligible assets on the exact dates the policy alerts.
    expected = selected.date.map(frame.groupby('date').y.mean()).sum() if n else 0
    all_events = frame.loc[frame.y.eq(1), 'event_id'].nunique()
    return {'rows': len(frame), 'positive_rows': int(frame.y.sum()), 'base_rate': float(baseline),
            'pr_auc': float(average_precision_score(frame.y, scores)),
            'roc_auc': float(roc_auc_score(frame.y, scores)) if frame.y.nunique() == 2 else None,
            'alerts': n, 'hits': hits, 'precision': hits/n if n else None,
            'precision_wilson95_descriptive': wilson(hits, n),
            'same_date_random_expected_hits': float(expected),
            'same_date_lift': hits/expected if expected > 0 else None,
            'captured_events': selected.loc[selected.y.eq(1), 'event_id'].nunique(),
            'evaluable_events': all_events,
            'event_recall': selected.loc[selected.y.eq(1), 'event_id'].nunique()/all_events if all_events else None,
            'clean_hits_before_10pct_drop': int(selected.y_clean.sum()),
            'close_confirmed_hits': int(selected.y_close.sum()),
            'down20_count': int(selected.y_down.sum()),
            'median_target_day_if_hit': float(selected.loc[selected.y.eq(1), 'first_hit_day'].median()) if hits else None}


def block_lift_ci(d, mask, repetitions=500):
    """14-day calendar block bootstrap; preserves within-block coin/day dependence."""
    work = d.assign(selected=np.asarray(mask, dtype=bool)).copy()
    work['block'] = (work.date - work.date.min()).dt.days // 14
    work['sel_y'] = work.y * work.selected
    a = work.groupby('block').agg(n=('y', 'size'), hits=('y', 'sum'), selected=('selected', 'sum'), sh=('sel_y', 'sum')).to_numpy(float)
    rng = np.random.default_rng(20261009)
    sampled = a[rng.integers(0, len(a), size=(repetitions, len(a)))].sum(axis=1)
    valid = (sampled[:, 2] > 0) & (sampled[:, 1] > 0)
    lifts = (sampled[valid, 3] / sampled[valid, 2]) / (sampled[valid, 1] / sampled[valid, 0])
    return np.quantile(lifts, [.025, .975]).tolist() if len(lifts) else [None, None]


def evidence(samples):
    """Discover rule direction on train only, carry unchanged cutoffs forward."""
    tr = samples[samples.split.eq('train')]
    records = []
    for feature in FEATURES:
        valid = tr[tr[feature].notna()]
        low, high = valid[feature].quantile([.25, .75])
        low_p = valid.loc[valid[feature] <= low, 'y'].mean()
        high_p = valid.loc[valid[feature] >= high, 'y'].mean()
        direction, cutoff = ('high', high) if high_p > low_p else ('low', low)
        result = {'feature': feature, 'direction': direction, 'cutoff': float(cutoff)}
        for split in ['train', 'validation', 'test']:
            d = samples[samples.split.eq(split) & samples[feature].notna()].copy()
            mask = d[feature] >= cutoff if direction == 'high' else d[feature] <= cutoff
            s = d.loc[mask]
            result.update({f'{split}_n': len(s), f'{split}_hits': int(s.y.sum()),
                           f'{split}_precision': float(s.y.mean()),
                           f'{split}_lift': float(s.y.mean()/d.y.mean()),
                           f'{split}_down20': float(s.y_down.mean()),
                           f'{split}_base_down20': float(d.y_down.mean())})
            # Date, liquidity and volatility matched controls, excluding rule positives.
            d['liq_bin'] = np.minimum((d.groupby('date').log_liquidity_30.rank(pct=True)*3).astype(int), 2)
            d['vol_bin'] = np.minimum((d.groupby('date').volatility_20.rank(pct=True)*3).astype(int), 2)
            d['selected'] = mask
            keys = ['date', 'liq_bin', 'vol_bin']
            controls = d[~mask].groupby(keys).y.agg(['mean', 'size'])
            matched = d[mask].join(controls, on=keys)
            matched = matched[matched['size'].ge(3)]
            result[f'{split}_matched_coverage'] = len(matched)/len(s) if len(s) else 0
            result[f'{split}_matched_lift'] = float(matched.y.mean()/matched['mean'].mean()) if len(matched) and matched['mean'].mean() > 0 else None
            if split == 'test':
                ci = block_lift_ci(d, mask)
                result['test_lift_ci_low'], result['test_lift_ci_high'] = ci
        records.append(result)
    result = pd.DataFrame(records).sort_values('train_lift', ascending=False)
    result.to_csv(OUT / 'signal_evidence.csv', index=False)
    return result


def run(target):
    bars = pd.read_parquet(OUT / 'daily_bars.parquet')
    print('Computing causal features and labels...', flush=True)
    features = compute_features(bars)
    d = label_panel(features, target=target)
    d.to_parquet(OUT / 'samples.parquet', index=False)
    parts = {s: d[d.split.eq(s)].copy() for s in ['train', 'validation', 'test']}
    overview = {s: {'rows': len(x), 'positives': int(x.y.sum()), 'coins': x.symbol.nunique(),
                    'events': x.loc[x.y.eq(1), 'event_id'].nunique(),
                    'start': str(x.feature_at.min()), 'end': str(x.feature_at.max())} for s, x in parts.items()}
    print(json.dumps(overview, indent=2), flush=True)
    print('Evaluating predeclared signal families...', flush=True)
    signal_table = evidence(d)
    print(signal_table[['feature', 'direction', 'train_lift', 'validation_lift', 'test_lift']].head(10).to_string(index=False), flush=True)
    train, val, test = (parts[s] for s in ['train', 'validation', 'test'])
    models = {
        'logistic': make_pipeline(SimpleImputer(strategy='median', add_indicator=True),
                                  StandardScaler(), LogisticRegression(C=.1, max_iter=1500)),
        'small_boosting': HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=7,
            min_samples_leaf=100, l2_regularization=10, learning_rate=.05,
            early_stopping=False, random_state=42),
    }
    validation_results, score_sets = {}, {}
    for name, model in models.items():
        print('Training', name, flush=True)
        with threadpool_limits(limits=4):
            model.fit(train[FEATURES], train.y)
            score_sets[name] = {s: model.predict_proba(parts[s][FEATURES])[:, 1] for s in ['validation', 'test']}
        selected = alert_policy(val, score_sets[name]['validation'])
        validation_results[name] = metrics(val, score_sets[name]['validation'], selected)
    # Choose model solely from validation; test outputs never drive choices.
    winner = max(models, key=lambda name: validation_results[name]['pr_auc'])
    candidates = []
    for quantile in [.50, .70, .80, .90, .95, .98]:
        threshold = float(np.quantile(score_sets[winner]['validation'], quantile))
        selected = alert_policy(val, score_sets[winner]['validation'], threshold)
        if len(selected) >= 30:
            hits = int(selected.y.sum())
            candidates.append((wilson(hits, len(selected))[0], threshold, quantile, len(selected)))
    _, threshold, threshold_quantile, _ = max(candidates)
    results = {'protocol': {
        'target': target, 'horizon_days': 7,
        'first_24h_max_gain_below': .10,
        'past_day_gain_below': .05, 'past_week_gain_below': .10,
        'liquidity_proxy_range': [200000, 20000000],
        'entry': 'daily close; forecast available at next UTC midnight',
        'success': 'First target high touch on days 2..7, with first day high < +10%. No stop-loss condition in main label.',
        'clean_success': 'Main success and no -10% low touch on/before target day; ambiguous same-day order counted failure.',
        'validation_start': '2026-05-01', 'test_start': '2026-08-01',
        'purge': 'Exclude any train/validation row whose 7-day label ends at/after next partition start.',
        'model_selection': 'Validation PR-AUC only; fixed hyperparameters; no class reweighting.',
        'threshold_selection': 'Validation-only quantile grid, highest Wilson lower bound with >=30 de-duplicated alerts.',
        'alert_policy': 'Maximum 3 new alerts/day above cutoff; 7-day per-symbol cooldown; each evaluation partition starts with empty cooldown state.',
        'hypothesis_caution': '35 exploratory features; individual bootstrap intervals are not corrected for multiple comparisons. Temporal stability is not evidence of causation.',
        'lead_time_caution': 'A full first day without +10% touch is guaranteed by positive labels; daily candles cannot establish exact breakout onset time.',
        'deployment': 'Research model only; probabilities not calibrated; no trading or notification integration.'},
        'overview': overview, 'validation_models': validation_results, 'selected_model': winner,
        'threshold': threshold, 'threshold_validation_quantile': threshold_quantile,
        'test_models_fixed_budget': {}, 'primary': {}}
    for name in models:
        selected = alert_policy(test, score_sets[name]['test'])
        results['test_models_fixed_budget'][name] = metrics(test, score_sets[name]['test'], selected)
    for split in ['validation', 'test']:
        frame = parts[split]
        selected = alert_policy(frame, score_sets[winner][split], threshold)
        results['primary'][split] = metrics(frame, score_sets[winner][split], selected)
        selected.to_csv(OUT / f'{split}_alerts.csv', index=False)
    # Fixed, simple ranking baselines, on the same eligible rows and alert budget.
    baselines = {'volatility_only': test.volatility_20.to_numpy(),
                 'volume_build_only': test.volume_ratio_3_20.fillna(0).to_numpy(),
                 'compression_only': -test.bb_relative_60.fillna(1).to_numpy(),
                 'recent_decline_only': -test.ret_7.to_numpy()}
    results['test_baselines_fixed_budget'] = {}
    for name, scores in baselines.items():
        results['test_baselines_fixed_budget'][name] = metrics(test, scores, alert_policy(test, scores))
    # Repeat fixed chosen design in disjoint test months; do not refit or retune.
    test_scored = test.assign(score=score_sets[winner]['test'])
    primary_alerts = alert_policy(test, score_sets[winner]['test'], threshold)
    results['test_months'] = {}
    for month, g in test_scored.groupby(test_scored.feature_at.dt.strftime('%Y-%m')):
        selected = primary_alerts.loc[primary_alerts.index.intersection(g.index)]
        results['test_months'][month] = metrics(g, g.score, selected)
    model_bundle = {'model': models[winner], 'features': FEATURES, 'threshold': threshold,
                    'protocol': results['protocol'], 'trained_through': overview['train']['end']}
    joblib.dump(model_bundle, OUT / 'research_model.joblib')
    (OUT / 'results.json').write_text(json.dumps(results, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x)))
    print('Selected:', winner, 'threshold:', threshold, flush=True)
    print(json.dumps(results['primary'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=float, default=.30)
    run(parser.parse_args().target)
