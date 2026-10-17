"""NEAR futures onset census, matched antecedents and locked chronological models."""
from pathlib import Path
import gzip
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
OUT = BASE / 'output'
PRICE = ['ret1', 'ret4', 'ret24', 'volume_build', 'compression', 'range_position', 'btc_excess1', 'eth_excess1']
DERIV = ['oi_ret1', 'oi_ret4', 'funding']
HOUR = pd.Timedelta(hours=1)


def price_frame(rows, start, end):
    f = pd.DataFrame(rows, columns=['ms', 'open', 'high', 'low', 'close', 'volume', 'turnover']).astype(float)
    f.index = pd.to_datetime(f.ms, unit='ms').dt.as_unit('ns') + pd.Timedelta(minutes=15)
    f = f.sort_index()
    expected = pd.date_range(pd.Timestamp(start, unit='ms') + pd.Timedelta(minutes=15), pd.Timestamp(end, unit='ms'), freq='15min')
    if not f.index.equals(expected) or not np.isfinite(f.to_numpy()).all():
        raise ValueError('Missing, duplicate or invalid candles')
    if ((f.low <= 0) | (f.high < f[['open', 'close']].max(axis=1)) | (f.low > f[['open', 'close']].min(axis=1)) | (f.turnover < 0)).any():
        raise ValueError('Invalid OHLC')
    return f


def features(frames, oi, funding):
    f = frames['NEARUSDT'].copy()
    f.index = f.index.as_unit('ns')
    for hours in (1, 4, 24):
        f[f'ret{hours}'] = f.close / f.close.shift(hours * 4) - 1
    r = np.log(f.close).diff()
    f['rv24'] = r.rolling(96).std()
    f['compression'] = r.rolling(4).std() / f.rv24
    f['volume_build'] = f.turnover.rolling(4).mean() / f.turnover.shift(4).rolling(96).mean()
    lo, hi = f.low.shift(1).rolling(96).min(), f.high.shift(1).rolling(96).max()
    f['range_position'] = (f.close - lo) / (hi - lo)
    for symbol, key in [('BTCUSDT', 'btc_excess1'), ('ETHUSDT', 'eth_excess1')]:
        other = frames[symbol].close
        f[key] = f.ret1 - (other / other.shift(4) - 1)
    f['oi'] = np.nan
    f['funding'] = np.nan
    for records, timestamp, value, dest, tolerance in [(oi, 'timestamp', 'openInterest', 'oi', '30min'),
                                                       (funding, 'fundingRateTimestamp', 'fundingRate', 'funding', '12h')]:
        if not records:
            continue
        data = pd.DataFrame(records)
        data['available'] = pd.to_datetime(data[timestamp].astype('int64'), unit='ms').dt.as_unit('ns') + pd.Timedelta(minutes=15)
        data[dest] = data[value].astype(float)
        if data.available.duplicated().any() or not np.isfinite(data[dest]).all():
            raise ValueError('Invalid derivatives history')
        merged = pd.merge_asof(pd.DataFrame({'time': f.index}), data.sort_values('available')[['available', dest]],
                               left_on='time', right_on='available', direction='backward', tolerance=pd.Timedelta(tolerance))
        f[dest] = merged[dest].to_numpy()
    f['oi_ret1'] = f.oi / f.oi.shift(4) - 1
    f['oi_ret4'] = f.oi / f.oi.shift(16) - 1
    return f.replace([np.inf, -np.inf], np.nan)


def future_extreme(series, periods, maximum=True):
    rolling = series.shift(-1).iloc[::-1].rolling(periods, min_periods=periods)
    return (rolling.max() if maximum else rolling.min()).iloc[::-1]


def census(f, direction, target, start):
    sign = 1 if direction == 'long' else -1
    jump = sign * (f.close.shift(-4) - f.close * (1 + sign * .02))
    extreme = future_extreme(f.high if sign == 1 else f.low, 96, sign == 1)
    move = sign * (extreme - f.close * (1 + sign * target))
    raw = f.index[(jump >= 0) & (move >= 0) & (f.index >= start)]
    kept = []
    for time in raw:
        if not kept or time - kept[-1] >= 24 * HOUR:
            kept.append(time)
    return pd.DatetimeIndex(kept), raw


def outcomes(f, events):
    y = pd.Series(0., index=f.index)
    event = pd.Series(pd.NaT, index=f.index, dtype='datetime64[ns]')
    for t in events:
        window = (f.index >= t - HOUR) & (f.index <= t - pd.Timedelta(minutes=15))
        y.loc[window] = 1.
        event.loc[window] = t
    y.loc[f.index + 25 * HOUR > f.index.max()] = np.nan
    return y, event


def matched_comparisons(f, events, train_start, train_end):
    finite = np.isfinite(f[PRICE]).all(axis=1)
    pool = f[(f.index >= train_start) & (f.index + 25 * HOUR < train_end) & finite & f.ret1.abs().lt(.01)].copy()
    for event in events:
        pool = pool[(pool.index < event - 24 * HOUR) | (pool.index > event + 24 * HOUR)]
    pairs = []
    for event in events[(events >= train_start) & (events + 24 * HOUR < train_end)]:
        t = event - HOUR
        if t not in f.index or not finite.loc[t] or abs(f.loc[t, 'ret1']) >= .01:
            continue
        possible = pool[(pool.index.hour == t.hour) & (abs(pool.index - t) <= pd.Timedelta(days=28))]
        if possible.empty:
            continue
        match = abs(np.log(possible.rv24) - np.log(f.loc[t, 'rv24'])).idxmin()
        pairs.append((t, match))
        pool = pool.drop(match)
    result = []
    for key in PRICE + DERIV:
        a = np.array([f.loc[t, key] for t, _ in pairs])
        b = np.array([f.loc[t, key] for _, t in pairs])
        valid = np.isfinite(a) & np.isfinite(b)
        if not valid.any():
            continue
        result.append({'feature': key, 'pairs': int(valid.sum()), 'before_onset_median': float(np.median(a[valid])),
                       'control_median': float(np.median(b[valid])), 'case_greater_fraction': float(np.mean(a[valid] > b[valid])),
                       'case_less_fraction': float(np.mean(a[valid] < b[valid])), 'tie_fraction': float(np.mean(a[valid] == b[valid])),
                       'median_paired_difference': float(np.median(a[valid] - b[valid]))})
    return {'pairs': len(pairs), 'comparisons': result}


def replay(scores, threshold):
    selected, last = [], None
    for t, score in scores.items():
        if score >= threshold and (last is None or t - last >= 6 * HOUR):
            selected.append(t)
            last = t
    return pd.DatetimeIndex(selected)


def metrics(times, y, event, all_events, f, direction, target):
    hits = y.reindex(times).fillna(0).astype(bool)
    count, success = len(times), int(hits.sum())
    ci = binomtest(success, count).proportion_ci(method='exact') if count else None
    touched, adverse, duration = [], [], []
    sign = 1 if direction == 'long' else -1
    for t in times:
        future = f.loc[(f.index > t) & (f.index <= t + 24 * HOUR)]
        entry = f.loc[t, 'close']
        progress = sign * ((future.high if sign == 1 else future.low) - entry * (1 + sign * target))
        reaches = future.index[progress >= 0]
        hit = len(reaches) > 0
        touched.append(hit)
        end = reaches[0] if hit else future.index[-1]
        window = future.loc[:end]
        # Include the entire target-touch candle: ordering within15min is unknown, so adverse excursion is conservative.
        worst = window.low.min() / entry - 1 if sign == 1 else 1 - window.high.max() / entry
        adverse.append(min(0., float(worst)))
        if hit:
            duration.append(float((end - t) / HOUR))
    lead = [(event.loc[t] - t) / HOUR for t in times if hits.loc[t]]
    return {'alerts': count, 'onset_hits': success, 'onset_precision': success / count if count else None,
            'precision_exact95': [ci.low, ci.high] if ci else None,
            'events': len(all_events), 'events_caught': int(event.reindex(times).dropna().nunique()),
            'event_recall': event.reindex(times).dropna().nunique() / len(all_events) if len(all_events) else None,
            'median_lead_hours': float(np.median(lead)) if lead else None,
            'target_touch_24h_rate': float(np.mean(touched)) if count else None,
            'median_target_touch_hours': float(np.median(duration)) if duration else None,
            'median_adverse_return': float(np.median(adverse)) if adverse else None,
            'scope': 'Per-direction retrospective replay. Exact interval assumes independent alerts; clustered events can make it optimistic. Target touch is not a trade win rate.'}


def main():
    raw = gzip.decompress((OUT / 'data.json.gz').read_bytes())
    data = json.loads(raw)
    manifest = json.loads((OUT / 'manifest.json').read_text())
    if hashlib.sha256((OUT / 'data.json.gz').read_bytes()).hexdigest() != manifest['data_sha256']:
        raise ValueError('Data hash mismatch')
    if hashlib.sha256((BASE / 'protocol.json').read_bytes()).hexdigest() != manifest['protocol_sha256']:
        raise ValueError('Protocol changed since data collection')
    frames = {s: price_frame(rows, data['start'], data['end']) for s, rows in data['candles'].items()}
    f = features(frames, data['oi'], data['funding'])
    start = pd.Timestamp(data['start'], unit='ms') + pd.Timedelta(days=3)
    train_end, cal_end, end = start + pd.Timedelta(days=90), start + pd.Timedelta(days=120), pd.Timestamp(data['end'], unit='ms')
    splits = {'train': (start, train_end), 'calibration': (train_end + 48 * HOUR, cal_end), 'holdout': (cal_end + 48 * HOUR, end)}
    quiet = f.ret1.abs().lt(.01)
    price_ok = np.isfinite(f[PRICE]).all(axis=1)
    derivative_ok = np.isfinite(f[DERIV]).all(axis=1)
    derivative_coverage = float(derivative_ok[f.index >= start].mean())
    # Paired comparison only if derivatives history has enough coverage; never fill missing derivatives with zero.
    paired = derivative_coverage >= .9
    candidates = quiet & price_ok & (derivative_ok if paired else True)
    report = {'manifest': manifest, 'windows': {k: [str(a), str(b)] for k, (a, b) in splits.items()},
              'source_hashes': {name: hashlib.sha256((BASE / name).read_bytes()).hexdigest() for name in ('study.py', 'protocol.json', 'fetch.mjs')},
              'derivatives_coverage': derivative_coverage, 'paired_models': paired, 'studies': []}
    locks = []
    for direction in ('long', 'short'):
        for target in (.05, .10):
            events, raw_events = census(f, direction, target, start)
            y, target_event = outcomes(f, events)
            masks = {name: (f.index >= a) & (f.index + 25 * HOUR < b) for name, (a, b) in splits.items()}
            event_sets = {name: events[(events >= a) & (events + 24 * HOUR < b)] for name, (a, b) in splits.items()}
            study = {'direction': direction, 'target': target, 'raw_onset_windows': len(raw_events),
                     'distinct_events': {k: len(v) for k, v in event_sets.items()},
                     'training_antecedents': matched_comparisons(f, events, start, train_end), 'models': []}
            train, cal, test = [f.index[masks[k] & candidates & y.notna()] for k in ('train', 'calibration', 'holdout')]
            study['candidate_counts'] = dict(zip(('train', 'calibration', 'holdout'), map(len, (train, cal, test))))
            study['reachable_events'] = {k: int(target_event.reindex(t).dropna().nunique()) for k, t in zip(('train', 'calibration', 'holdout'), (train, cal, test))}
            study['holdout_candidate_base_rate'] = float(y.loc[test].mean()) if len(test) else None
            support = study['reachable_events']['train'] >= 20 and study['reachable_events']['calibration'] >= 5 and y.loc[train].nunique() == 2 and y.loc[cal].nunique() == 2
            if not support:
                study['status'] = 'insufficient_event_support'
                report['studies'].append(study)
                continue
            for name, cols in [('price_volume', PRICE)] + ([('with_derivatives', PRICE + DERIV)] if paired else []):
                model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000))
                model.fit(f.loc[train, cols], y.loc[train])
                threshold = float(np.quantile(model.predict_proba(f.loc[cal, cols])[:, 1], .95))
                scaler, estimator = model.steps[0][1], model.steps[1][1]
                lock = {'direction': direction, 'target': target, 'name': name, 'features': cols, 'threshold': threshold,
                        'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(), 'coef': estimator.coef_[0].tolist(),
                        'intercept': float(estimator.intercept_[0]), 'calibration_end': str(cal_end), 'deployment': 'not_approved'}
                locks.append(lock)
                # Save parameters before looking at this model's heldout predictions.
                (OUT / 'model_locks.json').write_text(json.dumps(locks, indent=2))
                scores = pd.Series(model.predict_proba(f.loc[test, cols])[:, 1], index=test)
                alerts = replay(scores, threshold)
                result = {'name': name, 'threshold': threshold, **metrics(alerts, y, target_event, event_sets['holdout'], f, direction, target)}
                # A uniformly spaced quiet-candidate baseline uses the identical6h replay cooldown.
                baseline_alerts = replay(pd.Series(1., index=test), 1.)
                result['scheduled_baseline'] = metrics(baseline_alerts, y, target_event, event_sets['holdout'], f, direction, target)
                study['models'].append(result)
            study['status'] = 'retrospective_only_no_live_promotion'
            report['studies'].append(study)
    (OUT / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    f.to_parquet(OUT / 'features.parquet')
    print(json.dumps({k: report[k] for k in ('windows', 'derivatives_coverage', 'paired_models')}, indent=2))
    for s in report['studies']:
        print(json.dumps({k: s[k] for k in ('direction', 'target', 'distinct_events', 'status', 'models')}, indent=2))


if __name__ == '__main__':
    main()
