"""Causal episode replay. Outcome columns never enter run_symbol()."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / 'output'
POLICIES = ['entry', 'delay15', 'hold15', 'hold_rebuy']
CUTOFF = 0.8338692277262161
CAUSAL = ['time', 'symbol', 'close', 'low', 'quality', 'eligible_price', 'quote_accel', 'net5', 'netprev10']


def hashes():
    paths = ['study.py', 'protocol.json', '../minute_onset/study.py',
             '../minute_onset/output/frozen.json', '../flow_replay/output/universe.json',
             '../flow_replay/output/download_audit.json']
    return {p: hashlib.sha256((BASE / p).read_bytes()).hexdigest() for p in paths}


def freeze():
    p = OUT / 'frozen.json'
    if p.exists():
        assert json.loads(p.read_text())['source_hashes'] == hashes()
    else:
        p.write_text(json.dumps({'frozen_at': pd.Timestamp.now(tz='UTC').isoformat(),
                                'source_hashes': hashes()}, indent=2))


def flow_features(bars, index):
    # Bar bucket is START; every value is first available at bucket+5min.
    b = bars.set_index('bucket')[['quote', 'buy', 'sell']].sort_index()
    b.index = b.index + pd.Timedelta(minutes=5)
    x = b.reindex(index).fillna(0)
    prior = x.rolling(2, min_periods=2).sum().shift(1)
    return pd.DataFrame({'net5': (x.buy-x.sell)/x.quote.replace(0, np.nan),
                         'netprev10': (prior.buy-prior.sell)/prior.quote.replace(0, np.nan)}, index=index)


def run_symbol(frame):
    """No event/outcome fields accepted; frame is a complete regular observation grid."""
    rows = frame[CAUSAL].sort_values('time')
    below = 0
    armed = False
    episode = None
    previous = None
    alerts = []
    starts = 0
    resets = 0
    for r in rows.itertuples(index=False):
        gap = previous is not None and r.time-previous != pd.Timedelta(minutes=5)
        previous = r.time
        valid = bool(r.quality) and np.isfinite(r.close) and np.isfinite(r.low) and np.isfinite(r.quote_accel)
        if gap or not valid:
            resets += episode is not None
            episode = None
            armed = False
            below = 0
            if not valid:
                continue
        below = below+1 if r.quote_accel < CUTOFF else 0
        if episode is not None and r.time-episode['start'] > pd.Timedelta(minutes=60):
            episode = None
            below = 1 if r.quote_accel < CUTOFF else 0
        if episode is None:
            if below >= 6:
                armed = True
            if not (armed and r.eligible_price and r.quote_accel >= CUTOFF):
                continue
            episode = {'start': r.time, 'price': r.close, 'low': r.close, 'fired': set()}
            armed = False
            starts += 1
        else:
            episode['low'] = min(episode['low'], r.low)
        elapsed = (r.time-episode['start']).total_seconds()/60
        if not r.eligible_price:
            continue
        held = episode['low'] >= episode['price']*.99 and episode['price'] <= r.close < episode['price']*1.02
        conditions = {'entry': elapsed == 0, 'delay15': elapsed == 15,
                      'hold15': elapsed == 15 and held,
                      'hold_rebuy': 15 <= elapsed <= 60 and held and r.netprev10 <= 0 and r.net5 >= .20}
        for policy, condition in conditions.items():
            if condition and policy not in episode['fired']:
                alerts.append({'symbol': r.symbol, 'time': r.time, 'policy': policy,
                               'episode_start': episode['start'], 'wait_minutes': elapsed,
                               'entry_price': episode['price'], 'close_at_alert': r.close,
                               'episode_min_return': episode['low']/episode['price']-1,
                               'net5': r.net5, 'netprev10': r.netprev10})
                episode['fired'].add(policy)
    return alerts, {'episodes': starts, 'quality_or_gap_resets': resets}


def period(d, name):
    if name == 'discovery':
        return d[d.time.ge('2026-04-02') & (d.time+pd.Timedelta(hours=2) < pd.Timestamp('2026-07-01'))]
    return d[d.time.between('2026-08-01', '2026-09-30 23:55')]


def event_keys(d):
    return set(zip(d.symbol, d.next_event))


def metrics(pool, raw_alerts):
    a = raw_alerts[raw_alerts.label_complete]
    hits = a[a.strict_hit]
    events = event_keys(pool[pool.strict_hit])
    found = event_keys(hits)
    return {'alerts_generated': len(raw_alerts), 'alerts_evaluated': len(a),
            'unscorable_alerts': len(raw_alerts)-len(a), 'strict_hits': int(a.strict_hit.sum()),
            'precision': float(a.strict_hit.mean()) if len(a) else None,
            'lead0_hits_excluded': int((a.hit6 & ~a.strict_hit).sum()),
            'eligible_events': len(events), 'detected_events': len(found),
            'recall': len(found)/len(events) if events else None,
            'median_lead_minutes': float(hits.lead_minutes.median()) if len(hits) else None,
            'down5_within60': int(a.min60.le(-.05).sum()),
            'median_return60': float(a.return60.median()) if len(a) else None,
            'median_wait_minutes': float(a.wait_minutes.median()) if len(a) else None}


def random_reference(pool, a):
    a = a[a.label_complete].copy()
    a['day'] = a.time.dt.floor('D')
    p = pool.copy()
    p['day'] = p.time.dt.floor('D')
    lookup = {key: g.strict_hit.to_numpy(dtype=int) for key, g in p.groupby(['symbol', 'day'])}
    draws = np.zeros(200)
    expected = 0.
    rng = np.random.default_rng(20261009)
    for key, g in a.groupby(['symbol', 'day']):
        choices = lookup[key]
        assert len(choices) >= len(g)
        expected += choices.mean()*len(g)
        for i in range(len(draws)):
            draws[i] += rng.choice(choices, size=len(g), replace=False).sum()
    return {'count_matched': len(a), 'expected_hits': float(expected),
            'expected_precision': float(expected/len(a)) if len(a) else None,
            'random_hit_count_2_5pct': float(np.quantile(draws, .025)),
            'random_hit_count_97_5pct': float(np.quantile(draws, .975)),
            'warning': 'Simulation range for count-matched random timestamps; not a confidence interval on policy lift.'}


def main():
    OUT.mkdir(exist_ok=True)
    freeze()
    panel = pd.read_parquet(BASE.parent/'minute_onset/output/panel.parquet')
    symbols = json.loads((BASE.parent/'flow_replay/output/universe.json').read_text())['symbols']
    prepared = []
    for number, symbol in enumerate(symbols, 1):
        g = panel[panel.symbol.eq(symbol)].copy().sort_values('time')
        files = sorted((BASE.parent/'tradeflow/output/bars').glob(symbol+'_*.parquet'))
        files = [p for p in files if '2026-03-31' <= p.stem[len(symbol)+1:] <= '2026-10-01']
        bars = pd.concat([pd.read_parquet(p, columns=['bucket','quote','buy','sell']) for p in files])
        flow = flow_features(bars, pd.DatetimeIndex(g.time))
        for key in ['net5','netprev10']:
            g[key] = flow[key].to_numpy()
        prepared.append(g)
        if number % 6 == 0:
            print(f'Prepared {number}/{len(symbols)} coins', flush=True)
    d = pd.concat(prepared, ignore_index=True)
    all_alerts, result = [], {}
    for name in ['discovery','replication']:
        part = period(d, name)
        alerts, audits = [], {}
        for symbol, g in part.groupby('symbol'):
            generated, audit = run_symbol(g[CAUSAL])
            alerts.extend(generated)
            audits[symbol] = audit
        a = pd.DataFrame(alerts)
        labels = ['symbol','time','label_complete','strict_hit','hit6','lead_minutes','next_event','min60','return60']
        a = a.merge(part[labels], on=['symbol','time'], how='left', validate='many_to_one')
        assert a.label_complete.notna().all()
        a['split'] = name
        all_alerts.append(a)
        pool = part[part.quality & part.eligible_price & part.label_complete]
        result[name] = {'eligible_observations': len(pool), 'episode_count': sum(v['episodes'] for v in audits.values()),
                        'coin_audit': audits, 'policies': {}}
        for policy in POLICIES:
            z = a[a.policy.eq(policy)]
            result[name]['policies'][policy] = {'metrics': metrics(pool,z), 'random': random_reference(pool,z)}
    pd.concat(all_alerts).to_csv(OUT/'alerts.csv', index=False)
    (OUT/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: {p: v['metrics'] for p,v in r['policies'].items()} for k,r in result.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
