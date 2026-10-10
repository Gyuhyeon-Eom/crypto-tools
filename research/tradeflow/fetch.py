"""Freeze matched observations, download official daily spot trades, aggregate causally."""
import concurrent.futures as cf
import hashlib
import io
import json
import ssl
import time
import urllib.request
from pathlib import Path

import certifi
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / 'output'
CONTEXT = ssl.create_default_context(cafile=certifi.where())


def matching_features(d):
    return np.column_stack([d.log_liquidity.astype(float), np.log(d.vol168.astype(float).clip(lower=1e-10)), d.ret24.astype(float)])


def select_samples():
    d = pd.read_parquet(BASE.parent / 'microstructure/output/samples.parquet')
    train = matching_features(d[d.split.eq('train')])
    center, scale = np.mean(train, axis=0), np.std(train, axis=0)
    rng = np.random.default_rng(20261009)
    rows, counts = [], {}
    for split in ['train', 'validation', 'test']:
        part = d[d.split.eq(split)].copy()
        positives = part[part.y.eq(1)].assign(anchor_distance=lambda x: (x.first_hit_hour - 48).abs())
        anchors = positives.sort_values(['anchor_distance', 'time', 'symbol']).drop_duplicates('event_id').sort_values('event_id')
        total = len(anchors)
        if len(anchors) > 80:
            anchors = anchors.iloc[np.sort(rng.choice(len(anchors), 80, replace=False))]
        counts[split] = {'available_events': total, 'selected_events': len(anchors)}
        for _, case in anchors.iterrows():
            controls = part[(part.time.eq(case.time)) & part.first_hit_hour.eq(0) & part.symbol.ne(case.symbol)].copy()
            point = (matching_features(case.to_frame().T).astype(float) - center) / scale
            controls['match_distance'] = np.linalg.norm((matching_features(controls) - center) / scale - point, axis=1)
            controls = controls.sort_values(['match_distance', 'symbol']).head(3)
            if len(controls) < 3:
                continue
            group = split + ':' + case.event_id
            for role, record in [('case', case), *[('control', x) for _, x in controls.iterrows()]]:
                row = record.to_dict()
                row.update(group=group, role=role, match_distance=0.0 if role == 'case' else float(record.match_distance))
                rows.append(row)
    out = pd.DataFrame(rows)
    out.to_parquet(OUT / 'matched_samples.parquet', index=False)
    out[['group','role','symbol','time','split','first_hit_hour','match_distance','log_liquidity','vol168','ret24']].to_csv(OUT / 'matched_samples.csv', index=False)
    audit = {'sampling': counts, 'rows':len(out), 'groups':out.group.nunique(), 'match_center':center.tolist(), 'match_scale':scale.tolist(),
             'protocol_sha256':hashlib.sha256((BASE / 'protocol.json').read_bytes()).hexdigest(),
             'upstream_sha256':hashlib.sha256((BASE.parent / 'microstructure/output/samples.parquet').read_bytes()).hexdigest()}
    (OUT / 'selection_audit.json').write_text(json.dumps(audit, indent=2))
    return out


def aggregate(raw):
    """Each label is the START of its [start,start+5min) bucket. No backward fill."""
    x = raw.sort_values('timestamp', kind='stable').copy()
    if not x.side.str.lower().isin(['buy','sell']).all():
        raise ValueError('Unknown side')
    if not ((x.price > 0) & (x.volume >= 0)).all():
        raise ValueError('Invalid trade price or size')
    x['bucket'] = pd.to_datetime(x.timestamp, unit='ms').dt.floor('5min')
    x['quote'] = x.price * x.volume
    x['buy'] = x.quote.where(x.side.str.lower().eq('buy'), 0)
    x['sell'] = x.quote - x.buy
    x['rpi_quote'] = x.quote.where(x.rpi.eq(1), 0) if 'rpi' in x else 0
    return x.groupby('bucket').agg(open=('price','first'), high=('price','max'), low=('price','min'), close=('price','last'),
                                  quote=('quote','sum'), buy=('buy','sum'), sell=('sell','sum'), trades=('price','size'),
                                  last_ms=('timestamp','max'), rpi_quote=('rpi_quote','sum')).reset_index()


def get_day(task):
    symbol, day = task
    stem = f'{symbol}_{day}'
    raw_path, bars_path = OUT / 'raw' / (stem + '.csv.gz'), OUT / 'bars' / (stem + '.parquet')
    url = f'https://public.bybit.com/spot/{symbol}/{stem}.csv.gz'
    try:
        if not raw_path.exists():
            for attempt in range(4):
                try:
                    req = urllib.request.Request(url, headers={'User-Agent':'crypto-tools-offline-research/1'})
                    data = urllib.request.urlopen(req, context=CONTEXT, timeout=45).read()
                    raw_path.write_bytes(data)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
        data = raw_path.read_bytes()
        raw = pd.read_csv(io.BytesIO(data), compression='gzip')
        dates = pd.to_datetime(raw.timestamp, unit='ms').dt.strftime('%Y-%m-%d')
        if not dates.eq(day).all():
            raise ValueError('Trades outside requested UTC day')
        bars = aggregate(raw)
        bars.to_parquet(bars_path, index=False)
        return {'symbol':symbol,'day':day,'url':url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
                'trades':len(raw),'bars':len(bars),'fetched_at':pd.Timestamp.now(tz='UTC').isoformat()}
    except Exception as e:
        return {'symbol':symbol,'day':day,'url':url,'error':str(e)}


def main():
    for p in [OUT / 'raw', OUT / 'bars']:
        p.mkdir(parents=True, exist_ok=True)
    selected = select_samples()
    tasks = set()
    for r in selected.itertuples():
        for day in pd.date_range((r.time-pd.Timedelta(hours=24)).floor('D'), (r.time-pd.Timedelta(nanoseconds=1)).floor('D'), freq='D'):
            tasks.add((r.symbol, day.strftime('%Y-%m-%d')))
    tasks = sorted(tasks)
    print(f'{selected.group.nunique()} matched sets, {len(selected)} windows, {len(tasks)} daily files', flush=True)
    results = []
    with cf.ThreadPoolExecutor(max_workers=6) as pool:
        for result in pool.map(get_day, tasks):
            results.append(result)
            if len(results) % 50 == 0 or len(results) == len(tasks):
                print(f'{len(results)}/{len(tasks)} files, {sum("error" in x for x in results)} errors, {sum(x.get("trades",0) for x in results):,} trades', flush=True)
                (OUT / 'download_audit.json').write_text(json.dumps(results, indent=2))
    print('Complete', flush=True)


if __name__ == '__main__':
    main()
