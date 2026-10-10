"""Causal microstructure aggregation and nullable, unchanged surge labels."""
from pathlib import Path
import argparse
import gzip
import hashlib
import importlib.util
import json
import subprocess
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('frontier', BASE.parent/'precision_frontier/study.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
WINDOWS = [1, 5, 15, 60]
MEASURES = ['ask_depletion_buy', 'bid_recovery_sell', 'ask_cost_depletion_buy', 'mid_return_60s', 'spread_bps', 'no_trade']
MICRO = [f'{key}_{minutes}m' for minutes in WINDOWS for key in MEASURES]
TARGETS = ['JTOUSDT', 'TREEUSDT', 'WLDUSDT']


def utc(value):
    t = pd.Timestamp(value)
    return t.tz_localize('UTC').tz_convert(None) if t.tzinfo is None else t.tz_convert('UTC').tz_localize(None)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_captures(paths):
    frames, intervals, manifests = [], {}, []
    for directory in sorted(map(Path, paths)):
        summary = json.loads((directory/'summary.json').read_text())
        # Re-run integrity audit; do not trust a stale passed flag after file edits.
        result = subprocess.run(['node', str(BASE.parent/'orderbook_lab/audit.mjs'), str(directory.resolve())], capture_output=True, text=True)
        if result.returncode:
            raise ValueError(f'Capture replay failed: {directory}: {result.stderr or result.stdout}')
        start, end = utc(summary['started_at']), utc(summary['finished_at'])
        for symbol in summary['symbols']:
            for a, b in intervals.get(symbol, []):
                if start < b and end > a:
                    raise ValueError(f'Overlapping runs for {symbol}')
            intervals.setdefault(symbol, []).append((start, end))
        with gzip.open(directory/'features.jsonl.gz', 'rt') as f:
            rows = [json.loads(line) for line in f]
        frame = pd.DataFrame(rows)
        frame['run'] = directory.name
        frame['received_time'] = pd.to_datetime(frame.received_ms, unit='ms', utc=True).dt.tz_localize(None)
        # Reject wall-clock regressions or material jumps relative to monotonic time.
        for _, group in frame.groupby('symbol', sort=False):
            if group.received_time.diff().dropna().lt(pd.Timedelta(0)).any():
                raise ValueError('Non-monotone receiver wall clock')
            offset = group.received_ms - group.t
            if offset.max() - offset.min() > 250:
                raise ValueError('Receiver wall clock moved >250ms during run')
        frames.append(frame)
        manifests.append({'path': str(directory.resolve()), 'summary_sha256': sha(directory/'summary.json'),
                          'features_sha256': sha(directory/'features.jsonl.gz')})
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), manifests


def aggregate(samples, at):
    """Use exact arrival cutoff before second-binning; never carry a future sample back."""
    empty = {key: np.nan for key in MICRO}
    result = {**empty, 'micro_eligible': False, 'micro_reason': 'no_samples'}
    at = utc(at)
    if samples.empty:
        return result
    x = samples[(samples.received_time > at-pd.Timedelta(hours=1)) & (samples.received_time <= at)].copy()
    if x.empty:
        return result
    x = x.sort_values('received_time').drop_duplicates('received_time', keep='last')
    if x[['run', 'connection', 'generation']].drop_duplicates().shape[0] != 1:
        return {**result, 'micro_reason': 'reset_or_run_boundary'}
    required = ['ask_depth_change', 'bid_recovery', 'ask_cost_change', 'mid_return_60s', 'spread_bps',
                'buy_quote_60s', 'sell_quote_60s', 'net_taker_fraction']
    for key in required:
        if key not in x:
            x[key] = np.nan
    zero = x.buy_quote_60s.eq(0) & x.sell_quote_60s.eq(0)
    flow = x.net_taker_fraction.where(~zero, 0)
    x['ask_depletion_buy'] = (-x.ask_depth_change).clip(lower=0) * flow.clip(lower=0)
    x['bid_recovery_sell'] = x.bid_recovery * (-flow).clip(lower=0)
    x['ask_cost_depletion_buy'] = (-x.ask_cost_change).clip(lower=0) * flow.clip(lower=0)
    x['no_trade'] = zero.astype(float)
    x['valid'] = x.status.eq('ready') & np.isfinite(x[MEASURES]).all(axis=1)
    # Keep latest actual observation in each second, not an average with an earlier reset.
    x['second'] = x.received_time.dt.floor('s')
    x = x.drop_duplicates('second', keep='last')
    for minutes in WINDOWS:
        left = at-pd.Timedelta(minutes=minutes)
        window = x[x.received_time > left]
        good = window[window.valid]
        coverage = min(1., len(good)/(minutes*60))
        result[f'coverage_{minutes}m'] = coverage
        if coverage < .95:
            return {**result, 'micro_reason': f'coverage_{minutes}m'}
        times = [left, *good.received_time, at]
        if max(b-a for a, b in zip(times, times[1:])) > pd.Timedelta(seconds=2.5):
            return {**result, 'micro_reason': f'gap_{minutes}m'}
        for key in MEASURES:
            result[f'{key}_{minutes}m'] = float(good[key].mean() if key == 'no_trade' else good[key].median())
    return {**result, 'micro_eligible': True, 'micro_reason': 'ready'}


def prices(raw, asof):
    raw = raw.copy()
    raw['time'] = pd.to_datetime(raw.time, utc=True).dt.tz_localize(None)
    raw = raw[raw.time.le(utc(asof).floor('h'))]
    if raw.duplicated(['symbol', 'time']).any():
        raise ValueError('Duplicate price rows')
    if not raw.time.eq(raw.time.dt.floor('h')).all():
        raise ValueError('Price timestamps must be hourly candle END')
    if 'open_ms' in raw and not (pd.to_datetime(raw.open_ms, unit='ms')+pd.Timedelta(hours=1)).eq(raw.time).all():
        raise ValueError('Candle timestamp convention mismatch')
    if not (np.isfinite(raw[['open','high','low','close','turnover']]).all().all() and raw[['open','high','low','close']].gt(0).all().all()):
        raise ValueError('Invalid candle values')
    if not (raw.high.ge(raw[['open','close','low']].max(axis=1)).all() and raw.low.le(raw[['open','close','high']].min(axis=1)).all()):
        raise ValueError('Invalid OHLC range')
    frames = []
    for symbol, group in raw.groupby('symbol'):
        f = legacy.outcomes(legacy.hard.census.labels(legacy.hard.features(group)))
        f['symbol'] = symbol
        f['log_volume'] = np.log(f.volume_build.where(f.volume_build.gt(0)))
        f['label_status'] = np.where(f.index+pd.Timedelta(hours=30) > utc(asof), 'pending_time',
                                    np.where(f.label_complete, 'mature', 'missing_price'))
        f['hit'] = f.hit.astype('boolean').where(f.label_status.eq('mature'), pd.NA)
        f.loc[~f.label_status.eq('mature'), ['target_event', 'lead_hours']] = [pd.NaT, np.nan]
        frames.append(f.reset_index(drop=True))
    return pd.concat(frames, ignore_index=True)


def build(raw, samples, asof, start, end, symbols):
    p = prices(raw[raw.symbol.isin(symbols)], asof)
    grid = pd.MultiIndex.from_product([symbols, pd.date_range(utc(start), utc(end), freq='h', inclusive='left')], names=['symbol','time']).to_frame(index=False)
    if grid.time.gt(utc(asof)).any():
        raise ValueError('Cannot build future decision timestamps')
    data = grid.merge(p, on=['symbol','time'], how='left', validate='one_to_one')
    missing_status = pd.Series(np.where(data.time+pd.Timedelta(hours=30)>utc(asof), 'pending_time', 'missing_price'), index=data.index)
    data['label_status'] = data.label_status.fillna(missing_status)
    data['hit'] = data.hit.astype('boolean')
    data['base_candidate'] = legacy.candidate_mask(data.fillna({'past_complete':False}))
    micro = []
    groups = {s:g for s,g in samples.groupby('symbol')} if not samples.empty else {}
    for row in data[['symbol','time']].itertuples(index=False):
        micro.append(aggregate(groups.get(row.symbol, pd.DataFrame()), row.time))
    return pd.concat([data.reset_index(drop=True), pd.DataFrame(micro)], axis=1)


def summarize(data):
    return {'hourly_observations':len(data), 'base_candidates':int(data.base_candidate.sum()),
            'micro_eligible':int(data.micro_eligible.sum()),
            'paired_candidates':int((data.base_candidate & data.micro_eligible).sum()),
            'mature_paired_candidates':int((data.base_candidate & data.micro_eligible & data.label_status.eq('mature')).sum()),
            'label_status':data.label_status.value_counts().to_dict(), 'micro_status':data.micro_reason.value_counts().to_dict(),
            'precision':None, 'reason':'Dataset readiness only; no trained or evaluated model.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prices', type=Path, required=True)
    parser.add_argument('--captures', nargs='+', type=Path, required=True)
    parser.add_argument('--asof', required=True)
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    parser.add_argument('--symbols', default=','.join(TARGETS))
    parser.add_argument('--universe', type=Path, help='JSON containing the fixed symbols; overrides --symbols')
    parser.add_argument('--output', type=Path, default=BASE/'output')
    args = parser.parse_args()
    if args.universe:
        args.symbols = ','.join(json.loads(args.universe.read_text())['symbols'])
    if not args.symbols or len(set(args.symbols.split(','))) != len(args.symbols.split(',')):
        parser.error('Empty or duplicate target symbols')
    if utc(args.start) != utc(args.start).floor('h') or utc(args.end) != utc(args.end).floor('h') or utc(args.end)<=utc(args.start):
        parser.error('start/end must be increasing UTC hourly boundaries')
    if utc(args.asof)>pd.Timestamp.now(tz='UTC').tz_localize(None):
        parser.error('asof cannot be in the future')
    samples, manifest = load_captures(args.captures)
    data = build(pd.read_parquet(args.prices), samples, args.asof, args.start, args.end, args.symbols.split(','))
    args.output.mkdir(parents=True, exist_ok=True)
    data.to_parquet(args.output/'dataset.parquet', index=False)
    result = {**summarize(data), 'asof':args.asof, 'captures':manifest, 'prices_sha256':sha(args.prices),
              'dataset_sha256':sha(args.output/'dataset.parquet'), 'sources':{p:sha(BASE/p) for p in ['dataset.py','protocol.json']}}
    result['capture_end_window_check'] = []
    for symbol, group in samples.groupby('symbol'):
        if symbol not in args.symbols.split(','):
            continue
        end=group.received_time.max()
        r=aggregate(group,end)
        result['capture_end_window_check'].append({'symbol':symbol,'checked_at':end.isoformat(),
            'micro_eligible':r['micro_eligible'],'reason':r['micro_reason'],
            **{k:v for k,v in r.items() if k.startswith('coverage_')}})
    (args.output/'readiness.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
