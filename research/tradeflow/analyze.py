"""Matched comparisons: these results are not alert precision or a backtest."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / 'output'
SCORES = ['buy_shift', 'sell_resilience', 'compression', 'quiet_activity', 'sequence']


def flow(x):
    return (x.buy.sum() - x.sell.sum()) / x.quote.sum() if x.quote.sum() > 0 else np.nan


def impact(x, min_bars):
    mask = (x.buy-x.sell) / x.quote.replace(0,np.nan) <= -.20
    if mask.sum() < min_bars:
        return np.nan
    z = x[mask]
    return (-z.ret).clip(lower=0).sum() / ((z.sell-z.buy)/z.quote).sum()


def features(bars, anchor, historical_hourly_range):
    start = anchor - pd.Timedelta(hours=24)
    x = bars[(bars.bucket >= start) & (bars.bucket < anchor)].copy().set_index('bucket')
    if not x.index.is_unique:
        raise ValueError('Duplicate 5min buckets')
    x = x.reindex(pd.date_range(start, anchor, freq='5min', inclusive='left'))
    active = x.quote.notna()
    result = {'active_fraction':float(active.mean()),'quote24':float(x.quote.sum()),'trades24':int(x.trades.sum()),
              'last_trade_age_minutes':(anchor.value/1e6-x.last_ms.max())/60000,
              'rpi_share':float(x.rpi_quote.sum()/x.quote.sum()) if x.quote.sum()>0 else np.nan}
    if active.mean()<.80 or result['last_trade_age_minutes']>5 or not active.iloc[0]:
        result['quality_error'] = 'Sparse or stale trades (including missing first bucket)'
        return result
    x['close'] = x.close.ffill()
    for col in ['open','high','low']:
        x[col] = x[col].fillna(x.close)
    for col in ['quote','buy','sell','trades']:
        x[col] = x[col].fillna(0)
    previous = x.close.shift(1)
    previous.iloc[0] = x.open.iloc[0]
    x['ret'] = x.close / previous - 1
    baseline, absorption, reversal = x.iloc[:216], x.iloc[216:264], x.iloc[264:]
    absorb_return = absorption.close.iloc[-1]/baseline.close.iloc[-1]-1
    reversal_return = reversal.close.iloc[-1]/absorption.close.iloc[-1]-1
    sell_ratio = (absorption.sell.sum()/4)/(baseline.sell.sum()/18) if baseline.sell.sum()>0 else np.nan
    base_impact, recent_impact = impact(baseline,8), impact(absorption,3)
    resilience = 1-recent_impact/base_impact if base_impact>0 else np.nan
    shift = flow(reversal)-flow(absorption)
    range18 = ((baseline.high-baseline.low)/baseline.close).mean()
    compression = -np.log(range18/(historical_hourly_range/12)) if range18>0 and historical_hourly_range>0 else np.nan
    activity = (absorption.quote.sum()/4)/(baseline.quote.sum()/18)
    absorb = flow(absorption)<=-.10 and sell_ratio>=1.25 and absorb_return>=-.005
    sequence = absorb and flow(reversal)>=.10 and shift>=.15 and reversal_return>0
    result.update(buy_shift=shift, sell_resilience=resilience, compression=compression,
                  quiet_activity=np.log(activity)-abs(absorb_return)/.01 if activity>0 else np.nan,
                  sequence=int(sequence), absorption_rule=int(absorb), buy_flow2=flow(reversal), sell_flow4=flow(absorption),
                  sell_activity_ratio=sell_ratio, absorption_return=absorb_return, reversal_return=reversal_return,
                  downside_impact_baseline=base_impact, downside_impact_recent=recent_impact)
    return result


def block_ci(values, times):
    """Resample UTC 7-day clusters, preserving all within-block matched comparisons."""
    f = pd.DataFrame({'value':values,'time':times}).dropna()
    if not len(f):
        return {'n':0}
    f['block'] = (f.time-pd.Timestamp('2026-01-01')).dt.days//7
    blocks = f.groupby('block').value.agg(['sum','count']).to_numpy()
    rng = np.random.default_rng(20261009)
    sums = blocks[rng.integers(0,len(blocks),(10000,len(blocks)))].sum(axis=1)
    bootstrap = sums[:,0]/sums[:,1]
    return {'n':len(f),'blocks':len(blocks),'concordance':float(f.value.mean()),
            'lower95':float(np.quantile(bootstrap,.025)),'upper95':float(np.quantile(bootstrap,.975)),
            'lower99':float(np.quantile(bootstrap,.005)),'upper99':float(np.quantile(bootstrap,.995))}


def compare(d, score):
    values,times,cases,controls = [],[],[],[]
    for _, g in d.groupby('group'):
        if len(g)!=4 or not np.isfinite(g[score]).all():
            continue
        case = float(g.loc[g.role.eq('case'),score].iloc[0])
        control = g.loc[g.role.eq('control'),score].to_numpy()
        values.append(float(((case>control)+.5*(case==control)).mean()))
        times.append(g.time.iloc[0]);cases.append(case);controls.extend(control.tolist())
    result = block_ci(values,times)
    if cases:
        result.update(case_mean=float(np.mean(cases)),control_mean=float(np.mean(controls)),
                      case_median=float(np.median(cases)),control_median=float(np.median(controls)))
        if score in ['sequence','absorption_rule']:
            result.update(case_exposed=int(sum(cases)),control_exposed=int(sum(controls)),controls=len(controls))
    return result


def main():
    selected = pd.read_parquet(OUT/'matched_samples.parquet')
    hourly = pd.read_parquet(BASE.parent/'microstructure/output/bybit.parquet')
    bysymbol = {s:g.set_index('time').sort_index() for s,g in hourly.groupby('symbol')}
    cache, rows = {},[]
    for r in selected.itertuples():
        key = (r.symbol,r.time)
        if key not in cache:
            days = pd.date_range((r.time-pd.Timedelta(hours=24)).floor('D'),(r.time-pd.Timedelta(nanoseconds=1)).floor('D'))
            files = [OUT/'bars'/f'{r.symbol}_{day:%Y-%m-%d}.parquet' for day in days]
            if not all(p.exists() for p in files):
                cache[key] = {'quality_error':'Missing daily archive'}
            else:
                b = pd.concat([pd.read_parquet(p) for p in files])
                h = bysymbol[r.symbol]
                historic = h[(h.index>r.time-pd.Timedelta(hours=192)) & (h.index<=r.time-pd.Timedelta(hours=24))]
                ref = ((historic.high-historic.low)/historic.close).mean() if len(historic)==168 else np.nan
                f = features(b,r.time,ref)
                current = h[(h.index>r.time-pd.Timedelta(hours=24)) & (h.index<=r.time)]
                f['hourly_quote24'] = float(current.turnover.sum())
                f['turnover_relative_error'] = abs(f['quote24']/f['hourly_quote24']-1) if f['hourly_quote24']>0 else np.nan
                if len(current)!=24 or not f['turnover_relative_error']<=.01:
                    f['quality_error'] = 'Archive/candle turnover mismatch or missing candle'
                cache[key] = f
        rows.append({**r._asdict(),**cache[key]})
    d = pd.DataFrame(rows)
    if 'quality_error' not in d:
        d['quality_error'] = None
    good_groups = d.groupby('group').quality_error.apply(lambda x:x.isna().all())
    d['group_valid'] = d.group.map(good_groups)
    d.to_parquet(OUT/'features.parquet',index=False)
    d.to_csv(OUT/'features.csv',index=False)
    good = d[d.group_valid]
    result = {'selected_groups':int(d.group.nunique()),'valid_groups':int(good.group.nunique()),
              'quality_errors':d.quality_error.value_counts().to_dict(),'unique_windows':len(cache),
              'matching_distance_max':float(selected.match_distance.max()),'splits':{}}
    summaries = []
    for split in ['train','validation','test']:
        part = good[good.split.eq(split)]
        cases = part[part.role.eq('case')]
        entry = {'groups':int(part.group.nunique()),'symbols':int(part.symbol.nunique()),
                 'median_lead_hours':float(cases.first_hit_hour.median()),'min_lead_hours':float(cases.first_hit_hour.min()),
                 'max_lead_hours':float(cases.first_hit_hour.max()),'scores':{}}
        for score in SCORES+['absorption_rule']:
            entry['scores'][score] = compare(part,score)
            summaries.append({'split':split,'score':score,**entry['scores'][score]})
        # Check how well covariates are balanced. Means of matched set case-control differences in training SD units.
        audit = json.loads((OUT/'selection_audit.json').read_text())
        balance = {}
        for col, scale in zip(['log_liquidity','log_vol168','ret24'],audit['match_scale']):
            p = part.copy();p['log_vol168'] = np.log(p.vol168)
            differences = [float(g[g.role.eq('case')][col].iloc[0]-g[g.role.eq('control')][col].mean())/scale for _,g in p.groupby('group')]
            balance[col] = float(np.mean(differences))
        entry['matching_standardized_difference'] = balance
        result['splits'][split] = entry
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
