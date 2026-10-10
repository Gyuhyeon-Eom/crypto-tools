"""Start with objective surges, not previously selected alert candidates."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE/'output'
SOURCE = BASE.parent/'microstructure/output/bybit.parquet'
FEATURES = ['volume_build','compression','near_high','prior_runup','relative_strength',
            'recent_spike','market_up','market_breadth']
THRESHOLDS = dict(zip(FEATURES,[2.,.7,.98,.05,.05,1.,.01,.6]))


def freeze():
    hashes = {name: hashlib.sha256(p.read_bytes()).hexdigest() for name,p in
              [('study.py',BASE/'study.py'),('protocol.json',BASE/'protocol.json'),('hourly_source',SOURCE)]}
    path = OUT/'frozen.json'
    if path.exists():
        assert json.loads(path.read_text())['hashes'] == hashes
    else:
        path.write_text(json.dumps({'at':pd.Timestamp.now(tz='UTC').isoformat(),'hashes':hashes},indent=2))


def price_features(g):
    g = g.set_index('time').sort_index()
    g = g.reindex(pd.date_range(g.index.min(),g.index.max(),freq='h'))
    f = g.copy()
    f['ret1'] = f.close/f.close.shift(1)-1
    f['prior_runup'] = f.close/f.close.shift(24)-1
    f['ret6'] = f.close/f.close.shift(6)-1
    f['volume_build'] = f.turnover.rolling(6).mean()/f.turnover.rolling(168).mean().shift(6)
    normalized_range = (f.high-f.low)/f.close.shift(1)
    f['compression'] = normalized_range.rolling(6).mean()/normalized_range.rolling(168).mean().shift(6)
    f['near_high'] = f.close/f.high.rolling(168).max().shift(1)
    f['recent_spike'] = f.ret1.ge(.03).astype(float).rolling(72).sum()
    f['match_vol'] = f.ret1.rolling(168).std().shift(24)
    f['match_liquidity'] = (f.turnover.rolling(336).mean()*24).shift(24)
    f['past_complete'] = f.close.notna().rolling(720).min().eq(1)
    f['time'] = f.index
    return f


def labels(f):
    f = f.copy()
    hi = np.column_stack([f.high.shift(-i) for i in range(1,25)])
    lo = np.column_stack([f.low.shift(-i) for i in range(1,25)])
    cl = np.column_stack([f.close.shift(-i) for i in range(1,25)])
    op = np.column_stack([f.open.shift(-i) for i in range(1,25)])
    f['future_complete'] = np.isfinite(hi).all(axis=1)&np.isfinite(lo).all(axis=1)&np.isfinite(cl).all(axis=1)&np.isfinite(op).all(axis=1)
    f['max24'] = np.max(hi,axis=1)/f.close-1
    f['return24'] = f.close.shift(-24)/f.close-1
    f['jump1'] = f.close.shift(-1)/f.close-1
    f['during_volume'] = f.turnover.shift(-1)/f.turnover.rolling(168).mean()
    f['raw_event'] = f.past_complete&f.future_complete&f.jump1.ge(.05)&f.max24.ge(.15)
    keep=[];last=None
    for t in f.index[f.raw_event]:
        if last is None or t-last>=pd.Timedelta(hours=24):
            keep.append(t);last=t
    f['event'] = f.index.isin(keep)
    f['event_nearby'] = f.raw_event.astype(int).rolling(49,center=True,min_periods=1).max().eq(1)
    f['context'] = np.select([f.prior_runup.lt(-.05),f.prior_runup.ge(.05)],['rebound','continuation'],default='flat')
    return f


def part(d,name):
    if name=='discovery':
        return d[d.time.between('2026-04-02','2026-06-29 23:00')]
    return d[d.time.between('2026-08-01','2026-09-30 23:00')]


def covariates(d):
    return np.column_stack([np.log(d.match_vol.clip(lower=1e-10)),np.log(d.match_liquidity.clip(lower=1e-10))])


def match(d,split,scale):
    p = part(d,split)
    cases=p[p.event & p.symbol.ne('BTCUSDT')]
    rows=[];drops=[]
    for case in cases.itertuples(index=False):
        pool=p[p.symbol.eq(case.symbol)&p.past_complete&p.future_complete&~p.event_nearby&p.max24.lt(.15)&p.time.dt.hour.eq(case.time.hour)].copy()
        gap=(pool.time-case.time).abs()
        pool=pool[gap.between(pd.Timedelta(hours=48),pd.Timedelta(days=28))]
        pool['distance']=np.linalg.norm((covariates(pool)-covariates(pd.DataFrame([case._asdict()])))/scale,axis=1)
        pool=pool[pool.distance.le(3)].sort_values(['distance','time']).head(3)
        if len(pool)!=3:
            drops.append({'symbol':case.symbol,'time':str(case.time),'context':case.context,'reason':'fewer than3 matched controls'})
            continue
        group=f'{case.symbol}:{case.time.isoformat()}'
        for role,row in [('case',case._asdict())]+[('control',x._asdict()) for x in pool.itertuples(index=False)]:
            rows.append({**row,'group':group,'anchor':case.time,'role':role,'split':split})
    return pd.DataFrame(rows), drops


def flag(x,key):
    return x.le(THRESHOLDS[key]) if key=='compression' else x.ge(THRESHOLDS[key])


def comparisons(m):
    result={}
    for key in FEATURES:
        good=m.groupby('group')[key].apply(lambda s:len(s)==4 and np.isfinite(s).all())
        z=m[m.group.isin(good[good].index)].copy();z['flag']=flag(z[key],key)
        pairs=[]
        for _,g in z.groupby('group'):
            pairs.append({'time':g.anchor.iloc[0], 'difference':float(g[g.role.eq('case')].flag.iloc[0])-float(g[g.role.eq('control')].flag.mean())})
        if not pairs:
            result[key]={'n':0};continue
        p=pd.DataFrame(pairs);p['block']=(p.time-pd.Timestamp('2026-01-01')).dt.days//7
        b=p.groupby('block').difference.agg(['sum','count']).to_numpy()
        rng=np.random.default_rng(20261009)
        boot=b[rng.integers(0,len(b),(10000,len(b)))].sum(axis=1)
        boot=boot[:,0]/boot[:,1]
        ca=z[z.role.eq('case')];co=z[z.role.eq('control')]
        result[key]={'n':len(p),'blocks':len(b),'event_prevalence':float(ca.flag.mean()),
                     'control_prevalence':float(co.flag.mean()),'difference':float(p.difference.mean()),
                     'lower':float(np.quantile(boot,.05/8/2)),'upper':float(np.quantile(boot,1-.05/8/2)),
                     'event_median':float(ca[key].median()),'control_median':float(co[key].median())}
    return result


def describe(cases):
    return {'events':len(cases),'coins':cases.symbol.nunique(),'days':cases.time.dt.floor('D').nunique(),
            'contexts':cases.context.value_counts().to_dict(),
            'pre_feature_prevalence':{k:float(flag(cases[k].dropna(),k).mean()) for k in FEATURES},
            'pre_feature_available':{k:int(cases[k].notna().sum()) for k in FEATURES},
            'quantiles':{k:{str(q):float(v) for q,v in cases[k].quantile([.25,.5,.75]).items()}
                         for k in ['prior_runup','volume_build','compression','near_high','relative_strength','during_volume','jump1','max24','return24']},
            'during_volume_ge2':int(cases.during_volume.ge(2).sum()),
            'during_volume_ge5':int(cases.during_volume.ge(5).sum()),
            'close24_below_anchor':int(cases.return24.lt(0).sum()),
            'close24_below_plus5':int(cases.return24.lt(.05).sum())}


def main():
    OUT.mkdir(exist_ok=True);freeze()
    raw=pd.read_parquet(SOURCE)
    assert not raw.duplicated(['symbol','time']).any()
    assert raw[['open','high','low','close']].gt(0).all().all()
    assert raw.high.ge(raw[['open','close','low']].max(axis=1)).all()
    assert raw.low.le(raw[['open','close','high']].min(axis=1)).all()
    assert (pd.to_datetime(raw.open_ms,unit='ms')+pd.Timedelta(hours=1)).eq(raw.time).all()
    d=pd.concat([price_features(g).reset_index(drop=True) for _,g in raw.groupby('symbol')],ignore_index=True)
    btc=d[d.symbol.eq('BTCUSDT')].set_index('time').prior_runup
    other=d[d.symbol.ne('BTCUSDT')]
    market=other.groupby('time').ret6.median()
    breadth=other[other.ret6.notna()].groupby('time').ret6.apply(lambda s:s.gt(0).mean())
    d['relative_strength']=d.prior_runup-d.time.map(btc)
    d['market_up']=d.time.map(market)
    d['market_breadth']=d.time.map(breadth)
    d=pd.concat([labels(g.set_index('time',drop=False)).reset_index(drop=True) for _,g in d.groupby('symbol')],ignore_index=True)
    d[FEATURES]=d[FEATURES].replace([np.inf,-np.inf],np.nan)
    early=part(d,'discovery');early=early[early.past_complete&early.symbol.ne('BTCUSDT')]
    scale=np.nanstd(covariates(early),axis=0)
    results={};matched=[];events=[];paths=[]
    lookup={symbol:g.set_index('time') for symbol,g in d.groupby('symbol')}
    for split in ['discovery','replication']:
        cases=part(d,split);cases=cases[cases.event&cases.symbol.ne('BTCUSDT')].copy();cases['split']=split
        m,drops=match(d,split,scale)
        matched.append(m);events.append(cases)
        co=m[m.role.eq('control')]
        results[split]={'all_events':describe(cases),'matched_events':describe(m[m.role.eq('case')]),
                        'dropped_events':drops,'control_rows':len(co),'unique_controls':len(co.drop_duplicates(['symbol','time'])),
                        'comparisons':comparisons(m)}
        for row in m.itertuples(index=False):
            g=lookup[row.symbol];base=g.loc[row.time,'close'];volbase=g.loc[row.time-pd.Timedelta(hours=168):row.time-pd.Timedelta(hours=1),'turnover'].mean()
            for offset in range(-24,25):
                t=row.time+pd.Timedelta(hours=offset)
                if t not in g.index:continue
                v=g.loc[t]
                paths.append({'split':split,'group':row.group,'role':row.role,'symbol':row.symbol,'time':row.time,
                              'offset':offset,'price_return':v.close/base-1,'quote_ratio':v.turnover/volbase,
                              **{k:v[k] for k in FEATURES}})
    pd.concat(events).to_csv(OUT/'events.csv',index=False)
    pd.concat(matched).to_csv(OUT/'matched.csv',index=False)
    pd.DataFrame(paths).to_csv(OUT/'trajectories.csv',index=False)
    (OUT/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()
