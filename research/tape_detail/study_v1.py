"""Raw-tape hard-negative study, deliberately separate from production alerts."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parent
OUT=BASE/'output'
RAW=BASE.parent/'tradeflow/output/raw'
spec=importlib.util.spec_from_file_location('minute_helpers',BASE.parent/'minute_onset/study.py')
upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
FEATURES=['buy_persistence','buy_acceleration','sell_resilience','price_efficiency','arrival_acceleration','large_buy_change']


def source_hashes():
    paths=['study.py','protocol.json','../minute_onset/study.py','../onset_study/study.py','../flow_replay/study.py','../minute_onset/output/frozen.json']
    return {p:hashlib.sha256((BASE/p).read_bytes()).hexdigest() for p in paths}


def covariates(d):
    return np.column_stack([d.quote_accel.astype(float),np.log(d.match_vol168.astype(float).clip(lower=1e-10)),
                            np.log(d.match_liquidity.astype(float)),np.log1p(d.burst_age.astype(float))])


def candidates():
    panel=pd.read_parquet(BASE.parent/'minute_onset/output/panel.parquet')
    cutoff=json.loads((BASE.parent/'minute_onset/output/frozen.json').read_text())['rules'][0]['cutoff']
    d=panel[panel.eligible_price&panel.quality&panel.label_complete&(panel.quote_accel>=cutoff)].copy()
    d=d.sort_values(['symbol','time'])
    ages=[]
    for _,g in d.groupby('symbol',sort=False):
        new=g.time.diff().gt(pd.Timedelta(minutes=10))|g.time.diff().isna()
        start=g.time.where(new).ffill()
        ages.extend(((g.time-start).dt.total_seconds()/60).tolist())
    d['burst_age']=ages
    return d


def select():
    d=candidates();early=upstream.split(d,'discovery')
    scale=np.std(covariates(early),axis=0)
    allrows=[];audit={}
    for split in ['discovery','replication']:
        p=upstream.split(d,split)
        cases=p[p.strict_hit].sort_values('time').drop_duplicates(['symbol','next_event'])
        chosen_count=0
        for case in cases.itertuples():
            gap=(p.time-case.time).abs()
            minute=p.time.dt.hour*60+p.time.dt.minute;cminute=case.time.hour*60+case.time.minute
            circular=(minute-cminute).abs();circular=np.minimum(circular,1440-circular)
            controls=p[p.symbol.eq(case.symbol)&~p.hit6&(p.max60<.08)&gap.ge(pd.Timedelta(hours=24))&gap.le(pd.Timedelta(days=14))&(circular<=240)].copy()
            controls['distance']=np.linalg.norm((covariates(controls)-covariates(pd.DataFrame([case._asdict()])))/scale,axis=1)
            controls=controls[controls.distance<=3].sort_values(['distance','time'])
            controls['day']=controls.time.dt.floor('D')
            controls=controls.drop_duplicates('day').head(3)
            if len(controls)!=3:continue
            group=f'{split}:{case.symbol}:{case.next_event.isoformat()}'
            for role,item in [('case',case)]+[('control',x) for x in controls.itertuples()]:
                row=item._asdict();row.update(group=group,role=role,split=split,anchor=case.time,offset=0,match_distance=0 if role=='case' else item.distance)
                allrows.append(row)
            chosen_count+=1
        audit[split]={'positive_events_with_volume_trigger':len(cases),'matched_sets':chosen_count,'unmatched':len(cases)-chosen_count,'candidate_observations':len(p)}
    selected=pd.DataFrame(allrows)
    selected.to_parquet(OUT/'selected.parquet',index=False)
    selected[['group','role','split','symbol','time','anchor','next_event','quote_accel','burst_age','match_distance']].to_csv(OUT/'selected.csv',index=False)
    result={'source_hashes':source_hashes(),'selected_at':pd.Timestamp.now(tz='UTC').isoformat(),'scale':scale.tolist(),'sampling':audit}
    (OUT/'selection.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


def net(x):
    total=x.quote.sum()
    return (x.buy.sum()-x.sell.sum())/total if total>0 else np.nan


def impact(x,minimum):
    imbalance=(x.buy-x.sell)/x.quote.replace(0,np.nan)
    z=x[imbalance<=-.2]
    if len(z)<minimum:return np.nan
    return (-z.ret).clip(lower=0).sum()/(-imbalance.loc[z.index]).sum()


def gap_median(x):
    unique=np.unique(x.timestamp.to_numpy())
    delta=np.diff(unique)
    return float(np.median(delta)) if len(delta)>=10 else np.nan


def tape_features(raw,anchor,expected_quote):
    start=anchor-pd.Timedelta(minutes=60)
    ms=anchor.value/1e6;start_ms=start.value/1e6
    x=raw[(raw.timestamp>=start_ms)&(raw.timestamp<ms)].sort_values('timestamp',kind='stable').copy()
    if x.empty:return {'quality_error':'Empty raw window'}
    if not x.side.str.lower().isin(['buy','sell']).all():return {'quality_error':'Unknown trade side'}
    x['quote']=x.price*x.volume;x['buy']=x.quote.where(x.side.str.lower().eq('buy'),0);x['sell']=x.quote-x.buy
    x['minute']=pd.to_datetime(x.timestamp,unit='ms').dt.floor('min')
    b=x.groupby('minute').agg(close=('price','last'),first=('price','first'),quote=('quote','sum'),buy=('buy','sum'),sell=('sell','sum'),trades=('price','size'))
    b=b.reindex(pd.date_range(start,anchor,freq='min',inclusive='left'))
    active=b.trades.notna()
    result={'raw_trades':len(x),'quote60':float(x.quote.sum()),'quote_error':float(abs(x.quote.sum()/expected_quote-1)) if expected_quote>0 else np.nan,
            'last_age_seconds':float((ms-x.timestamp.max())/1000),'baseline_active':float(active.iloc[:45].mean()),'recent_active':float(active.iloc[45:].mean())}
    if result['quote_error']>.01 or not np.isfinite(result['quote_error']) or result['last_age_seconds']>60 or result['baseline_active']<.8 or result['recent_active']<.8 or not active.iloc[0]:
        result['quality_error']='Sparse/stale raw tape or turnover discrepancy';return result
    b['close']=b.close.ffill()
    prior=b.close.shift(1);prior.iloc[0]=b['first'].iloc[0]
    b['ret']=b.close/prior-1
    b[['quote','buy','sell','trades']]=b[['quote','buy','sell','trades']].fillna(0)
    base,recent=b.iloc[:45],b.iloc[45:]
    result['buy_persistence']=float(((recent.buy>recent.sell)&recent.trades.gt(0)).sum()/recent.trades.gt(0).sum()-((base.buy>base.sell)&base.trades.gt(0)).sum()/base.trades.gt(0).sum())
    result['buy_acceleration']=net(b.iloc[-5:])-net(b.iloc[-15:-5])
    result['sell_resilience']=impact(base,6)-impact(recent,3)
    path=recent.ret.abs().sum()
    result['price_efficiency']=(b.close.iloc[-1]/b.close.iloc[44]-1)/path if path>0 else np.nan
    baseline_raw=x[x.timestamp<ms-15*60000];last5=x[x.timestamp>=ms-5*60000];recent_raw=x[x.timestamp>=ms-15*60000]
    bg,rg=gap_median(baseline_raw),gap_median(last5)
    result['arrival_acceleration']=np.log(bg/rg) if bg>0 and rg>0 else np.nan
    cutoff=baseline_raw.quote.quantile(.95)
    result['large_buy_change']=recent_raw.buy.where(recent_raw.quote>cutoff,0).sum()/recent_raw.quote.sum()-baseline_raw.buy.where(baseline_raw.quote>cutoff,0).sum()/baseline_raw.quote.sum()
    result['large_print_cutoff']=float(cutoff)
    result['baseline_gap_ms']=bg;result['recent_gap_ms']=rg
    return result


def extract():
    selected=pd.read_parquet(OUT/'selected.parquet');results=[];cache={};bars_cache={};sources={}
    for row in selected.itertuples():
        pieces=[];expected=0.;missing=False
        start=row.time-pd.Timedelta(minutes=60)
        for day in pd.date_range(start.floor('D'),(row.time-pd.Timedelta(nanoseconds=1)).floor('D')):
            stem=f'{row.symbol}_{day:%Y-%m-%d}'
            path=RAW/f'{stem}.csv.gz'
            if not path.exists():missing=True;break
            if stem not in cache:
                cache[stem]=pd.read_csv(path,compression='gzip',usecols=['timestamp','price','volume','side'])
                bars_cache[stem]=pd.read_parquet(RAW.parent/'bars'/f'{stem}.parquet')
                sources[stem]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'trades':len(cache[stem])}
            pieces.append(cache[stem]);b=bars_cache[stem]
            expected+=b.loc[(b.bucket>=start)&(b.bucket<row.time),'quote'].sum()
        v={'quality_error':'Missing raw file'} if missing else tape_features(pd.concat(pieces,ignore_index=True),row.time,expected)
        results.append({**row._asdict(),**v})
    out=pd.DataFrame(results)
    if 'quality_error' not in out:out['quality_error']=None
    valid=out.groupby('group').quality_error.apply(lambda s:s.isna().all())
    out['group_valid']=out.group.map(valid)
    out.to_parquet(OUT/'features.parquet',index=False);out.to_csv(OUT/'features.csv',index=False)
    (OUT/'source_audit.json').write_text(json.dumps(sources,indent=2))
    print('Raw features extracted; no later-period comparisons printed',flush=True)


def comparison(d,key,direction=1,alpha=.05):
    groups=d.groupby('group')[key].apply(lambda s:len(s)==4 and np.isfinite(s).all())
    return upstream.common.compare(d[d.group.isin(groups[groups].index)],key,direction,alpha)


def discover():
    if (OUT/'frozen.json').exists():raise RuntimeError('Already frozen')
    d=pd.read_parquet(OUT/'features.parquet');d=d[d.group_valid&d.split.eq('discovery')]
    stats={};candidates=[]
    for key in FEATURES:
        initial=comparison(d,key);direction=1 if initial.get('concordance',.5)>=.5 else -1
        s=comparison(d,key,direction);s['direction']=direction;stats[key]=s
        if s['n']>=10 and s['concordance']>=.60 and s['lower']>.50:candidates.append((s['concordance'],key,direction))
    result={'frozen_at':pd.Timestamp.now(tz='UTC').isoformat(),'source_hashes':source_hashes(),'discovery':stats,
            'quality_matched_sets':d.group.nunique(),'candidates':[{'feature':k,'direction':di} for _,k,di in sorted(candidates,reverse=True)[:2]]}
    (OUT/'frozen.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2),flush=True)


def replicate():
    frozen=json.loads((OUT/'frozen.json').read_text());assert frozen['source_hashes']==source_hashes()
    allrows=pd.read_parquet(OUT/'features.parquet');d=allrows[allrows.group_valid&allrows.split.eq('replication')]
    result={'replication':{k:comparison(d,k,frozen['discovery'][k]['direction'],alpha=.05/6) for k in FEATURES},
            'quality_matched_sets':d.group.nunique(),'quality_errors':allrows.quality_error.value_counts().to_dict(),
            'warning':'Matched conditional comparison only, no new alert precision estimate'}
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['select','extract','discover','replicate']);a=p.parse_args()
    {'select':select,'extract':extract,'discover':discover,'replicate':replicate}[a.mode]()
