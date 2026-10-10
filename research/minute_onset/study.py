"""5-minute pre-onset study. Discovery/freeze and replication are separate commands."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parent
OUT=BASE/'output'
RAW=BASE.parent/'tradeflow/output/bars'
spec=importlib.util.spec_from_file_location('hour_onset_helpers',BASE.parent/'onset_study/study.py')
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
FEATURES=common.FEATURES
OFFSETS=[-60,-30,-15,-5,0]


def safe_log(x):
    return np.log(x.where(x>0))


def build_features(bars,hourly,days):
    index=pd.date_range(min(days),max(days)+pd.Timedelta(days=1),freq='5min',inclusive='left')
    b=bars.set_index('bucket').sort_index().reindex(index)
    active=b.quote.notna()
    known=pd.Series(index.floor('D').isin(days),index=index)
    close=b.close.ffill().where(known)
    f=pd.DataFrame(index=index+pd.Timedelta(minutes=5))
    f['close']=close.to_numpy()
    for col in ['open','high','low']:
        f[col]=b[col].fillna(close).where(known).to_numpy()
    x=b[['quote','buy','sell','trades']].fillna(0).set_axis(f.index)
    recent=x.rolling(3).sum();baseline=x.rolling(60).sum().shift(12)
    f['quote_accel']=safe_log(recent.quote/(baseline.quote/20))
    f['count_accel']=safe_log(recent.trades/(baseline.trades/20))
    f['size_accel']=safe_log((recent.quote/recent.trades)/(baseline.quote/baseline.trades))
    f['net_change']=(recent.buy-recent.sell)/recent.quote-(baseline.buy-baseline.sell)/baseline.quote
    r15=f.close/f.close.shift(3)-1
    f['price_accel']=r15-(f.close.shift(3)/f.close.shift(12)-1)/3
    range15=(f.high.rolling(3).max()-f.low.rolling(3).min())/f.close
    f['range_accel']=safe_log(range15/range15.rolling(60).mean().shift(12))
    r5=f.close/f.close.shift(1)-1
    f['quiet']=(r5.rolling(6).max()<.01)&(f.close/f.close.shift(6)-1<.02)
    f['known']=known.to_numpy();f['active']=active.to_numpy()
    f['last_age']=(f.index.as_unit('ms').astype('int64')-b.last_ms.ffill().to_numpy())/60000
    h=hourly.set_index('time').sort_index()
    h=h.reindex(pd.date_range(h.index.min(),h.index.max(),freq='h'))
    market=pd.DataFrame(index=h.index)
    market['liquidity']=h.turnover.rolling(720).mean()*24
    market['vol168']=h.close.pct_change(fill_method=None).rolling(168).std()
    market['ret24']=h.close/h.close.shift(24)-1
    market['continuous']=h.close.notna().rolling(720).min().eq(1)
    # Last closed hour only: do not use the partially formed hour's final turnover.
    quote_hour=x.quote.resample('h',closed='right',label='right').sum(min_count=1)
    compare=((quote_hour.reindex(h.index)/h.turnover-1).abs()<=.01)
    market['hourly_verified']=compare.rolling(6).min().eq(1)
    current=market.reindex(f.index.floor('h')).set_axis(f.index)
    f['eligible_price']=current.continuous.fillna(False).astype(bool)&current.liquidity.between(200000,20000000)&f.quiet
    for col in ['liquidity','vol168','ret24']:
        f['match_'+col]=market[col].reindex((f.index-pd.Timedelta(hours=1)).floor('h')).to_numpy()
    f[FEATURES]=f[FEATURES].replace([np.inf,-np.inf],np.nan)
    f['quality']=f.known.rolling(72).min().eq(1)&f.active.rolling(72).mean().ge(.8)&f.active&(f.last_age<=5)&current.hourly_verified.fillna(False).astype(bool)&f[FEATURES].notna().all(axis=1)
    f['time']=f.index
    return f


def label(f):
    f=f.copy()
    hi=np.column_stack([f.high.shift(-i) for i in range(1,13)])
    lo=np.column_stack([f.low.shift(-i) for i in range(1,13)])
    future_known=np.column_stack([f.known.shift(-i,fill_value=False) for i in range(1,13)]).all(axis=1)
    complete=future_known&np.isfinite(hi).all(axis=1)&np.isfinite(lo).all(axis=1)
    f['max60']=np.max(hi,axis=1)/f.close-1
    f['min60']=np.minimum(0,np.min(lo,axis=1)/f.close-1)
    f['return60']=f.close.shift(-12)/f.close-1
    f['future_complete']=complete
    candidates=f.eligible_price&complete&(f.close.shift(-1)/f.close-1>=.02)&(f.max60>=.08)&f.active.shift(-1,fill_value=False)
    onsets=[];last=None
    for t in f.index[candidates]:
        if last is None or t-last>=pd.Timedelta(hours=24):
            onsets.append(t);last=t
    f['event']=f.index.isin(onsets)
    target=pd.Series(pd.NaT,index=f.index,dtype='datetime64[ns]')
    for t in onsets:target.loc[t]=t
    target=target.bfill()
    lead=(target-pd.Series(f.index,index=f.index)).dt.total_seconds()/60
    f['hit6']=lead.between(0,60)  # shared replay helper consumes this boolean name
    f['strict_hit']=lead.between(5,60)
    f['lead_minutes']=lead.where(f.hit6)
    f['next_event']=target.where(f.hit6)
    f['label_complete']=f.known.rolling(25).min().shift(-24).eq(1)&f.close.notna().rolling(25).min().shift(-24).eq(1)
    # Shift the event-detection window by5min; this is timing sensitivity, not PnL.
    f['delay5_hit']=lead.between(5,65)
    return f


def split(d,name):
    if name=='discovery':
        return d[d.time.between('2026-04-02','2026-06-30 23:55')&(d.time+pd.Timedelta(hours=2)<pd.Timestamp('2026-07-01'))]
    return d[d.time.between('2026-08-01','2026-09-30 23:55')]


def source_hashes():
    return {p:hashlib.sha256((BASE/p).read_bytes()).hexdigest() for p in ['study.py','protocol.json','../onset_study/study.py','../flow_replay/study.py']}


def prepare():
    universe=json.loads((BASE.parent/'flow_replay/output/universe.json').read_text())['symbols']
    h=pd.read_parquet(BASE.parent/'microstructure/output/bybit.parquet');frames=[]
    for symbol in universe:
        files=[p for p in sorted(RAW.glob(symbol+'_*.parquet')) if '2026-03-31'<=p.stem[len(symbol)+1:]<='2026-10-01']
        days=[pd.Timestamp(p.stem[len(symbol)+1:]) for p in files]
        f=build_features(pd.concat([pd.read_parquet(p) for p in files]),h[h.symbol.eq(symbol)],days)
        f=label(f);f['symbol']=symbol;frames.append(f.reset_index(drop=True))
    panel=pd.concat(frames,ignore_index=True)
    panel.to_parquet(OUT/'panel.parquet',index=False)
    (OUT/'data_audit.json').write_text(json.dumps({'symbols':universe,'rows':len(panel),'prepared_at':pd.Timestamp.now(tz='UTC').isoformat(),
        'source_hashes':source_hashes(),'upstream_download_audit_sha256':hashlib.sha256((BASE.parent/'flow_replay/output/download_audit.json').read_bytes()).hexdigest()},indent=2))
    print('Panel prepared; no replication summaries inspected',flush=True)


def matched(d,name,scale):
    p=split(d,name);cases=p[p.event].sort_values(['time','symbol'])
    lookup=d.set_index(['symbol','time']);bytime=p.set_index('time',drop=False)
    rows=[];drops={'bad_case':0,'missing_controls':0}
    def trajectory(symbol,t):
        keys=[(symbol,t+pd.Timedelta(minutes=m)) for m in OFFSETS]
        if not all(k in lookup.index for k in keys):return None
        z=lookup.loc[keys]
        return z if z.quality.all() else None
    for case in cases.itertuples():
        if trajectory(case.symbol,case.time) is None:
            drops['bad_case']+=1;continue
        g=bytime.loc[[case.time]]
        controls=g[g.eligible_price&g.quality&g.future_complete&(g.max60<.08)&g.symbol.ne(case.symbol)].copy()
        controls['distance']=np.linalg.norm((common.covariates(controls)-common.covariates(pd.DataFrame([case._asdict()])))/scale,axis=1)
        selected=[]
        for c in controls[controls.distance<=3].sort_values(['distance','symbol']).itertuples():
            if trajectory(c.symbol,case.time) is not None:selected.append(c)
            if len(selected)==3:break
        if len(selected)!=3:
            drops['missing_controls']+=1;continue
        for role,item in [('case',case)]+[('control',c) for c in selected]:
            z=trajectory(item.symbol,case.time)
            for offset,(_,row) in zip(OFFSETS,z.iterrows()):
                rows.append({'group':f'{case.symbol}:{case.time.isoformat()}','anchor':case.time,'symbol':item.symbol,'role':role,'offset':offset,
                    'match_distance':0 if role=='case' else item.distance,**{k:float(row[k]) for k in FEATURES}})
    m=pd.DataFrame(rows,columns=['group','anchor','symbol','role','offset','match_distance',*FEATURES])
    return m,{'price_events':len(cases),'matched_events':m.group.nunique(),'drops':drops}


def discover():
    if (OUT/'frozen.json').exists():raise RuntimeError('Already frozen; no silent refit')
    d=pd.read_parquet(OUT/'panel.parquet');p=split(d,'discovery')
    p=p[p.eligible_price&p.quality&p.label_complete]
    scale=np.std(common.covariates(p),axis=0)
    m,audit=matched(d,'discovery',scale);m.to_csv(OUT/'discovery_trajectories.csv',index=False)
    stats={};candidates=[]
    for key in FEATURES:
        initial=common.compare(m,key);direction=1 if initial.get('concordance',.5)>=.5 else -1
        s=common.compare(m,key,direction);s['direction']=direction;stats[key]=s
        if s['n']>=10 and s['concordance']>=.60 and s['lower']>.5:candidates.append((s['concordance'],key,direction))
    rules=[];activity=False
    for _,key,direction in sorted(candidates,reverse=True):
        if key in ['quote_accel','count_accel'] and activity:continue
        rules.append({'feature':key,'direction':direction,'cutoff':float((p[key]*direction).quantile(.90))})
        activity|=key in ['quote_accel','count_accel']
        if len(rules)==2:break
    result={'frozen_at':pd.Timestamp.now(tz='UTC').isoformat(),'source_hashes':source_hashes(),'scale':scale.tolist(),
        'matching':audit,'discovery':stats,'rules':rules,'eligible_hours_equivalent':len(p)/12,'eligible_rows':len(p)}
    (OUT/'frozen.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2),flush=True)


def metrics(d,a):
    def keys(z):return set(zip(z.symbol,z.next_event))
    hits=a[a.hit6];events=keys(d[d.hit6])
    return {'alerts':len(a),'hits':int(a.hit6.sum()),'precision':float(a.hit6.mean()) if len(a) else None,
        'strict5min_hits':int(a.strict_hit.sum()),'delayed5min_hits':int(a.delay5_hit.sum()),
        'eligible_events':len(events),'detected_events':len(keys(hits)),
        'recall':len(keys(hits))/len(events) if events else None,'median_lead_minutes':float(hits.lead_minutes.median()) if len(hits) else None,
        'median_min60':float(a.min60.median()) if len(a) else None,'down5':int((a.min60<=-.05).sum()),
        'median_return60':float(a.return60.median()) if len(a) else None}


def replicate():
    frozen=json.loads((OUT/'frozen.json').read_text());assert frozen['source_hashes']==source_hashes()
    d=pd.read_parquet(OUT/'panel.parquet');m,audit=matched(d,'replication',np.array(frozen['scale']))
    m.to_csv(OUT/'replication_trajectories.csv',index=False)
    stats={k:common.compare(m,k,frozen['discovery'][k]['direction'],alpha=.05/6) for k in FEATURES}
    p=split(d,'replication');price=int(p.eligible_price.sum());p=p[p.eligible_price&p.quality&p.label_complete]
    result={'matching':audit,'replication':stats,'eligible_rows':len(p),'price_eligible_rows':price,'base_rate':float(p.hit6.mean()),'replay':{}}
    masks=[];scores=[]
    for rule in frozen['rules']:
        key=rule['feature'];score=p[key]*rule['direction'];mask=score>=rule['cutoff']
        a=common.replay(p,mask,score);a.to_csv(OUT/f'alerts_{key}.csv',index=False)
        result['replay'][key]={'metrics':metrics(p,a),'random':common.random_baseline(p,a)}
        masks.append(mask);scores.append(score-rule['cutoff'])
    if len(masks)==2:
        a=common.replay(p,masks[0]&masks[1],scores[0]+scores[1]);a.to_csv(OUT/'alerts_combined.csv',index=False)
        result['replay']['combined']={'metrics':metrics(p,a),'random':common.random_baseline(p,a)}
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','discover','replicate']);a=p.parse_args()
    {'prepare':prepare,'discover':discover,'replicate':replicate}[a.mode]()
