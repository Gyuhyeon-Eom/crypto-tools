"""Event-aligned discovery followed by frozen, later-period hypothesis replay."""
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
spec=importlib.util.spec_from_file_location('previous_replay',BASE.parent/'flow_replay/study.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
FEATURES=['quote_accel','count_accel','size_accel','net_change','range_accel','price_accel']
OFFSETS=[-24,-12,-6,-3,-2,-1,0]


def features(h,bars,days):
    f=previous.price_features(h)
    f['ret1']=f.close/f.close.shift(1)-1
    f['ret6']=f.close/f.close.shift(6)-1
    f['vol168']=f.ret1.rolling(168).std()
    f['quiet']=(f.ret1.rolling(6).max()<.03)&(f.ret6<.05)
    f['eligible_price']=f.close.notna().rolling(720).min().eq(1)&f.liquidity.between(200000,20000000)&f.quiet
    # Aggregate only completed [start,end) buckets, with hour END labels.
    b=bars.set_index('bucket').sort_index()
    x=b[['quote','buy','sell','trades']].resample('h').sum(min_count=1)
    x.index=x.index+pd.Timedelta(hours=1)
    x=x.reindex(f.index)
    base=x.rolling(18).sum().shift(6)
    f['quote_accel']=np.log(x.quote/(base.quote/18))
    f['count_accel']=np.log(x.trades/(base.trades/18))
    f['size_accel']=np.log((x.quote/x.trades)/(base.quote/base.trades))
    f['net_change']=(x.buy-x.sell)/x.quote-(base.buy-base.sell)/base.quote
    normalized_range=(f.high-f.low)/f.close
    f['range_accel']=np.log(normalized_range/normalized_range.rolling(18).mean().shift(6))
    f['price_accel']=f.ret1-f.ret1.rolling(3).mean().shift(1)
    q=previous.flow_features(bars,days,min(days),max(days)+pd.Timedelta(days=1))
    f=f.join(q[['raw_quote24','active_fraction','files_complete','first_active','last_trade_age_minutes']])
    f['quality']=(f.active_fraction>=.8)&f.files_complete.fillna(False).astype(bool)&f.first_active.fillna(False).astype(bool)&(f.last_trade_age_minutes<=5)&((f.raw_quote24/f.turnover24-1).abs()<=.01)
    f[FEATURES]=f[FEATURES].replace([np.inf,-np.inf],np.nan)
    f['quality']&=f[FEATURES].notna().all(axis=1)
    for col in ['liquidity','vol168','ret24']:
        f['match_'+col]=f[col].shift(6)
    f['time']=f.index
    return f


def label_events(f):
    f=f.copy()
    hi=np.column_stack([f.high.shift(-h).to_numpy() for h in range(1,25)])
    lo=np.column_stack([f.low.shift(-h).to_numpy() for h in range(1,25)])
    f['future24_complete']=np.isfinite(hi).all(axis=1)&np.isfinite(lo).all(axis=1)
    f['max24']=np.max(hi,axis=1)/f.close-1
    f['min24']=np.minimum(0,np.min(lo,axis=1)/f.close-1)
    f['return24']=f.close.shift(-24)/f.close-1
    raw=f.eligible_price&f.future24_complete&(f.close.shift(-1)/f.close-1>=.05)&(f.max24>=.15)
    chosen=[];last=None
    for t in f.index[raw]:
        if last is None or t-last>=pd.Timedelta(hours=24):
            chosen.append(t);last=t
    f['event']=f.index.isin(chosen)
    f['next_event']=pd.NaT
    # A positive target requires a price-defined event, irrespective of future trade-file quality.
    target=pd.Series(pd.NaT,index=f.index,dtype='datetime64[ns]')
    for t in chosen:
        target.loc[t]=t
    target=target.bfill()
    lead=(target-pd.Series(f.index,index=f.index)).dt.total_seconds()/3600
    f['hit6']=lead.between(0,6)
    f['hit6_strict']=lead.between(1,6)
    f['next_event']=target.where(f.hit6)
    f['lead_hours']=lead.where(f.hit6)
    # All labels need max30h future coverage to avoid censoring onset+24h validation.
    f['label_complete']=f.close.notna().rolling(31).min().shift(-30).eq(1)
    return f


def prepare():
    u=json.loads((BASE.parent/'flow_replay/output/universe.json').read_text())
    hourly=pd.read_parquet(BASE.parent/'microstructure/output/bybit.parquet')
    parts=[]
    for symbol in u['symbols']:
        paths=sorted(RAW.glob(symbol+'_*.parquet'))
        paths=[p for p in paths if '2026-03-31'<=p.stem[len(symbol)+1:]<='2026-10-01']
        days=[pd.Timestamp(p.stem[len(symbol)+1:]) for p in paths]
        b=pd.concat([pd.read_parquet(p) for p in paths])
        f=features(hourly[hourly.symbol.eq(symbol)],b,days)
        f=label_events(f);f['symbol']=symbol
        parts.append(f.reset_index(drop=True))
    panel=pd.concat(parts,ignore_index=True)
    panel.to_parquet(OUT/'panel.parquet',index=False)
    (OUT/'preparation.json').write_text(json.dumps({'protocol_sha256':hashlib.sha256((BASE/'protocol.json').read_bytes()).hexdigest(),
        'prepared_at':pd.Timestamp.now(tz='UTC').isoformat(),'symbols':u['symbols'],'rows':len(panel)},indent=2))
    print('Prepared hourly panel; replication results not summarized',flush=True)


def window(d,split):
    if split=='discovery':
        return d[d.time.between('2026-04-02','2026-06-29 23:00:00')&(d.time+pd.Timedelta(hours=30)<pd.Timestamp('2026-07-01'))]
    return d[d.time.between('2026-08-01','2026-09-30 23:00:00')]


def covariates(d):
    return np.column_stack([np.log(d.match_liquidity.astype(float)),np.log(d.match_vol168.astype(float).clip(lower=1e-10)),d.match_ret24.astype(float)])


def matched(d,split,scale):
    p=window(d,split)
    cases=p[p.event].sort_values(['time','symbol'])
    lookup=d.set_index(['symbol','time'])
    rows=[];drops={'bad_case_trajectory':0,'insufficient_controls':0}
    def trajectory(symbol,t):
        keys=[(symbol,t+pd.Timedelta(hours=h)) for h in OFFSETS]
        if not all(k in lookup.index for k in keys):
            return None
        z=lookup.loc[keys].copy()
        if not z.quality.all():
            return None
        return z
    for case in cases.itertuples():
        if trajectory(case.symbol,case.time) is None:
            drops['bad_case_trajectory']+=1;continue
        controls=p[p.time.eq(case.time)&p.eligible_price&p.quality&p.future24_complete&(p.max24<.15)&p.symbol.ne(case.symbol)].copy()
        controls['distance']=np.linalg.norm((covariates(controls)-covariates(pd.DataFrame([case._asdict()])))/scale,axis=1)
        controls=controls[controls.distance<=3].sort_values(['distance','symbol'])
        selected=[]
        for control in controls.itertuples():
            if trajectory(control.symbol,case.time) is not None:
                selected.append(control)
            if len(selected)==3:break
        if len(selected)!=3:
            drops['insufficient_controls']+=1;continue
        group=f'{case.symbol}:{case.time.isoformat()}'
        for role,item in [('case',case)]+[('control',x) for x in selected]:
            z=trajectory(item.symbol,case.time).reset_index()
            for offset,(_,row) in zip(OFFSETS,z.iterrows()):
                rows.append({'group':group,'anchor':case.time,'symbol':item.symbol,'role':role,'offset':offset,
                             'match_distance':0 if role=='case' else item.distance,**{k:float(row[k]) for k in FEATURES}})
    result=pd.DataFrame(rows,columns=['group','anchor','symbol','role','offset','match_distance',*FEATURES])
    return result,{'price_defined_events':len(cases),'matched_events':result.group.nunique(),'drops':drops}


def compare(m,feature,direction=1,alpha=.05):
    at=m[m.offset.eq(0)]
    pairs=[]
    for group,g in at.groupby('group'):
        case=g[g.role.eq('case')][feature].iloc[0]*direction
        controls=g[g.role.eq('control')][feature].to_numpy()*direction
        value=float(((case>controls)+.5*(case==controls)).mean())
        pairs.append({'time':g.anchor.iloc[0],'value':value})
    if not pairs:return {'n':0}
    p=pd.DataFrame(pairs);p['block']=(p.time-pd.Timestamp('2026-01-01')).dt.days//7
    b=p.groupby('block').value.agg(['sum','count']).to_numpy()
    rng=np.random.default_rng(20261009)
    z=b[rng.integers(0,len(b),(10000,len(b)))].sum(axis=1)
    boot=z[:,0]/z[:,1]
    return {'n':len(p),'blocks':len(b),'concordance':float(p.value.mean()),'lower':float(np.quantile(boot,alpha/2)),
            'upper':float(np.quantile(boot,1-alpha/2)),
            'case_median':float(at[at.role.eq('case')][feature].median()),'control_median':float(at[at.role.eq('control')][feature].median())}


def discover():
    if (OUT/'frozen_hypotheses.json').exists():
        raise RuntimeError('Hypotheses already frozen; do not silently refit')
    d=pd.read_parquet(OUT/'panel.parquet')
    eligible=window(d,'discovery');eligible=eligible[eligible.eligible_price&eligible.quality&eligible.label_complete]
    scale=np.std(covariates(eligible),axis=0)
    m,audit=matched(d,'discovery',scale)
    m.to_csv(OUT/'discovery_trajectories.csv',index=False)
    stats={};candidates=[]
    for key in FEATURES:
        raw=compare(m,key)
        direction=1 if raw.get('concordance',.5)>=.5 else -1
        s=compare(m,key,direction);s['direction']=direction;stats[key]=s
        if s['n']>=10 and s['concordance']>=.60 and s['lower']>.50:
            candidates.append((s['concordance'],key,direction))
    rules=[];used_activity=False
    for _,key,direction in sorted(candidates,reverse=True):
        if key in ['quote_accel','count_accel'] and used_activity:continue
        cutoff=float((eligible[key]*direction).quantile(.90))
        rules.append({'feature':key,'direction':direction,'cutoff':cutoff})
        used_activity|=key in ['quote_accel','count_accel']
        if len(rules)==2:break
    result={'frozen_at':pd.Timestamp.now(tz='UTC').isoformat(),'protocol_sha256':hashlib.sha256((BASE/'protocol.json').read_bytes()).hexdigest(),
            'study_sha256':hashlib.sha256((BASE/'study.py').read_bytes()).hexdigest(),
            'scale':scale.tolist(),'matching':audit,'discovery':stats,'rules':rules,'eligible_discovery_hours':len(eligible)}
    (OUT/'frozen_hypotheses.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2),flush=True)


def replay(d,mask,score):
    w=d.assign(_mask=mask,_score=score).sort_values(['time','_score','symbol'],ascending=[True,False,True])
    selected=[];last={};counts={}
    for idx,row in w[w._mask].iterrows():
        day=row.time.date()
        if counts.get(day,0)>=3 or (row.symbol in last and row.time-last[row.symbol]<pd.Timedelta(hours=24)):continue
        selected.append(idx);last[row.symbol]=row.time;counts[day]=counts.get(day,0)+1
    return d.loc[selected]


def metrics(d,a):
    hit=a[a.hit6]
    def eventkeys(x):
        return set(zip(x.symbol,x.next_event))
    events=eventkeys(d[d.hit6])
    return {'alerts':len(a),'hits':int(a.hit6.sum()),'precision':float(a.hit6.mean()) if len(a) else None,
            'strict_lead_hits':int(a.hit6_strict.sum()),'eligible_events':len(events),'detected_events':len(eventkeys(hit)),
            'recall':len(eventkeys(hit))/len(events) if events else None,
            'median_lead_hours':float(hit.lead_hours.median()) if len(hit) else None,
            'median_min24':float(a.min24.median()) if len(a) else None,
            'down10':int((a.min24<=-.10).sum()),'median_return24':float(a.return24.median()) if len(a) else None}


def random_baseline(d,a):
    rng=np.random.default_rng(20261009);symbols=sorted(d.symbol.unique());sid={s:i for i,s in enumerate(symbols)}
    last=np.full((1000,len(symbols)),-1e9);counts=np.zeros(1000);hits=counts.copy();rows=[]
    for t,n in a.groupby('time').size().items():
        g=d[d.time.eq(t)].sort_values('symbol');ids=np.array([sid[s] for s in g.symbol]);hour=t.value/3.6e12
        valid=hour-last[:,ids]>=24;draw=rng.random(valid.shape);draw[~valid]=np.inf
        pick=np.argsort(draw,axis=1)[:,:n];good=np.take_along_axis(valid,pick,axis=1);rr,cc=np.where(good)
        last[rr,ids[pick][rr,cc]]=hour
        y=g.hit6.to_numpy()[pick]*good
        counts+=good.sum(axis=1);hits+=y.sum(axis=1)
        rows.append({'time':t,'expected':float(y.sum(axis=1).mean()),'actual':int(a[a.time.eq(t)].hit6.sum())})
    e=pd.DataFrame(rows)
    interval=None
    if len(e) and e.expected.sum()>0:
        e['block']=(e.time-pd.Timestamp('2026-01-01')).dt.days//7
        b=e.groupby('block')[['actual','expected']].sum().to_numpy();z=b[rng.integers(0,len(b),(10000,len(b)))].sum(axis=1)
        v=z[z[:,1]>0,0]/z[z[:,1]>0,1]
        interval={'lift':float(e.actual.sum()/e.expected.sum()),'lower95':float(np.quantile(v,.025)),'upper95':float(np.quantile(v,.975)),'blocks':len(b)}
    return {'mean_alerts':float(counts.mean()),'mean_hits':float(hits.mean()),'mean_precision':float(np.divide(hits,counts,out=np.zeros_like(hits),where=counts>0).mean()),'comparison':interval}


def replicate():
    frozen=json.loads((OUT/'frozen_hypotheses.json').read_text())
    assert frozen['protocol_sha256']==hashlib.sha256((BASE/'protocol.json').read_bytes()).hexdigest()
    assert frozen['study_sha256']==hashlib.sha256((BASE/'study.py').read_bytes()).hexdigest()
    d=pd.read_parquet(OUT/'panel.parquet')
    m,audit=matched(d,'replication',np.array(frozen['scale']))
    m.to_csv(OUT/'replication_trajectories.csv',index=False)
    stats={k:compare(m,k,frozen['discovery'][k]['direction'],alpha=.05/6) for k in FEATURES}
    p=window(d,'replication');p=p[p.eligible_price&p.quality&p.label_complete].copy()
    result={'matching':audit,'replication':stats,'eligible_hours':len(p),'base_rate':float(p.hit6.mean()),'replay':{}}
    masks=[];scores=[]
    for rule in frozen['rules']:
        key=rule['feature'];score=p[key]*rule['direction'];mask=score>=rule['cutoff']
        a=replay(p,mask,score);a.to_csv(OUT/f'alerts_{key}.csv',index=False)
        result['replay'][key]={'metrics':metrics(p,a),'random':random_baseline(p,a)}
        masks.append(mask);scores.append(score-rule['cutoff'])
    if len(masks)==2:
        a=replay(p,masks[0]&masks[1],scores[0]+scores[1]);a.to_csv(OUT/'alerts_combined.csv',index=False)
        result['replay']['combined']={'metrics':metrics(p,a),'random':random_baseline(p,a)}
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','discover','replicate']);args=parser.parse_args()
    {'prepare':prepare,'discover':discover,'replicate':replicate}[args.mode]()
