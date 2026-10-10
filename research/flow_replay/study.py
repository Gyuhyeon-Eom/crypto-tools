"""Causal fixed-rule replay; deliberately no model fitting or threshold search."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parent
OUT=BASE/'output'
RAW=BASE.parent/'tradeflow/output/bars'


def price_features(h):
    h=h.sort_values('time').set_index('time')
    h=h.reindex(pd.date_range(h.index.min(),h.index.max(),freq='h'))
    f=h.copy()
    f['ret24']=h.close/h.close.shift(24)-1
    f['ret168']=h.close/h.close.shift(168)-1
    f['liquidity']=h.turnover.rolling(720).mean()*24
    f['turnover24']=h.turnover.rolling(24).sum()
    f['volume_ratio6']=h.turnover.rolling(6).mean()/h.turnover.rolling(168).mean().shift(6)
    f['price_eligible']=h.close.notna().rolling(720).min().eq(1)&f.liquidity.between(200000,20000000)&(f.ret24<.05)&(f.ret168<.10)
    return f


def flow_features(bars,available_days,start,end):
    """Index is completed interval END. Rolling windows only use past trades."""
    index=pd.date_range(start,end,freq='5min',inclusive='left')
    b=bars.set_index('bucket').reindex(index)
    if not b.index.is_unique:
        raise ValueError('Duplicate bars')
    active=b.quote.notna()
    sums=b[['quote','buy','sell']].fillna(0)
    r2=sums.rolling(24).sum()
    r4=sums.rolling(48).sum().shift(24)
    f=pd.DataFrame(index=index)
    f['net2']=(r2.buy-r2.sell)/r2.quote.replace(0,np.nan)
    f['buy_shift']=f.net2-(r4.buy-r4.sell)/r4.quote.replace(0,np.nan)
    f['raw_quote24']=sums.quote.rolling(288).sum()
    f['active_fraction']=active.rolling(288).mean()
    known=pd.Series(index.floor('D').isin(available_days),index=index)
    f['files_complete']=known.rolling(288).min().eq(1)
    f['first_active']=active.shift(287,fill_value=False)
    f.index=f.index+pd.Timedelta(minutes=5)
    end_ms=f.index.as_unit('ms').astype('int64')
    f['last_trade_age_minutes']=(end_ms-b.last_ms.ffill().to_numpy())/60000
    return f


def add_outcomes(f):
    f=f.copy()
    hi=np.column_stack([f.high.shift(-k) for k in range(1,169)])
    lo=np.column_stack([f.low.shift(-k) for k in range(1,169)])
    mature=np.isfinite(hi).all(axis=1)&np.isfinite(lo).all(axis=1)
    entry=f.close.to_numpy()[:,None]
    touch=hi>=entry*1.30
    first=np.where(touch.any(axis=1),touch.argmax(axis=1)+1,0)
    primary=mature&(hi[:,:24]<entry*1.10).all(axis=1)&(first>24)
    before=(np.arange(1,169)[None,:]<=first[:,None])
    stop=((lo<=entry*.90)&before).any(axis=1)
    f['mature']=mature;f['y']=primary.astype(int);f['y_clean']=(primary&~stop).astype(int)
    f['y_short']=(mature&(hi[:,:3]<entry*1.03).all(axis=1)&(hi[:,:24]>=entry*1.15).any(axis=1)).astype(int)
    f['lead_hours']=np.where(primary,first,np.nan)
    f['target_time']=f.index+pd.to_timedelta(first,unit='h')
    f['min_return7']=np.minimum(0,np.min(lo/entry-1,axis=1))
    f['pretarget_min_return']=np.where(primary,np.minimum(0,np.min(np.where(before,lo/entry-1,np.inf),axis=1)),np.nan)
    f['return7']=f.close.shift(-168)/f.close-1
    next_entry=f.open.shift(-1).to_numpy()[:,None]
    next_touch=hi>=next_entry*1.30
    next_first=np.where(next_touch.any(axis=1),next_touch.argmax(axis=1)+1,0)
    next_y=mature&(hi[:,:24]<next_entry*1.10).all(axis=1)&(next_first>24)
    next_stop=((lo<=next_entry*.90)&(np.arange(1,169)[None,:]<=next_first[:,None])).any(axis=1)
    f['next_open_y']=next_y.astype(int);f['next_open_clean']=(next_y&~next_stop).astype(int)
    return f


def assign_events(d):
    d=d.copy();d['event_id']=''
    for symbol,g in d[d.y.eq(1)].groupby('symbol'):
        anchor=None;n=0
        for idx,row in g.sort_values(['target_time','time']).iterrows():
            if anchor is None or row.target_time-anchor>pd.Timedelta(hours=168):
                n+=1;anchor=row.target_time
            d.loc[idx,'event_id']=f'{symbol}:{n}'
    return d


def build_panel(hourly,symbols,start,end,raw_dir=RAW):
    frames=[]
    raw_start=pd.Timestamp(start)-pd.Timedelta(days=1)
    # End is inclusive for observations; only trades strictly before end needed.
    raw_end=pd.Timestamp(end)
    for symbol in symbols:
        h=hourly[hourly.symbol.eq(symbol)]
        if h.empty:
            raise ValueError(f'Missing hourly data for {symbol}')
        f=price_features(h)
        f=add_outcomes(f)
        bars=[];days=[]
        for day in pd.date_range(raw_start.floor('D'),(raw_end-pd.Timedelta(nanoseconds=1)).floor('D')):
            path=raw_dir/f'{symbol}_{day:%Y-%m-%d}.parquet'
            if path.exists():
                bars.append(pd.read_parquet(path));days.append(day)
        if not bars:
            raise ValueError(f'No archive files for {symbol}')
        flow=flow_features(pd.concat(bars),days,raw_start,raw_end)
        f=f.join(flow)
        discrepancy=(f.raw_quote24/f.turnover24-1).abs()
        f['turnover_error']=discrepancy
        f['quality']=(f.active_fraction>=.80)&f.files_complete.fillna(False).astype(bool)&f.first_active.fillna(False).astype(bool)&(f.last_trade_age_minutes<=5)&(discrepancy<=.01)&f.buy_shift.notna()&f.net2.notna()
        f['trigger']=(f.net2>=.10)&(f.buy_shift>=.15)
        f['time']=f.index;f['symbol']=symbol
        f=f[(f.time>=pd.Timestamp(start))&(f.time<=pd.Timestamp(end))&(f.time.dt.hour%4==0)].reset_index(drop=True)
        frames.append(f)
    return assign_events(pd.concat(frames,ignore_index=True).sort_values(['time','symbol']).reset_index(drop=True))


def replay(d):
    last={};counts={};selected=[]
    for idx,r in d[d.trigger].sort_values(['time','buy_shift','symbol'],ascending=[True,False,True]).iterrows():
        day=r.time.date()
        if counts.get(day,0)>=3 or (r.symbol in last and r.time-last[r.symbol]<pd.Timedelta(hours=168)):
            continue
        selected.append(idx);last[r.symbol]=r.time;counts[day]=counts.get(day,0)+1
    return d.loc[selected].copy()


def comparisons(d,alerts,repetitions=1000):
    rng=np.random.default_rng(20261009)
    symbols=sorted(d.symbol.unique());sid={s:i for i,s in enumerate(symbols)}
    schedule=alerts.groupby('time').size()
    last=np.full((repetitions,len(symbols)),-1e9)
    simple_last={};simple=[];expected=[];random_counts=np.zeros((repetitions,3));random_hits=random_counts.copy()
    splitnames=['early','july','late']
    for t,n in schedule.items():
        group=d[d.time.eq(t)].sort_values('symbol')
        ids=np.array([sid[s] for s in group.symbol]);hour=t.value/3.6e12
        eligible=hour-last[:,ids]>=168
        draw=rng.random(eligible.shape)
        draw[~eligible]=np.inf
        pick=np.argsort(draw,axis=1)[:,:n]
        valid=np.take_along_axis(eligible,pick,axis=1)
        # No selections are fabricated when there are too few available coins.
        selected_ids=ids[pick]
        rr,cc=np.where(valid)
        last[rr,selected_ids[rr,cc]]=hour
        ys=group.y.to_numpy()[pick]*valid
        counts=valid.sum(axis=1);hits=ys.sum(axis=1)
        split=period(t);j=splitnames.index(split)
        random_counts[:,j]+=counts;random_hits[:,j]+=hits
        expected.append({'time':t,'expected':float(hits.mean()),'expected_count':float(counts.mean()),'actual':int(alerts[alerts.time.eq(t)].y.sum()),'actual_count':int(n),'period':split})
        candidates=group[group.symbol.map(lambda s:s not in simple_last or t-simple_last[s]>=pd.Timedelta(hours=168))]
        choice=candidates.sort_values(['volume_ratio6','symbol'],ascending=[False,True]).head(n)
        for idx,row in choice.iterrows():
            simple.append(idx);simple_last[row.symbol]=t
    simulations=[]
    for j,name in enumerate(splitnames+['all']):
        count=random_counts[:,j] if j<3 else random_counts.sum(axis=1)
        hit=random_hits[:,j] if j<3 else random_hits.sum(axis=1)
        precision=np.divide(hit,count,out=np.zeros_like(hit),where=count>0)
        simulations.append({'period':name,'mean_alerts':float(count.mean()),'minimum_alerts':int(count.min()),'mean_hits':float(hit.mean()),
                            'mean_precision':float(precision.mean()),'hit_lower95':float(np.quantile(hit,.025)),'hit_upper95':float(np.quantile(hit,.975))})
    return d.loc[simple].copy(),pd.DataFrame(expected,columns=['time','expected','expected_count','actual','actual_count','period']),simulations


def period(t):
    return 'early' if t<pd.Timestamp('2026-07-01') else ('july' if t<pd.Timestamp('2026-08-01') else 'late')


def metrics(d,alerts):
    n=len(alerts);hit=alerts[alerts.y.eq(1)]
    events=d.loc[d.y.eq(1),'event_id'].nunique()
    detected=hit.event_id.nunique()
    def quantile(series,q):
        return float(series.quantile(q)) if len(series.dropna()) else None
    return {'eligible_observations':len(d),'base_rate':float(d.y.mean()) if len(d) else None,
            'raw_trigger_observations':int(d.trigger.sum()),'alerts':n,'hits':int(alerts.y.sum()),
            'precision':float(alerts.y.mean()) if n else None,'eligible_episodes':events,'detected_episodes':detected,
            'episode_recall':detected/events if events else None,'clean_hits':int(alerts.y_clean.sum()),
            'secondary_hits':int(alerts.y_short.sum()),'next_open_hits':int(alerts.next_open_y.sum()),
            'next_open_clean_hits':int(alerts.next_open_clean.sum()),'median_lead_hours':quantile(hit.lead_hours,.5),
            'mae7_median':quantile(alerts.min_return7,.5),'mae7_worst10pct':quantile(alerts.min_return7,.1),
            'down10_count':int((alerts.min_return7<=-.10).sum()),'down20_count':int((alerts.min_return7<=-.20).sum()),
            'median_pretarget_low':quantile(hit.pretarget_min_return,.5),'median_close_return7':quantile(alerts.return7,.5)}


def lift_interval(e):
    if e.empty or e.expected.sum()==0:
        return None
    e=e.copy();e['block']=(e.time-pd.Timestamp('2026-01-01')).dt.days//7
    b=e.groupby('block')[['actual','expected']].sum().to_numpy()
    rng=np.random.default_rng(20261009)
    z=b[rng.integers(0,len(b),(10000,len(b)))].sum(axis=1)
    ratio=z[z[:,1]>0,0]/z[z[:,1]>0,1]
    return {'lift':float(e.actual.sum()/e.expected.sum()),'lower95':float(np.quantile(ratio,.025)),
            'upper95':float(np.quantile(ratio,.975)),'blocks':len(b)}


def main():
    u=json.loads((OUT/'universe.json').read_text())
    assert u['protocol_sha256']==hashlib.sha256((BASE/'protocol.json').read_bytes()).hexdigest()
    h=pd.read_parquet(BASE.parent/'microstructure/output/bybit.parquet')
    panel=build_panel(h,u['symbols'],'2026-04-01','2026-10-02')
    panel['period']=panel.time.map(period)
    panel.to_parquet(OUT/'panel.parquet',index=False)
    d=panel[panel.price_eligible&panel.quality&panel.mature].copy()
    alerts=replay(d)
    simple,expected,random=comparisons(d,alerts)
    alerts.to_csv(OUT/'alerts.csv',index=False);simple.to_csv(OUT/'volume_baseline.csv',index=False)
    expected.to_csv(OUT/'random_expectation.csv',index=False)
    result={'coverage':{'all_grid_rows':len(panel),'price_eligible':int(panel.price_eligible.sum()),'quality_and_mature':len(d),
                        'symbols':int(d.symbol.nunique()),'observed_times':int(d.time.nunique()),
                        'maximum_turnover_error':float(panel.turnover_error.max()),
                        'price_eligible_episodes':int(panel[panel.price_eligible&panel.y.eq(1)].event_id.nunique())},'periods':{},'random':random}
    for name in ['early','july','late','all']:
        p=d if name=='all' else d[d.period.eq(name)]
        a=alerts if name=='all' else alerts[alerts.period.eq(name)]
        s=simple if name=='all' else simple[simple.period.eq(name)]
        e=expected if name=='all' else expected[expected.period.eq(name)]
        result['periods'][name]={'flow':metrics(p,a),'volume':metrics(p,s),'vs_random':lift_interval(e)}
    monthly=[]
    for month,p in d.groupby(d.time.dt.strftime('%Y-%m')):
        a=alerts[alerts.time.dt.strftime('%Y-%m').eq(month)]
        monthly.append({'month':month,**metrics(p,a)})
    pd.DataFrame(monthly).to_csv(OUT/'monthly.csv',index=False)
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
