"""Post-fit evidence checks only: no model or threshold changes."""
import json
import numpy as np
import pandas as pd
from features import OUT
from evaluate import matched_random


def calendar_sums(a,key,start,end):
    anchor=pd.Timestamp('2020-01-06');weeks=((a.index-anchor).days//7).to_numpy()
    x=pd.DataFrame({'week':weeks,'value':a[key].to_numpy()}).groupby('week').value.agg(['sum','count'])
    return x.reindex(range((start-anchor).days//7,(end-anchor).days//7+1),fill_value=0).to_numpy(float)


def alert_difference(a,b,key,start,end):
    if a.empty or b.empty:return None
    x=calendar_sums(a,key,start,end);y=calendar_sums(b,key,start,end)
    rng=np.random.default_rng(31);pick=rng.integers(0,len(x),(3000,len(x)))
    xx=x[pick].sum(axis=1);yy=y[pick].sum(axis=1)
    diff=xx[:,0]/np.where(xx[:,1]>0,xx[:,1],np.nan)-yy[:,0]/np.where(yy[:,1]>0,yy[:,1],np.nan)
    return {'difference':float(a[key].mean()-b[key].mean()),'paired_week95':np.nanquantile(diff,[.025,.975]).tolist(),
            'note':'Same calendar weeks resampled together; alert counts and timestamps can differ. Individual intervals, not corrected for selection.'}


def corr_blocks(x,y):
    pair=pd.concat([x.rename('x'),y.rename('y')],axis=1).dropna()
    week=((pair.index-pd.Timestamp('2020-01-06')).days//7).to_numpy()
    z=pd.DataFrame({'week':week,'n':1.,'x':pair.x,'y':pair.y,'xx':pair.x**2,'yy':pair.y**2,'xy':pair.x*pair.y}).groupby('week').sum().to_numpy()
    def calc(v):
        n,sx,sy,xx,yy,xy=np.moveaxis(v,-1,0)
        denom=np.sqrt(np.maximum(0,(xx-sx*sx/n)*(yy-sy*sy/n)))
        return (xy-sx*sy/n)/np.where(denom>0,denom,np.nan)
    rng=np.random.default_rng(31);sample=z[rng.integers(0,len(z),(3000,len(z)))].sum(axis=1)
    return {'n':len(pair),'correlation':float(calc(z.sum(axis=0))),'block95':np.nanquantile(calc(sample),[.025,.975]).tolist()}


def main():
    r=json.loads((OUT/'results.json').read_text());f=pd.read_parquet(OUT/'features.parquet')
    start=pd.Timestamp(r['folds'][0]['test'][0]);end=pd.Timestamp(r['folds'][-1]['test'][1]);results=[]
    for t in r['tasks']:
      key=t['key'];outcome=pd.read_parquet(OUT/'outcomes'/f'{key}_prior_4h.parquet')
      for m in t['aggregate']:
        if m['name']=='scheduled':continue
        name=m['name'];kind=name.rsplit('_',1)[1]
        a=pd.read_parquet(OUT/'alerts'/f'{key}_{name}.parquet');b=pd.read_parquet(OUT/'alerts'/f'{key}_baseline_{kind}.parquet')
        item={'task':key,'name':name,'precision_delta_vs_baseline':alert_difference(a,b,'target_before_stop',start,end),
              'net_delta_vs_baseline':alert_difference(a,b,'net20',start,end)}
        if name.startswith(('baseline_','all48_')):
            pool=pd.DatetimeIndex([])
            for path in sorted((OUT/'scores').glob(f'{key}_*_{name}.parquet')):pool=pool.append(pd.read_parquet(path).index)
            aa=a.copy();aa['y']=aa.target_before_stop
            item['matched_random']=matched_random(aa,pool.sort_values(),outcome.target_before_stop,repeats=499)
        results.append(item)
      print(key,'diagnostics complete',flush=True)
    # Same held-out quiet pool; label-only lead/lag study, no new features sent to models.
    pool=pd.DatetimeIndex([])
    for path in sorted((OUT/'scores').glob('long_5_72_*_baseline_boosting.parquet')):pool=pool.append(pd.read_parquet(path).index)
    beta=f.ret1.rolling(2880).cov(f.btc_ret1).shift()/f.btc_ret1.rolling(2880).var().shift()
    near=f.close;btc_ret={0:f.btc_ret1}
    # Reconstruct BTC close from original source rather than summing overlapping returns.
    from features import load_verified
    from data import first
    _,_,raw,_=load_verified();btc=first.price_frame(raw['candles']['BTCUSDT'],raw['start'],raw['end']).close
    lead=[]
    for feature in ['l1_btc_residual1','ai_btc_residual1','spot_delta1','chain48_stable7','chain48_dex_build','accounts_change4']:
        for hours in (0,1,4,24):
            target=f.ret1-beta*f.btc_ret1 if hours==0 else near.shift(-4*hours)/near-1-beta*(btc.shift(-4*hours)/btc-1)
            lead.append({'feature':feature,'future_hours':hours,**corr_blocks(f[feature].reindex(pool),target.reindex(pool))})
    daily=pd.read_parquet(OUT/'daily.parquet')
    result={'comparisons':results,'lead_lag':lead,'study_window_chain_gaps':daily.loc['2024-10-17':'2026-10-16'].isna().sum().astype(int).to_dict(),
            'lead_lag_limits':'Exploratory linear correlations against BTC-beta-adjusted NEAR returns in the held-out quiet pool.0h is preceding1h concurrent return, positive hours future return. Block CIs unadjusted across24correlations. Neither causal evidence nor nonlinear predictor test.',
            'random_limits':'Same-week budget greedy random times,499attempts; not uniform or a causal randomization test. Only baseline/all48 diagnostic; no model refit.'}
    (OUT/'diagnostics.json').write_text(json.dumps(result,indent=2,allow_nan=False));print('Saved diagnostics.json',flush=True)

if __name__=='__main__':main()
