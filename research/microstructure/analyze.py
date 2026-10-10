"""Frozen hypothesis tests using hourly flow, lagged OI/funding and two venues."""
from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

BASE = Path(__file__).resolve().parent
OUT = BASE / 'output'
PRICE = ['ret1','ret6','ret24','ret168','vol24','vol168','bybit_volume_ratio6',
         'range_ratio24','position24','log_liquidity','market_ret24','market_breadth','btc_ret24','relative_market24']
MICRO = ['buy_share1','buy_share6','buy_share24','buy_change6','binance_volume_ratio6',
         'oi_change6','oi_change24','funding24','funding_change24','basis_anomaly',
         'venue_return_gap1','prior_compression','absorption_strength','binance_ret6']
RULES = ['H1_buying_flat','H2_oi_funding','H3_absorption','H4_venue_lead','H3b_absorption_breakout']


def lagged_derivatives(index, oi, funding):
    """Availability timestamps, not nearest timestamps; never use future observations."""
    result = pd.DataFrame(index=index)
    left = pd.DataFrame({'time':index})
    if len(oi):
        oi = oi.sort_values('time').copy()
        oi['observed_at'] = oi.time
        oi['time'] = oi.time + pd.Timedelta(hours=1)
        matched = pd.merge_asof(left, oi[['time','observed_at','oi']], on='time', direction='backward', tolerance=pd.Timedelta(hours=8))
        result['oi'] = matched.oi.to_numpy()
        result['oi_source_time'] = matched.observed_at.to_numpy()
    else:
        result['oi'],result['oi_source_time'] = np.nan,pd.NaT
    if len(funding):
        funding = funding.sort_values('time').copy()
        funding['time'] = (funding.time + pd.Timedelta(hours=1)).dt.ceil('h')
        events = funding.groupby('time').funding.sum()
        # The hourly grid is an interval sum, not a forward-filled individual rate.
        event_values = events.reindex(index, fill_value=0)
        matches = pd.merge_asof(left, events.rename('value').reset_index().rename(columns={'time':'last_event'}),
                                left_on='time',right_on='last_event',direction='backward',tolerance=pd.Timedelta(hours=12))
        known = matches.last_event.notna().to_numpy()
        valid48 = pd.Series(known,index=index).rolling(48).min().eq(1)
        result['funding24'] = event_values.rolling(24).sum().where(valid48)
        result['funding_change24'] = (event_values.rolling(24).sum()-event_values.rolling(24).sum().shift(24)).where(valid48)
    else:
        result['funding24'],result['funding_change24'] = np.nan,np.nan
    # Endpoint interpolation is explicitly backward-only, with an age limit.
    result['oi_change6'] = result.oi/result.oi.shift(6)-1
    result['oi_change24'] = result.oi/result.oi.shift(24)-1
    good_oi = result.oi.notna().rolling(25).min().eq(1)
    result.loc[~good_oi,['oi_change6','oi_change24']] = np.nan
    return result


def symbol_features(bybit, binance, oi, funding):
    y = bybit.sort_values('time').set_index('time')
    index = pd.date_range(y.index.min(),y.index.max(),freq='h')
    y = y.reindex(index)
    b = binance.sort_values('time').set_index('time').reindex(index)
    f = pd.DataFrame(index=index)
    c,bc = y.close,b.close
    for hours in [1,6,24,168]:
        f[f'ret{hours}'] = c/c.shift(hours)-1
    for hours in [1,6]:
        f[f'binance_ret{hours}'] = bc/bc.shift(hours)-1
    returns = c.pct_change(fill_method=None)
    f['vol24'],f['vol168'] = returns.rolling(24).std(),returns.rolling(168).std()
    f['liquidity'] = y.turnover.rolling(720).mean()*24
    f['log_liquidity'] = np.log1p(f.liquidity)
    f['bybit_volume_ratio6'] = y.turnover.rolling(6).mean()/y.turnover.rolling(168).mean().shift(6)
    yrange = (y.high-y.low)/c
    f['range_ratio24'] = yrange.rolling(24).mean()/yrange.rolling(168).mean().shift(24)
    f['position24'] = (c-y.low.rolling(24).min())/(y.high.rolling(24).max()-y.low.rolling(24).min())
    for hours in [1,6,24]:
        f[f'buy_share{hours}'] = b.buy_turnover.rolling(hours).sum()/b.turnover.rolling(hours).sum()
    previous18 = b.buy_turnover.rolling(18).sum().shift(6)/b.turnover.rolling(18).sum().shift(6)
    f['buy_change6'] = f.buy_share6-previous18
    f['binance_volume_ratio6'] = b.turnover.rolling(6).mean()/b.turnover.rolling(168).mean().shift(6)
    brange = (b.high-b.low)/bc
    f['prior_compression'] = brange.rolling(18).mean().shift(6)/brange.rolling(168).mean().shift(24)
    f['absorption_strength'] = (.5-f.buy_share6)-(-f.binance_ret6).clip(lower=0)*5
    f['breakout_after_absorption'] = bc > b.high.rolling(24).max().shift(6)
    basis = np.log(bc/c)
    f['basis_anomaly'] = basis-basis.rolling(168).median().shift(1)
    f['venue_return_gap1'] = f.binance_ret1-f.ret1
    deriv = lagged_derivatives(index,oi,funding)
    f = f.join(deriv)
    f['close'],f['high'],f['low'] = c,y.high,y.low
    f['continuous168'] = c.notna().rolling(169).min().eq(1)
    f['time'] = index
    f['symbol'] = bybit.symbol.iloc[0]
    return f.replace([np.inf,-np.inf],np.nan).reset_index(drop=True)


def label_symbol(f):
    f = f.sort_values('time').copy()
    # Hourly reindexing happens in feature construction, preserving calendar gaps.
    highs = np.column_stack([f.high.shift(-h).to_numpy() for h in range(1,169)])
    lows = np.column_stack([f.low.shift(-h).to_numpy() for h in range(1,169)])
    c = f.close.to_numpy()[:,None]
    mature = np.isfinite(highs).all(axis=1)&np.isfinite(lows).all(axis=1)
    touch = highs >= c*1.30
    first = np.where(touch.any(axis=1),touch.argmax(axis=1)+1,0)
    quiet24 = (highs[:,:24] < c*1.10).all(axis=1)
    y = mature&quiet24&(first>24)
    stopped = ((lows <= c*.90)&(np.arange(1,169)[None,:] <= first[:,None])).any(axis=1)
    f['y'] = y.astype(int)
    f['y_clean'] = (y&~stopped).astype(int)
    f['y_down'] = (mature&(lows <= c*.80).any(axis=1)).astype(int)
    f['y_short'] = (mature&(highs[:,:3] < c*1.03).all(axis=1)&(highs[:,:24] >= c*1.15).any(axis=1)).astype(int)
    f['first_hit_hour'] = first
    f['target_time'] = f.time+pd.to_timedelta(first,unit='h')
    f['label_end'] = f.time+pd.Timedelta(hours=168)
    f['mature'] = mature
    return f


def rules(f):
    flat = f.ret6.abs() <= .02
    f = f.copy()
    f['H1_buying_flat'] = flat&(f.buy_share6>=.55)&(f.buy_change6>=.05)&(f.binance_volume_ratio6>=1.25)
    f['H2_oi_funding'] = flat&(f.oi_change24>=.05)&(f.funding24<=0)&(f.funding_change24<0)
    f['H3_absorption'] = (f.prior_compression<=.75)&(f.buy_share6<=.45)&(f.binance_ret6>=-.005)&(f.binance_volume_ratio6>=1.25)
    f['H4_venue_lead'] = (f.basis_anomaly>=.002)&(f.venue_return_gap1>0)&(f.buy_share1>=.55)
    f['H3b_absorption_breakout'] = f.H3_absorption&f.breakout_after_absorption
    f['score_H1_buying_flat'] = f.buy_change6
    f['score_H2_oi_funding'] = f.oi_change24
    f['score_H3_absorption'] = f.absorption_strength
    f['score_H4_venue_lead'] = f.basis_anomaly
    f['score_H3b_absorption_breakout'] = f.absorption_strength
    return f


def replay(d,mask,score,threshold=-np.inf,cooldown_hours=168,cap=3):
    """Process times in order. Later daily signals cannot replace earlier alerts."""
    work = d.assign(_fire=np.asarray(mask,dtype=bool),_score=np.asarray(score)).sort_values(['time','_score','symbol'],ascending=[True,False,True])
    last,counts,selected = {},{},[]
    for idx,row in work[work._fire & work._score.ge(threshold)].iterrows():
        day = row.time.date()
        if counts.get(day,0)>=cap:
            continue
        if row.symbol in last and row.time-last[row.symbol]<pd.Timedelta(hours=cooldown_hours):
            continue
        last[row.symbol]=row.time
        counts[day]=counts.get(day,0)+1
        selected.append(idx)
    return work.loc[selected]


def block_bounds(d):
    """d columns time, observed, expected; paired 7-day calendar blocks."""
    if len(d)==0 or d.expected.sum()<=0:
        return {'lift':None,'lower_98_75':None,'upper_98_75':None,'blocks':0}
    d=d.copy()
    d['block']=(d.time-d.time.min()).dt.days//7
    a=d.groupby('block')[['observed','expected']].sum().to_numpy()
    rng=np.random.default_rng(20261009)
    boot=a[rng.integers(0,len(a),(2000,len(a)))].sum(axis=1)
    valid=boot[:,1]>0
    ratio=boot[valid,0]/boot[valid,1]
    return {'lift':float(d.observed.sum()/d.expected.sum()),'lower_98_75':float(np.quantile(ratio,.0125)),
            'upper_98_75':float(np.quantile(ratio,.9875)),'blocks':len(a)}


def conditional(d,rule):
    d=d.copy()
    d['fire']=d[rule]
    chosen=d[d.fire]
    if not len(chosen):
        return {'n':0,'hits':0,'precision':None}
    out={'n':len(chosen),'hits':int(chosen.y.sum()),'precision':float(chosen.y.mean()),
         'base_rate':float(d.y.mean()),'secondary_precision':float(chosen.y_short.mean()),
         'secondary_base_rate':float(d.y_short.mean()),'down20_rate':float(chosen.y_down.mean())}
    for feature,col in [('vol168','volbin'),('liquidity','liqbin')]:
        d[col]=np.minimum((d.groupby('time')[feature].rank(pct=True)*3).astype(int),2)
    for label,keys in [('time_matched',['time']),('risk_matched',['time','volbin','liqbin'])]:
        control=d[~d.fire].groupby(keys).y.agg(['mean','size'])
        matched=d[d.fire].join(control,on=keys)
        matched=matched[matched['size']>=3]
        paired=pd.DataFrame({'time':matched.time,'observed':matched.y,'expected':matched['mean']})
        out[label]={**block_bounds(paired),'coverage':len(matched)/len(chosen)}
    return out


def alert_metrics(d,selected):
    out={'alerts':len(selected),'hits':int(selected.y.sum()),
         'precision':float(selected.y.mean()) if len(selected) else None,
         'clean_hits':int(selected.y_clean.sum()),'down20_count':int(selected.y_down.sum()),
         'secondary_hits':int(selected.y_short.sum()),
         'median_lead_to_target_hours':float(selected.loc[selected.y.eq(1),'first_hit_hour'].median()) if selected.y.sum() else None}
    expected=selected.time.map(d.groupby('time').y.mean())
    paired=pd.DataFrame({'time':selected.time,'observed':selected.y,'expected':expected})
    out['random_expected_hits']=float(expected.sum())
    out['random_comparison']=block_bounds(paired)
    events=d.loc[d.y.eq(1),'event_id'].nunique()
    out['captured_events']=selected.loc[selected.y.eq(1),'event_id'].nunique()
    out['evaluable_events']=events
    out['event_recall']=out['captured_events']/events if events else None
    return out


def same_budget_baseline(d,schedule,feature):
    """Select simple-price candidates at exactly the rule's firing times/budgets."""
    last,selected={},[]
    for time,n in schedule.groupby('time').size().items():
        group=d[d.time.eq(time)].sort_values([feature,'symbol'],ascending=[False,True])
        count=0
        for idx,row in group.iterrows():
            if row.symbol in last and time-last[row.symbol]<pd.Timedelta(hours=168):continue
            selected.append(idx);last[row.symbol]=time;count+=1
            if count>=n:break
    return d.loc[selected]


def wilson(hits,n):
    if n==0:return 0
    p,z=hits/n,1.96
    return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)


def main():
    data={k:pd.read_parquet(OUT/f'{k}.parquet') for k in ['bybit','binance','oi','funding']}
    symbols=json.loads((OUT/'universe.json').read_text())['symbols']
    groups={k:{s:g for s,g in d.groupby('symbol')} for k,d in data.items()}
    features=[]
    for n,s in enumerate(symbols,1):
        if any(s not in groups[k] for k in data):
            continue
        f=symbol_features(*(groups[k][s] for k in ['bybit','binance','oi','funding']))
        features.append(f)
        if n%25==0:print('Features:',n,'/',len(symbols),flush=True)
    all_features=pd.concat(features,ignore_index=True)
    market=all_features.groupby('time').agg(market_ret24=('ret24','median'),market_breadth=('ret24',lambda x:(x>0).mean()))
    btc=groups['bybit']['BTCUSDT'].sort_values('time').set_index('time').close
    btc=btc.reindex(pd.date_range(btc.index.min(),btc.index.max(),freq='h'))
    market['btc_ret24']=btc/btc.shift(24)-1
    all_features=all_features.merge(market,on='time',how='left',validate='many_to_one')
    all_features['relative_market24']=all_features.ret24-all_features.market_ret24
    labelled=pd.concat([label_symbol(g) for _,g in all_features.groupby('symbol')],ignore_index=True)
    base=(labelled.time.dt.hour.mod(4).eq(0)&labelled.time.ge('2026-04-01')&labelled.mature
          &labelled.continuous168&labelled.liquidity.between(200000,20000000)&labelled.ret24.lt(.05)&labelled.ret168.lt(.10))
    complete=labelled[PRICE+MICRO].notna().all(axis=1)
    coverage={'price_eligible':int(base.sum()),'complete_case':int((base&complete).sum()),
              'missing_by_feature':labelled.loc[base,PRICE+MICRO].isna().sum().to_dict()}
    d=rules(labelled[base&complete].copy()).sort_values(['time','symbol']).reset_index(drop=True)
    d['split']='purged'
    d.loc[d.label_end<'2026-07-01','split']='train'
    d.loc[(d.time>='2026-07-01')&(d.label_end<'2026-08-01'),'split']='validation'
    d.loc[d.time>='2026-08-01','split']='test'
    d['event_id']=''
    for symbol,g in d[d.y.eq(1)].groupby('symbol'):
        anchor,event=None,0
        for idx,row in g.sort_values('target_time').iterrows():
            if anchor is None or row.target_time-anchor>pd.Timedelta(hours=168):
                anchor,event=row.target_time,event+1
            d.at[idx,'event_id']=f'{symbol}:{event}'
    d.to_parquet(OUT/'samples.parquet',index=False)
    parts={s:d[d.split.eq(s)] for s in ['train','validation','test']}
    overview={s:{'rows':len(p),'coins':p.symbol.nunique(),'positives':int(p.y.sum()),
                  'base_rate':float(p.y.mean()),'events':p.loc[p.y.eq(1),'event_id'].nunique(),
                  'start':str(p.time.min()),'end':str(p.time.max())} for s,p in parts.items()}
    print(json.dumps(overview,indent=2),flush=True)
    result={'coverage':coverage,'overview':overview,'rules':{}}
    for rule in RULES:
        result['rules'][rule]={}
        for split,p in parts.items():
            selected=replay(p,p[rule],p['score_'+rule])
            outcome={'conditional':conditional(p,rule),'alerts':alert_metrics(p,selected)}
            result['rules'][rule][split]=outcome
            if split=='test':selected.to_csv(OUT/f'{rule}_test_alerts.csv',index=False)
        print(rule,json.dumps(result['rules'][rule]['test']),flush=True)
    # Pick the simple comparator on validation, not by looking at final outcomes.
    for rule in RULES:
        comparators={}
        for feature in ['vol168','bybit_volume_ratio6','ret24']:
            comparators[feature]={}
            for split in ['validation','test']:
                p=parts[split]
                schedule=replay(p,p[rule],p['score_'+rule])
                comparator=same_budget_baseline(p,schedule,feature)
                comparators[feature][split]=alert_metrics(p,comparator)
        winner=max(comparators,key=lambda k:comparators[k]['validation']['precision'] or 0)
        result['rules'][rule]['simple_comparators']=comparators
        result['rules'][rule]['selected_simple_comparator']=winner
    # Same-sample ablation: no class weighting and no test-based hyperparameter selection.
    tr,val,te=(parts[s] for s in ['train','validation','test'])
    result['models']={}
    for name,columns in [('price_only',PRICE),('price_plus_micro',PRICE+MICRO)]:
        model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=.1,max_iter=2000))
        with threadpool_limits(limits=4):
            model.fit(tr[columns],tr.y)
            pv=model.predict_proba(val[columns])[:,1]
            pt=model.predict_proba(te[columns])[:,1]
        choices=[]
        for q in [.50,.70,.80,.90,.95,.98]:
            cutoff=float(np.quantile(pv,q))
            a=replay(val,np.ones(len(val),dtype=bool),pv,cutoff)
            if len(a)>=20:choices.append((wilson(int(a.y.sum()),len(a)),cutoff,q))
        _,cutoff,q=max(choices) if choices else (0,float(np.quantile(pv,.9)),.9)
        result['models'][name]={'cutoff':cutoff,'validation_quantile':q}
        for split,p,scores in [('validation',val,pv),('test',te,pt)]:
            selected=replay(p,np.ones(len(p),dtype=bool),scores,cutoff)
            result['models'][name][split]={'pr_auc':float(average_precision_score(p.y,scores)),
                'roc_auc':float(roc_auc_score(p.y,scores)),**alert_metrics(p,selected)}
            if split=='test':selected.to_csv(OUT/f'{name}_test_alerts.csv',index=False)
        joblib.dump({'model':model,'features':columns,'cutoff':cutoff},OUT/f'{name}.joblib')
        print(name,json.dumps(result['models'][name]),flush=True)
    # Never interpret unsupported/rare results as evidence of absence.
    for rule in RULES:
        rv=result['rules'][rule]
        final=rv['test'];validation=rv['validation']
        ca=final['conditional'];alerts=final['alerts']
        enough=alerts['alerts']>=30
        matched=ca.get('risk_matched',{})
        consistent=(validation['conditional'].get('time_matched',{}).get('lift') or 0)>1
        comparator=rv['simple_comparators'][rv['selected_simple_comparator']]['test']
        better=(alerts['precision'] or 0)>(comparator['precision'] or 0)
        supported=(enough and (matched.get('lift') or 0)>1 and (matched.get('lower_98_75') or 0)>1
                   and (alerts['random_comparison'].get('lower_98_75') or 0)>1 and consistent and better)
        rv['status']='candidate_supported_needs_fresh_confirmation' if supported else ('insufficient_alert_sample' if not enough else 'not_supported')
    (OUT/'results.json').write_text(json.dumps(result,indent=2,default=lambda x:int(x) if isinstance(x,np.integer) else float(x)))
    print('Analysis complete',flush=True)


if __name__=='__main__':
    main()
