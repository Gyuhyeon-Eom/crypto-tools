"""Frozen chronological models, independent baselines and path-risk accounting."""
from pathlib import Path
import gzip
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from data import make_features, zigzag, precursor_labels, touch_outcomes, funding_cumulative, candidates, fold_windows, split_index, PRICE, DERIV, HOUR

BASE=Path(__file__).resolve().parent
OUT=BASE/'output'
QUANTILES=[.8,.9,.95,.975,.99]


def replay(scores, threshold):
    result=[];last=None
    for t,s in scores.items():
        if s>=threshold and (last is None or t-last>=6*HOUR):result.append(t);last=t
    return pd.DatetimeIndex(result)


def calibrate(scores,y):
    results=[]
    for q in QUANTILES:
        threshold=float(np.quantile(scores,q));chosen=replay(scores,threshold)
        if len(chosen)<20:continue
        hits=int(y.reindex(chosen).sum());ci=binomtest(hits,len(chosen)).proportion_ci(method='exact')
        results.append({'quantile':q,'threshold':threshold,'alerts':len(chosen),'hits':hits,'lower95':float(ci.low)})
    return max(results,key=lambda r:(r['lower95'],r['alerts'])) if results else None


def weekly_ci(alerts,key,all_start,all_end,seed=17):
    if alerts.empty:return [None,None]
    anchor=pd.Timestamp('2020-01-06')
    week=((alerts.index-anchor).days//7).to_numpy()
    frame=pd.DataFrame({'week':week,'value':alerts[key].to_numpy()})
    n0=int((all_start-anchor).days//7);n1=int((all_end-anchor).days//7)
    grouped=frame.groupby('week').value.agg(['sum','count']).reindex(range(n0,n1+1),fill_value=0).to_numpy(float)
    rng=np.random.default_rng(seed)
    sample=grouped[rng.integers(0,len(grouped),(3000,len(grouped)))].sum(axis=1)
    value=sample[:,0]/np.where(sample[:,1]>0,sample[:,1],np.nan)
    return np.nanquantile(value,[.025,.975]).tolist()


def describe_alerts(alerts,events,start,end):
    if alerts.empty:return {'alerts':0}
    n=len(alerts);hits=int(alerts.y.sum());ci=binomtest(hits,n).proportion_ci(method='exact')
    caught=set(pd.to_datetime(alerts.event.dropna()))&set(events)
    result={'alerts':n,'hits':hits,'precision':hits/n,'exact95':[float(ci.low),float(ci.high)],
            'block95':weekly_ci(alerts,'y',start,end),'active_weeks':int(alerts.resample('W').size().gt(0).sum()),
            'events':len(events),'events_caught':len(caught),'pre_onset_recall':len(caught)/len(events) if len(events) else None,
            'pre_onset_alert_fraction':float(alerts.pre_onset.mean()),'target_touch_rate':float(alerts.touch.mean()),
            'median_touch_hours':float(alerts.first_touch_hours.median()) if alerts.first_touch_hours.notna().any() else None,
            'median_adverse_pct':float(alerts.adverse.median()*100),'worst_adverse_pct':float(alerts.adverse.min()*100),
            'mean_net_10bps_pct':float(alerts.net_10bps.mean()*100),'mean_net_20bps_pct':float(alerts.net_20bps.mean()*100),
            'net_20bps_block95_pct':[x*100 for x in weekly_ci(alerts,'net_20bps',start,end)]}
    for stop in (2,3,5):
        result[f'target_before_{stop}pct_adverse']=float(alerts[f'before_adverse_{stop}'].mean())
        result[f'same_bar_ambiguous_{stop}pct']=int(alerts[f'ambiguous_{stop}'].sum())
    return result


def pack_alerts(times, y, ids, outcomes, f):
    a=outcomes.reindex(times).copy();a['y']=y.reindex(times);a['event']=ids.reindex(times);a['pre_onset']=a.event.notna().astype(float)
    a['btc_regime']=np.select([f.btc_ret7d.reindex(times)>.05,f.btc_ret7d.reindex(times)<-.05],['up','down'],default='flat')
    return a


def make_model(kind):
    if kind=='logistic':return make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000))
    return HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,learning_rate=.05,l2_regularization=1,
                                         early_stopping=False,random_state=17)


def matched_random(alerts,pool,y,repeats=200):
    """Same-week alert budgets; enforce6h spacing both within and across week boundaries."""
    from bisect import bisect_left, insort
    if alerts.empty:return None
    rng=np.random.default_rng(17)
    week=lambda times:((times-pd.Timestamp('2020-01-06')).days//7).to_numpy()
    counts=pd.Series(week(alerts.index)).value_counts().sort_index()
    groups={w:pool[week(pool)==w] for w in counts.index}
    null=[]
    for _ in range(repeats):
        chosen=[];success=True
        for w,count in counts.items():
            picked=0
            for j in rng.permutation(len(groups[w])):
                t=groups[w][j];pos=bisect_left(chosen,t)
                if (pos and t-chosen[pos-1]<6*HOUR) or (pos<len(chosen) and chosen[pos]-t<6*HOUR):continue
                insort(chosen,t);picked+=1
                if picked==count:break
            if picked<count:success=False;break
        if success:null.append(float(y.reindex(chosen).mean()))
    if not null:return {'valid_repeats':0}
    observed=float(alerts.y.mean())
    return {'valid_repeats':len(null),'null_precision_median':float(np.median(null)),
            'null_precision95':np.quantile(null,[.025,.975]).tolist(),'one_sided_p':(1+sum(v>=observed for v in null))/(1+len(null)),
            'scope':'Exploratory matched-week random timing. Multiple model comparisons require correction; this is not causal evidence.'}


def main():
    raw=json.loads(gzip.decompress((OUT/'data.json.gz').read_bytes()))
    manifest=json.loads((OUT/'manifest.json').read_text())
    for file,want in [('data.json.gz',manifest['data_sha256'])]:
        if hashlib.sha256((OUT/file).read_bytes()).hexdigest()!=want:raise ValueError('Data changed')
    if hashlib.sha256((BASE/'protocol.json').read_bytes()).hexdigest()!=manifest['protocol_sha256']:raise ValueError('Protocol changed')
    f=make_features(raw);f.to_parquet(OUT/'features.parquet')
    end=pd.Timestamp(raw['end'],unit='ms');folds=fold_windows(end)
    all_candidate=candidates(f);price_candidate=candidates(f,False)
    funding=funding_cumulative(f,raw['funding']);events={t:zigzag(f,t) for t in (.05,.1)}
    locks=OUT/'locks';locks.mkdir(exist_ok=True)
    scores_dir=OUT/'scores';scores_dir.mkdir(exist_ok=True)
    report={'manifest':manifest,'source_hashes':{n:hashlib.sha256((BASE/n).read_bytes()).hexdigest() for n in ('data.py','evaluate.py','protocol.json')},
            'folds':[{k:[str(a),str(b)] for k,(a,b) in fold.items()} for fold in folds],
            'paired_candidate_count':int(all_candidate.sum()),'price_candidate_count':int(price_candidate.sum()),'tasks':[]}
    events_all=pd.concat(events.values(),ignore_index=True);events_all.to_parquet(OUT/'events.parquet',index=False)
    for direction in ('long','short'):
      for target in (.05,.1):
        onset_y,ids,event_frame=precursor_labels(f,events[target],direction)
        for hours in (24,72):
          outcome=touch_outcomes(f,direction,target,hours,funding)
          outcome.to_parquet(OUT/f'outcome_{direction}_{target}_{hours}.parquet')
          for task_type in (['touch','onset'] if hours==72 else ['touch']):
            task_key=f'{direction}_{int(target*100)}_{hours}_{task_type}'
            y=outcome.touch if task_type=='touch' else onset_y
            task={'key':task_key,'direction':direction,'target':target,'hours':hours,'type':task_type,'fold_results':[], 'aggregate':[]}
            combined={};pools=[];event_lists=[]
            for fold_id,windows in enumerate(folds):
              indices={name:split_index(f,w,all_candidate&y.notna()) for name,w in windows.items()}
              train,cal,test=[indices[k] for k in ('train','calibration','test')]
              a,b=windows['test'];test_events=pd.DatetimeIndex(event_frame.loc[(event_frame.onset>=a)&(event_frame.onset+72*HOUR<b),'onset'])
              event_lists.extend(test_events);pools.extend(test)
              record={'fold':fold_id,'counts':{k:len(v) for k,v in indices.items()},'test_base_rate':float(y.reindex(test).mean()),'models':[]}
              if y.reindex(train).nunique()<2 or y.reindex(cal).nunique()<2 or (task_type=='onset' and (ids.reindex(train).nunique()<20 or ids.reindex(cal).nunique()<5)):
                record['status']='insufficient_support';task['fold_results'].append(record);continue
              for group,cols in [('price',PRICE),('derivatives',PRICE+DERIV)]:
                for kind in ('logistic','boosting'):
                  name=group+'_'+kind;model=make_model(kind);model.fit(f.loc[train,cols],y.loc[train])
                  cal_score=pd.Series(model.predict_proba(f.loc[cal,cols])[:,1],index=cal)
                  choice=calibrate(cal_score,y)
                  if choice is None:record['models'].append({'name':name,'status':'insufficient_calibration_alerts'});continue
                  lock={'task':task_key,'fold':fold_id,'name':name,'features':cols,'calibration':choice,'train_end':str(windows['train'][1]),
                        'calibration_end':str(windows['calibration'][1]),'deployment':'not_approved'}
                  path=locks/f'{task_key}_{fold_id}_{name}'
                  joblib.dump(model,str(path)+'.joblib');lock['model_sha256']=hashlib.sha256(Path(str(path)+'.joblib').read_bytes()).hexdigest()
                  Path(str(path)+'.json').write_text(json.dumps(lock,indent=2))
                  scores=pd.Series(model.predict_proba(f.loc[test,cols])[:,1],index=test)
                  chosen=replay(scores,choice['threshold']);alerts=pack_alerts(chosen,y,ids,outcome,f)
                  combined.setdefault(name,[]).append(alerts)
                  pd.DataFrame({'score':scores,'y':y.loc[test],'pre_onset':onset_y.loc[test]}).to_parquet(scores_dir/f'{task_key}_{fold_id}_{name}.parquet')
                  m={'name':name,'calibration':choice,**describe_alerts(alerts,test_events,a,b),
                     'auc':float(roc_auc_score(y.loc[test],scores)) if y.loc[test].nunique()==2 else None,
                     'brier':float(brier_score_loss(y.loc[test],scores))}
                  record['models'].append(m)
              # Baselines use the same decision pool, maturity and6h cooldown.
              for name,score,choice in [('scheduled',pd.Series(1.,index=test),{'threshold':1.}),
                    ('momentum', (1 if direction=='long' else -1)*f.ret4.loc[test],
                     calibrate((1 if direction=='long' else -1)*f.ret4.loc[cal],y))]:
                if choice is None:continue
                alerts=pack_alerts(replay(score,choice['threshold']),y,ids,outcome,f)
                combined.setdefault(name,[]).append(alerts)
                record['models'].append({'name':name,**describe_alerts(alerts,test_events,a,b)})
              task['fold_results'].append(record)
              print(task_key,'fold',fold_id,'complete',flush=True)
            test_start=folds[0]['test'][0];test_end=folds[-1]['test'][1];pool=pd.DatetimeIndex(pools);all_events=pd.DatetimeIndex(sorted(set(event_lists)))
            for name,frames in combined.items():
              alerts=pd.concat(frames).sort_index();alerts.to_parquet(OUT/f'alerts_{task_key}_{name}.parquet')
              item={'name':name,**describe_alerts(alerts,all_events,test_start,test_end)}
              if name not in ('scheduled','momentum'):item['matched_random']=matched_random(alerts,pool,y)
              item['regimes']={regime:{'alerts':len(g),'precision':float(g.y.mean()),'mean_net20bps_pct':float(g.net_20bps.mean()*100)} for regime,g in alerts.groupby('btc_regime')}
              # Concentration check removes3 calendar days with largest absolute NEAR daily close return, without refitting.
              daily=f.close.resample('D').last().pct_change().abs();worst_days=set(daily.loc[test_start:test_end].nlargest(3).index)
              filtered=alerts[~alerts.index.normalize().isin(worst_days)]
              item['excluding_top3_market_days']={'alerts':len(filtered),'precision':float(filtered.y.mean()) if len(filtered) else None}
              task['aggregate'].append(item)
            report['tasks'].append(task)
            (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('Saved',OUT/'report.json',flush=True)


if __name__=='__main__':main()
