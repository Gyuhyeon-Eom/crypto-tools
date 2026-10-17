"""Frozen external-feature ablation against a directly learned structural-stop target."""
from pathlib import Path
import json,hashlib
import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score,brier_score_loss
from features import BASE,OUT,groups,load_verified,build,PRICE,DERIV
from data import fold_windows,split_index,candidates,precursor_labels,zigzag,funding_cumulative,HOUR
from stops import levels,evaluate_stop
from evaluate import calibrate,replay,make_model,weekly_ci


def paired_week_statistics(values,permutations=9999):
    """Positive values mean lower squared error than baseline, evaluated on identical candidates."""
    if values.empty:return None
    week=((values.index-pd.Timestamp('2020-01-06')).days//7).to_numpy()
    g=pd.DataFrame({'week':week,'v':values.to_numpy()}).groupby('week').v.agg(['sum','count']).to_numpy(float)
    rng=np.random.default_rng(29);observed=g[:,0].sum()/g[:,1].sum()
    null=(rng.choice([-1,1],(permutations,len(g)))*g[:,0]).sum(axis=1)/g[:,1].sum()
    sample=g[rng.integers(0,len(g),(3000,len(g)))].sum(axis=1)
    return {'mean_brier_improvement':float(observed),'paired_week95':np.quantile(sample[:,0]/sample[:,1],[.025,.975]).tolist(),
            'week_sign_p':float((1+(null>=observed).sum())/(permutations+1)),'week_blocks':len(g)}


def holm(ps):
    order=np.argsort(ps);out=np.empty(len(ps));running=0.
    for rank,i in enumerate(order):running=max(running,min(1.,(len(ps)-rank)*ps[i]));out[i]=running
    return out


def alert_metrics(a,events,start,end):
    if a.empty:return {'alerts':0}
    n=len(a);hits=int(a.target_before_stop.sum());ci=binomtest(hits,n).proportion_ci(method='exact')
    caught=set(a.event.dropna())&set(events)
    return {'alerts':n,'hits':hits,'precision':hits/n,'exact95':[float(ci.low),float(ci.high)],'block95':weekly_ci(a,'target_before_stop',start,end),
            'mean_net20_pct':float(a.net20.mean()*100),'mean_net10_pct':float(a.net20.mean()*100+.1),
            'net20_block95_pct':[v*100 for v in weekly_ci(a,'net20',start,end)],'median_stop_distance_pct':float(a.stop_distance_pct.median()),
            'stop_rate':float(a.stopped.mean()),'timeout_rate':float(a.timeout.mean()),'ambiguous':int(a.ambiguous.sum()),
            'median_exit_hours':float(a.exit_hours.median()),'pre_onset_precision':float(a.event.notna().mean()),
            'events':len(events),'events_caught':len(caught),'onset_recall':len(caught)/len(events) if len(events) else None,
            'active_weeks':int(a.resample('W').size().gt(0).sum())}


def pack(times,outcome,ids,f):
    a=outcome.reindex(times).copy();a['event']=ids.reindex(times)
    a['regime']=np.select([f.btc_ret7d.reindex(times)>.05,f.btc_ret7d.reindex(times)<-.05],['up','down'],default='flat')
    return a


def main():
    p,manifest,raw,_=load_verified()
    result_path=OUT/'results.json'
    if result_path.exists():raise ValueError('Finished research already exists; preserve outputs')
    f=build();columns=groups();union=sorted(set(sum(columns.values(),[]))-{'stop_distance'})
    complete=np.isfinite(f[union]).all(axis=1);quiet=candidates(f)
    end=f.index[-1];folds=fold_windows(end);funding=funding_cumulative(f,raw['funding'])
    for d in ['models','scores','alerts','outcomes']:(OUT/d).mkdir(exist_ok=True)
    report={'protocol_sha256':manifest['protocol_sha256'],'amendment_sha256':manifest['amendment_sha256'],
            'scope':p['status'],'groups':columns,'folds':[{k:list(map(str,v)) for k,v in x.items()} for x in folds],
            'source_hashes':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in BASE.glob('*.py')},
            'tasks':[]}
    for direction in ('long','short'):
      sl=levels(f,direction);sign=1 if direction=='long' else -1
      f['stop_distance']=sign*(f.close-sl['prior_4h'])/f.close
      initial_eligible=quiet&complete&f.stop_distance.gt(0)
      for target in (.05,.1):
        _,ids,events=precursor_labels(f,zigzag(f,target),direction)
        for hours in (24,72):
          key=f'{direction}_{int(target*100)}_{hours}'
          outcomes={name:evaluate_stop(f,direction,target,hours,s,funding) for name,s in sl.items()}
          for name,o in outcomes.items():o.to_parquet(OUT/'outcomes'/f'{key}_{name}.parquet')
          outcome=outcomes['prior_4h'];y=outcome.target_before_stop
          eligible=initial_eligible&outcome.valid_stop
          task={'key':key,'direction':direction,'target':target,'hours':hours,'primary_stop':'prior_4h','folds':[],'aggregate':[],
                'execution_cancelled_next_open_stop_invalid':int((initial_eligible&~outcome.valid_stop&(f.index+73*HOUR<end)).sum())}
          collected={};predictions={};all_events=[]
          for fold_id,windows in enumerate(folds):
            indices={k:split_index(f,w,eligible) for k,w in windows.items()};train,cal,test=[indices[k] for k in ('train','calibration','test')]
            a,b=windows['test'];test_events=pd.DatetimeIndex(events.loc[(events.onset>=a)&(events.onset+72*HOUR<b),'onset']);all_events.extend(test_events)
            record={'fold':fold_id,'counts':{k:len(v) for k,v in indices.items()},'base_rate':float(y.loc[test].mean()),'models':[]}
            if y.loc[train].nunique()<2 or y.loc[cal].nunique()<2:raise ValueError('Insufficient target class support '+key)
            for kind in ('logistic','boosting'):
              for group,cols in columns.items():
                name=group+'_'+kind;model=make_model(kind);model.fit(f.loc[train,cols],y.loc[train])
                choice=calibrate(pd.Series(model.predict_proba(f.loc[cal,cols])[:,1],index=cal),y)
                if choice is None:raise ValueError('Insufficient calibration alerts '+key+' '+name)
                stem=OUT/'models'/f'{key}_{fold_id}_{name}'
                joblib.dump(model,str(stem)+'.joblib')
                lock={'task':key,'fold':fold_id,'group':group,'kind':kind,'features':cols,'calibration':choice,'protocol_sha256':manifest['protocol_sha256'],
                      'source_hashes':report['source_hashes'],'model_sha256':hashlib.sha256(Path(str(stem)+'.joblib').read_bytes()).hexdigest(),'promotion':'not_approved'}
                Path(str(stem)+'.json').write_text(json.dumps(lock,indent=2))
                # Only after the lock is persisted, predict and evaluate the later test window.
                score=pd.Series(model.predict_proba(f.loc[test,cols])[:,1],index=test)
                prediction=pd.DataFrame({'score':score,'y':y.loc[test]});prediction.to_parquet(OUT/'scores'/f'{key}_{fold_id}_{name}.parquet')
                predictions.setdefault(name,[]).append(prediction)
                times=replay(score,choice['threshold']);alerts=pack(times,outcome,ids,f);collected.setdefault(name,[]).append(alerts)
                m={'name':name,'calibration':choice,**alert_metrics(alerts,test_events,a,b),'brier':float(brier_score_loss(y.loc[test],score)),
                   'auc':float(roc_auc_score(y.loc[test],score)) if y.loc[test].nunique()==2 else None}
                record['models'].append(m)
            scheduled=pack(replay(pd.Series(1.,index=test),1.),outcome,ids,f);collected.setdefault('scheduled',[]).append(scheduled)
            record['models'].append({'name':'scheduled',**alert_metrics(scheduled,test_events,a,b)})
            task['folds'].append(record);print(key,'fold',fold_id,'done',flush=True)
          overall_start=folds[0]['test'][0];overall_end=folds[-1]['test'][1];all_events=pd.DatetimeIndex(sorted(set(all_events)))
          combined_scores={name:pd.concat(v).sort_index() for name,v in predictions.items()}
          for name,frames in collected.items():
            alerts=pd.concat(frames).sort_index();alerts.to_parquet(OUT/'alerts'/f'{key}_{name}.parquet')
            m={'name':name,**alert_metrics(alerts,all_events,overall_start,overall_end),'stop_sensitivity':[]}
            if name!='scheduled':
                scores=combined_scores[name];m['candidate_count']=len(scores);m['brier']=float(brier_score_loss(scores.y,scores.score));m['auc']=float(roc_auc_score(scores.y,scores.score))
                if not name.startswith('baseline_'):
                    kind=name.rsplit('_',1)[1];control=combined_scores['baseline_'+kind]
                    if not scores.index.equals(control.index) or not scores.y.equals(control.y):raise ValueError('Nonpaired comparison')
                    m['incremental']=paired_week_statistics((control.y-control.score)**2-(scores.y-scores.score)**2)
                    m['incremental']['fold_brier_improvements']=[next(x['brier'] for x in fold['models'] if x['name']=='baseline_'+kind)-next(x['brier'] for x in fold['models'] if x['name']==name) for fold in task['folds']]
            for policy,o in outcomes.items():
                tested=pack(alerts.index,o,ids,f);tested=tested[tested.valid_stop]
                sensitivity={'policy':policy,'original_alerts':len(alerts),**alert_metrics(tested,all_events,overall_start,overall_end),'folds':[]}
                for fold in folds:
                    a,b=fold['test'];part=tested.loc[(tested.index>=a)&(tested.index<b)]
                    sensitivity['folds'].append({'alerts':len(part),'precision':float(part.target_before_stop.mean()) if len(part) else None,'mean_net20_pct':float(part.net20.mean()*100) if len(part) else None})
                m['stop_sensitivity'].append(sensitivity)
            m['regimes']={reg:{'alerts':len(a),'precision':float(a.target_before_stop.mean()),'mean_net20_pct':float(a.net20.mean()*100)} for reg,a in alerts.groupby('regime')}
            task['aggregate'].append(m)
          report['tasks'].append(task)
          (OUT/'progress.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    items=[m['incremental'] for t in report['tasks'] for m in t['aggregate'] if 'incremental' in m]
    for x,p_adjusted in zip(items,holm([x['week_sign_p'] for x in items])):x['holm_p']=float(p_adjusted)
    report['multiple_comparison_count']=len(items)
    result_path.write_text(json.dumps(report,indent=2,allow_nan=False));print('Saved',result_path,flush=True)

if __name__=='__main__':main()
