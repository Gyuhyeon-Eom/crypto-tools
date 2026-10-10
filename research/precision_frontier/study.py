"""Full-frequency precision frontier, thresholds fixed using June only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE=Path(__file__).resolve().parent
OUT=BASE/'output'
spec=importlib.util.spec_from_file_location('hard',BASE.parent/'continuation_controls/study.py')
hard=importlib.util.module_from_spec(spec);spec.loader.exec_module(hard)
CONTEXT=['prior_runup','log_volume','near_high']
FULL=CONTEXT+['price_efficiency','volume_acceleration']
QUANTILES=[0,.5,.8,.9,.95,.99,.995,.999]
MODELS=['volume','efficiency','context3','full5']


def freeze():
 paths=['study.py','protocol.json','../continuation_controls/study.py','../event_census/study.py',
        '../microstructure/output/bybit.parquet']
 hashes={p:hashlib.sha256((BASE/p).read_bytes()).hexdigest() for p in paths}
 target=OUT/'source_manifest.json'
 if target.exists():assert json.loads(target.read_text())['hashes']==hashes
 else:target.write_text(json.dumps({'at':pd.Timestamp.now(tz='UTC').isoformat(),'hashes':hashes},indent=2))


def outcomes(f):
 f=f.copy()
 event_times=f.index[f.event]
 # Ignore onset at the observation itself. Only genuinely later events can count.
 positions=event_times.searchsorted(f.index,side='right')
 target=pd.Series(pd.NaT,index=f.index,dtype='datetime64[ns]')
 valid=positions<len(event_times)
 target.loc[valid]=event_times.take(positions[valid]).to_numpy()
 lead=(target-pd.Series(f.index,index=f.index)).dt.total_seconds()/3600
 f['hit']=lead.between(1,6)
 f['target_event']=target.where(f.hit)
 f['lead_hours']=lead.where(f.hit)
 f['label_complete']=f[['open','high','low','close']].notna().all(axis=1).rolling(31).min().shift(-30).eq(1)
 lows=np.column_stack([f.low.shift(-i) for i in range(1,25)])
 f['min24']=np.minimum(0,np.min(lows,axis=1)/f.close-1)
 return f


def windows(d,name):
 if name=='train':return d[d.time.ge('2026-04-02')&(d.time+pd.Timedelta(hours=30)<pd.Timestamp('2026-06-01'))]
 if name=='calibration':return d[d.time.ge('2026-06-01')&(d.time+pd.Timedelta(hours=30)<pd.Timestamp('2026-07-01'))]
 return d[d.time.between('2026-08-01','2026-09-30 23:00')]


def candidate_mask(d):
 return d.symbol.ne('BTCUSDT')&d.past_complete&d.prior_runup.ge(.05)&np.isfinite(d[FULL]).all(axis=1)


def replay(frame,threshold):
 """Accept only known candidate timestamps, identifiers and scores, not outcomes."""
 d=frame[['time','symbol','score']].sort_values(['time','symbol'])
 chosen=[];last={}
 for row in d.itertuples():
  if row.score<threshold:continue
  if row.symbol in last and row.time-last[row.symbol]<pd.Timedelta(hours=24):continue
  chosen.append(row.Index);last[row.symbol]=row.time
 return chosen


def event_keys(d):return set(zip(d.symbol,d.target_event))


def uncertainty(a):
 if len(a)==0:return {'active_weeks':0,'block_lower':None,'block_upper':None,'iid_lower':None,'iid_upper':None}
 x=a.copy();x['week']=(x.time-pd.Timestamp('2026-01-01')).dt.days//7
 b=x.groupby('week').hit.agg(['sum','count']).to_numpy(dtype=float)
 rng=np.random.default_rng(20261010)
 sampled=b[rng.integers(0,len(b),(4000,len(b)))].sum(axis=1)
 ratios=sampled[:,0]/sampled[:,1]
 ci=binomtest(int(a.hit.sum()),len(a)).proportion_ci(confidence_level=.95,method='exact')
 return {'active_weeks':len(b),'block_lower':float(np.quantile(ratios,.025)),
         'block_upper':float(np.quantile(ratios,.975)),'iid_lower':float(ci.low),'iid_upper':float(ci.high)}


def metrics(part,pool,a):
 evaluable=a[a.label_complete];hits=evaluable[evaluable.hit]
 all_events=set(zip(part.loc[part.event&part.symbol.ne('BTCUSDT'),'symbol'],part.loc[part.event&part.symbol.ne('BTCUSDT'),'time']))
 reachable=event_keys(pool[pool.hit&pool.label_complete])&all_events
 found=event_keys(hits)&all_events
 # Recall excludes anchors outside this split, but alert precision evaluates all mature labels.
 days=(part.time.max()-part.time.min()).total_seconds()/86400+1/24
 return {'alerts_generated':len(a),'alerts':len(evaluable),'unscorable':len(a)-len(evaluable),
         'hits':int(evaluable.hit.sum()),'precision':float(evaluable.hit.mean()) if len(evaluable) else None,
         'alerts_per_day':len(a)/days,'all_events':len(all_events),'reachable_events':len(reachable),
         'detected_events':len(found),'all_recall':len(found)/len(all_events) if all_events else 0.,
         'reachable_recall':len(found)/len(reachable) if reachable else 0.,
         'median_lead_hours':float(hits.lead_hours.median()) if len(hits) else None,
         'median_return24':float(evaluable.return24.median()) if len(evaluable) else None,
         'down5_24h':int(evaluable.min24.le(-.05).sum()),**uncertainty(evaluable)}


def select_policy(rows):
 eligible=[r for r in rows if r['alerts']>=30 and r['all_recall']>=.10]
 if not eligible:return None
 return sorted(eligible,key=lambda r:(-r['precision'],-r['all_recall'],r['alerts'],r['model'],r['threshold']))[0]['policy_id']


def evidence_gate(m):
 return bool(m['precision'] is not None and m['precision']>=.8 and m['alerts']>=100 and
             m['all_recall']>=.10 and m['active_weeks']>=8 and m['block_lower']>=.8 and m['iid_lower']>=.8)


def main():
 OUT.mkdir(exist_ok=True);freeze()
 raw=pd.read_parquet(hard.census.SOURCE)
 d=pd.concat([outcomes(hard.census.labels(hard.features(g))).reset_index(drop=True) for _,g in raw.groupby('symbol')],ignore_index=True)
 d['log_volume']=np.log(d.volume_build.where(d.volume_build>0))
 parts={s:windows(d,s) for s in ['train','calibration','evaluation']}
 pools={s:p[candidate_mask(p)].copy() for s,p in parts.items()}
 train=pools['train'];train=train[train.label_complete]
 classifiers={};metadata={}
 for model,columns in [('context3',CONTEXT),('full5',FULL)]:
  pipe=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=20261010))
  pipe.fit(train[columns],train.hit)
  classifiers[model]=(pipe,columns)
  scaler,reg=pipe.steps[0][1],pipe.steps[1][1]
  metadata[model]={'columns':columns,'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
                   'coef':reg.coef_[0].tolist(),'intercept':reg.intercept_.tolist()}
 def scores(p,model):
  if model=='volume':return p.volume_build.to_numpy()
  if model=='efficiency':return p.price_efficiency.to_numpy()
  pipe,cols=classifiers[model];return pipe.predict_proba(p[cols])[:,1]
 allrows=[];policies=[];alerts=[]
 cal=pools['calibration']
 for model in MODELS:
  p=cal.copy();p['score']=scores(p,model)
  seen=set()
  for q in QUANTILES:
   threshold=float(p.score.quantile(q))
   if threshold in seen:continue
   seen.add(threshold)
   pid=f'{model}_q{q:g}'
   policy={'policy_id':pid,'model':model,'calibration_quantile':q,'threshold':threshold}
   policies.append(policy)
   indexes=replay(p,threshold);a=p.loc[indexes].copy();a['policy_id']=pid;a['split']='calibration';alerts.append(a)
   allrows.append({**policy,'split':'calibration',**metrics(parts['calibration'],p,a)})
 selected=select_policy(allrows)
 lock={'locked_at':pd.Timestamp.now(tz='UTC').isoformat(),'selected_policy':selected,'policies':policies,
       'models':metadata,'train_candidates':len(train),'train_positive_rows':int(train.hit.sum()),
       'note':'Locked before evaluation scoring; selected with calibration only. Historical periods previously inspected.'}
 (OUT/'model_lock.json').write_text(json.dumps(lock,indent=2,allow_nan=False))
 # Evaluation only starts after writing fixed thresholds, coefficients and selection.
 for model in MODELS:
  p=pools['evaluation'].copy();p['score']=scores(p,model)
  for policy in [z for z in policies if z['model']==model]:
   a=p.loc[replay(p,policy['threshold'])].copy();a['policy_id']=policy['policy_id'];a['split']='evaluation';alerts.append(a)
   allrows.append({**policy,'split':'evaluation',**metrics(parts['evaluation'],p,a)})
 result=pd.DataFrame(allrows);result.to_csv(OUT/'frontier.csv',index=False)
 cols=['policy_id','split','time','symbol','score','hit','label_complete','target_event','lead_hours','return24','min24']
 pd.concat(alerts)[cols].to_csv(OUT/'alerts.csv',index=False)
 summary={'candidate_rows':{s:len(p) for s,p in pools.items()},
          'candidate_positive_row_rate':{s:float(p.loc[p.label_complete,'hit'].mean()) for s,p in pools.items()},
          'selected_policy':selected,'selected_results':[r for r in allrows if r['policy_id']==selected],
          'evaluation_80pct_rows':[r for r in allrows if r['split']=='evaluation' and r['precision'] is not None and r['precision']>=.8],
          'selected_passes_evidence_gate':any(evidence_gate(r) for r in allrows if r['split']=='evaluation' and r['policy_id']==selected),
          'warning':'These are4 tested models,not a theoretical upper bound; no uninspected final test period.'}
 (OUT/'results.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
 print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
