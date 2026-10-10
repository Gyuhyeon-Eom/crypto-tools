"""Same-strength, same-activity controls for continuation surge events."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parent
OUT=BASE/'output'
spec=importlib.util.spec_from_file_location('census',BASE.parent/'event_census/study.py')
census=importlib.util.module_from_spec(spec);spec.loader.exec_module(census)
FEATURES=['volume_acceleration','price_efficiency','close_retention_change']


def freeze():
 paths=['study.py','protocol.json','../event_census/study.py','../event_census/output/frozen.json','../microstructure/output/bybit.parquet']
 hashes={p:hashlib.sha256((BASE/p).read_bytes()).hexdigest() for p in paths}
 target=OUT/'frozen.json'
 if target.exists():assert json.loads(target.read_text())['hashes']==hashes
 else:target.write_text(json.dumps({'at':pd.Timestamp.now(tz='UTC').isoformat(),'hashes':hashes},indent=2))


def features(g):
 f=census.price_features(g)
 recent=f.turnover.rolling(2).mean()
 prior=f.turnover.rolling(4).mean().shift(2)
 f['volume_acceleration']=np.log(recent.where(recent>0)/prior.where(prior>0))
 ret=np.log(f.close/f.close.shift(1))
 f['price_efficiency']=ret.rolling(6).sum()/ret.abs().rolling(6).sum().replace(0,np.nan)
 giveback=f.high-f.close
 scale=(f.high-f.low).rolling(168).mean().shift(6).replace(0,np.nan)
 f['close_retention_change']=(giveback.rolling(3).mean().shift(3)-giveback.rolling(3).mean())/scale
 f[FEATURES]=f[FEATURES].replace([np.inf,-np.inf],np.nan)
 return f


def baseline_cov(d):
 return np.column_stack([np.log(d.match_vol.clip(lower=1e-10)),np.log(d.match_liquidity.clip(lower=1e-10))])


def choose_controls(case,pool,scale):
 p=pool[pool.symbol.eq(case.symbol)&pool.past_complete&pool.future_complete&pool.prior_runup.ge(.05)&pool.max24.lt(.15)&~pool.event_nearby].copy()
 distance=(p.time-case.time).abs()
 hour=(p.time.dt.hour-case.time.hour).abs()
 p=p[distance.between(pd.Timedelta(hours=48),pd.Timedelta(days=28))&(np.minimum(hour,24-hour)<=4)]
 ret_tolerance=max(.03,.25*case.prior_runup)
 p['return_gap']=p.prior_runup-case.prior_runup
 p['volume_log_gap']=np.log(p.volume_build/case.volume_build)
 p['high_gap']=p.near_high-case.near_high
 p=p[p.return_gap.abs().le(ret_tolerance)&p.volume_log_gap.abs().le(np.log(1.5))&p.high_gap.abs().le(.05)]
 b=(baseline_cov(p)-baseline_cov(pd.DataFrame([case._asdict()])))/scale
 p['baseline_distance']=np.linalg.norm(b,axis=1)
 p['match_distance']=np.sqrt((p.return_gap/ret_tolerance)**2+(p.volume_log_gap/np.log(1.5))**2+(p.high_gap/.05)**2+(b*b).sum(axis=1))
 p=p[p.baseline_distance.le(3)].sort_values(['match_distance','time'])
 p['day']=p.time.dt.floor('D')
 return p.drop_duplicates('day').head(3)


def match(d,split,scale):
 p=census.part(d,split)
 cases=p[p.event&p.prior_runup.ge(.05)&p.symbol.ne('BTCUSDT')]
 rows=[];drops=[]
 for c in cases.itertuples(index=False):
  controls=choose_controls(c,p,scale)
  if len(controls)==0:
   drops.append({'symbol':c.symbol,'time':str(c.time),'prior_runup':c.prior_runup,'volume_build':c.volume_build,'near_high':c.near_high})
   continue
  group=f'{c.symbol}:{c.time.isoformat()}'
  for role,row in [('case',c._asdict())]+[('control',x._asdict()) for x in controls.itertuples(index=False)]:
   rows.append({**row,'group':group,'anchor':c.time,'role':role,'split':split,'control_count':len(controls)})
 m=pd.DataFrame(rows)
 return m,{'original_cases':len(cases),'matched_cases':len(cases)-len(drops),'dropped':drops}


def compare(m,key):
 pairs=[]
 for group,g in m.groupby('group'):
  if not np.isfinite(g[key]).all():continue
  case=g[g.role.eq('case')][key].iloc[0]
  controls=g[g.role.eq('control')][key].to_numpy()
  pairs.append({'group':group,'time':g.anchor.iloc[0],'value':float(((case>controls)+.5*(case==controls)).mean()),
                'difference':float(case-np.median(controls))})
 p=pd.DataFrame(pairs)
 if p.empty:return {'n':0},p
 p['block']=(p.time-pd.Timestamp('2026-01-01')).dt.days//7
 b=p.groupby('block').value.agg(['sum','count']).to_numpy()
 rng=np.random.default_rng(20261010)
 boot=b[rng.integers(0,len(b),(10000,len(b)))].sum(axis=1)
 boot=boot[:,0]/boot[:,1]
 z=m[m.group.isin(p.group)]
 result={'n':len(p),'blocks':len(b),'concordance':float(p.value.mean()),
         'lower':float(np.quantile(boot,.05/3/2)),'upper':float(np.quantile(boot,1-.05/3/2)),
         'case_median':float(z[z.role.eq('case')][key].median()),
         'control_median':float(z[z.role.eq('control')][key].median()),
         'median_paired_difference':float(p.difference.median())}
 result['passes_gate']=result['n']>=20 and result['blocks']>=6 and result['concordance']>=.6 and result['lower']>.5
 return result,p


def main():
 OUT.mkdir(exist_ok=True);freeze()
 raw=pd.read_parquet(census.SOURCE)
 d=pd.concat([census.labels(features(g)).reset_index(drop=True) for _,g in raw.groupby('symbol')],ignore_index=True)
 early=census.part(d,'discovery')
 early=early[early.symbol.ne('BTCUSDT')&early.past_complete&early.prior_runup.ge(.05)]
 scale=np.nanstd(baseline_cov(early),axis=0)
 result={'scale':scale.tolist(),'periods':{}};all_matches=[];pairs=[]
 original=pd.read_csv(BASE.parent/'event_census/output/events.csv',parse_dates=['time'])
 for split in ['discovery','replication']:
  p=census.part(d,split)
  found=p[p.event&p.prior_runup.ge(.05)&p.symbol.ne('BTCUSDT')]
  expected=original[original.split.eq(split)&original.prior_runup.ge(.05)]
  assert set(zip(found.symbol,found.time))==set(zip(expected.symbol,expected.time))
  m,audit=match(d,split,scale);all_matches.append(m)
  cases=m[m.role.eq('case')];controls=m[m.role.eq('control')]
  audit['control_counts']=cases.control_count.value_counts().to_dict()
  audit['control_rows']=len(controls);audit['unique_controls']=len(controls.drop_duplicates(['symbol','time']))
  audit['matched_coins']=cases.symbol.nunique()
  audit['balance']={key:{'case_median':float(cases[key].median()),'control_median':float(controls[key].median())}
                    for key in ['prior_runup','volume_build','near_high','match_vol','match_liquidity']}
  stats={}
  for key in FEATURES:
   stats[key],p=compare(m,key);p['split']=split;p['feature']=key;pairs.append(p)
  result['periods'][split]={'audit':audit,'features':stats}
 result['candidates']=[key for key in FEATURES if all(result['periods'][s]['features'][key].get('passes_gate',False) for s in ['discovery','replication'])]
 pd.concat(all_matches).to_csv(OUT/'matched.csv',index=False)
 pd.concat(pairs).to_csv(OUT/'paired_scores.csv',index=False)
 (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
 print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
