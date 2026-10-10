"""Pre-register, fit before test, then evaluate a paired prospective ablation."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from dataset import BASE, MICRO, TARGETS, legacy, sha, utc

SOURCES = ['protocol.json','universe.json','dataset.py','study.py','fetch_prices.py','../orderbook_lab/engine.mjs',
           '../orderbook_lab/capture.mjs','../precision_frontier/study.py',
           '../continuation_controls/study.py','../event_census/study.py']
MODELS = {'context3':legacy.CONTEXT, 'context3_micro':legacy.CONTEXT+MICRO}
UNIVERSE = json.loads((BASE/'universe.json').read_text())['symbols']


def hashes():
    return {p:sha(BASE/p) for p in SOURCES}


def register(start, destination, now=None):
    now=utc(now or pd.Timestamp.now(tz='UTC'));start=utc(start)
    if start <= now or start != start.floor('h'):
        raise ValueError('Registration start must be a future UTC whole hour')
    train_end=start+pd.Timedelta(days=28)
    cal_start=train_end+pd.Timedelta(hours=48)
    cal_end=cal_start+pd.Timedelta(days=14)
    test_start=cal_end+pd.Timedelta(hours=48)
    test_end=test_start+pd.Timedelta(days=56)
    result={'registered_at':now.isoformat(),'symbols':UNIVERSE,'sources':hashes(),
            'train':[start.isoformat(),train_end.isoformat()],
            'calibration':[cal_start.isoformat(),cal_end.isoformat()],
            'evaluation':[test_start.isoformat(),test_end.isoformat()],
            'evaluation_mature_at':(test_end+pd.Timedelta(hours=30)).isoformat(),
            'state':'registered_not_scheduled'}
    with Path(destination).open('x') as f:json.dump(result,f,indent=2)
    return result


def validate_registration(reg):
    if reg['sources'] != hashes():raise ValueError('Registered source code changed; use a separately registered version')
    if reg['symbols'] != UNIVERSE:raise ValueError('Unexpected universe')
    if utc(reg['registered_at']) >= utc(reg['train'][0]):raise ValueError('Registration was not prospective')


def split(data, reg, name):
    start,end=map(utc,reg[name])
    return data[data.time.ge(start)&data.time.lt(end)&data.symbol.isin(reg['symbols'])].copy()


def pool(part):
    return part[part.base_candidate & part.micro_eligible & part.label_status.eq('mature')].copy()


def support(data):
    p=pool(data)
    positives=p[p.hit.fillna(False)]
    events=len(positives[['symbol','target_event']].drop_duplicates())
    return {'mature_paired_candidates':len(p),'positive_target_events':events,
            'classes':int(p.hit.nunique()),'base_candidates':int(data.base_candidate.sum()),
            'base_candidates_without_micro':int((data.base_candidate&~data.micro_eligible).sum())}


def complete_grid(part, bounds, symbols):
    expected=pd.MultiIndex.from_product([symbols,pd.date_range(utc(bounds[0]),utc(bounds[1]),freq='h',inclusive='left')])
    if part.duplicated(['symbol','time']).any():return False
    actual=pd.MultiIndex.from_frame(part[['symbol','time']])
    return len(expected.difference(actual))==0 and part.label_status.eq('mature').all()


def fit(data, reg, now=None):
    validate_registration(reg)
    now=utc(now or pd.Timestamp.now(tz='UTC'))
    train,cal=split(data,reg,'train'),split(data,reg,'calibration')
    readiness={'train':support(train),'calibration':support(cal)}
    errors=[]
    if now < utc(reg['calibration'][1])+pd.Timedelta(hours=30):errors.append('calibration outcomes not yet mature')
    if now >= utc(reg['evaluation'][0]):errors.append('model must be locked before evaluation starts')
    for name,part,minimum,events in [('train',train,1000,20),('calibration',cal,200,10)]:
        if not complete_grid(part,reg[name],reg['symbols']):errors.append(name+': incomplete hourly price/label grid')
        s=readiness[name]
        if s['mature_paired_candidates']<minimum or s['positive_target_events']<events or s['classes']!=2:
            errors.append(name+': insufficient predeclared sample support')
    if errors:return {'status':'insufficient_data','reasons':errors,'support':readiness,'models':{}}
    train,cal=pool(train),pool(cal)
    models={}
    for name,columns in MODELS.items():
        pipe=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=20261010))
        pipe.fit(train[columns],train.hit.astype(int))
        scaler,model=pipe.steps[0][1],pipe.steps[1][1]
        scores=pipe.predict_proba(cal[columns])[:,1]
        models[name]={'columns':columns,'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
                      'coef':model.coef_[0].tolist(),'intercept':float(model.intercept_[0]),
                      'threshold':float(np.quantile(scores,.8))}
    return {'status':'locked','locked_at':now.isoformat(),'registration':reg,'support':readiness,'models':models}


def score(part, model):
    x=(part[model['columns']].to_numpy(dtype=float)-np.array(model['mean']))/np.array(model['scale'])
    z=x@np.array(model['coef'])+model['intercept']
    return 1/(1+np.exp(-np.clip(z,-700,700)))


def paired_interval(alerts, start, end):
    start,end=utc(start),utc(end)
    weeks=int(np.ceil((end-start)/pd.Timedelta(days=7)))
    totals=[]
    for name in MODELS:
        a=alerts[name].copy();a['week']=((a.time-start)/pd.Timedelta(days=7)).astype(int)
        g=a.groupby('week').hit.agg(['sum','count']).reindex(range(weeks),fill_value=0).to_numpy(dtype=float)
        totals.append(g)
    rng=np.random.default_rng(20261010)
    indexes=rng.integers(0,weeks,(4000,weeks))
    a,b=[g[indexes].sum(axis=1) for g in totals]
    good=(a[:,1]>0)&(b[:,1]>0)
    if not good.any():return {'lower':None,'upper':None,'valid_draws':0}
    difference=b[good,0]/b[good,1]-a[good,0]/a[good,1]
    return {'lower':float(np.quantile(difference,.025)),'upper':float(np.quantile(difference,.975)),'valid_draws':int(good.sum())}


def evaluate(data, lock, now=None):
    if lock.get('status')!='locked':return {'status':'not_fitted','precision':None}
    reg=lock['registration'];validate_registration(reg)
    now=utc(now or pd.Timestamp.now(tz='UTC'))
    if utc(lock['locked_at'])>=utc(reg['evaluation'][0]):raise ValueError('Model locked after test started')
    if now<utc(reg['evaluation_mature_at']):return {'status':'pending_outcomes','precision':None}
    part=split(data,reg,'evaluation')
    if not complete_grid(part,reg['evaluation'],reg['symbols']):return {'status':'incomplete_price_labels','precision':None}
    candidates=pool(part)
    if candidates.empty:return {'status':'no_covered_candidates','precision':None,'support':support(part)}
    alerts={};results={}
    for name,model in lock['models'].items():
        p=candidates.copy();p['score']=score(p,model)
        # Replay only sees timestamp, symbol and score; join outcomes afterwards.
        a=p.loc[legacy.replay(p[['time','symbol','score']],model['threshold'])].copy()
        alerts[name]=a
        results[name]=legacy.metrics(part,p,a)
        results[name]['passes_80pct_evidence_gate']=legacy.evidence_gate(results[name])
    baseline=results['context3']['alerts'];augmented=results['context3_micro']['alerts']
    ratio=augmented/baseline if baseline else None
    comparable=ratio is not None and .8<=ratio<=1.25
    interval=paired_interval(alerts,*reg['evaluation'])
    return {'status':'evaluated','models':results,'support':support(part),'alert_count_ratio':ratio,
            'comparable_alert_budget':comparable,'paired_precision_difference_95pct':interval,
            'incremental_benefit_supported':bool(comparable and interval['lower'] is not None and interval['lower']>0
                and all(m['alerts']>=100 and m['active_weeks']>=8 for m in results.values())),
            'scope':'Fixed original114-coin universe; report missing-book coverage explicitly.'}


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    r=sub.add_parser('register');r.add_argument('--start',required=True);r.add_argument('--output',type=Path,required=True)
    for command in ['fit','evaluate']:
        p=sub.add_parser(command);p.add_argument('--data',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='register':result=register(args.start,args.output)
    else:
        data=pd.read_parquet(args.data);metadata=json.loads(args.input.read_text())
        result=fit(data,metadata) if args.command=='fit' else evaluate(data,metadata)
        result['dataset_sha256']=sha(args.data)
        with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
