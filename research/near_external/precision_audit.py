"""Secondary post-fit precision audit across ALL added-group comparisons, no refit."""
import json
import numpy as np
import pandas as pd
from features import OUT
from diagnostics import calendar_sums
from train import holm

def main():
    r=json.loads((OUT/'results.json').read_text());start=pd.Timestamp(r['folds'][0]['test'][0]);end=pd.Timestamp(r['folds'][-1]['test'][1]);rows=[]
    for t in r['tasks']:
      for model in t['aggregate']:
        name=model['name']
        if name=='scheduled' or name.startswith('baseline_'):continue
        kind=name.rsplit('_',1)[1]
        a=pd.read_parquet(OUT/'alerts'/f"{t['key']}_{name}.parquet")
        b=pd.read_parquet(OUT/'alerts'/f"{t['key']}_baseline_{kind}.parquet")
        x=calendar_sums(a,'target_before_stop',start,end);y=calendar_sums(b,'target_before_stop',start,end)
        rng=np.random.default_rng(41);swap=rng.integers(0,2,(9999,len(x))).astype(bool)
        xx=np.where(swap[:,:,None],x[None,:,:],y[None,:,:]).sum(axis=1)
        yy=np.where(swap[:,:,None],y[None,:,:],x[None,:,:]).sum(axis=1)
        null=xx[:,0]/xx[:,1]-yy[:,0]/yy[:,1]
        observed=float(a.target_before_stop.mean()-b.target_before_stop.mean())
        rows.append({'task':t['key'],'name':name,'precision_difference':observed,'week_swap_p':float((1+(null>=observed).sum())/10000)})
    for x,p in zip(rows,holm([x['week_swap_p'] for x in rows])):x['holm_p']=float(p)
    (OUT/'precision_audit.json').write_text(json.dumps({'status':'Secondary exploratory analysis specified after inspecting aggregate results; all112added-group comparisons included. No model or threshold changes.',
        'assumption':'Calendar-week exchangeability of model alert outcomes/budgets, not causal assignment. Swapping preserves within-week clusters but can change aggregate alert counts. Do not treat as prospective confirmation.', 'comparisons':rows},indent=2))
    print('Secondary precision Holm<.05:',[(x['task'],x['name'],x['holm_p']) for x in rows if x['holm_p']<.05])
    print('Chain short5 72 boosting:',next(x for x in rows if x['task']=='short_5_72' and x['name']=='chain48_boosting'))
if __name__=='__main__':main()
