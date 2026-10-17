"""Increase Monte Carlo resolution uniformly, without changing any fitted model."""
import json
import pandas as pd
from data import precursor_labels
from evaluate import OUT, matched_random
from antecedents import holm


def main():
    r=json.loads((OUT/'report.json').read_text());f=pd.read_parquet(OUT/'features.parquet');events=pd.read_parquet(OUT/'events.parquet')
    results=[]
    for task in r['tasks']:
        if task['type']=='onset':y,_,_=precursor_labels(f,events[events.target==task['target']],task['direction'])
        else:y=pd.read_parquet(OUT/f"outcome_{task['direction']}_{task['target']}_{task['hours']}.parquet").touch
        for model in task['aggregate']:
            if model['name'] in ('scheduled','momentum'):continue
            name=model['name'];alerts=pd.read_parquet(OUT/f"alerts_{task['key']}_{name}.parquet")
            pools=[pd.read_parquet(path).index for path in sorted((OUT/'scores').glob(f"{task['key']}_*_{name}.parquet"))]
            pool=pools[0].append(pools[1:]).unique().sort_values()
            comparison=matched_random(alerts,pool,y,repeats=1999)
            results.append({'task':task['key'],'model':name,**comparison})
        print(task['key'],'random timing complete',flush=True)
    ps=[x.get('one_sided_p',1.) for x in results]
    for x,p in zip(results,holm(ps)):x['holm_p']=float(p)
    (OUT/'null_audit.json').write_text(json.dumps({'repetitions':1999,'comparisons':results,
        'limits':'Exploratory same-week count and6h spacing matched greedy randomization; not uniform over all valid alert sets and not a causal randomization test. Failed complete-budget draws discarded and counted. Holm adjusts all40 fitted variants; no model or threshold changes.'},indent=2))
    print('Saved null_audit.json',flush=True)
if __name__=='__main__':main()
