"""Onset-aligned continuation after the same first-stage activity trigger."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from study import OUT,RAW,FEATURES,tape_features,comparison,source_hashes


def main():
    frozen=json.loads((OUT/'frozen.json').read_text());assert frozen['source_hashes']==source_hashes()
    selected=pd.read_parquet(OUT/'selected.parquet');cache={};bars={};windows={};rows=[]
    for group,g in selected.groupby('group'):
        case=g[g.role.eq('case')].iloc[0]
        lead=(case.next_event-case.time).total_seconds()/60
        for before in [15,5]:
            if lead<before:continue
            elapsed=lead-before
            for row in g.itertuples():
                t=row.time+pd.Timedelta(minutes=elapsed);start=t-pd.Timedelta(minutes=60);key=(row.symbol,t)
                if key not in windows:
                    pieces=[];expected=0.;missing=False
                    for day in pd.date_range(start.floor('D'),(t-pd.Timedelta(nanoseconds=1)).floor('D')):
                        stem=f'{row.symbol}_{day:%Y-%m-%d}';path=RAW/f'{stem}.csv.gz'
                        if not path.exists():missing=True;break
                        if stem not in cache:
                            cache[stem]=pd.read_csv(path,compression='gzip',usecols=['timestamp','price','volume','side'])
                            bars[stem]=pd.read_parquet(RAW.parent/'bars'/f'{stem}.parquet')
                        pieces.append(cache[stem]);b=bars[stem]
                        expected+=b.loc[(b.bucket>=start)&(b.bucket<t),'quote'].sum()
                    windows[key]={'quality_error':'Missing source'} if missing else tape_features(pd.concat(pieces,ignore_index=True),t,expected)
                rows.append({**row._asdict(),'checkpoint':f'minus{before}','observed_at':t,'elapsed_minutes':elapsed,
                             'case_onset':case.next_event,**windows[key]})
    d=pd.DataFrame(rows)
    if 'quality_error' not in d:d['quality_error']=None
    d['group_valid']=d.groupby(['checkpoint','group']).quality_error.transform(lambda s:s.isna().all())
    d.to_parquet(OUT/'monitor_features.parquet',index=False);d.to_csv(OUT/'monitor_features.csv',index=False)
    result={}
    for period in ['discovery','replication']:
        result[period]={}
        for checkpoint in ['minus15','minus5']:
            part=d[d.split.eq(period)&d.checkpoint.eq(checkpoint)&d.group_valid]
            result[period][checkpoint]={'matched_sets':part.group.nunique(),
                'scores':{k:comparison(part,k,frozen['discovery'][k]['direction'],alpha=.05/18) for k in FEATURES}}
    (OUT/'monitor_results.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
