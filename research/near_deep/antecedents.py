"""Exploratory discovery/replication of antecedents with matched controls and week blocks."""
import json
import numpy as np
import pandas as pd
from data import PRICE, DERIV, HOUR, candidates
from evaluate import OUT


def pairs(f,events,start,end):
    eligible=candidates(f)
    pool=f.index[eligible&(f.index>=start)&(f.index+73*HOUR<end)]
    # Control observations must be away from retrospective onsets, both directions.
    for t in events.onset:
        pool=pool[(pool<t-6*HOUR)|(pool>t+6*HOUR)]
    used=set();rows=[];eligible_events=0
    for _,event in events[(events.onset>=start)&(events.confirmed<end)].iterrows():
        if event.confirmed-event.onset>72*HOUR:continue
        t=(event.onset-pd.Timedelta(minutes=15)).floor('h')
        if t not in f.index or t<start or t+73*HOUR>=end or not eligible.loc[t]:continue
        eligible_events+=1
        available=pool[(pool.hour==t.hour)&(abs(pool-t)<=pd.Timedelta(days=28))&~pool.isin(used)]
        if len(available)==0:continue
        ratio=f.rv24.loc[available]/f.loc[t,'rv24']
        available=available[ratio.between(.5,2)]
        if len(available)==0:continue
        distance=abs(np.log(f.rv24.loc[available]/f.loc[t,'rv24']))+.1*abs((available-t).total_seconds())/(28*86400)
        control=distance.idxmin();used.add(control)
        rows.append({'direction':event.direction,'onset':event.onset,'case':t,'control':control})
    return pd.DataFrame(rows,columns=['direction','onset','case','control']),eligible_events


def holm(values):
    order=np.argsort(values);out=np.empty(len(values));running=0
    for rank,i in enumerate(order):
        running=max(running,min(1,(len(values)-rank)*values[i]));out[i]=running
    return out


def main():
    f=pd.read_parquet(OUT/'features.parquet');events=pd.read_parquet(OUT/'events.parquet')
    end=f.index[-1];start=f.index[0]+pd.Timedelta(days=7)
    windows={'discovery':(start,start+pd.Timedelta(days=360)), 'replication':(end-pd.Timedelta(days=270),end)}
    report={'status':'Exploratory matched case-control, not alert precision. Control exclusion uses future labels solely for descriptive comparison. Direction of each feature hypothesis comes from discovery only. Weekly sign flips preserve within-week dependence; Holm corrects all96 hypotheses. No causal claims.', 'windows':{k:list(map(str,v)) for k,v in windows.items()},'studies':[]}
    all_comparisons=[]
    for target in (.05,.1):
        ev=events[events.target==target]
        match={k:pairs(f,ev,*w) for k,w in windows.items()}
        for direction in ('long','short'):
            sets={k:v[0][v[0].direction==direction] for k,v in match.items()}
            item={'direction':direction,'target':target,'pairs':{k:len(v) for k,v in sets.items()},'features':[]}
            for split,p in sets.items():p.to_parquet(OUT/f'pairs_{direction}_{target}_{split}.parquet',index=False)
            for feature in PRICE+DERIV:
                values={}
                for split,p in sets.items():
                    x=f.loc[p.case,feature].to_numpy();c=f.loc[p.control,feature].to_numpy();diff=x-c
                    values[split]={'case_median':float(np.median(x)),'control_median':float(np.median(c)), 'paired_difference_median':float(np.median(diff)),
                                  'higher_fraction':float(np.mean(diff>0)), 'equal_fraction':float(np.mean(diff==0)), 'lower_fraction':float(np.mean(diff<0))}
                    if split=='discovery':hypothesis=np.sign(np.median(diff))
                    else:
                        week=((pd.DatetimeIndex(p.case)-pd.Timestamp('2020-01-06')).days//7).to_numpy()
                        signs=pd.Series(np.sign(diff)*hypothesis,index=week).groupby(level=0).sum().to_numpy()
                        rng=np.random.default_rng(17)
                        null=(rng.choice([-1,1],size=(5000,len(signs)))*signs).sum(axis=1)
                        values[split]['weekly_one_sided_p']=float((1+(null>=signs.sum()).sum())/5001) if hypothesis else 1.
                        values[split]['independent_week_blocks']=len(signs)
                        # Confidence interval on paired directional consistency, resampling whole weeks.
                        grouped=pd.DataFrame({'week':week,'sign':np.sign(diff)}).groupby('week')['sign'].agg(['sum','count']).to_numpy()
                        sample=grouped[rng.integers(0,len(grouped),size=(3000,len(grouped)))].sum(axis=1)
                        values[split]['paired_sign_block95'] = np.quantile(sample[:,0]/sample[:,1],[.025,.975]).tolist()
                row={'feature':feature,'discovery_direction':int(hypothesis),**values}
                item['features'].append(row);all_comparisons.append(row)
            report['studies'].append(item)
    for row,p in zip(all_comparisons,holm([x['replication']['weekly_one_sided_p'] for x in all_comparisons])):
        row['replication']['holm_p']=float(p)
    (OUT/'antecedents.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('Antecedent pairs',[(x['direction'],x['target'],x['pairs']) for x in report['studies']],flush=True)
    print('Holm<.05',[(x['direction'],x['target'],y['feature']) for x in report['studies'] for y in x['features'] if y['replication']['holm_p']<.05],flush=True)

if __name__=='__main__':main()
