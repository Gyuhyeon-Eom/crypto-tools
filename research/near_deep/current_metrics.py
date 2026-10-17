import json
import pandas as pd
from data import precursor_labels
from evaluate import OUT, describe_alerts, pack_alerts, weekly_ci

def main():
    d=json.loads((OUT/'current_rules.json').read_text());r=json.loads((OUT/'report.json').read_text())
    f=pd.read_parquet(OUT/'features.parquet');events=pd.read_parquet(OUT/'events.parquet')
    start=pd.Timestamp(r['folds'][0]['test'][0]);end=f.index[-1];d['metrics']=[]
    for direction in ('long','short'):
      times=pd.DatetimeIndex(pd.to_datetime([x['time'] for x in d['rows'] if x['direction']==direction],unit='ms'))
      for target in (.05,.1):
        y,ids,ev=precursor_labels(f,events[events.target==target],direction)
        eligible_events=[]
        for fold in r['folds']:
            a,b=map(pd.Timestamp,fold['test']);eligible_events.extend(ev.loc[(ev.onset>=a)&(ev.onset+pd.Timedelta(hours=72)<b),'onset'])
        for hours in (24,72):
            o=pd.read_parquet(OUT/f'outcome_{direction}_{target}_{hours}.parquet')
            alerts=pack_alerts(times,y,ids,o,f)
            item={'direction':direction,'target':target,'hours':hours,**describe_alerts(alerts,eligible_events,start,end),'stops':[]}
            for policy in ('prior_4h','prior_12h','prior_24h','confirmed_pivot'):
                s=pd.read_parquet(OUT/f'stop_outcome_{direction}_{target}_{hours}_{policy}.parquet').reindex(times)
                s=s[s.valid_stop]
                item['stops'].append({'policy':policy,'valid_stops':len(s),'target_before_stop':float(s.target_before_stop.mean()),
                                     'mean_net20bps_pct':float(s.net20.mean()*100),'block95':weekly_ci(s,'target_before_stop',start,end)})
            d['metrics'].append(item)
    (OUT/'current_rules_metrics.json').write_text(json.dumps(d,indent=2,allow_nan=False))
    for m in d['metrics']:
        print(m['direction'],m['target'],m['hours'],m['alerts'],'pre',m['precision'],'touch',m['target_touch_rate'],'stop4',m['stops'][0]['target_before_stop'])
if __name__=='__main__':main()
