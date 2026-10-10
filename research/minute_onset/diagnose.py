"""Post-result diagnostic of saturation; not a new confirmatory experiment."""
import json
import numpy as np
import pandas as pd
from study import OUT,split,metrics


def main():
    frozen=json.loads((OUT/'frozen.json').read_text())
    d=split(pd.read_parquet(OUT/'panel.parquet'),'replication')
    d=d[d.eligible_price&d.quality&d.label_complete]
    results={}
    for rule in frozen['rules']:
        key=rule['feature'];score=d[key]*rule['direction']
        raw=d[score>=rule['cutoff']].copy()
        raw['_score']=score.loc[raw.index]
        hits=raw[raw.hit6]
        alerts=pd.read_csv(OUT/f'alerts_{key}.csv',parse_dates=['time'])
        last={};selected=[]
        for idx,row in raw.sort_values(['time','_score','symbol'],ascending=[True,False,True]).iterrows():
            if row.symbol in last and row.time-last[row.symbol]<pd.Timedelta(hours=24):continue
            last[row.symbol]=row.time;selected.append(idx)
        uncapped=raw.loc[selected]
        uncapped.to_csv(OUT/f'diagnostic_uncapped_{key}.csv',index=False)
        minute=alerts.time.dt.hour*60+alerts.time.dt.minute
        daily=raw.groupby(raw.time.dt.date).size()
        strict=hits[hits.strict_hit]
        events=[]
        for (symbol,target),g in hits.groupby(['symbol','next_event']):
            events.append({'symbol':symbol,'onset_utc':target,'earliest_candidate_utc':g.time.min(),
                           'maximum_lead_minutes':float(g.lead_minutes.max()),'candidate_rows':len(g),
                           'max_quote_ratio':float(np.exp(g.quote_accel.max()))})
        pd.DataFrame(events).to_csv(OUT/f'raw_event_candidates_{key}.csv',index=False)
        results[key]={'status':'posthoc_saturation_diagnostic_not_confirmatory',
            'raw_rows':len(raw),'raw_hit_rows':int(raw.hit6.sum()),'raw_precision':float(raw.hit6.mean()),
            'raw_detected_events':hits[['symbol','next_event']].drop_duplicates().shape[0],
            'raw_strict_detected_events':strict[['symbol','next_event']].drop_duplicates().shape[0],
            'raw_daily_median':float(daily.median()),'raw_daily_min':int(daily.min()),'raw_daily_max':int(daily.max()),
            'actual_alert_utc_minute_median':float(minute.median()),'actual_alert_utc_minute_max':int(minute.max()),
            'remove_daily_cap_keep24h_cooldown':metrics(d,uncapped)}
    (OUT/'diagnostics.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
