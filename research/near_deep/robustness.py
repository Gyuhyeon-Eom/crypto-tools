"""Fixed structural-stop sensitivity, data audit and peer transfer diagnostics."""
import gzip,json,hashlib
import joblib
import numpy as np
import pandas as pd
from data import make_features, candidates, PRICE, HOUR, touch_outcomes, funding_cumulative, precursor_labels
from evaluate import OUT, weekly_ci, replay, describe_alerts, pack_alerts
from stops import levels,evaluate_stop


def main():
    r=json.loads((OUT/'report.json').read_text());raw=json.loads(gzip.decompress((OUT/'data.json.gz').read_bytes()))
    f=pd.read_parquet(OUT/'features.parquet');funding=funding_cumulative(f,raw['funding'])
    start=pd.Timestamp(r['folds'][0]['test'][0]);end=f.index[-1]
    report={'stop_protocol_sha256':hashlib.sha256((OUT.parent/'stop_protocol.json').read_bytes()).hexdigest(),
            'status':'All stop variants are retrospective sensitivity tests, not selected policies. Returns are per-alert overlapping hypothetical trades. Peer results are postselection diagnostics.', 'stops':[], 'peer_transfer':[], 'coverage':[]}
    ev=pd.read_parquet(OUT/'events.parquet');eligible=candidates(f)
    for direction in ('long','short'):
      ls=levels(f,direction)
      for target in (.05,.1):
        _,ids,events=precursor_labels(f,ev[ev.target==target],direction)
        for fold in r['folds']:
            a,b=map(pd.Timestamp,fold['test']);pool=f.index[eligible&(f.index>=a)&(f.index+73*HOUR<b)]
            event=events[(events.onset>=a)&(events.onset+72*HOUR<b)]
            reached=set(ids.reindex(pool).dropna())&set(event.onset)
            report['coverage'].append({'direction':direction,'target':target,'start':str(a),'events':len(event),'reachable':len(reached),
                                       'eligible_decisions':len(pool),'all_hourly_decisions':int(((f.index>=a)&(f.index+73*HOUR<b)&(f.index.minute==0)).sum())})
        for hours in (24,72):
          for policy,stop in ls.items():
            outcome=evaluate_stop(f,direction,target,hours,stop,funding)
            outcome.to_parquet(OUT/f'stop_outcome_{direction}_{target}_{hours}_{policy}.parquet')
            for task in r['tasks']:
                if task['direction']!=direction or task['target']!=target or task['hours']!=hours:continue
                for model in task['aggregate']:
                    alerts=pd.read_parquet(OUT/f"alerts_{task['key']}_{model['name']}.parquet")
                    results=outcome.reindex(alerts.index);a=results[results.valid_stop]
                    item={'task':task['key'],'model':model['name'],'policy':policy,'alerts':len(alerts),'valid_stops':len(a),'folds':[]}
                    if len(a):
                        item.update({'target_before_stop':float(a.target_before_stop.mean()),'block95':weekly_ci(a,'target_before_stop',start,end),
                                     'median_stop_distance_pct':float(a.stop_distance_pct.median()),'stop_distance_p90_pct':float(a.stop_distance_pct.quantile(.9)),
                                     'stop_rate':float(a.stopped.mean()),'timeout_rate':float(a.timeout.mean()),'same_bar_ambiguous':int(a.ambiguous.sum()),
                                     'mean_net20bps_pct':float(a.net20.mean()*100),'net20bps_block95_pct':[v*100 for v in weekly_ci(a,'net20',start,end)],
                                     'mean_net10bps_pct':float(a.net20.mean()*100+.1),'median_exit_hours':float(a.exit_hours.median())})
                        for fold in r['folds']:
                            lo,hi=map(pd.Timestamp,fold['test']);g=a.loc[(a.index>=lo)&(a.index<hi)]
                            item['folds'].append({'alerts':len(g),'target_before_stop':float(g.target_before_stop.mean()) if len(g) else None,'mean_net20bps_pct':float(g.net20.mean()*100) if len(g) else None})
                    report['stops'].append(item)
          print('Stops',direction,target,hours,'complete',flush=True)
    # NEAR-trained price model, original calibrated threshold, no peer refit or threshold tuning.
    for symbol in ('SUIUSDT','APTUSDT','SEIUSDT'):
        peer=make_features(raw,symbol);eligible_peer=candidates(peer,False)
        for task in r['tasks']:
            if task['type']!='touch':continue
            outcome=touch_outcomes(peer,task['direction'],task['target'],task['hours'])
            for name in ('price_logistic','price_boosting'):
                frames=[]
                for i,fold in enumerate(r['folds']):
                    lock=OUT/'locks'/f"{task['key']}_{i}_{name}.json"
                    if not lock.exists():continue
                    info=json.loads(lock.read_text());model=joblib.load(lock.with_suffix('.joblib'))
                    a,b=map(pd.Timestamp,fold['test']);pool=peer.index[eligible_peer&(peer.index>=a)&(peer.index+73*HOUR<b)]
                    score=pd.Series(model.predict_proba(peer.loc[pool,PRICE])[:,1],index=pool)
                    times=replay(score,info['calibration']['threshold']);frames.append(outcome.reindex(times))
                if frames:
                    a=pd.concat(frames).sort_index()
                    report['peer_transfer'].append({'symbol':symbol,'task':task['key'],'model':name,'alerts':len(a),
                                                    'target_touch':float(a.touch.mean()) if len(a) else None,'block95':weekly_ci(a,'touch',start,end),
                                                    'scope':'Exploratory postselected peers; target-touch only, no funding or stop-adjusted profit claim.'})
        print('Peer transfer',symbol,'complete',flush=True)
    report['audit']={'data_sha256':r['manifest']['data_sha256'],'start':str(f.index[0]),'end':str(end),
                     'price_rows':len(f),'oi_observations':len(raw['oi']),'funding_observations':len(raw['funding']),
                     'complete_candidate_fraction':int(eligible.sum())/int(((f.index.minute==0)&np.isfinite(f[PRICE]).all(axis=1)).sum()),
                     'price_derivatives_candidate_match':r['paired_candidate_count']==r['price_candidate_count'],
                     'future_execution_limits':['next-open proxy has no measured latency','historical OI timestamps plus15min assumed availability','trade-price structural stops, not exchange fill records','no historical orderbook, liquidation or taker-flow data for NEAR','unlevered per-alert outcomes, no liquidation/portfolio simulation']}
    (OUT/'robustness.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('Saved robustness.json',flush=True)

if __name__=='__main__':main()
