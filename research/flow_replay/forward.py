"""Frozen future-period audit. Nothing schedules itself or sends alerts.

python3 research/flow_replay/forward.py status
python3 research/flow_replay/forward.py run  # available after 2026-11-17 00:00 UTC
"""
import argparse
import concurrent.futures as cf
import hashlib
import gzip
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

import study
from fetch import raw

BASE=Path(__file__).resolve().parent
OUT=BASE/'output'
DEST=OUT/'forward'
START=pd.Timestamp('2026-10-10')
END=pd.Timestamp('2026-11-10')
MATURE=pd.Timestamp('2026-11-17')


def hashes():
    paths=['protocol.json','study.py','fetch.py','forward.py','output/universe.json','../tradeflow/fetch.py']
    return {p:hashlib.sha256((BASE/p).read_bytes()).hexdigest() for p in paths}


def freeze():
    path=OUT/'forward_manifest.json'
    if path.exists():
        return json.loads(path.read_text())
    result={'registered_at':pd.Timestamp.now(tz='UTC').isoformat(),
            'start_utc':START.isoformat(),'end_exclusive_utc':END.isoformat(),'mature_at_utc':MATURE.isoformat(),
            'status':'protocol_frozen_not_scheduled','source_hashes':hashes(),
            'claim':'Future archived-data audit, not live signal recording. Run command manually after outcome maturity.'}
    path.write_text(json.dumps(result,indent=2))
    return result


def run():
    manifest=freeze()
    if manifest['source_hashes']!=hashes():
        raise RuntimeError('Frozen research code/universe changed; refuse silently altered forward test')
    now=pd.Timestamp.now(tz='UTC').tz_localize(None)
    if now<MATURE:
        print(json.dumps({'status':'pending_outcome_maturity','now_utc':now.isoformat(),'run_after_utc':MATURE.isoformat(),
                          'explanation':'Future results do not exist yet. No downloads, fitting, or alerts were executed.'},indent=2))
        return
    DEST.mkdir(parents=True,exist_ok=True)
    u=json.loads((OUT/'universe.json').read_text())
    with cf.ThreadPoolExecutor(max_workers=6) as pool:
        frames=list(pool.map(collect_hourly,u['symbols']))
    hourly=pd.concat(frames,ignore_index=True)
    hourly.to_parquet(DEST/'hourly.parquet',index=False)
    tasks=[(s,f'{day:%Y-%m-%d}') for s in u['symbols'] for day in pd.date_range(START-pd.Timedelta(days=1),END-pd.Timedelta(days=1))]
    audit=[]
    with cf.ThreadPoolExecutor(max_workers=6) as pool:
        for item in pool.map(raw.get_day,tasks):
            audit.append(item)
    (DEST/'download_audit.json').write_text(json.dumps(audit,indent=2))
    panel=study.build_panel(hourly,u['symbols'],START,END-pd.Timedelta(hours=4))
    d=panel[panel.price_eligible&panel.quality&panel.mature].copy()
    alerts=study.replay(d)
    baseline,e,random=study.comparisons(d,alerts)
    result={'status':'completed_frozen_future_audit','evaluated_at':pd.Timestamp.now(tz='UTC').isoformat(),
            'flow':study.metrics(d,alerts),'volume':study.metrics(d,baseline),
            'vs_random':study.lift_interval(e),'random':random[-1],
            'coverage':{'grid_rows':len(panel),'price_eligible':int(panel.price_eligible.sum()),'quality_mature':len(d)},
            'source_hashes':hashes()}
    panel.to_parquet(DEST/'panel.parquet',index=False)
    alerts.to_csv(DEST/'alerts.csv',index=False)
    baseline.to_csv(DEST/'volume_baseline.csv',index=False)
    e.to_csv(DEST/'random_expectation.csv',index=False)
    (DEST/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2))


def collect_hourly(symbol):
    """Cache and paginate spot candles; never include an unfinished hour."""
    cache=DEST/'hourly_cache';cache.mkdir(exist_ok=True)
    path=cache/f'{symbol}.json.gz'
    start_ms=int((START-pd.Timedelta(days=31)).timestamp()*1000)
    end_ms=int(MATURE.timestamp()*1000)
    if path.exists():
        with gzip.open(path,'rt') as f:
            rows=json.load(f)['rows']
    else:
        cursor=end_ms-1;rows=[]
        while cursor>=start_ms:
            params={'category':'spot','symbol':symbol,'interval':'60','limit':1000,'start':start_ms,'end':cursor}
            url='https://api.bybit.com/v5/market/kline?'+urllib.parse.urlencode(params)
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(url,context=raw.CONTEXT,timeout=30) as response:
                        payload=json.load(response)
                    if payload.get('retCode')!=0:
                        raise ValueError(str(payload.get('retMsg')))
                    break
                except Exception:
                    if attempt==3:
                        raise
                    time.sleep(2**attempt)
            batch=payload['result']['list']
            if not batch:
                break
            rows.extend(batch)
            next_cursor=min(int(x[0]) for x in batch)-1
            if next_cursor>=cursor:
                raise RuntimeError('Nonprogressing pagination')
            cursor=next_cursor;time.sleep(.12)
        with gzip.open(path,'wt') as f:
            json.dump({'symbol':symbol,'rows':rows,'fetched_at':pd.Timestamp.now(tz='UTC').isoformat(),'start_ms':start_ms,'end_ms':end_ms},f)
    frame=pd.DataFrame(rows,columns=['open_ms','open','high','low','close','volume','turnover']).astype(float)
    frame=frame[(frame.open_ms>=start_ms)&(frame.open_ms+3600000<=end_ms)].drop_duplicates('open_ms')
    frame['time']=pd.to_datetime(frame.open_ms,unit='ms')+pd.Timedelta(hours=1)
    frame['symbol']=symbol
    return frame.sort_values('time')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['status','run'],default='status',nargs='?')
    args=p.parse_args()
    if args.mode=='status':
        print(json.dumps(freeze(),indent=2))
    else:
        run()
