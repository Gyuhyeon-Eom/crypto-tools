"""Public historical data only; cached, rate-limited, no credentials or writes to services."""
from __future__ import annotations
import concurrent.futures
import gzip
import hashlib
import json
import ssl
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import certifi
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / 'output'
CACHE = OUT / 'cache'
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE.parent / 'precursor'))
from study import STABLE, LARGE

CONTEXT = ssl.create_default_context(cafile=certifi.where())
START = int(pd.Timestamp('2026-03-01', tz='UTC').timestamp()*1000)
END = int(pd.Timestamp('2026-10-09', tz='UTC').timestamp()*1000)


def get(host, endpoint, params=None):
    url = host + endpoint + ('?' + urllib.parse.urlencode(params) if params else '')
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'precursor-hypothesis-research/1'})
            with urllib.request.urlopen(req, context=CONTEXT, timeout=30) as resp:
                data = json.load(resp)
            if isinstance(data, dict) and data.get('retCode', 0) != 0:
                raise ValueError(f"{data.get('retCode')} {data.get('retMsg')}")
            return data
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1 + attempt*2)
        finally:
            time.sleep(.12)


def collect(symbol, kind):
    path = CACHE / f'{symbol}_{kind}.json.gz'
    if path.exists():
        with gzip.open(path, 'rt') as f:
            return json.load(f)
    rows, requests = [], 0
    if kind == 'binance':
        cursor = START
        while cursor < END:
            batch = get('https://data-api.binance.vision', '/api/v3/klines', {
                'symbol': symbol, 'interval': '1h', 'limit': 1000, 'startTime': cursor, 'endTime': END-1})
            requests += 1
            if not batch:
                break
            rows.extend(batch)
            next_cursor = int(batch[-1][0]) + 3600000
            if next_cursor <= cursor:
                raise RuntimeError('Non-progressing Binance pagination')
            cursor = next_cursor
    else:
        cursor = END-1
        while cursor >= START:
            if kind == 'bybit':
                endpoint = '/v5/market/kline'
                params = dict(category='spot', symbol=symbol, interval='60', limit=1000, start=START, end=cursor)
                time_key = 0
            elif kind == 'oi':
                endpoint = '/v5/market/open-interest'
                params = dict(category='linear', symbol=symbol, intervalTime='4h', limit=200, startTime=START, endTime=cursor)
                time_key = 'timestamp'
            else:
                endpoint = '/v5/market/funding/history'
                params = dict(category='linear', symbol=symbol, limit=200, startTime=START, endTime=cursor)
                time_key = 'fundingRateTimestamp'
            result = get('https://api.bybit.com', endpoint, params)
            requests += 1
            batch = result.get('result', {}).get('list', [])
            if not batch:
                break
            rows.extend(batch)
            next_cursor = min(int(row[time_key]) for row in batch)-1
            if next_cursor >= cursor:
                raise RuntimeError(f'Non-progressing {kind} pagination')
            cursor = next_cursor
    payload = {'symbol': symbol, 'kind': kind, 'fetched_at': datetime.now(timezone.utc).isoformat(),
               'start_ms': START, 'end_ms_exclusive': END, 'requests': requests, 'rows': rows}
    temp = path.with_suffix('.tmp')
    with gzip.open(temp, 'wt') as f:
        json.dump(payload, f)
    temp.replace(path)
    return payload


def parse(payload):
    rows, kind = payload['rows'], payload['kind']
    if not rows:
        return pd.DataFrame()
    if kind == 'binance':
        d = pd.DataFrame(rows, columns=['open_ms','open','high','low','close','volume','close_ms','turnover',
                                         'trades','buy_volume','buy_turnover','unused']).astype(float)
        d['time'] = pd.to_datetime(d.open_ms, unit='ms', utc=True).dt.tz_localize(None) + pd.Timedelta(hours=1)
        d = d[(d.open_ms >= START) & (d.open_ms + 3600000 <= END)]
    elif kind == 'bybit':
        d = pd.DataFrame(rows, columns=['open_ms','open','high','low','close','volume','turnover']).astype(float)
        d['time'] = pd.to_datetime(d.open_ms, unit='ms', utc=True).dt.tz_localize(None) + pd.Timedelta(hours=1)
        d = d[(d.open_ms >= START) & (d.open_ms + 3600000 <= END)]
    elif kind == 'oi':
        d = pd.DataFrame(rows)
        d['oi'] = pd.to_numeric(d['singleOpenInterest']) if 'singleOpenInterest' in d else pd.to_numeric(d['openInterest']) / 2
        d['time'] = pd.to_datetime(pd.to_numeric(d.timestamp), unit='ms', utc=True).dt.tz_localize(None)
        d = d[['time', 'oi']]
    else:
        d = pd.DataFrame(rows)
        d['funding'] = pd.to_numeric(d.fundingRate)
        d['time'] = pd.to_datetime(pd.to_numeric(d.fundingRateTimestamp), unit='ms', utc=True).dt.tz_localize(None)
        d = d[['time', 'funding']]
    d['symbol'] = payload['symbol']
    return d.drop_duplicates(['time','symbol']).sort_values('time')


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    universe_file = OUT / 'universe.json'
    if universe_file.exists():
        universe = json.loads(universe_file.read_text())
    else:
        b = get('https://data-api.binance.vision', '/api/v3/exchangeInfo')
        y = get('https://api.bybit.com', '/v5/market/tickers', {'category': 'linear'})
        bs = {x['symbol'] for x in b['symbols']}
        ys = {x['symbol'] for x in y['result']['list']}
        p = pd.read_parquet(BASE.parent / 'precursor/output/daily_bars.parquet')
        p = p[p.date <= '2026-03-31'].sort_values(['symbol','date'])
        past = []
        for symbol,g in p.groupby('symbol'):
            if (len(g) >= 90 and g.date.max() == pd.Timestamp('2026-03-31')
                    and symbol.removesuffix('USDT') not in STABLE|LARGE
                    and 200000 <= g.tail(30).usd_volume_proxy.mean() <= 20000000):
                past.append(symbol)
        universe = {'selected_at': datetime.now(timezone.utc).isoformat(), 'past_eligible': past,
                    'symbols': sorted(set(past)&bs&ys), 'binance_missing': sorted(set(past)-bs),
                    'linear_missing': sorted(set(past)-ys)}
        universe_file.write_text(json.dumps(universe, indent=2))
    symbols = universe['symbols']
    jobs = [(s,k) for s in symbols for k in ['bybit','binance','oi','funding']]
    jobs.extend([('BTCUSDT','bybit')])
    frames = {k: [] for k in ['bybit','binance','oi','funding']}
    manifest, errors = [], []

    def task(job):
        s,k = job
        try:
            p = collect(s,k)
            d = parse(p)
            meta = {x:p[x] for x in ['symbol','kind','requests','fetched_at']}
            meta.update(rows=len(d), sha256=hashlib.sha256((CACHE/f'{s}_{k}.json.gz').read_bytes()).hexdigest())
            return k,d,meta,None
        except Exception as exc:
            return k,None,None,{'symbol':s,'kind':k,'error':str(exc)}

    print(f'Fetching {len(symbols)} symbols / {len(jobs)} data streams', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(task,job) for job in jobs]
        for n,future in enumerate(concurrent.futures.as_completed(futures),1):
            k,d,meta,error = future.result()
            if error:
                errors.append(error)
                print('ERROR',error,flush=True)
            else:
                frames[k].append(d)
                manifest.append(meta)
            if n % 25 == 0:
                print(f'Completed {n}/{len(jobs)} streams',flush=True)
    counts = {}
    for kind,parts in frames.items():
        p = pd.concat([d for d in parts if len(d)],ignore_index=True).sort_values(['symbol','time'])
        p.to_parquet(OUT/f'{kind}.parquet',index=False)
        counts[kind] = {'rows':len(p),'symbols':p.symbol.nunique(),'start':str(p.time.min()),'end':str(p.time.max())}
    audit = {'counts':counts,'errors':errors,'sources':manifest,
             'protocol_sha256':hashlib.sha256((BASE/'protocol.json').read_bytes()).hexdigest()}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(counts,indent=2),flush=True)


if __name__ == '__main__':
    main()
