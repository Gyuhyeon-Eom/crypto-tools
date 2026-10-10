"""Append public, closed spot candles in a new file; never overwrite frozen history."""
from pathlib import Path
import argparse
import gzip
import json
import ssl
import time
import urllib.parse
import urllib.request
import certifi
import pandas as pd
from dataset import BASE, TARGETS, sha


def fetch(symbol, start_ms, end_ms):
    rows, responses, cursor = [], [], end_ms-1
    while cursor >= start_ms:
        params = dict(category='spot',symbol=symbol,interval='60',limit=1000,start=start_ms,end=cursor)
        url = 'https://api.bybit.com/v5/market/kline?' + urllib.parse.urlencode(params)
        request = urllib.request.Request(url,headers={'User-Agent':'orderbook-evaluation-research/1'})
        with urllib.request.urlopen(request,context=ssl.create_default_context(cafile=certifi.where()),timeout=30) as response:
            payload = json.load(response)
        if payload.get('retCode') != 0 or payload['result'].get('category') != 'spot' or payload['result'].get('symbol') != symbol:
            raise ValueError(f'Unexpected candle response for {symbol}')
        responses.append(payload)
        batch = payload['result']['list']
        if not batch:
            break
        # Current unclosed candle must never become a historical close.
        server_ms = int(payload['time'])
        rows.extend(r for r in batch if start_ms <= int(r[0]) and int(r[0])+3600000 <= min(end_ms,server_ms))
        new_cursor = min(int(r[0]) for r in batch)-1
        if new_cursor >= cursor:
            raise ValueError('Non-progressing pagination')
        cursor = new_cursor
        time.sleep(.15)
    frame = pd.DataFrame(rows,columns=['open_ms','open','high','low','close','volume','turnover']).astype(float)
    frame['symbol'] = symbol
    frame['time'] = pd.to_datetime(frame.open_ms,unit='ms')+pd.Timedelta(hours=1)
    return frame, responses


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--history',type=Path,default=BASE.parent/'microstructure/output/bybit.parquet')
    parser.add_argument('--output',type=Path,default=BASE/'output')
    parser.add_argument('--universe',type=Path,help='JSON containing fixed symbols; omit for the3-coin infrastructure pilot')
    args=parser.parse_args()
    symbols=json.loads(args.universe.read_text())['symbols'] if args.universe else TARGETS
    if not symbols or len(set(symbols))!=len(symbols):raise ValueError('Empty or duplicate universe')
    args.output.mkdir(parents=True,exist_ok=True)
    cache=args.output/'price_cache';cache.mkdir(exist_ok=True)
    old=pd.read_parquet(args.history)
    old=old[old.symbol.isin(symbols)].copy()
    if set(old.symbol)!=set(symbols):raise ValueError('Missing historical context for a registered symbol')
    end_ms=int(pd.Timestamp.now(tz='UTC').floor('h').timestamp()*1000)
    frames=[old];requests=0
    for symbol in symbols:
        start_ms=int(pd.Timestamp(old.loc[old.symbol.eq(symbol),'time'].max()).timestamp()*1000)
        new,payloads=fetch(symbol,start_ms,end_ms)
        with gzip.open(cache/f'{symbol}_{end_ms}.json.gz','wt') as f:json.dump(payloads,f)
        frames.append(new);requests+=len(payloads)
    merged=pd.concat(frames,ignore_index=True).sort_values(['symbol','time'])
    if merged.duplicated(['symbol','time']).any():raise ValueError('Overlapping candle inputs')
    merged.to_parquet(args.output/'prices.parquet',index=False)
    result={'fetched_at':pd.Timestamp.now(tz='UTC').isoformat(),'history_sha256':sha(args.history),
            'closed_through':str(merged.time.max()),'rows':len(merged),'new_rows':len(merged)-len(old),
            'requests':requests,'symbols':symbols,'prices_sha256':sha(args.output/'prices.parquet')}
    (args.output/'prices_manifest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
