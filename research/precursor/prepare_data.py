"""Build an auditable daily spot OHLCV panel. No credentials or notifications."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import ssl
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import certifi
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'output'


def main():
    OUT.mkdir(exist_ok=True)
    frames, sources = [], []
    # Later archive wins overlaps; fresh exchange candles win all archive overlaps.
    for path in [ROOT / 'surge-alert/ml_data/raw_data.json',
                 ROOT / 'surge-alert/ml_data/raw_data_extended.json']:
        payload = path.read_bytes()
        sources.append({'path': str(path.relative_to(ROOT)),
                        'sha256': hashlib.sha256(payload).hexdigest()})
        for symbol, values in json.loads(payload).items():
            df = pd.DataFrame(values).rename(columns={
                'ts': 'open_ms', 'o': 'open', 'h': 'high', 'l': 'low',
                'c': 'close', 'v': 'volume'})
            df['symbol'] = symbol
            df['turnover'] = df.volume * df.close
            df['turnover_exact'] = False
            df['source'] = path.name
            frames.append(df)
    # BTC is needed as a market-control feature; use the archived spot panel.
    path = ROOT / 'crypto-surge-detector/data/BTCUSDT_1d.parquet'
    df = pd.read_parquet(path)
    # pandas can preserve microsecond resolution: use explicit millisecond conversion.
    df['open_ms'] = [int(x.timestamp() * 1000) for x in pd.to_datetime(df.open_time, utc=True)]
    df['symbol'] = 'BTCUSDT'
    df['turnover'] = df.quote_volume
    df['turnover_exact'] = True
    df['source'] = 'BTCUSDT_1d.parquet'
    frames.append(df.drop(columns=['open_time', 'quote_volume']))
    sources.append({'path': str(path.relative_to(ROOT)),
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    archive = pd.concat(frames, ignore_index=True)
    symbols = sorted(archive.symbol.unique())
    cache = OUT / 'fresh_daily.json'
    if cache.exists():
        fresh = json.loads(cache.read_text())
        print('Using cached fresh data:', fresh['fetched_at'], flush=True)
    else:
        context = ssl.create_default_context(cafile=certifi.where())

        def fetch(symbol):
            params = urllib.parse.urlencode(dict(category='spot', symbol=symbol,
                                                 interval='D', limit=200))
            url = 'https://api.bybit.com/v5/market/kline?' + params
            for attempt in range(3):
                try:
                    request = urllib.request.Request(url, headers={'User-Agent': 'precursor-research/1'})
                    with urllib.request.urlopen(request, context=context, timeout=20) as response:
                        result = json.load(response)
                    if result.get('retCode') != 0:
                        return symbol, None, str(result.get('retMsg'))
                    return symbol, result, None
                except Exception as exc:
                    if attempt == 2:
                        return symbol, None, str(exc)
                    time.sleep(attempt + 1)
                finally:
                    time.sleep(.25)

        fresh = {'fetched_at': datetime.now(timezone.utc).isoformat(), 'responses': {}, 'errors': {}}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for count, (symbol, result, error) in enumerate(pool.map(fetch, symbols), 1):
                if error:
                    fresh['errors'][symbol] = error
                else:
                    fresh['responses'][symbol] = result
                if count % 40 == 0:
                    print(f'Fetched {count}/{len(symbols)}', flush=True)
        cache.write_text(json.dumps(fresh))
    frames = [archive]
    discarded_open = 0
    for symbol, response in fresh['responses'].items():
        rows = response.get('result', {}).get('list', [])
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=['open_ms', 'open', 'high', 'low', 'close', 'volume', 'turnover']).astype(float)
        df['open_ms'] = df.open_ms.astype('int64')
        closed = df.open_ms + 86_400_000 <= response['time']
        discarded_open += int((~closed).sum())
        df = df.loc[closed].copy()
        df['symbol'] = symbol
        df['turnover_exact'] = True
        df['source'] = 'bybit_api'
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True).drop_duplicates(['symbol', 'open_ms'], keep='last')
    panel['date'] = pd.to_datetime(panel.open_ms, unit='ms', utc=True).dt.tz_localize(None)
    # Use one identical volume proxy across old and fresh observations to avoid source shifts.
    panel['usd_volume_proxy'] = panel.volume * panel.close
    valid = ((panel.close > 0) & (panel.open > 0) & (panel.low > 0)
             & (panel.high >= panel[['open', 'close', 'low']].max(axis=1))
             & (panel.low <= panel[['open', 'close']].min(axis=1))
             & (panel.volume >= 0))
    invalid = int((~valid).sum())
    panel = panel.loc[valid].sort_values(['symbol', 'date']).reset_index(drop=True)
    panel.to_parquet(OUT / 'daily_bars.parquet', index=False)
    audit = {'sources': sources, 'fresh_sha256': hashlib.sha256(cache.read_bytes()).hexdigest(),
             'fetch_time': fresh['fetched_at'], 'symbols': int(panel.symbol.nunique()),
             'rows': len(panel), 'date_start': str(panel.date.min()), 'date_end': str(panel.date.max()),
             'dropped_open_candles': discarded_open, 'invalid_bars': invalid,
             'fetch_errors': fresh['errors'],
             'universe_caveat': 'Historical archived universe selected by earlier collectors; delisted/unrecorded assets may be absent. No claim of a survivorship-free exchange universe.',
             'volume_units': 'All model volume features use base volume * daily close, an approximate USD turnover; exact API turnover retained separately.'}
    (OUT / 'data_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps({k: v for k, v in audit.items() if k not in ['sources', 'fetch_errors']}, indent=2), flush=True)
    print('Fetch errors:', len(fresh['errors']), flush=True)


if __name__ == '__main__':
    main()
