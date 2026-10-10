"""Outcome-blind fixed universe and exhaustive daily raw trade coverage."""
import concurrent.futures as cf
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parent
OUT=BASE/'output'
FLOW=BASE.parent/'tradeflow'
spec=importlib.util.spec_from_file_location('raw_trade_fetch',FLOW/'fetch.py')
raw=importlib.util.module_from_spec(spec);spec.loader.exec_module(raw)


def select_universe():
    path=OUT/'universe.json'
    if path.exists():
        return json.loads(path.read_text())
    universe=json.loads((BASE.parent/'microstructure/output/universe.json').read_text())['symbols']
    daily=pd.read_parquet(BASE.parent/'precursor/output/daily_bars.parquet')
    daily=daily[daily.symbol.isin(universe)&daily.date.between('2026-03-02','2026-03-31')]
    liq=daily.groupby('symbol').usd_volume_proxy.mean().sort_values()
    rng=np.random.default_rng(20261009)
    selected=[]
    for i,group in enumerate(np.array_split(liq.index.to_numpy(),3)):
        for symbol in sorted(rng.choice(group,12,replace=False)):
            selected.append({'symbol':str(symbol),'liquidity_tercile':i+1,'march_liquidity':float(liq.loc[symbol])})
    result={'selected_at':pd.Timestamp.now(tz='UTC').isoformat(),'seed':20261009,'selection':selected,
            'symbols':sorted(x['symbol'] for x in selected),
            'protocol_sha256':hashlib.sha256((BASE/'protocol.json').read_bytes()).hexdigest()}
    path.write_text(json.dumps(result,indent=2))
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    u=select_universe()
    days=pd.date_range('2026-03-31','2026-10-01')
    tasks=[(symbol,f'{day:%Y-%m-%d}') for symbol in u['symbols'] for day in days]
    old=json.loads((FLOW/'output/download_audit.json').read_text())
    old={(r['symbol'],r['day']):r for r in old if 'error' not in r}
    results=[]
    def one(task):
        if task in old and (FLOW/'output/bars'/f'{task[0]}_{task[1]}.parquet').exists():
            return {**old[task],'reused':True}
        return raw.get_day(task)
    print(f'Fixed {len(u["symbols"])} symbols, {len(tasks)} symbol-days',flush=True)
    with cf.ThreadPoolExecutor(max_workers=6) as pool:
        for r in pool.map(one,tasks):
            results.append(r)
            if len(results)%200==0 or len(results)==len(tasks):
                print(f'{len(results)}/{len(tasks)} files, errors {sum("error" in a for a in results)}, trades {sum(a.get("trades",0) for a in results):,}',flush=True)
                (OUT/'download_audit.json').write_text(json.dumps(results,indent=2))
    print('Complete',flush=True)


if __name__=='__main__':
    main()
