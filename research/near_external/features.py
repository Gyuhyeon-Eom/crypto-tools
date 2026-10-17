"""External data alignment with explicit availability lags, no retrospective filling."""
from pathlib import Path
import sys,gzip,json,hashlib
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'near_deep'))
from data import first,make_features,PRICE,DERIV,candidates

BASE=Path(__file__).resolve().parent
OUT=BASE/'output/study_v1'
SECTOR=[s+'_'+v for s in ('l1','ai') for v in ('ret1','ret4','breadth1','volume_breadth','btc_residual1','near_excess4')]
SPOT=['spot_delta15','spot_delta1','spot_delta4','spot_delta_acc4','spot_volume_build','spot_basis','spot_excess1','spot_share1']
CHAIN=lambda lag:[f'chain{lag}_'+v for v in ('stable1','stable7','dex_build','tvl7')]
ACCOUNT=['accounts_long','accounts_change1','accounts_change4']

def groups():
    base=PRICE+DERIV+['stop_distance']
    return {'baseline':base,'sector':base+SECTOR,'spot':base+SPOT,'chain48':base+CHAIN(48),
            'account':base+ACCOUNT,'sector_spot':base+SECTOR+SPOT,'all48':base+SECTOR+SPOT+CHAIN(48)+ACCOUNT,
            'all24':base+SECTOR+SPOT+CHAIN(24)+ACCOUNT}

def read_json_gz(path):return json.loads(gzip.decompress(path.read_bytes()))

def load_verified():
    p=json.loads((BASE/'protocol.json').read_text());m=json.loads((OUT/'manifest.json').read_text())
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    if digest(BASE/'protocol.json')!=m['protocol_sha256']:raise ValueError('Protocol mutated')
    if digest(BASE/'acquisition_amendment.json')!=m['amendment_sha256']:raise ValueError('Amendment mutated')
    base_path=BASE/p['base_snapshot']
    if digest(base_path)!=p['base_sha256']:raise ValueError('Base data mutated')
    records={}
    for e in m['entries']:
        path=OUT/e['file']
        if digest(path)!=e['sha256']:raise ValueError('External data mutated: '+e['name'])
        value=read_json_gz(path);records[e['name']]=value.get('data',value) if isinstance(value,dict) else value
    return p,m,read_json_gz(base_path),records

def align_available(frame,index,lag,tolerance):
    if frame.index.duplicated().any():raise ValueError('Duplicate external times')
    right=frame.copy().sort_index();right['available_at']=right.index+pd.Timedelta(lag)
    right['available_at']=right.available_at.dt.as_unit('ns')
    result=pd.merge_asof(pd.DataFrame({'decision_at':index.as_unit('ns')}),right.reset_index(drop=True),
                         left_on='decision_at',right_on='available_at',direction='backward',tolerance=pd.Timedelta(tolerance))
    if (result.available_at>result.decision_at).any():raise ValueError('Future availability joined')
    result.index=index
    return result.drop(columns=['decision_at','available_at'])

def sector_features(frames,f,baskets):
    result=pd.DataFrame(index=f.index);btc=frames['BTCUSDT'].close.pct_change(4,fill_method=None)
    variance=btc.rolling(2880).var().shift(1)
    for sector,members in baskets.items():
        if 'NEARUSDT' in members or len(set(members))!=len(members):raise ValueError('Invalid sector basket')
        close=pd.DataFrame({s:frames[s].close for s in members});r1=close.pct_change(4,fill_method=None);r4=close.pct_change(16,fill_method=None)
        turnover=pd.DataFrame({s:frames[s].turnover for s in members})
        vb=turnover.rolling(4).mean()/turnover.shift(4).rolling(96).mean()
        residual=pd.DataFrame({s:r1[s]-(r1[s].rolling(2880).cov(btc).shift(1)/variance)*btc for s in members})
        result[sector+'_ret1']=r1.mean(axis=1,skipna=False)
        result[sector+'_ret4']=r4.mean(axis=1,skipna=False)
        result[sector+'_breadth1']=(r1>0).mean(axis=1).where(r1.notna().all(axis=1))
        result[sector+'_volume_breadth']=(vb>1).mean(axis=1).where(vb.notna().all(axis=1))
        result[sector+'_btc_residual1']=residual.mean(axis=1,skipna=False)
        result[sector+'_near_excess4']=f.ret4-result[sector+'_ret4']
    return result

def spot_frame(rows,start,end):
    if any(len(r)!=12 for r in rows):raise ValueError('Invalid Binance candle schema')
    cols=['ms','open','high','low','close','volume','end_ms','quote','trades','buy_base','buy_quote','ignored']
    s=pd.DataFrame(rows,columns=cols).astype(float).sort_values('ms')
    s.index=pd.to_datetime(s.ms,unit='ms').dt.as_unit('ns')+pd.Timedelta(minutes=15)
    expected=pd.date_range(pd.Timestamp(start,unit='ms')+pd.Timedelta(minutes=15),pd.Timestamp(end,unit='ms'),freq='15min')
    if not s.index.equals(expected) or not np.isfinite(s.to_numpy()).all():raise ValueError('Missing/duplicate/nonfinite spot bars')
    if ((s.end_ms!=s.ms+899999)|(s.close<=0)|(s.low<=0)|(s.high<s[['open','close']].max(axis=1))|
        (s.low>s[['open','close']].min(axis=1))|(s.quote<0)|(s.buy_quote<0)|(s.buy_quote>s.quote)|(s.buy_base<0)|(s.buy_base>s.volume)).any():
        raise ValueError('Invalid spot OHLC/side volume')
    return s

def spot_features(s,f):
    result=pd.DataFrame(index=f.index);delta=2*s.buy_quote-s.quote
    for bars,suffix in [(1,'15'),(4,'1'),(16,'4')]:result['spot_delta'+suffix]=delta.rolling(bars).sum()/s.quote.rolling(bars).sum()
    result['spot_delta_acc4']=result.spot_delta4-result.spot_delta4.shift(16)
    result['spot_volume_build']=s.quote.rolling(4).mean()/s.quote.shift(4).rolling(96).mean()
    result['spot_basis']=s.close/f.close-1
    result['spot_excess1']=s.close.pct_change(4,fill_method=None)-f.ret1
    result['spot_share1']=s.quote.rolling(4).sum()/(s.quote.rolling(4).sum()+f.turnover.rolling(4).sum())
    return result

def daily_series(rows,column,value):
    dates=pd.to_datetime([int(r['date']) if isinstance(r,dict) else int(r[0]) for r in rows],unit='s').as_unit('ns')
    if dates.duplicated().any() or not dates.equals(dates.normalize()):raise ValueError('Non-daily/duplicate chain history')
    values=np.array([value(r) for r in rows],dtype=float)
    if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid chain values')
    return pd.Series(values,index=dates,name=column).sort_index()

def chain_features(records,index):
    stable=daily_series(records['near_stablecoins'],'stable',lambda x:x['totalCirculating'].get('peggedUSD',np.nan))
    tvl=daily_series(records['near_tvl'],'tvl',lambda x:x['tvl'])
    dex=daily_series(records['near_dex_volume']['totalDataChart'],'dex',lambda x:x[1])
    daily=pd.concat([stable,tvl,dex],axis=1,sort=True).sort_index()
    daily=daily.reindex(pd.date_range(daily.index.min(),daily.index.max(),freq='D'))
    values=pd.DataFrame({'stable1':daily.stable.pct_change(1,fill_method=None),'stable7':daily.stable.pct_change(7,fill_method=None),
                         'dex_build':daily.dex.rolling(7).mean()/daily.dex.shift(7).rolling(28).mean(),'tvl7':daily.tvl.pct_change(7,fill_method=None)})
    result=pd.DataFrame(index=index)
    for lag in (24,48):
        joined=align_available(values,index,f'{lag}h','25h')
        for c in values:result[f'chain{lag}_'+c]=joined[c]
    return result,daily

def account_features(rows,index):
    if any(x['symbol']!='NEARUSDT' for x in rows):raise ValueError('Wrong account-ratio symbol')
    r=pd.DataFrame(rows);r.index=pd.to_datetime(r.timestamp.astype('int64'),unit='ms').dt.as_unit('ns');r=r.sort_index()
    buy=r.buyRatio.astype(float);sell=r.sellRatio.astype(float)
    if r.index.duplicated().any() or not np.isfinite(buy).all() or not buy.between(0,1).all() or not sell.between(0,1).all() or not np.allclose(buy+sell,1,atol=.00011):raise ValueError('Invalid account ratios')
    r=buy.reindex(pd.date_range(r.index.min(),r.index.max(),freq='h'))
    values=pd.DataFrame({'accounts_long':r,'accounts_change1':r.diff(1),'accounts_change4':r.diff(4)})
    return align_available(values,index,'1h','2h')

def build():
    p,m,raw,records=load_verified();f=make_features(raw)
    frames={s:first.price_frame(rows,raw['start'],raw['end']) for s,rows in raw['candles'].items()}
    for s in set(p['sector_baskets']['l1'])-set(frames):frames[s]=first.price_frame(records['sector_'+s]['rows'],raw['start'],raw['end'])
    for s in p['sector_baskets']['ai']:
        frames[s]=spot_frame(records['spot_'+s]['rows'],raw['start'],raw['end'])
        frames[s]['turnover']=frames[s]['quote']
    sector=sector_features(frames,f,p['sector_baskets']);spot=spot_frame(records['spot_NEARUSDT']['rows'],raw['start'],raw['end'])
    chain,daily=chain_features(records,f.index);account=account_features(records['account_NEARUSDT']['rows'],f.index)
    f=pd.concat([f,sector,spot_features(spot,f),chain,account],axis=1).replace([np.inf,-np.inf],np.nan)
    features=sorted(set(SECTOR+SPOT+CHAIN(24)+CHAIN(48)+ACCOUNT))
    complete=np.isfinite(f[PRICE+DERIV+features]).all(axis=1);quiet=candidates(f)
    quality={'rows':len(f),'external_columns':len(features),'quiet_base_candidates':int(quiet.sum()),'paired_complete_candidates':int((quiet&complete).sum()),
             'finite_fraction':{x:float(f[x].notna().mean()) for x in features},'chain_daily_gaps':{x:int(daily[x].isna().sum()) for x in daily},
             'chain_dates':[str(daily.index.min()),str(daily.index.max())], 'spot_rows':len(spot),'account_rows':len(records['account_NEARUSDT']['rows']),
             'sector_members':p['sector_baskets'],'limitations':['current historical revisions, not original publication vintages','fixed present-day sector proxies with survivor bias','chain lag24/48h is assumption','single external spot venue; no true CEX cash inflow']}
    f.to_parquet(OUT/'features.parquet');daily.to_parquet(OUT/'daily.parquet')
    (OUT/'quality.json').write_text(json.dumps(quality,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in quality.items() if k!='finite_fraction'},indent=2))
    return f

if __name__=='__main__':build()
