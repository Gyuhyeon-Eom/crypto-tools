"""Causal inputs and separately computed future outcomes for the deeper study."""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('near_first', Path(__file__).resolve().parent.parent/'near_precursor/study.py')
first = importlib.util.module_from_spec(spec)
spec.loader.exec_module(first)
HOUR = pd.Timedelta(hours=1)
PRICE = ['ret15','ret1','ret4','ret24','ret72','volume_build','volume_acceleration','compression','vol_regime',
         'range_position','range_compression','wick_balance','efficiency4','btc_excess1','eth_excess1','btc_excess4',
         'eth_excess4','btc_ret1','btc_ret24']
DERIV = ['oi_ret1','oi_ret4','oi_ret24','funding','mark_basis']


def make_features(raw, symbol='NEARUSDT'):
    frames = {s:first.price_frame(raw['candles'][s],raw['start'],raw['end']) for s in {symbol,'BTCUSDT','ETHUSDT'}}
    # The earlier feature function names its target NEAR; alias only the target input for price-only peer diagnostics.
    frames['NEARUSDT'] = frames[symbol]
    f = first.features(frames, raw['oi'] if symbol=='NEARUSDT' else [], raw['funding'] if symbol=='NEARUSDT' else [])
    r = np.log(f.close).diff()
    f['ret15'] = f.close/f.close.shift()-1
    f['ret72'] = f.close/f.close.shift(288)-1
    f['volume_acceleration'] = f.turnover/f.turnover.shift().rolling(4).mean()
    f['vol_regime'] = f.rv24/r.rolling(672).std()
    span = (f.high-f.low)/f.close
    f['range_compression'] = span.rolling(4).mean()/span.shift(4).rolling(96).mean()
    lower = f[['open','close']].min(axis=1)-f.low
    upper = f.high-f[['open','close']].max(axis=1)
    f['wick_balance'] = (lower-upper).rolling(4).sum()/(f.high-f.low).rolling(4).sum()
    f['efficiency4'] = r.rolling(16).sum().abs()/r.abs().rolling(16).sum()
    f['oi_ret24'] = f.oi/f.oi.shift(96)-1
    for sym, prefix in [('BTCUSDT','btc'),('ETHUSDT','eth')]:
        other = frames[sym].close
        f[prefix+'_excess4'] = f.ret4-(other/other.shift(16)-1)
        if prefix=='btc':
            f['btc_ret1'] = other/other.shift(4)-1
            f['btc_ret24'] = other/other.shift(96)-1
            f['btc_ret7d'] = other/other.shift(672)-1
    f['mark_basis'] = np.nan
    if symbol=='NEARUSDT' and raw.get('mark'):
        mark = pd.DataFrame(raw['mark'],columns=['ms','open','high','low','close']).astype(float)
        mark.index = pd.to_datetime(mark.ms,unit='ms').dt.as_unit('ns')+pd.Timedelta(minutes=15)
        mark = mark.sort_index()
        if not mark.index.equals(f.index) or not np.isfinite(mark.to_numpy()).all() or (mark.close<=0).any():
            raise ValueError('Invalid mark-price history')
        f['mark_basis'] = mark.close.shift()/f.close.shift()-1
    return f.replace([np.inf,-np.inf],np.nan)


def zigzag(f, target):
    """Alternating close-only directional changes; all anchors are retrospective labels."""
    c=f.close.to_numpy();i_low=i_high=0;mode=0;events=[]
    for i in range(1,len(c)):
        if mode<=0:
            if c[i]<c[i_low]:i_low=i
            if c[i]>=c[i_low]*(1+target):
                events.append({'direction':'long','onset':f.index[i_low],'confirmed':f.index[i],'target':target})
                mode=1;i_high=i
                continue
        if mode>=0:
            if c[i]>c[i_high]:i_high=i
            if c[i]<=c[i_high]*(1-target):
                events.append({'direction':'short','onset':f.index[i_high],'confirmed':f.index[i],'target':target})
                mode=-1;i_low=i
    return pd.DataFrame(events,columns=['direction','onset','confirmed','target'])


def precursor_labels(f, events, direction):
    y=pd.Series(0.,index=f.index);ids=pd.Series(pd.NaT,index=f.index,dtype='datetime64[ns]')
    chosen=events[(events.direction==direction)&((events.confirmed-events.onset)<=72*HOUR)]
    for t in chosen.onset:
        mask=(f.index>=t-HOUR)&(f.index<=t-pd.Timedelta(minutes=15))
        y.loc[mask]=1.;ids.loc[mask]=t
    y.loc[f.index+73*HOUR>f.index.max()]=np.nan
    return y,ids,chosen


def funding_cumulative(f, records):
    if not records:return np.zeros(len(f))
    rates=pd.DataFrame(records)
    times=pd.to_datetime(rates.fundingRateTimestamp.astype('int64'),unit='ms').dt.as_unit('ns')
    s=pd.Series(rates.fundingRate.astype(float).to_numpy(),index=times).sort_index()
    if s.index.duplicated().any():raise ValueError('Duplicate funding settlements')
    return s.cumsum().reindex(f.index,method='ffill').fillna(0).to_numpy()


def touch_outcomes(f, direction, target, hours, funding=None):
    """First target/adverse touches and path risk, with conservative intrabar ordering."""
    n=len(f);steps=hours*4;sign=1 if direction=='long' else -1
    entry=f.open.shift(-1).to_numpy();high=f.high.to_numpy();low=f.low.to_numpy();close=f.close.to_numpy()
    first_hit=np.full(n,-1,dtype=int);worst=np.zeros(n);stops={p:np.full(n,-1,dtype=int) for p in (.02,.03,.05)}
    for k in range(1,steps+1):
        size=n-k;e=entry[:size]
        favorable=high[k:] if sign==1 else low[k:]
        adverse=low[k:]/e-1 if sign==1 else 1-high[k:]/e
        active=first_hit[:size]<0
        worst[:size]=np.where(active,np.minimum(worst[:size],adverse),worst[:size])
        touched=sign*(favorable-e*(1+sign*target))>=0
        first_hit[:size]=np.where(active&touched,k,first_hit[:size])
        for barrier,first_stop in stops.items():
            bad=(low[k:]<=e*(1-barrier)) if sign==1 else (high[k:]>=e*(1+barrier))
            first_stop[:size]=np.where((first_stop[:size]<0)&bad,k,first_stop[:size])
    complete=np.arange(n)+steps<n
    hit=first_hit>0
    duration=np.where(hit,first_hit,steps)
    exit_indices=np.minimum(np.arange(n)+duration,n-1)
    gross=np.where(hit,target,sign*(close[exit_indices]/entry-1))
    cumulative=np.zeros(n) if funding is None else funding
    funding_cost=sign*(cumulative[exit_indices]-cumulative)
    result=pd.DataFrame({'touch':np.where(complete,hit.astype(float),np.nan),'first_touch_hours':np.where(hit,first_hit/4,np.nan),
                         'adverse':np.where(complete,worst,np.nan),'gross_return':np.where(complete,gross,np.nan),
                         'funding_cost':funding_cost,'net_10bps':np.where(complete,gross-funding_cost-.001,np.nan),
                         'net_20bps':np.where(complete,gross-funding_cost-.002,np.nan)},index=f.index)
    for barrier,stop in stops.items():
        key=str(int(barrier*100))
        result['before_adverse_'+key]=np.where(complete,hit&((stop<0)|(first_hit<stop)),np.nan)
        result['ambiguous_'+key]=complete&hit&(first_hit==stop)
    return result


def candidates(f, with_derivatives=True):
    columns=PRICE+(DERIV if with_derivatives else [])
    return (f.index.minute==0)&f.ret1.abs().lt(.01)&f.ret4.abs().lt(.02)&np.isfinite(f[columns]).all(axis=1)


def fold_windows(end):
    result=[]
    for i in range(3):
        test_start=end-pd.Timedelta(days=270-i*90)
        cal_start=test_start-pd.Timedelta(days=90)
        train_start=cal_start-pd.Timedelta(days=360)
        result.append({'train':(train_start,cal_start),'calibration':(cal_start+72*HOUR,test_start),
                       'test':(test_start+72*HOUR,test_start+pd.Timedelta(days=90))})
    return result


def split_index(f,window,eligible):
    a,b=window
    return f.index[eligible&(f.index>=a)&(f.index+73*HOUR<b)]
