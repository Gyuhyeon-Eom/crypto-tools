"""Past-only structural stops, evaluated after model selection is frozen."""
import numpy as np
import pandas as pd


def levels(f, direction):
    low = direction == 'long'
    s = f.low if low else f.high
    result = {}
    for hours in (4, 12, 24):
        result[f'prior_{hours}h'] = s.rolling(hours * 4).min() if low else s.rolling(hours * 4).max()
    # At time t, a pivot at t-2 is confirmed by two fully closed right bars.
    center = s.shift(2)
    conditions = [(center < s.shift(k) if low else center > s.shift(k)) for k in (0, 1, 3, 4)]
    confirmed = np.logical_and.reduce(conditions)
    pivot = center.where(confirmed).ffill()
    when = pd.Series(f.index, index=f.index).where(confirmed).ffill() - pd.Timedelta(minutes=30)
    result['confirmed_pivot'] = pivot.where((pd.Series(f.index,index=f.index)-when) <= pd.Timedelta(hours=24))
    return result


def evaluate_stop(f, direction, target, hours, stop, funding=None):
    n = len(f); sign = 1 if direction == 'long' else -1; steps = hours * 4
    entry = f.open.shift(-1).to_numpy(); sl = stop.to_numpy(); tp = entry * (1 + sign * target)
    valid = np.isfinite(sl) & (sign*(entry-sl)>0) & (np.arange(n)+steps<n)
    state = np.zeros(n, dtype=int); exit_step = np.full(n, steps); exit_price = np.full(n,np.nan); ambiguous = np.zeros(n,bool)
    for k in range(1, steps+1):
        size=n-k
        active=valid[:size] & (state[:size]==0)
        if sign==1:
            win=f.high.to_numpy()[k:]>=tp[:size]; lose=f.low.to_numpy()[k:]<=sl[:size]
            fill=np.minimum(sl[:size], f.open.to_numpy()[k:])
        else:
            win=f.low.to_numpy()[k:]<=tp[:size]; lose=f.high.to_numpy()[k:]>=sl[:size]
            fill=np.maximum(sl[:size], f.open.to_numpy()[k:])
        done=active & (win|lose)
        ambiguous[:size] |= active & win & lose
        state[:size]=np.where(done,np.where(lose,-1,1),state[:size])
        exit_step[:size]=np.where(done,k,exit_step[:size])
        exit_price[:size]=np.where(done,np.where(lose,fill,tp[:size]),exit_price[:size])
    end=np.minimum(np.arange(n)+exit_step,n-1)
    exit_price=np.where(state==0,f.close.to_numpy()[end],exit_price)
    cum=np.zeros(n) if funding is None else funding
    net=sign*(exit_price/entry-1)-sign*(cum[end]-cum)-.002
    return pd.DataFrame({'valid_stop':valid,'stop_distance_pct':np.where(valid,sign*(entry-sl)/entry*100,np.nan),
                         'target_before_stop':np.where(valid,state==1,np.nan),'stopped':np.where(valid,state==-1,np.nan),
                         'timeout':np.where(valid,state==0,np.nan),'ambiguous':valid&ambiguous,
                         'net20':np.where(valid,net,np.nan),'exit_hours':np.where(valid,exit_step/4,np.nan)},index=f.index)
