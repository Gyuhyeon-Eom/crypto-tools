#!/usr/bin/env python3
"""
Crypto Surge Scanner - Standalone
Loads validated model from data/thresholds.json and scans cached market data.
Run pipeline_v2.py first to generate data and model.
"""
import json, os, time, math
import urllib.request, urllib.error
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

def load_thresholds():
    p = DATA_DIR / "thresholds.json"
    if not p.exists():
        print("Error: thresholds.json not found. Run pipeline_v2.py first.")
        return {}
    with open(p) as f:
        return json.load(f)

def compute_rsi(closes, period=14):
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    rsi = np.full(len(closes), np.nan)
    if len(deltas) < period: return rsi
    ag = np.mean(gains[:period]); al = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        ag = (ag*(period-1)+gains[i])/period; al = (al*(period-1)+losses[i])/period
        rsi[i+1] = 100.0 if al==0 else 100.0-100.0/(1.0+ag/al)
    fg=np.mean(gains[:period]); fl=np.mean(losses[:period])
    rsi[period] = 100.0 if fl==0 else 100.0-100.0/(1.0+fg/fl)
    return rsi

def compute_atr(highs, lows, closes, period=14):
    n=len(closes); tr=np.zeros(n); tr[0]=highs[0]-lows[0]
    for i in range(1,n): tr[i]=max(highs[i]-lows[i],abs(highs[i]-closes[i-1]),abs(lows[i]-closes[i-1]))
    atr=np.full(n,np.nan)
    if n>=period:
        atr[period-1]=np.mean(tr[:period])
        for i in range(period,n): atr[i]=(atr[i-1]*(period-1)+tr[i])/period
    return atr

def extract_features(df, idx, btc_df=None):
    if idx<44: return None
    c=df["close"].values; h=df["high"].values; l=df["low"].values; v=df["volume"].values; o=df["open"].values
    pe=idx; ps=idx-14
    if ps<0: return None
    f={}
    v7=np.mean(v[pe-7:pe]); v30=np.mean(v[max(0,pe-30):pe]); v20=np.mean(v[max(0,pe-20):pe])
    f["vol_7d_30d_ratio"]=v7/v30 if v30>0 else 1.0
    f["vol_spike_3d"]=np.max(v[pe-3:pe])/v20 if v20>0 else 1.0
    if pe>=7:
        vv=v[pe-7:pe]; x=np.arange(7,dtype=float)
        f["vol_trend_7d"]=(np.polyfit(x,vv,1)[0]/np.mean(vv)) if np.std(vv)>0 and np.mean(vv)>0 else 0.0
    else: f["vol_trend_7d"]=0.0
    af=compute_atr(h,l,c,14)
    a14=af[pe-1] if not np.isnan(af[pe-1]) else 0
    f["atr_14d_ratio"]=a14/c[pe-1] if c[pe-1]>0 else 0
    a7v=[af[j] for j in range(pe-7,pe) if not np.isnan(af[j])]
    a30v=[af[j] for j in range(max(0,pe-30),pe) if not np.isnan(af[j])]
    a7=np.mean(a7v) if a7v else a14; a30=np.mean(a30v) if a30v else a14
    f["volatility_contraction"]=a7/a30 if a30>0 else 1.0
    if pe>=20:
        bc=c[pe-20:pe]; bm=np.mean(bc); bs=np.std(bc)
        f["bb_width"]=(2*bs)/bm if bm>0 else 0
    else: f["bb_width"]=0
    f["price_vs_60d_high"]=c[pe-1]/np.max(h[pe-60:pe]) if pe>=60 and np.max(h[pe-60:pe])>0 else 1.0
    f["price_ret_7d"]=c[pe-1]/c[pe-8]-1 if pe>=8 and c[pe-8]>0 else 0
    f["price_vs_ma20"]=c[pe-1]/np.mean(c[pe-20:pe])-1 if pe>=20 and np.mean(c[pe-20:pe])>0 else 0
    rf=compute_rsi(c,14); rv=rf[pe-1]
    f["rsi_14"]=rv if not np.isnan(rv) else 50.0
    rw=rf[ps:pe]; vr=rw[~np.isnan(rw)]
    f["rsi_consolidation_days"]=int(np.sum((vr>=40)&(vr<=55)))
    f["rsi_trend_7d"]=rf[pe-1]-rf[pe-8] if pe>=8 and not np.isnan(rf[pe-1]) and not np.isnan(rf[pe-8]) else 0.0
    f["rsi_oversold_days"]=int(np.sum(vr<30)) if len(vr)>0 else 0
    cr=0
    for j in range(pe-1,max(ps-1,-1),-1):
        if c[j]<o[j]: cr+=1
        else: break
    f["consecutive_red"]=cr
    f["green_ratio_3d"]=sum(1 for j in range(pe-3,pe) if c[j]>=o[j])/3.0
    dh=0
    for j in range(pe-3,pe):
        bd=abs(c[j]-o[j]); tr2=h[j]-l[j]
        if tr2>0:
            if bd/tr2<0.1: dh+=1
            elif (min(o[j],c[j])-l[j])>2*bd and (min(o[j],c[j])-l[j])>0.5*tr2: dh+=1
    f["doji_hammer_count"]=dh
    bodies=[abs(c[j]-o[j]) for j in range(max(0,pe-7),pe)]
    ranges=[h[j]-l[j] for j in range(max(0,pe-7),pe)]
    f["body_range_ratio"]=np.mean(bodies)/np.mean(ranges) if np.mean(ranges)>0 else 0
    if btc_df is not None and pe>=8:
        cd=df["open_time"].dt.date.values; bd=btc_df["open_time"].dt.date.values
        bpn=bp7=None
        for bi in range(len(bd)-1,-1,-1):
            if bd[bi]<=cd[pe-1] and bpn is None: bpn=btc_df["close"].values[bi]
            if bd[bi]<=cd[pe-8] and bp7 is None: bp7=btc_df["close"].values[bi]
            if bpn is not None and bp7 is not None: break
        cr7=c[pe-1]/c[pe-8]-1 if c[pe-8]>0 else 0
        f["vs_btc_7d"]=(cr7-(bpn/bp7-1)) if bpn and bp7 and bp7>0 else 0.0
        f["coin_ret_7d"]=cr7
    else: f["vs_btc_7d"]=0.0; f["coin_ret_7d"]=0.0
    if pe>=14:
        pc14=abs(c[pe-1]/c[pe-14]-1); vc14=np.mean(v[pe-7:pe])/np.mean(v[pe-14:pe-7]) if np.mean(v[pe-14:pe-7])>0 else 1
        f["accumulation_signal"]=vc14/(1+pc14*10)
    else: f["accumulation_signal"]=1.0
    return f

def compute_score(features, thresholds):
    """Gaussian scoring: peak score at surge_mean, decays as value moves away.
    Values beyond surge_mean (overshoot) are penalized, not rewarded.
    This catches 'precursors' not 'already surging' coins."""
    if not features or not thresholds: return 0.0
    sc=0.0; mx=0.0
    for fn,info in thresholds.items():
        if fn not in features: continue
        w=info["weight"]; mx+=w; val=features[fn]
        sm=info["surge_mean"]; ss=info["surge_std"]
        nm=info["normal_mean"]
        d=info["direction"]
        # Gaussian: score peaks at surge_mean, decays with distance
        if ss > 0:
            z = (val - sm) / ss
            gauss = math.exp(-0.5 * z * z)  # 1.0 at surge_mean, decays
        else:
            gauss = 1.0 if val == sm else 0.0
        # Bonus: if value is between normal_mean and surge_mean, still decent
        # Penalty: if value is on wrong side of normal_mean, extra penalty
        if d == "higher":
            if val < nm:
                gauss *= 0.3  # below normal = strong penalty
        else:  # direction == "lower"
            if val > nm:
                gauss *= 0.3  # above normal = strong penalty
        sc += w * gauss
    return (sc/mx*100) if mx>0 else 0.0

def main():
    print("="*50)
    print("  Crypto Surge Precursor Scanner")
    print("="*50)
    TH=load_thresholds()
    if not TH: return
    pf=DATA_DIR/"top_pairs.json"
    if not pf.exists(): print("No pairs. Run pipeline_v2.py first."); return
    with open(pf) as ff: pairs=json.load(ff)
    bf=DATA_DIR/"BTCUSDT_1d.parquet"
    btc=pd.read_parquet(bf) if bf.exists() else None
    results=[]
    for sym in pairs:
        df_path=DATA_DIR/(sym+"_1d.parquet")
        if not df_path.exists(): continue
        df=pd.read_parquet(df_path)
        if len(df)<60: continue
        feat=extract_features(df,len(df),btc)
        if not feat: continue
        sc=compute_score(feat,TH)
        results.append({"symbol":sym,"score":sc,"features":feat,"last_close":float(df["close"].values[-1])})
    results.sort(key=lambda x:x["score"],reverse=True)
    print("\nScanned {} coins\n".format(len(results)))
    print("{:<5} {:<12} {:<8} {:<12}".format("Rank","Coin","Score","Price"))
    print("-"*40)
    for i,r in enumerate(results[:20]):
        cn=r["symbol"].replace("USDT","")
        print("{:<5} {:<12} {:<8.1f} ${:<12.4f}".format(i+1,cn,r["score"],r["last_close"]))
    print("\n--- Top 10 Detail ---\n")
    for i,r in enumerate(results[:10]):
        cn=r["symbol"].replace("USDT","")
        print("{}. {} (Score: {:.1f})".format(i+1,cn,r["score"]))
        for fn in sorted(TH.keys(),key=lambda x:TH[x]["weight"],reverse=True):
            if fn in r["features"]:
                print("   {}: {:.4f} (surge: {:.4f}, normal: {:.4f})".format(
                    fn,r["features"][fn],TH[fn]["surge_mean"],TH[fn]["normal_mean"]))
        print()

if __name__=="__main__": main()
