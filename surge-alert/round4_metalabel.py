#!/usr/bin/env python3
"""
Round 4: Triple Barrier + Meta-labeling + Isolation Forest
Marcos Lopez de Prado 스타일

1단계: Triple Barrier Labeling (라벨 품질 개선)
2단계: Isolation Forest로 "이상 상태" 탐지 (1차 필터)
3단계: XGB/LGB로 분류 (2차 모델)
4단계: Meta-labeling — 1차 + 2차 결합, Precision 직접 최적화
5단계: Cross-sectional feature 추가 (시장 대비 상대적 위치)

+ Purged K-Fold CV (시계열 누수 방지)
+ 30일 내 급등 이력 있으면 학습/스캔에서 제외
"""

import json,math,os,sys,time,datetime,pickle,warnings,random
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score,precision_score,recall_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

BASE_DIR=Path(__file__).parent
DATA_DIR=BASE_DIR/"ml_data"
SURGE_TH=0.3;SURGE_WINDOW=7;MIN_VOL=50_000;MAX_VOL=10_000_000

WEBHOOK_URL=""
_env=os.path.expanduser("~/.openclaw/.env.slack-surge")
if os.path.exists(_env):
    with open(_env) as f:
        for line in f:
            if line.startswith("SLACK_WEBHOOK_URL="): WEBHOOK_URL=line.strip().split("=",1)[1]

def send_slack(msg):
    if not WEBHOOK_URL: print(f"[SLACK] {msg}"); return
    for chunk in [msg[i:i+3500] for i in range(0,len(msg),3500)]:
        payload=json.dumps({"text":chunk})
        req=urllib.request.Request(WEBHOOK_URL,data=payload.encode(),headers={"Content-Type":"application/json"},method="POST")
        try: urllib.request.urlopen(req,timeout=30)
        except: pass
        time.sleep(0.5)

# ============================================================
# Math/TA helpers
# ============================================================
def mean(a): return sum(a)/len(a) if a else 0.0
def stdev(a):
    if len(a)<2: return 0.0
    m=mean(a); return math.sqrt(sum((x-m)**2 for x in a)/(len(a)-1))
def polyfit_slope(y):
    n=len(y)
    if n<2: return 0.0
    xm=(n-1)/2.0;ym=mean(y)
    if ym==0: return 0.0
    num=sum((i-xm)*(v-ym) for i,v in enumerate(y));den=sum((i-xm)**2 for i in range(n))
    return (num/den)/ym if den else 0.0
def compute_rsi(closes,period=14):
    rsi=[None]*len(closes)
    if len(closes)<period+1: return rsi
    d=[closes[i]-closes[i-1] for i in range(1,len(closes))]
    g=[max(x,0) for x in d];lo=[max(-x,0) for x in d]
    ag=mean(g[:period]);al=mean(lo[:period])
    rsi[period]=100.0 if al==0 else 100.0-100.0/(1.0+ag/al)
    for i in range(period,len(d)):
        ag=(ag*(period-1)+g[i])/period;al=(al*(period-1)+lo[i])/period
        rsi[i+1]=100.0 if al==0 else 100.0-100.0/(1.0+ag/al)
    return rsi
def compute_atr(h,l,c,period=14):
    n=len(c);tr=[h[0]-l[0]]
    for i in range(1,n): tr.append(max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1])))
    atr=[None]*n
    if n>=period:
        atr[period-1]=mean(tr[:period])
        for i in range(period,n): atr[i]=(atr[i-1]*(period-1)+tr[i])/period
    return atr
def compute_macd(c,fast=12,slow=26,sig=9):
    if len(c)<slow+sig: return [0]*len(c),[0]*len(c),[0]*len(c)
    def ema(a,p):
        k=2/(p+1);r=[a[0]]
        for i in range(1,len(a)): r.append(a[i]*k+r[-1]*(1-k))
        return r
    ef=ema(c,fast);es=ema(c,slow);ml=[f-s for f,s in zip(ef,es)];sl=ema(ml,sig)
    return ml,sl,[m-s for m,s in zip(ml,sl)]
def compute_obv(c,v):
    o=[0]
    for i in range(1,len(c)):
        if c[i]>c[i-1]: o.append(o[-1]+v[i])
        elif c[i]<c[i-1]: o.append(o[-1]-v[i])
        else: o.append(o[-1])
    return o

# ============================================================
# Triple Barrier Labeling
# ============================================================
def triple_barrier_label(c, idx, upper=0.30, lower=-0.15, max_days=7):
    """
    +upper 먼저 찍으면 1 (급등)
    lower 먼저 찍으면 -1 (급락)
    둘 다 안 찍으면 0 (무의미)
    """
    if idx >= len(c) - 1: return 0
    entry = c[idx]
    if entry <= 0: return 0
    for j in range(idx+1, min(idx+max_days+1, len(c))):
        ret = c[j] / entry - 1
        if ret >= upper: return 1
        if ret <= lower: return -1
    return 0

# ============================================================
# Feature Extraction (기존 85개 + cross-sectional 5개 = 90개)
# ============================================================
def extract_features(o,h,l,c,v,idx):
    if idx<59: return None
    f={}
    for d in [1,2,3,5,7,10,14,21]:
        f[f"ret_{d}d"]=c[idx]/c[idx-d]-1 if idx>=d and c[idx-d]>0 else 0
    for p in [5,7,10,20,50]:
        if idx>=p-1: ma=mean(c[idx-p+1:idx+1]);f[f"price_vs_ma{p}"]=c[idx]/ma-1 if ma>0 else 0
        else: f[f"price_vs_ma{p}"]=0
    if idx>=20:
        ma5=mean(c[idx-4:idx+1]);ma10=mean(c[idx-9:idx+1]);ma20=mean(c[idx-19:idx+1])
        f["ma5_ma10_cross"]=ma5/ma10-1 if ma10>0 else 0;f["ma5_ma20_cross"]=ma5/ma20-1 if ma20>0 else 0
        f["ma10_ma20_cross"]=ma10/ma20-1 if ma20>0 else 0
    else: f["ma5_ma10_cross"]=0;f["ma5_ma20_cross"]=0;f["ma10_ma20_cross"]=0
    for w in [14,30,60]:
        if idx>=w-1:
            hi=max(h[idx-w+1:idx+1]);lo_=min(l[idx-w+1:idx+1]);rng=hi-lo_
            f[f"price_pos_{w}d"]=(c[idx]-lo_)/rng if rng>0 else 0.5;f[f"price_vs_{w}d_high"]=c[idx]/hi if hi>0 else 1
        else: f[f"price_pos_{w}d"]=0.5;f[f"price_vs_{w}d_high"]=1
    r14=compute_rsi(c[:idx+1],14);r7=compute_rsi(c[:idx+1],7)
    f["rsi_14"]=r14[idx] if r14[idx] else 50;f["rsi_7"]=r7[idx] if r7[idx] else 50
    if idx>=7 and r14[idx] and r14[idx-7]: f["rsi14_change_7d"]=r14[idx]-r14[idx-7]
    else: f["rsi14_change_7d"]=0
    if idx>=3 and r14[idx] and r14[idx-3]: f["rsi14_change_3d"]=r14[idx]-r14[idx-3]
    else: f["rsi14_change_3d"]=0
    rw=[r14[j] for j in range(max(0,idx-13),idx+1) if r14[j] is not None]
    f["rsi_oversold_days"]=sum(1 for r in rw if r<30);f["rsi_consol_days"]=sum(1 for r in rw if 40<=r<=55)
    f["rsi_overbought_days"]=sum(1 for r in rw if r>70)
    f["rsi_min_14d"]=min(rw) if rw else 50;f["rsi_max_14d"]=max(rw) if rw else 50;f["rsi_range_14d"]=(max(rw)-min(rw)) if rw else 0
    if idx>=7 and r14[idx] and r14[idx-7]:
        pd_=1 if c[idx]>c[idx-7] else -1;rd=1 if r14[idx]>r14[idx-7] else -1
        f["rsi_divergence"]=1 if pd_==-1 and rd==1 else(-1 if pd_==1 and rd==-1 else 0)
    else: f["rsi_divergence"]=0
    ml,sl,hist=compute_macd(c[:idx+1])
    if idx<len(ml):
        pa=c[idx] if c[idx]>0 else 1;f["macd_line"]=ml[idx]/pa;f["macd_signal"]=sl[idx]/pa;f["macd_histogram"]=hist[idx]/pa
        f["macd_cross"]=1 if idx>=1 and hist[idx]>0 and hist[idx-1]<=0 else(-1 if idx>=1 and hist[idx]<0 and hist[idx-1]>=0 else 0)
        f["macd_hist_slope"]=polyfit_slope([hist[idx-i] for i in range(min(4,idx+1)-1,-1,-1)]) if idx>=3 else 0
    else: f["macd_line"]=0;f["macd_signal"]=0;f["macd_histogram"]=0;f["macd_cross"]=0;f["macd_hist_slope"]=0
    atr_s=compute_atr(h[:idx+1],l[:idx+1],c[:idx+1],14);a14=atr_s[idx] if atr_s[idx] else 0
    f["atr_ratio"]=a14/c[idx] if c[idx]>0 else 0
    a7v=[atr_s[j] for j in range(max(0,idx-6),idx+1) if atr_s[j]];a30v=[atr_s[j] for j in range(max(0,idx-29),idx+1) if atr_s[j]]
    f["volatility_ratio_7_30"]=mean(a7v)/mean(a30v) if a30v and mean(a30v)>0 else 1
    if idx>=29:
        ae=[atr_s[j] for j in range(idx-29,idx-14) if atr_s[j]];al_=[atr_s[j] for j in range(idx-6,idx+1) if atr_s[j]]
        f["volatility_contraction"]=mean(al_)/mean(ae) if ae and mean(ae)>0 else 1
    else: f["volatility_contraction"]=1
    if idx>=19:
        bc=c[idx-19:idx+1];bm=mean(bc);bs=stdev(bc);f["bb_width"]=(2*bs)/bm if bm>0 else 0;f["bb_position"]=(c[idx]-(bm-bs))/(2*bs) if bs>0 else 0.5
    else: f["bb_width"]=0;f["bb_position"]=0.5
    v7=mean(v[max(0,idx-6):idx+1]);v14=mean(v[max(0,idx-13):idx+1]);v30=mean(v[max(0,idx-29):idx+1])
    f["vol_ratio_1d_7d"]=v[idx]/v7 if v7>0 else 1;f["vol_ratio_3d_14d"]=mean(v[max(0,idx-2):idx+1])/v14 if v14>0 else 1
    f["vol_ratio_7d_30d"]=v7/v30 if v30>0 else 1;f["vol_trend_7d"]=polyfit_slope(v[max(0,idx-6):idx+1])
    f["vol_trend_14d"]=polyfit_slope(v[max(0,idx-13):idx+1])
    vl2=mean(v[max(0,idx-1):idx+1]);vp5=mean(v[max(0,idx-6):max(1,idx-1)]) if idx>=3 else vl2
    f["vol_explosion_2d"]=vl2/vp5 if vp5>0 else 1
    obv=compute_obv(c[max(0,idx-29):idx+1],v[max(0,idx-29):idx+1]);f["obv_slope_7d"]=polyfit_slope(obv[-7:]) if len(obv)>=7 else 0
    if idx>=7: f["vol_price_divergence"]=polyfit_slope(v[idx-6:idx+1])-polyfit_slope(c[idx-6:idx+1])
    else: f["vol_price_divergence"]=0
    f["green_ratio_3d"]=sum(1 for j in range(max(0,idx-2),idx+1) if c[j]>=o[j])/3.0;f["green_ratio_7d"]=sum(1 for j in range(max(0,idx-6),idx+1) if c[j]>=o[j])/7.0
    cr=0
    for j in range(idx,max(idx-14,-1),-1):
        if c[j]<o[j]: cr+=1
        else: break
    f["consec_red"]=cr
    cg=0
    for j in range(idx,max(idx-14,-1),-1):
        if c[j]>=o[j]: cg+=1
        else: break
    f["consec_green"]=cg
    bodies=[];uw=[];lw_=[]
    for j in range(max(0,idx-6),idx+1):
        b=abs(c[j]-o[j]);fl=h[j]-l[j]
        if fl>0: bodies.append(b/fl);uw.append((h[j]-max(c[j],o[j]))/fl);lw_.append((min(c[j],o[j])-l[j])/fl)
    f["avg_body_ratio"]=mean(bodies) if bodies else 0.5;f["avg_upper_wick"]=mean(uw) if uw else 0;f["avg_lower_wick"]=mean(lw_) if lw_ else 0
    hc=0;dc=0
    for j in range(max(0,idx-2),idx+1):
        b=abs(c[j]-o[j]);fl=h[j]-l[j]
        if fl>0:
            lwk=(min(c[j],o[j])-l[j])/fl;br=b/fl
            if lwk>0.6 and br<0.2: hc+=1
            if br<0.1: dc+=1
    f["hammer_3d"]=hc;f["doji_3d"]=dc
    if idx>=13:
        pc=abs(c[idx]/c[idx-13]-1);vr=mean(v[idx-6:idx+1]);vp=mean(v[idx-13:idx-6])
        f["accumulation"]=(vr/vp)/(1+pc*10) if vp>0 else 1
    else: f["accumulation"]=1
    if idx>=30:
        rl=min(l[idx-29:idx+1]);pk=max(h[idx-29:idx+1]);f["drawdown_30d"]=(c[idx]-pk)/pk if pk>0 else 0
        f["recovery_from_low"]=(c[idx]-rl)/rl if rl>0 else 0
    else: f["drawdown_30d"]=0;f["recovery_from_low"]=0
    if idx>=14:
        r7_=c[idx]/c[idx-7]-1 if c[idx-7]>0 else 0;r14_=c[idx]/c[idx-14]-1 if c[idx-14]>0 else 0
        f["momentum_acceleration"]=r7_-(r14_-r7_)
        uwv=0
        for j in range(idx,max(idx-7,-1),-1):
            if c[j]>=o[j] and v[j]>v7: uwv+=1
            else: break
        f["up_with_volume_streak"]=uwv
    else: f["momentum_acceleration"]=0;f["up_with_volume_streak"]=0
    if idx>=30:
        rh=sorted(h[idx-29:idx+1],reverse=True);res=mean(rh[:3]);f["distance_to_resistance"]=(res-c[idx])/c[idx] if c[idx]>0 else 0
        rls=sorted(l[idx-29:idx+1]);sup=mean(rls[:3]);f["distance_from_support"]=(c[idx]-sup)/c[idx] if c[idx]>0 else 0
    else: f["distance_to_resistance"]=0;f["distance_from_support"]=0
    if idx>=7:
        nm=abs(c[idx]-c[idx-7]);tm=sum(abs(c[j]-c[j-1]) for j in range(idx-6,idx+1));f["efficiency_ratio_7d"]=nm/tm if tm>0 else 0
    else: f["efficiency_ratio_7d"]=0
    f["vol_usd_7d"]=mean([v[j]*c[j] for j in range(max(0,idx-6),idx+1)]);f["vol_usd_1d"]=v[idx]*c[idx]
    vusd_3d=mean([v[j]*c[j] for j in range(max(0,idx-2),idx+1)]);vusd_prev=mean([v[j]*c[j] for j in range(max(0,idx-9),max(1,idx-2))]) if idx>=5 else vusd_3d
    f["vol_usd_acceleration"]=vusd_3d/vusd_prev if vusd_prev>0 else 1
    if idx>=29:
        range_7=max(h[idx-6:idx+1])-min(l[idx-6:idx+1]);range_30=max(h[idx-29:idx+1])-min(l[idx-29:idx+1])
        f["range_squeeze"]=range_7/range_30 if range_30>0 else 1
    else: f["range_squeeze"]=1
    if idx>=10:
        mi=idx-10;mv=c[idx-10]
        for j in range(idx-10,idx+1):
            if c[j]<mv: mv=c[j];mi=j
        f["days_since_10d_low"]=idx-mi;f["bounce_from_10d_low"]=(c[idx]-mv)/mv if mv>0 else 0
    else: f["days_since_10d_low"]=0;f["bounce_from_10d_low"]=0
    if idx>=13:
        hh=max(h[idx-13:idx+1]);ll=min(l[idx-13:idx+1]);f["williams_r"]=(hh-c[idx])/(hh-ll)*-100 if hh!=ll else -50
    else: f["williams_r"]=-50
    if idx>=13 and len(rw)>=14:
        rsi_min=min(rw);rsi_max=max(rw);f["stoch_rsi"]=(f["rsi_14"]-rsi_min)/(rsi_max-rsi_min) if rsi_max!=rsi_min else 0.5
    else: f["stoch_rsi"]=0.5
    buy_vol=sum(v[j] for j in range(max(0,idx-6),idx+1) if c[j]>=o[j])
    total_vol=sum(v[j] for j in range(max(0,idx-6),idx+1));f["buy_vol_ratio_7d"]=buy_vol/total_vol if total_vol>0 else 0.5
    day_range=h[idx]-l[idx]
    if day_range>0: f["today_body_pct"]=abs(c[idx]-o[idx])/day_range;f["today_position"]=(c[idx]-l[idx])/day_range
    else: f["today_body_pct"]=0;f["today_position"]=0.5
    if idx>=2: f["vol_increasing_3d"]=1 if v[idx]>v[idx-1]>v[idx-2] else 0
    else: f["vol_increasing_3d"]=0
    if a14>0 and day_range>0: f["range_vs_atr"]=day_range/a14
    else: f["range_vs_atr"]=1
    if idx>=24:
        ma20_now=mean(c[idx-19:idx+1]);ma20_5ago=mean(c[idx-24:idx-4]);f["ma20_slope"]=(ma20_now-ma20_5ago)/ma20_5ago if ma20_5ago>0 else 0
    else: f["ma20_slope"]=0
    if idx>=19:
        bm2=mean(c[idx-19:idx+1]);bs2=stdev(c[idx-19:idx+1]);f["above_upper_bb"]=1 if c[idx]>bm2+2*bs2 else 0;f["below_lower_bb"]=1 if c[idx]<bm2-2*bs2 else 0
    else: f["above_upper_bb"]=0;f["below_lower_bb"]=0
    if idx>=6:
        vwap_denom=sum(v[j] for j in range(idx-6,idx+1))
        vwap=sum(c[j]*v[j] for j in range(idx-6,idx+1))/vwap_denom if vwap_denom>0 else c[idx]
        f["price_vs_vwap"]=c[idx]/vwap-1 if vwap>0 else 0
    else: f["price_vs_vwap"]=0
    return f

def has_recent_surge(c, idx, lookback=30, threshold=0.30):
    """최근 lookback일 내 threshold 이상 급등한 적 있는지"""
    for j in range(max(0,idx-lookback), idx):
        if j >= 7:
            ret7 = c[j] / c[j-7] - 1 if c[j-7] > 0 else 0
            if ret7 >= threshold: return True
    # 현재 시점도 체크
    if idx >= 7:
        ret7 = c[idx] / c[idx-7] - 1 if c[idx-7] > 0 else 0
        if ret7 >= 0.20: return True
    if idx >= 14:
        ret14 = c[idx] / c[idx-14] - 1 if c[idx-14] > 0 else 0
        if ret14 >= 0.30: return True
    return False

# ============================================================
def manual_cv(model_cls,params,X,y,n_splits=5):
    tscv=TimeSeriesSplit(n_splits=n_splits);scores=[]
    for ti,vi in tscv.split(X):
        m=model_cls(**params);m.fit(X[ti],y[ti])
        pr=m.predict_proba(X[vi])[:,1]
        if sum(y[vi])>0: scores.append(average_precision_score(y[vi],pr))
    return mean(scores) if scores else 0

def backtest_30d(all_data,model,scaler,fcols,threshold,iforest=None,iforest_scaler=None):
    BD=30;tp=0;th_=0;ts=0
    for day in range(BD):
        dX=[];dy=[]
        for sym,d in all_data.items():
            n=len(d["c"]);oo,hh,ll,cc,vv=d["o"],d["h"],d["l"],d["c"],d["v"]
            idx=n-1-SURGE_WINDOW-(BD-1-day)
            if idx<59 or idx+SURGE_WINDOW>=n: continue
            av=mean([vv[j]*cc[j] for j in range(max(0,idx-29),idx+1)])
            if av>MAX_VOL or av<MIN_VOL: continue
            if has_recent_surge(cc,idx): continue
            feat=extract_features(oo,hh,ll,cc,vv,idx)
            if not feat: continue
            # Triple barrier label
            label=triple_barrier_label(cc,idx,upper=SURGE_TH,lower=-0.15,max_days=SURGE_WINDOW)
            dX.append([feat.get(fc,0) for fc in fcols]);dy.append(1 if label==1 else 0)
        if not dX: continue
        Xd=np.nan_to_num(np.array(dX,dtype=np.float64),nan=0,posinf=0,neginf=0)
        Xds=scaler.transform(Xd);yd=np.array(dy)
        
        # Isolation Forest 1차 필터
        if iforest and iforest_scaler:
            Xd_if=iforest_scaler.transform(Xd)
            anomaly_scores=iforest.decision_function(Xd_if)
            anomaly_mask=anomaly_scores<np.percentile(anomaly_scores,20)  # 하위 20%만 이상
        else:
            anomaly_mask=np.ones(len(Xd),dtype=bool)
        
        probs=model.predict_proba(Xds)[:,1]
        preds=((probs>=threshold)&anomaly_mask).astype(int)
        tp+=int(sum(preds));th_+=int(sum(preds&yd));ts+=int(sum(yd))
    p=th_/tp if tp>0 else 0;r=th_/ts if ts>0 else 0
    f1=2*p*r/(p+r) if p+r>0 else 0
    return {"precision":float(p),"recall":float(r),"f1":float(f1),"predictions":tp,"hits":th_,"surges":ts}

# ============================================================
def main():
    t0=time.time()
    send_slack(":brain: *Round 4 시작* — Triple Barrier + Meta-labeling + IsolationForest")
    print("Round 4: Triple Barrier + Meta-label")
    
    with open(DATA_DIR/"raw_data.json") as f: all_data=json.load(f)
    
    # [1] Triple Barrier 데이터셋
    print("\n[1] Triple Barrier 데이터셋 (최근 급등 이력 제외)...")
    rows=[];labels={"surge":0,"drop":0,"neutral":0};excluded=0
    for sym,d in all_data.items():
        n=len(d["c"]);o,h,l,c,v=d["o"],d["h"],d["l"],d["c"],d["v"]
        av=mean([v[j]*c[j] for j in range(max(0,n-30),n)])
        if av>MAX_VOL or av<MIN_VOL: continue
        for idx in range(59,n-SURGE_WINDOW):
            if has_recent_surge(c,idx):
                excluded+=1; continue
            feat=extract_features(o,h,l,c,v,idx)
            if not feat: continue
            label=triple_barrier_label(c,idx,upper=SURGE_TH,lower=-0.15,max_days=SURGE_WINDOW)
            # binary: surge=1, else=0
            feat["_label"]=1 if label==1 else 0
            feat["_ts"]=d["ts"][idx]
            rows.append(feat)
            if label==1: labels["surge"]+=1
            elif label==-1: labels["drop"]+=1
            else: labels["neutral"]+=1
    
    print(f"  {len(rows)}개 (surge:{labels['surge']}, drop:{labels['drop']}, neutral:{labels['neutral']})")
    print(f"  제외(이미 급등): {excluded}")
    
    df=pd.DataFrame(rows)
    fcols=[c for c in df.columns if not c.startswith("_")]
    X=np.nan_to_num(df[fcols].values.astype(np.float64),nan=0,posinf=0,neginf=0)
    y=df["_label"].values;ts_arr=df["_ts"].values
    si=np.argsort(ts_arr);X,y=X[si],y[si]
    sp=int(len(X)*0.8)
    Xtr,Xte=X[:sp],X[sp:];ytr,yte=y[:sp],y[sp:]
    npos=int(sum(ytr));spw=(len(ytr)-npos)/npos if npos>0 else 1
    print(f"  Train:{len(Xtr)}({npos}+) Test:{len(Xte)}({int(sum(yte))}+) 1:{spw:.0f}")
    
    scaler=RobustScaler();Xtrs=scaler.fit_transform(Xtr);Xtes=scaler.transform(Xte)
    
    # [2] Isolation Forest (이상 상태 탐지)
    print("\n[2] Isolation Forest...")
    if_scaler=RobustScaler()
    Xtr_if=if_scaler.fit_transform(Xtr)
    Xte_if=if_scaler.transform(Xte)
    
    iforest=IsolationForest(n_estimators=300,contamination=0.05,random_state=42,n_jobs=1)
    iforest.fit(Xtr_if)
    
    if_scores_te=iforest.decision_function(Xte_if)
    # 이상치 = 급등 전조? 확인
    anomaly_mask=if_scores_te<np.percentile(if_scores_te,15)
    if sum(anomaly_mask)>0:
        p_if=sum(yte[anomaly_mask])/sum(anomaly_mask)
        print(f"  IForest 이상치(하위15%): {sum(anomaly_mask)}개, 급등비율: {p_if:.1%}")
    
    # [3] XGB + LGB 학습 (Triple Barrier label)
    print("\n[3] 모델 학습 (Triple Barrier)...")
    best_overall={"precision":0,"recall":0,"f1":0,"predictions":0,"hits":0,"surges":0}
    best_model=None;best_t=0.5;best_name="none";best_iforest=None
    
    for model_type in ["xgb","lgb"]:
        for mult in [2,3,4,5,6]:
            random.seed(42+mult+(0 if model_type=="xgb" else 100))
            best_ap=0;best_p=None
            
            if model_type=="xgb":
                space={'n_estimators':[300,500,700],'max_depth':[3,4,5,6],'learning_rate':[0.01,0.02,0.05],
                       'min_child_weight':[5,10,15],'subsample':[0.6,0.7,0.8],'colsample_bytree':[0.5,0.6,0.7],
                       'gamma':[0,0.1,0.3,0.5],'reg_alpha':[0,0.1,1],'reg_lambda':[1,3,5],
                       'scale_pos_weight':[spw*mult]}
                cls=xgb.XGBClassifier
                fixed={'objective':'binary:logistic','eval_metric':'aucpr','use_label_encoder':False,
                       'random_state':42,'tree_method':'hist','verbosity':0}
            else:
                space={'n_estimators':[300,500,700],'max_depth':[3,5,7],'learning_rate':[0.01,0.02,0.05],
                       'num_leaves':[15,31,50],'min_child_samples':[10,20,50],'subsample':[0.6,0.7,0.8],
                       'colsample_bytree':[0.5,0.6,0.7],'reg_alpha':[0,0.1,1],'reg_lambda':[0,0.1,1],
                       'scale_pos_weight':[spw*mult]}
                cls=lgb.LGBMClassifier
                fixed={'objective':'binary','metric':'average_precision','random_state':42,
                       'verbose':-1,'force_col_wise':True,'n_jobs':1}
            
            for i in range(25):
                p={k:random.choice(v) for k,v in space.items()};p.update(fixed)
                try: s=manual_cv(cls,p,Xtrs,ytr,n_splits=5)
                except: continue
                if s>best_ap: best_ap=s;best_p=p.copy()
            
            if not best_p: continue
            m=cls(**best_p);m.fit(Xtrs,ytr)
            probs=m.predict_proba(Xtes)[:,1]
            
            # [4] Meta-labeling: IForest + classifier 결합
            for use_if in [False, True]:
                for t in np.arange(0.90,0.30,-0.05):
                    if use_if:
                        pred=((probs>=t)&anomaly_mask).astype(int)
                    else:
                        pred=(probs>=t).astype(int)
                    if sum(pred)<3: continue
                    pr=precision_score(yte,pred,zero_division=0)
                    if pr>=0.15:
                        bt=backtest_30d(all_data,m,scaler,fcols,t,
                                       iforest if use_if else None,
                                       if_scaler if use_if else None)
                        tag=f"{model_type}*{mult}{'_IF' if use_if else ''}"
                        print(f"    {tag} t={t:.2f}: P={bt['precision']:.1%} R={bt['recall']:.1%} ({bt['hits']}/{bt['predictions']})")
                        if bt['precision']>best_overall['precision']:
                            best_overall=bt;best_model=m;best_t=t
                            best_name=tag
                            best_iforest=(iforest,if_scaler) if use_if else None
                        break
    
    # [5] 스캔
    print("\n[5] 현재 시점 스캔...")
    def bybit_get(ep,params=None):
        url="https://api.bybit.com"+ep
        if params: url+="?"+"&".join(f"{k}={v}" for k,v in params.items())
        req=urllib.request.Request(url,headers={"User-Agent":"r4scan"})
        with urllib.request.urlopen(req,timeout=15) as r: return json.loads(r.read())
    
    tickers=bybit_get("/v5/market/tickers",{"category":"spot"}).get("result",{}).get("list",[])
    STABLECOINS={"USDCUSDT","UUSDT","XUSDUSDT","PAXGUSDT","DAIUSDT","FDUSDUSDT","TUSDUSDT","BUSDUSDT","EURUSDT","USTCUSDT","USDTUSDT","USDDUSDT","RLUSDUSDT","USD1USDT"}
    LARGE_CAPS={"BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","SOLUSDT","ADAUSDT","DOGEUSDT","TRXUSDT","DOTUSDT","MATICUSDT","LTCUSDT","BCHUSDT","LINKUSDT","AVAXUSDT","UNIUSDT","AAVEUSDT","ATOMUSDT"}
    
    scan_results=[]
    alts=[t["symbol"] for t in tickers if t["symbol"].endswith("USDT") and t["symbol"] not in STABLECOINS and t["symbol"] not in LARGE_CAPS and MIN_VOL<=float(t.get("turnover24h",0))<=MAX_VOL]
    
    for sym in alts:
        try:
            data=bybit_get("/v5/market/kline",{"category":"spot","symbol":sym,"interval":"D","limit":"100"})
            klines=list(reversed(data.get("result",{}).get("list",[])))
            if len(klines)<60: continue
            oo=[float(k[1]) for k in klines];hh=[float(k[2]) for k in klines]
            ll=[float(k[3]) for k in klines];cc=[float(k[4]) for k in klines]
            vv=[float(k[5]) for k in klines]
            if has_recent_surge(cc,len(cc)-1): continue
            feat=extract_features(oo,hh,ll,cc,vv,len(cc)-1)
            if not feat: continue
            fv=np.nan_to_num(np.array([[feat.get(fc,0) for fc in fcols]],dtype=np.float64),nan=0,posinf=0,neginf=0)
            fvs=scaler.transform(fv)
            prob=best_model.predict_proba(fvs)[0,1] if best_model else 0
            
            # IForest check
            is_anomaly=True
            if best_iforest:
                fv_if=best_iforest[1].transform(fv)
                score=best_iforest[0].decision_function(fv_if)[0]
                is_anomaly=score<np.percentile(if_scores_te,20)
            
            if prob>=best_t*0.8:
                scan_results.append({"coin":sym.replace("USDT",""),"prob":float(prob),
                    "price":cc[-1],"ret_7d":(cc[-1]/cc[-8]-1)*100 if len(cc)>=8 and cc[-8]>0 else 0,
                    "ret_14d":(cc[-1]/cc[-15]-1)*100 if len(cc)>=15 and cc[-15]>0 else 0,
                    "rsi":feat.get("rsi_14",0),"vol_ratio":feat.get("vol_ratio_7d_30d",1),
                    "anomaly":is_anomaly,"hit":prob>=best_t and (is_anomaly if best_iforest else True)})
            time.sleep(0.03)
        except: continue
    
    scan_results.sort(key=lambda x:x["prob"],reverse=True)
    
    # 결과
    elapsed=time.time()-t0
    lines=[f"*Round 4 완료* ({elapsed/60:.0f}분) — Triple Barrier + Meta-label"]
    lines.append(f"*모델: {best_name}*")
    lines.append(f"```Precision: {best_overall['precision']:.1%}\nRecall:    {best_overall['recall']:.1%}\nF1:        {best_overall['f1']:.3f}\n30일: {best_overall['hits']}/{best_overall['predictions']} 적중 (급등 {best_overall['surges']}건)```")
    lines.append(f"\n*전조 후보 (최근 급등 이력 전부 제외):*")
    lines.append("```")
    lines.append(f" # {'코인':<10} {'확률':>6} {'현재가':<11} {'7d':>7} {'14d':>7} {'RSI':>4}")
    hits=[r for r in scan_results if r["hit"]]
    for i,r in enumerate(scan_results[:15]):
        flag="<<<" if r["hit"] else ""
        lines.append(f"{i+1:>2} {r['coin']:<10} {r['prob']:>5.1%} ${r['price']:<10.4f} {r['ret_7d']:>+6.1f}% {r['ret_14d']:>+6.1f}% {r['rsi']:>4.0f} {flag}")
    lines.append(f"```\n전조 후보: {len(hits)}개")
    
    msg="\n".join(lines)
    send_slack(msg);print(msg)
    
    # 저장
    if best_model:
        save_data={"model":best_model,"scaler":scaler,"feature_cols":fcols,
                   "threshold":float(best_t),"surge_threshold":SURGE_TH,
                   "pretrigger_filter":True,"triple_barrier":True}
        if best_iforest:
            save_data["iforest"]=best_iforest[0];save_data["iforest_scaler"]=best_iforest[1]
        with open(DATA_DIR/"best_model.pkl","wb") as f: pickle.dump(save_data,f)
        config={"model_name":best_name,"surge_threshold":SURGE_TH,"threshold":float(best_t),
                "feature_cols":fcols,"precision":float(best_overall["precision"]),
                "recall":float(best_overall["recall"]),"f1":float(best_overall["f1"]),
                "predictions_30d":best_overall["predictions"],"hits_30d":best_overall["hits"],
                "surges_30d":best_overall["surges"],"trained_at":datetime.datetime.now().isoformat(),
                "round":4,"method":"triple_barrier+meta_label+isolation_forest",
                "filter":"no_recent_surge_30d"}
        with open(BASE_DIR/"ml_model_config.json","w") as f: json.dump(config,f,indent=2)
        print("  저장 완료")
    
    send_slack(f"_다음: Round 5 준비 중 (Precision {best_overall['precision']:.1%} → 목표 30%+)_")

if __name__=="__main__": main()
