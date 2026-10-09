#!/usr/bin/env python3
"""
급등 전조 모델 — Fast version
XGBoost + LightGBM 앙상블 (RF 제외)
데이터 캐시 활용, 빠른 실행
"""

import json, math, os, sys, time, datetime, pickle, warnings, random
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "ml_data"
DATA_DIR.mkdir(exist_ok=True)

SURGE_THRESHOLDS = [0.3, 0.5]
SURGE_WINDOW = 7
MIN_VOL_USD = 50_000
MAX_VOL_USD = 10_000_000

STABLECOINS = {
    "USDCUSDT","UUSDT","XUSDUSDT","PAXGUSDT","DAIUSDT",
    "FDUSDUSDT","TUSDUSDT","BUSDUSDT","EURUSDT","USTCUSDT",
    "USDTUSDT","USDDUSDT","RLUSDUSDT","USD1USDT",
}
LARGE_CAPS = {
    "BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","SOLUSDT",
    "ADAUSDT","DOGEUSDT","TRXUSDT","DOTUSDT","MATICUSDT",
    "LTCUSDT","BCHUSDT","LINKUSDT","AVAXUSDT","UNIUSDT",
    "AAVEUSDT","ATOMUSDT",
}

# ============================================================
# Math helpers
# ============================================================
def mean(arr): return sum(arr)/len(arr) if arr else 0.0
def stdev(arr):
    if len(arr)<2: return 0.0
    m=mean(arr); return math.sqrt(sum((x-m)**2 for x in arr)/(len(arr)-1))
def polyfit_slope(y):
    n=len(y)
    if n<2: return 0.0
    xm=(n-1)/2.0; ym=mean(y)
    if ym==0: return 0.0
    num=sum((i-xm)*(v-ym) for i,v in enumerate(y))
    den=sum((i-xm)**2 for i in range(n))
    return (num/den)/ym if den else 0.0
def compute_rsi_series(closes, period=14):
    rsi=[None]*len(closes)
    if len(closes)<period+1: return rsi
    deltas=[closes[i]-closes[i-1] for i in range(1,len(closes))]
    gains=[max(d,0) for d in deltas]; losses=[max(-d,0) for d in deltas]
    ag=mean(gains[:period]); al=mean(losses[:period])
    rsi[period]=100.0 if al==0 else 100.0-100.0/(1.0+ag/al)
    for i in range(period,len(deltas)):
        ag=(ag*(period-1)+gains[i])/period; al=(al*(period-1)+losses[i])/period
        rsi[i+1]=100.0 if al==0 else 100.0-100.0/(1.0+ag/al)
    return rsi
def compute_atr_series(highs,lows,closes,period=14):
    n=len(closes); tr=[highs[0]-lows[0]]
    for i in range(1,n): tr.append(max(highs[i]-lows[i],abs(highs[i]-closes[i-1]),abs(lows[i]-closes[i-1])))
    atr=[None]*n
    if n>=period:
        atr[period-1]=mean(tr[:period])
        for i in range(period,n): atr[i]=(atr[i-1]*(period-1)+tr[i])/period
    return atr
def compute_macd(closes,fast=12,slow=26,signal=9):
    if len(closes)<slow+signal: return [0]*len(closes),[0]*len(closes),[0]*len(closes)
    def ema(a,p):
        k=2/(p+1); r=[a[0]]
        for i in range(1,len(a)): r.append(a[i]*k+r[-1]*(1-k))
        return r
    ef=ema(closes,fast); es=ema(closes,slow)
    ml=[f-s for f,s in zip(ef,es)]; sl=ema(ml,signal)
    hist=[m-s for m,s in zip(ml,sl)]
    return ml,sl,hist
def compute_obv(closes,volumes):
    obv=[0]
    for i in range(1,len(closes)):
        if closes[i]>closes[i-1]: obv.append(obv[-1]+volumes[i])
        elif closes[i]<closes[i-1]: obv.append(obv[-1]-volumes[i])
        else: obv.append(obv[-1])
    return obv

# ============================================================
# Feature Extraction
# ============================================================
def extract_features_at(o,h,l,c,v,idx):
    if idx<59: return None
    f={}
    for d in [1,2,3,5,7,10,14,21]:
        f[f"ret_{d}d"]=c[idx]/c[idx-d]-1 if idx>=d and c[idx-d]>0 else 0
    for p in [5,7,10,20,50]:
        if idx>=p-1:
            ma=mean(c[idx-p+1:idx+1]); f[f"price_vs_ma{p}"]=c[idx]/ma-1 if ma>0 else 0
        else: f[f"price_vs_ma{p}"]=0
    if idx>=20:
        ma5=mean(c[idx-4:idx+1]);ma10=mean(c[idx-9:idx+1]);ma20=mean(c[idx-19:idx+1])
        f["ma5_ma10_cross"]=ma5/ma10-1 if ma10>0 else 0
        f["ma5_ma20_cross"]=ma5/ma20-1 if ma20>0 else 0
        f["ma10_ma20_cross"]=ma10/ma20-1 if ma20>0 else 0
    else: f["ma5_ma10_cross"]=0;f["ma5_ma20_cross"]=0;f["ma10_ma20_cross"]=0
    for w in [14,30,60]:
        if idx>=w-1:
            hi=max(h[idx-w+1:idx+1]);lo=min(l[idx-w+1:idx+1]);rng=hi-lo
            f[f"price_pos_{w}d"]=(c[idx]-lo)/rng if rng>0 else 0.5
            f[f"price_vs_{w}d_high"]=c[idx]/hi if hi>0 else 1
        else: f[f"price_pos_{w}d"]=0.5;f[f"price_vs_{w}d_high"]=1
    rsi14=compute_rsi_series(c[:idx+1],14); rsi7=compute_rsi_series(c[:idx+1],7)
    f["rsi_14"]=rsi14[idx] if rsi14[idx] is not None else 50
    f["rsi_7"]=rsi7[idx] if rsi7[idx] is not None else 50
    if idx>=7 and rsi14[idx] and rsi14[idx-7]: f["rsi14_change_7d"]=rsi14[idx]-rsi14[idx-7]
    else: f["rsi14_change_7d"]=0
    if idx>=3 and rsi14[idx] and rsi14[idx-3]: f["rsi14_change_3d"]=rsi14[idx]-rsi14[idx-3]
    else: f["rsi14_change_3d"]=0
    rw=[rsi14[j] for j in range(max(0,idx-13),idx+1) if rsi14[j] is not None]
    f["rsi_oversold_days"]=sum(1 for r in rw if r<30)
    f["rsi_consol_days"]=sum(1 for r in rw if 40<=r<=55)
    f["rsi_overbought_days"]=sum(1 for r in rw if r>70)
    f["rsi_min_14d"]=min(rw) if rw else 50
    f["rsi_max_14d"]=max(rw) if rw else 50
    f["rsi_range_14d"]=(max(rw)-min(rw)) if rw else 0
    if idx>=7 and rsi14[idx] and rsi14[idx-7]:
        pd_=1 if c[idx]>c[idx-7] else -1; rd=1 if rsi14[idx]>rsi14[idx-7] else -1
        f["rsi_divergence"]=1 if pd_==-1 and rd==1 else(-1 if pd_==1 and rd==-1 else 0)
    else: f["rsi_divergence"]=0
    ml,sl,hist=compute_macd(c[:idx+1])
    if idx<len(ml):
        pa=c[idx] if c[idx]>0 else 1
        f["macd_line"]=ml[idx]/pa;f["macd_signal"]=sl[idx]/pa;f["macd_histogram"]=hist[idx]/pa
        f["macd_cross"]=1 if idx>=1 and hist[idx]>0 and hist[idx-1]<=0 else(-1 if idx>=1 and hist[idx]<0 and hist[idx-1]>=0 else 0)
        f["macd_hist_slope"]=polyfit_slope([hist[idx-i] for i in range(min(4,idx+1)-1,-1,-1)]) if idx>=3 else 0
    else: f["macd_line"]=0;f["macd_signal"]=0;f["macd_histogram"]=0;f["macd_cross"]=0;f["macd_hist_slope"]=0
    atr_s=compute_atr_series(h[:idx+1],l[:idx+1],c[:idx+1],14)
    a14=atr_s[idx] if atr_s[idx] else 0
    f["atr_ratio"]=a14/c[idx] if c[idx]>0 else 0
    a7v=[atr_s[j] for j in range(max(0,idx-6),idx+1) if atr_s[j]]
    a30v=[atr_s[j] for j in range(max(0,idx-29),idx+1) if atr_s[j]]
    f["volatility_ratio_7_30"]=mean(a7v)/mean(a30v) if a30v and mean(a30v)>0 else 1
    if idx>=29:
        ae=[atr_s[j] for j in range(idx-29,idx-14) if atr_s[j]]
        al_=[atr_s[j] for j in range(idx-6,idx+1) if atr_s[j]]
        f["volatility_contraction"]=mean(al_)/mean(ae) if ae and mean(ae)>0 else 1
    else: f["volatility_contraction"]=1
    if idx>=19:
        bc=c[idx-19:idx+1];bm=mean(bc);bs=stdev(bc)
        f["bb_width"]=(2*bs)/bm if bm>0 else 0
        f["bb_position"]=(c[idx]-(bm-bs))/(2*bs) if bs>0 else 0.5
    else: f["bb_width"]=0;f["bb_position"]=0.5
    v_7d=mean(v[max(0,idx-6):idx+1]);v_14d=mean(v[max(0,idx-13):idx+1]);v_30d=mean(v[max(0,idx-29):idx+1])
    f["vol_ratio_1d_7d"]=v[idx]/v_7d if v_7d>0 else 1
    f["vol_ratio_3d_14d"]=mean(v[max(0,idx-2):idx+1])/v_14d if v_14d>0 else 1
    f["vol_ratio_7d_30d"]=v_7d/v_30d if v_30d>0 else 1
    f["vol_trend_7d"]=polyfit_slope(v[max(0,idx-6):idx+1])
    f["vol_trend_14d"]=polyfit_slope(v[max(0,idx-13):idx+1])
    vl2=mean(v[max(0,idx-1):idx+1]);vp5=mean(v[max(0,idx-6):max(1,idx-1)]) if idx>=3 else vl2
    f["vol_explosion_2d"]=vl2/vp5 if vp5>0 else 1
    obv=compute_obv(c[max(0,idx-29):idx+1],v[max(0,idx-29):idx+1])
    f["obv_slope_7d"]=polyfit_slope(obv[-7:]) if len(obv)>=7 else 0
    if idx>=7: f["vol_price_divergence"]=polyfit_slope(v[idx-6:idx+1])-polyfit_slope(c[idx-6:idx+1])
    else: f["vol_price_divergence"]=0
    f["green_ratio_3d"]=sum(1 for j in range(max(0,idx-2),idx+1) if c[j]>=o[j])/3.0
    f["green_ratio_7d"]=sum(1 for j in range(max(0,idx-6),idx+1) if c[j]>=o[j])/7.0
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
    bodies=[];upwicks=[];lowwicks=[]
    for j in range(max(0,idx-6),idx+1):
        body=abs(c[j]-o[j]);full=h[j]-l[j]
        if full>0: bodies.append(body/full);upwicks.append((h[j]-max(c[j],o[j]))/full);lowwicks.append((min(c[j],o[j])-l[j])/full)
    f["avg_body_ratio"]=mean(bodies) if bodies else 0.5
    f["avg_upper_wick"]=mean(upwicks) if upwicks else 0
    f["avg_lower_wick"]=mean(lowwicks) if lowwicks else 0
    hc=0;dc=0
    for j in range(max(0,idx-2),idx+1):
        body=abs(c[j]-o[j]);full=h[j]-l[j]
        if full>0:
            lw=(min(c[j],o[j])-l[j])/full;br=body/full
            if lw>0.6 and br<0.2: hc+=1
            if br<0.1: dc+=1
    f["hammer_3d"]=hc;f["doji_3d"]=dc
    if idx>=13:
        pc14=abs(c[idx]/c[idx-13]-1);vr=mean(v[idx-6:idx+1]);vp=mean(v[idx-13:idx-6])
        f["accumulation"]=(vr/vp)/(1+pc14*10) if vp>0 else 1
    else: f["accumulation"]=1
    if idx>=30:
        rl=min(l[idx-29:idx+1]);pk=max(h[idx-29:idx+1])
        f["drawdown_30d"]=(c[idx]-pk)/pk if pk>0 else 0
        f["recovery_from_low"]=(c[idx]-rl)/rl if rl>0 else 0
    else: f["drawdown_30d"]=0;f["recovery_from_low"]=0
    if idx>=14:
        r7=c[idx]/c[idx-7]-1 if c[idx-7]>0 else 0
        r14=c[idx]/c[idx-14]-1 if c[idx-14]>0 else 0
        f["momentum_acceleration"]=r7-(r14-r7)
        uwv=0
        for j in range(idx,max(idx-7,-1),-1):
            if c[j]>=o[j] and v[j]>v_7d: uwv+=1
            else: break
        f["up_with_volume_streak"]=uwv
    else: f["momentum_acceleration"]=0;f["up_with_volume_streak"]=0
    if idx>=30:
        rh=sorted(h[idx-29:idx+1],reverse=True);res=mean(rh[:3])
        f["distance_to_resistance"]=(res-c[idx])/c[idx] if c[idx]>0 else 0
        rls=sorted(l[idx-29:idx+1]);sup=mean(rls[:3])
        f["distance_from_support"]=(c[idx]-sup)/c[idx] if c[idx]>0 else 0
    else: f["distance_to_resistance"]=0;f["distance_from_support"]=0
    if idx>=7:
        nm=abs(c[idx]-c[idx-7]);tm=sum(abs(c[j]-c[j-1]) for j in range(idx-6,idx+1))
        f["efficiency_ratio_7d"]=nm/tm if tm>0 else 0
    else: f["efficiency_ratio_7d"]=0
    return f

# ============================================================
# 데이터
# ============================================================
def load_cached_data():
    cache_file=DATA_DIR/"raw_data.json"
    if cache_file.exists():
        print(f"  캐시 로드...")
        with open(cache_file) as f: return json.load(f)
    print("  ERROR: 캐시 없음! build_model_v3.py 먼저 실행")
    sys.exit(1)

def build_dataset(all_data,surge_threshold):
    print(f"  데이터셋 구축 (+{surge_threshold*100:.0f}%)...")
    rows=[];sc=0
    for sym,d in all_data.items():
        n=len(d["c"]);o,h,l,c,v=d["o"],d["h"],d["l"],d["c"],d["v"]
        avg_vol=mean([v[j]*c[j] for j in range(max(0,n-30),n)])
        if avg_vol>MAX_VOL_USD or avg_vol<MIN_VOL_USD: continue
        for idx in range(59,n-SURGE_WINDOW):
            feat=extract_features_at(o,h,l,c,v,idx)
            if feat is None: continue
            mx=0
            for j in range(idx+1,min(idx+SURGE_WINDOW+1,n)):
                r=c[j]/c[idx]-1 if c[idx]>0 else 0; mx=max(mx,r)
            feat["_label"]=1 if mx>=surge_threshold else 0
            feat["_ts"]=d["ts"][idx];feat["_sym"]=sym
            rows.append(feat)
            if mx>=surge_threshold: sc+=1
    print(f"  {len(rows)}개 ({sc} surge, {sc/len(rows)*100:.2f}%)")
    return rows

# ============================================================
# 수동 CV
# ============================================================
def manual_cv(model_cls,params,X,y,n_splits=5):
    tscv=TimeSeriesSplit(n_splits=n_splits);scores=[]
    for ti,vi in tscv.split(X):
        Xt,Xv=X[ti],X[vi];yt,yv=y[ti],y[vi]
        m=model_cls(**params);m.fit(Xt,yt)
        p=m.predict_proba(Xv)[:,1]
        if sum(yv)>0: scores.append(average_precision_score(yv,p))
    return mean(scores) if scores else 0

def tune_xgb(X,y,spw,n_iter=60):
    print(f"    XGBoost ({n_iter}회)...")
    space={
        'n_estimators':[200,300,500],'max_depth':[4,5,6,7],
        'learning_rate':[0.02,0.05,0.1],'min_child_weight':[3,5,10],
        'subsample':[0.7,0.8],'colsample_bytree':[0.6,0.7,0.8],
        'gamma':[0,0.1,0.5],
        'scale_pos_weight':[spw,spw*1.5,spw*2],
    }
    best_s=0;best_p=None;random.seed(42)
    for i in range(n_iter):
        p={k:random.choice(v) for k,v in space.items()}
        p.update({'objective':'binary:logistic','eval_metric':'aucpr','use_label_encoder':False,
                  'random_state':42,'tree_method':'hist','verbosity':0})
        try: s=manual_cv(xgb.XGBClassifier,p,X,y)
        except: continue
        if s>best_s:
            best_s=s;best_p=p.copy()
            if (i+1)%20==0: print(f"      [{i+1}] AP={best_s:.4f}")
    print(f"      Best AP={best_s:.4f}")
    m=xgb.XGBClassifier(**best_p);m.fit(X,y)
    return m,best_s

def tune_lgb(X,y,spw,n_iter=60):
    print(f"    LightGBM ({n_iter}회)...")
    space={
        'n_estimators':[200,300,500],'max_depth':[5,7,-1],
        'learning_rate':[0.02,0.05,0.1],'num_leaves':[31,50,80],
        'min_child_samples':[10,20,50],'subsample':[0.7,0.8],
        'colsample_bytree':[0.6,0.7,0.8],
        'scale_pos_weight':[spw,spw*1.5,spw*2],
    }
    best_s=0;best_p=None;random.seed(42)
    for i in range(n_iter):
        p={k:random.choice(v) for k,v in space.items()}
        p.update({'objective':'binary','metric':'average_precision','random_state':42,'verbose':-1,'force_col_wise':True})
        try: s=manual_cv(lgb.LGBMClassifier,p,X,y)
        except: continue
        if s>best_s:
            best_s=s;best_p=p.copy()
            if (i+1)%20==0: print(f"      [{i+1}] AP={best_s:.4f}")
    print(f"      Best AP={best_s:.4f}")
    m=lgb.LGBMClassifier(**best_p);m.fit(X,y)
    return m,best_s

# ============================================================
# 앙상블
# ============================================================
class DualEnsemble:
    def __init__(self,m1,m2,w1=0.5,w2=0.5):
        self.m1=m1;self.m2=m2;self.w1=w1;self.w2=w2
    def predict_proba(self,X):
        p1=self.m1.predict_proba(X);p2=self.m2.predict_proba(X)
        return p1*self.w1+p2*self.w2
    def predict(self,X):
        return (self.predict_proba(X)[:,1]>=0.5).astype(int)

# ============================================================
# 평가
# ============================================================
def find_best_threshold(model,X,y):
    probs=model.predict_proba(X)[:,1]
    best_t=0.5;best_f1=0;best_p=0;best_r=0
    for t in np.arange(0.05,0.95,0.01):
        pred=(probs>=t).astype(int)
        if sum(pred)==0: continue
        p=precision_score(y,pred,zero_division=0)
        r=recall_score(y,pred,zero_division=0)
        f1=2*p*r/(p+r) if p+r>0 else 0
        if p>=0.08 and f1>best_f1: best_t=t;best_f1=f1;best_p=p;best_r=r
    return best_t,best_p,best_r,best_f1

def backtest_30d(all_data,model,scaler,fcols,threshold,surge_th):
    print(f"  30일 백테스트 (t={threshold:.2f})...")
    BD=30;tp=0;th=0;ts=0
    for day in range(BD):
        dX=[];dy=[]
        for sym,d in all_data.items():
            n=len(d["c"]);o,h,l,c,v=d["o"],d["h"],d["l"],d["c"],d["v"]
            idx=n-1-SURGE_WINDOW-(BD-1-day)
            if idx<59 or idx+SURGE_WINDOW>=n: continue
            av=mean([v[j]*c[j] for j in range(max(0,idx-29),idx+1)])
            if av>MAX_VOL_USD or av<MIN_VOL_USD: continue
            feat=extract_features_at(o,h,l,c,v,idx)
            if feat is None: continue
            mx=0
            for j in range(idx+1,min(idx+SURGE_WINDOW+1,n)):
                r=c[j]/c[idx]-1 if c[idx]>0 else 0; mx=max(mx,r)
            dX.append([feat.get(fc,0) for fc in fcols])
            dy.append(1 if mx>=surge_th else 0)
        if not dX: continue
        Xd=np.nan_to_num(np.array(dX,dtype=np.float64),nan=0,posinf=0,neginf=0)
        Xds=scaler.transform(Xd);yd=np.array(dy)
        probs=model.predict_proba(Xds)[:,1]
        preds=(probs>=threshold).astype(int)
        tp+=sum(preds);th+=sum(preds&yd);ts+=sum(yd)
    p=th/tp if tp>0 else 0;r=th/ts if ts>0 else 0
    f1=2*p*r/(p+r) if p+r>0 else 0
    print(f"    P={p:.1%} R={r:.1%} F1={f1:.3f} ({th}/{tp} hits, {ts} surges)")
    return {"precision":float(p),"recall":float(r),"f1":float(f1),"predictions":int(tp),"hits":int(th),"surges":int(ts)}

# ============================================================
# Main
# ============================================================
def main():
    t0=time.time()
    print("="*65)
    print("  급등 전조 모델 — XGB + LGB Fast Pipeline")
    print("="*65)
    
    all_data=load_cached_data()
    print(f"  {len(all_data)}개 코인 로드")
    
    best_overall={"f1":0}
    all_results={}
    
    for surge_th in SURGE_THRESHOLDS:
        print(f"\n{'='*65}")
        print(f"  +{surge_th*100:.0f}% 기준")
        print(f"{'='*65}")
        
        rows=build_dataset(all_data,surge_th)
        ns=sum(1 for r in rows if r["_label"]==1)
        if ns<20: print("  부족 → 스킵"); continue
        
        df=pd.DataFrame(rows)
        fcols=[c for c in df.columns if not c.startswith("_")]
        X=np.nan_to_num(df[fcols].values.astype(np.float64),nan=0,posinf=0,neginf=0)
        y=df["_label"].values;ts_arr=df["_ts"].values
        si=np.argsort(ts_arr);X,y=X[si],y[si]
        
        sp=int(len(X)*0.8)
        Xtr,Xte=X[:sp],X[sp:];ytr,yte=y[:sp],y[sp:]
        npos=sum(ytr);nneg=len(ytr)-npos;spw=nneg/npos if npos>0 else 1
        print(f"  Train:{len(Xtr)}({sum(ytr)}+) Test:{len(Xte)}({sum(yte)}+) 1:{spw:.0f}")
        
        scaler=RobustScaler();Xtrs=scaler.fit_transform(Xtr);Xtes=scaler.transform(Xte)
        
        xm,xap=tune_xgb(Xtrs,ytr,spw,60)
        lm,lap=tune_lgb(Xtrs,ytr,spw,60)
        
        tw=xap+lap
        w1,w2=(xap/tw,lap/tw) if tw>0 else (0.5,0.5)
        ens=DualEnsemble(xm,lm,w1,w2)
        print(f"    앙상블: XGB={w1:.2f} LGB={w2:.2f}")
        
        for name,model in [("xgboost",xm),("lightgbm",lm),("ensemble",ens)]:
            t,p,r,f1=find_best_threshold(model,Xtes,yte)
            print(f"\n    {name}: threshold={t:.2f} P={p:.1%} R={r:.1%} F1={f1:.3f}")
            bt=backtest_30d(all_data,model,scaler,fcols,t,surge_th)
            
            key=f"{name}_surge{int(surge_th*100)}"
            all_results[key]={"model":name,"surge":surge_th,"threshold":t,
                "test_p":p,"test_r":r,"test_f1":f1,"backtest":bt}
            
            if bt["f1"]>best_overall.get("f1",0):
                best_overall={"model_name":name,"surge_threshold":surge_th,
                    "threshold":t,"model_obj":model,"scaler":scaler,"feature_cols":fcols,**bt}
    
    elapsed=time.time()-t0
    print(f"\n{'='*65}")
    print(f"  결과 ({elapsed/60:.1f}분)")
    print(f"{'='*65}")
    print(f"\n  {'설정':<30} {'Prec':>7} {'Rec':>7} {'F1':>7} {'예측':>5} {'적중':>4}")
    print(f"  {'-'*63}")
    for k,r in sorted(all_results.items(),key=lambda x:x[1]["backtest"]["f1"],reverse=True):
        b=r["backtest"]
        print(f"  {k:<30} {b['precision']:>6.1%} {b['recall']:>6.1%} {b['f1']:>6.3f} {b['predictions']:>5} {b['hits']:>4}")
    
    if best_overall.get("model_obj"):
        print(f"\n  === BEST ===")
        print(f"  {best_overall['model_name']} +{best_overall['surge_threshold']*100:.0f}%")
        print(f"  Threshold={best_overall['threshold']:.2f}")
        print(f"  P={best_overall['precision']:.1%} R={best_overall['recall']:.1%} F1={best_overall['f1']:.3f}")
        
        # Feature importance
        if hasattr(best_overall["model_obj"], 'feature_importances_'):
            imp=best_overall["model_obj"].feature_importances_
            top=sorted(zip(best_overall["feature_cols"],imp),key=lambda x:x[1],reverse=True)[:15]
            print(f"\n  Top Features:")
            for fn,im in top: print(f"    {fn:<30} {im:.4f}")
        
        config={"model_name":best_overall["model_name"],"surge_threshold":float(best_overall["surge_threshold"]),
            "threshold":float(best_overall["threshold"]),"feature_cols":best_overall["feature_cols"],
            "precision":float(best_overall["precision"]),"recall":float(best_overall["recall"]),"f1":float(best_overall["f1"]),
            "predictions_30d":int(best_overall["predictions"]),"hits_30d":int(best_overall["hits"])}
        with open(BASE_DIR/"ml_model_config.json","w") as f: json.dump(config,f,indent=2)
        with open(DATA_DIR/"best_model.pkl","wb") as f:
            pickle.dump({"model":best_overall["model_obj"],"scaler":best_overall["scaler"],
                "feature_cols":best_overall["feature_cols"],"threshold":best_overall["threshold"],
                "surge_threshold":best_overall["surge_threshold"]},f)
        print(f"\n  저장 완료: ml_model_config.json + ml_data/best_model.pkl")
    
    sr={k:{kk:vv for kk,vv in v.items()} for k,v in all_results.items()}
    with open(BASE_DIR/"ml_results.json","w") as f: json.dump(sr,f,indent=2,default=str)
    print(f"  소요: {elapsed/60:.1f}분")

if __name__=="__main__": main()
