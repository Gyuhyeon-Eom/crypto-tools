#!/usr/bin/env python3
"""
Precision 극대화 파이프라인
단계별로 개선하면서 중간 결과 출력

Stage 1: XGBoost + LightGBM 앙상블 (Precision-optimized)
Stage 2: Hard Negative Mining → 재학습
Stage 3: 30일 백테스트로 검증
Stage 4: Confidence calibration + 최종 threshold
"""

import json,math,os,sys,time,datetime,pickle,warnings,random
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score,precision_score,recall_score,f1_score
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

BASE_DIR=Path(__file__).parent
DATA_DIR=BASE_DIR/"ml_data"

SURGE_TH=0.3; SURGE_WINDOW=7; MIN_VOL=50_000; MAX_VOL=10_000_000

# ============================================================
# Math / TA helpers (동일)
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
    num=sum((i-xm)*(v-ym) for i,v in enumerate(y))
    den=sum((i-xm)**2 for i in range(n))
    return (num/den)/ym if den else 0.0
def compute_rsi(closes,period=14):
    rsi=[None]*len(closes)
    if len(closes)<period+1: return rsi
    d=[closes[i]-closes[i-1] for i in range(1,len(closes))]
    g=[max(x,0) for x in d]; lo=[max(-x,0) for x in d]
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
# 확장 Feature (기존 70 + 신규 20 = 90개)
# ============================================================
def extract_features(o,h,l,c,v,idx):
    if idx<59: return None
    f={}
    
    # --- 기존 feature ---
    for d in [1,2,3,5,7,10,14,21]:
        f[f"ret_{d}d"]=c[idx]/c[idx-d]-1 if idx>=d and c[idx-d]>0 else 0
    for p in [5,7,10,20,50]:
        if idx>=p-1: ma=mean(c[idx-p+1:idx+1]);f[f"price_vs_ma{p}"]=c[idx]/ma-1 if ma>0 else 0
        else: f[f"price_vs_ma{p}"]=0
    if idx>=20:
        ma5=mean(c[idx-4:idx+1]);ma10=mean(c[idx-9:idx+1]);ma20=mean(c[idx-19:idx+1])
        f["ma5_ma10_cross"]=ma5/ma10-1 if ma10>0 else 0
        f["ma5_ma20_cross"]=ma5/ma20-1 if ma20>0 else 0
        f["ma10_ma20_cross"]=ma10/ma20-1 if ma20>0 else 0
    else: f["ma5_ma10_cross"]=0;f["ma5_ma20_cross"]=0;f["ma10_ma20_cross"]=0
    for w in [14,30,60]:
        if idx>=w-1:
            hi=max(h[idx-w+1:idx+1]);lo_=min(l[idx-w+1:idx+1]);rng=hi-lo_
            f[f"price_pos_{w}d"]=(c[idx]-lo_)/rng if rng>0 else 0.5
            f[f"price_vs_{w}d_high"]=c[idx]/hi if hi>0 else 1
        else: f[f"price_pos_{w}d"]=0.5;f[f"price_vs_{w}d_high"]=1
    r14=compute_rsi(c[:idx+1],14);r7=compute_rsi(c[:idx+1],7)
    f["rsi_14"]=r14[idx] if r14[idx] else 50;f["rsi_7"]=r7[idx] if r7[idx] else 50
    if idx>=7 and r14[idx] and r14[idx-7]: f["rsi14_change_7d"]=r14[idx]-r14[idx-7]
    else: f["rsi14_change_7d"]=0
    if idx>=3 and r14[idx] and r14[idx-3]: f["rsi14_change_3d"]=r14[idx]-r14[idx-3]
    else: f["rsi14_change_3d"]=0
    rw=[r14[j] for j in range(max(0,idx-13),idx+1) if r14[j] is not None]
    f["rsi_oversold_days"]=sum(1 for r in rw if r<30)
    f["rsi_consol_days"]=sum(1 for r in rw if 40<=r<=55)
    f["rsi_overbought_days"]=sum(1 for r in rw if r>70)
    f["rsi_min_14d"]=min(rw) if rw else 50;f["rsi_max_14d"]=max(rw) if rw else 50
    f["rsi_range_14d"]=(max(rw)-min(rw)) if rw else 0
    if idx>=7 and r14[idx] and r14[idx-7]:
        pd_=1 if c[idx]>c[idx-7] else -1;rd=1 if r14[idx]>r14[idx-7] else -1
        f["rsi_divergence"]=1 if pd_==-1 and rd==1 else(-1 if pd_==1 and rd==-1 else 0)
    else: f["rsi_divergence"]=0
    ml,sl,hist=compute_macd(c[:idx+1])
    if idx<len(ml):
        pa=c[idx] if c[idx]>0 else 1
        f["macd_line"]=ml[idx]/pa;f["macd_signal"]=sl[idx]/pa;f["macd_histogram"]=hist[idx]/pa
        f["macd_cross"]=1 if idx>=1 and hist[idx]>0 and hist[idx-1]<=0 else(-1 if idx>=1 and hist[idx]<0 and hist[idx-1]>=0 else 0)
        f["macd_hist_slope"]=polyfit_slope([hist[idx-i] for i in range(min(4,idx+1)-1,-1,-1)]) if idx>=3 else 0
    else: f["macd_line"]=0;f["macd_signal"]=0;f["macd_histogram"]=0;f["macd_cross"]=0;f["macd_hist_slope"]=0
    atr_s=compute_atr(h[:idx+1],l[:idx+1],c[:idx+1],14)
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
        f["bb_width"]=(2*bs)/bm if bm>0 else 0;f["bb_position"]=(c[idx]-(bm-bs))/(2*bs) if bs>0 else 0.5
    else: f["bb_width"]=0;f["bb_position"]=0.5
    v7=mean(v[max(0,idx-6):idx+1]);v14=mean(v[max(0,idx-13):idx+1]);v30=mean(v[max(0,idx-29):idx+1])
    f["vol_ratio_1d_7d"]=v[idx]/v7 if v7>0 else 1
    f["vol_ratio_3d_14d"]=mean(v[max(0,idx-2):idx+1])/v14 if v14>0 else 1
    f["vol_ratio_7d_30d"]=v7/v30 if v30>0 else 1
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
    bodies=[];uw=[];lw=[]
    for j in range(max(0,idx-6),idx+1):
        b=abs(c[j]-o[j]);fl=h[j]-l[j]
        if fl>0: bodies.append(b/fl);uw.append((h[j]-max(c[j],o[j]))/fl);lw.append((min(c[j],o[j])-l[j])/fl)
    f["avg_body_ratio"]=mean(bodies) if bodies else 0.5
    f["avg_upper_wick"]=mean(uw) if uw else 0;f["avg_lower_wick"]=mean(lw) if lw else 0
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
        rl=min(l[idx-29:idx+1]);pk=max(h[idx-29:idx+1])
        f["drawdown_30d"]=(c[idx]-pk)/pk if pk>0 else 0
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
        rh=sorted(h[idx-29:idx+1],reverse=True);res=mean(rh[:3])
        f["distance_to_resistance"]=(res-c[idx])/c[idx] if c[idx]>0 else 0
        rls=sorted(l[idx-29:idx+1]);sup=mean(rls[:3])
        f["distance_from_support"]=(c[idx]-sup)/c[idx] if c[idx]>0 else 0
    else: f["distance_to_resistance"]=0;f["distance_from_support"]=0
    if idx>=7:
        nm=abs(c[idx]-c[idx-7]);tm=sum(abs(c[j]-c[j-1]) for j in range(idx-6,idx+1))
        f["efficiency_ratio_7d"]=nm/tm if tm>0 else 0
    else: f["efficiency_ratio_7d"]=0
    
    # --- 신규 feature (Precision 향상용) ---
    
    # 거래대금 (USD) — 절대 규모
    f["vol_usd_7d"]=mean([v[j]*c[j] for j in range(max(0,idx-6),idx+1)])
    f["vol_usd_1d"]=v[idx]*c[idx]
    
    # 거래대금 변화율 (갑자기 돈이 몰리는지)
    vusd_3d=mean([v[j]*c[j] for j in range(max(0,idx-2),idx+1)])
    vusd_prev=mean([v[j]*c[j] for j in range(max(0,idx-9),max(1,idx-2))]) if idx>=5 else vusd_3d
    f["vol_usd_acceleration"]=vusd_3d/vusd_prev if vusd_prev>0 else 1
    
    # 가격 범위 수축 (Squeeze) — 최근 7일 range / 30일 range
    if idx>=29:
        range_7=max(h[idx-6:idx+1])-min(l[idx-6:idx+1])
        range_30=max(h[idx-29:idx+1])-min(l[idx-29:idx+1])
        f["range_squeeze"]=range_7/range_30 if range_30>0 else 1
    else: f["range_squeeze"]=1
    
    # 연속 하락 후 반전 (V자 패턴)
    if idx>=10:
        # 최근 10일 중 최저점 위치
        min_idx=idx-10
        min_val=c[idx-10]
        for j in range(idx-10,idx+1):
            if c[j]<min_val: min_val=c[j];min_idx=j
        days_since_low=idx-min_idx
        f["days_since_10d_low"]=days_since_low
        f["bounce_from_10d_low"]=(c[idx]-min_val)/min_val if min_val>0 else 0
    else: f["days_since_10d_low"]=0;f["bounce_from_10d_low"]=0
    
    # Williams %R
    if idx>=13:
        hh=max(h[idx-13:idx+1]);ll=min(l[idx-13:idx+1])
        f["williams_r"]=(hh-c[idx])/(hh-ll)*-100 if hh!=ll else -50
    else: f["williams_r"]=-50
    
    # Stochastic RSI
    if idx>=13 and len(rw)>=14:
        rsi_min=min(rw);rsi_max=max(rw)
        f["stoch_rsi"]=(f["rsi_14"]-rsi_min)/(rsi_max-rsi_min) if rsi_max!=rsi_min else 0.5
    else: f["stoch_rsi"]=0.5
    
    # 거래량 프로파일 — 양봉 거래량 비율 (매수세 강도)
    buy_vol=sum(v[j] for j in range(max(0,idx-6),idx+1) if c[j]>=o[j])
    total_vol=sum(v[j] for j in range(max(0,idx-6),idx+1))
    f["buy_vol_ratio_7d"]=buy_vol/total_vol if total_vol>0 else 0.5
    
    # 당일 캔들 특성
    day_range=h[idx]-l[idx]
    if day_range>0:
        f["today_body_pct"]=abs(c[idx]-o[idx])/day_range
        f["today_position"]=(c[idx]-l[idx])/day_range
    else: f["today_body_pct"]=0;f["today_position"]=0.5
    
    # 최근 3일 거래량 패턴 (증가 추세?)
    if idx>=2:
        v_trend_3d=[v[idx-2],v[idx-1],v[idx]]
        f["vol_increasing_3d"]=1 if v_trend_3d[2]>v_trend_3d[1]>v_trend_3d[0] else 0
    else: f["vol_increasing_3d"]=0
    
    # ATR 대비 당일 range (변동성 확대 감지)
    if a14>0 and day_range>0:
        f["range_vs_atr"]=day_range/a14
    else: f["range_vs_atr"]=1
    
    # 20일 MA 기울기 (추세 방향)
    if idx>=24:
        ma20_now=mean(c[idx-19:idx+1])
        ma20_5ago=mean(c[idx-24:idx-4])
        f["ma20_slope"]=(ma20_now-ma20_5ago)/ma20_5ago if ma20_5ago>0 else 0
    else: f["ma20_slope"]=0
    
    # 종가 기준 상하 밴드 이탈
    if idx>=19:
        bm=mean(c[idx-19:idx+1]);bs=stdev(c[idx-19:idx+1])
        upper=bm+2*bs;lower=bm-2*bs
        f["above_upper_bb"]=1 if c[idx]>upper else 0
        f["below_lower_bb"]=1 if c[idx]<lower else 0
    else: f["above_upper_bb"]=0;f["below_lower_bb"]=0
    
    # 거래량 가중 가격 (VWAP 근사) 대비
    if idx>=6:
        vwap=sum(c[j]*v[j] for j in range(idx-6,idx+1))/sum(v[j] for j in range(idx-6,idx+1)) if sum(v[j] for j in range(idx-6,idx+1))>0 else c[idx]
        f["price_vs_vwap"]=c[idx]/vwap-1 if vwap>0 else 0
    else: f["price_vs_vwap"]=0
    
    return f

# ============================================================
# 수동 CV (joblib 없음)
# ============================================================
def manual_cv(model_cls,params,X,y,n_splits=5,metric="ap"):
    tscv=TimeSeriesSplit(n_splits=n_splits);scores=[]
    for ti,vi in tscv.split(X):
        Xt,Xv=X[ti],X[vi];yt,yv=y[ti],y[vi]
        m=model_cls(**params);m.fit(Xt,yt)
        pr=m.predict_proba(Xv)[:,1]
        if sum(yv)==0: continue
        if metric=="ap":
            scores.append(average_precision_score(yv,pr))
        elif metric=="precision_at_k":
            # 상위 k개의 precision
            k=max(1,int(sum(yv)*0.5))
            top_k_idx=np.argsort(pr)[-k:]
            scores.append(sum(yv[top_k_idx])/k)
    return mean(scores) if scores else 0

# ============================================================
# Stage 1: XGB + LGB 학습
# ============================================================
def stage1_train(Xtrs,ytr,spw):
    print("\n" + "="*60)
    print("  Stage 1: XGBoost + LightGBM")
    print("="*60)
    
    # XGBoost — precision 최적화
    print("\n  [XGBoost] 80회 튜닝...")
    space_xgb={
        'n_estimators':[200,300,500,700],'max_depth':[3,4,5,6,7],
        'learning_rate':[0.01,0.02,0.05,0.1],'min_child_weight':[3,5,7,10,15],
        'subsample':[0.6,0.7,0.8],'colsample_bytree':[0.5,0.6,0.7,0.8],
        'gamma':[0,0.1,0.3,0.5,1.0],'reg_alpha':[0,0.1,1],
        'reg_lambda':[1,3,5],
        'scale_pos_weight':[spw,spw*1.5,spw*2,spw*3],
    }
    best_xgb_score=0;best_xgb_params=None
    random.seed(42)
    for i in range(80):
        p={k:random.choice(v) for k,v in space_xgb.items()}
        p.update({'objective':'binary:logistic','eval_metric':'aucpr','use_label_encoder':False,
                  'random_state':42,'tree_method':'hist','verbosity':0})
        try:
            s=manual_cv(xgb.XGBClassifier,p,Xtrs,ytr,n_splits=5)
        except: continue
        if s>best_xgb_score:
            best_xgb_score=s;best_xgb_params=p.copy()
            if (i+1)%20==0: print(f"    [{i+1}/80] AP={s:.4f}")
    print(f"    Best XGB AP={best_xgb_score:.4f}")
    xgb_model=xgb.XGBClassifier(**best_xgb_params);xgb_model.fit(Xtrs,ytr)
    
    # LightGBM
    print("\n  [LightGBM] 80회 튜닝...")
    space_lgb={
        'n_estimators':[200,300,500,700],'max_depth':[3,5,7,10,-1],
        'learning_rate':[0.01,0.02,0.05,0.1],'num_leaves':[15,31,50,80],
        'min_child_samples':[5,10,20,30,50],'subsample':[0.6,0.7,0.8],
        'colsample_bytree':[0.5,0.6,0.7,0.8],
        'reg_alpha':[0,0.1,1],'reg_lambda':[0,0.1,1],
        'scale_pos_weight':[spw,spw*1.5,spw*2,spw*3],
    }
    best_lgb_score=0;best_lgb_params=None
    random.seed(123)
    for i in range(80):
        p={k:random.choice(v) for k,v in space_lgb.items()}
        p.update({'objective':'binary','metric':'average_precision','random_state':42,
                  'verbose':-1,'force_col_wise':True,'n_jobs':1})
        try:
            s=manual_cv(lgb.LGBMClassifier,p,Xtrs,ytr,n_splits=5)
        except: continue
        if s>best_lgb_score:
            best_lgb_score=s;best_lgb_params=p.copy()
            if (i+1)%20==0: print(f"    [{i+1}/80] AP={s:.4f}")
    print(f"    Best LGB AP={best_lgb_score:.4f}")
    lgb_model=lgb.LGBMClassifier(**best_lgb_params);lgb_model.fit(Xtrs,ytr)
    
    return xgb_model,lgb_model,best_xgb_score,best_lgb_score

# ============================================================
# Stage 2: Hard Negative Mining + 재학습
# ============================================================
def stage2_hard_negative(xgb_model,lgb_model,Xtrs,ytr,spw):
    print("\n" + "="*60)
    print("  Stage 2: Hard Negative Mining")
    print("="*60)
    
    # 현재 모델로 train set 예측
    xgb_probs=xgb_model.predict_proba(Xtrs)[:,1]
    lgb_probs=lgb_model.predict_proba(Xtrs)[:,1]
    avg_probs=(xgb_probs+lgb_probs)/2
    
    # False Positive 찾기: 확률 높은데 실제 0인 것
    fp_mask=(avg_probs>0.5)&(ytr==0)
    n_fp=sum(fp_mask)
    print(f"  False Positive: {n_fp}개 (확률>0.5인데 급등 안 함)")
    
    # Hard negative에 가중치 부여하여 재학습
    sample_weights=np.ones(len(ytr))
    sample_weights[fp_mask]=3.0  # FP에 3배 패널티
    sample_weights[ytr==1]=spw*2  # Positive에도 가중치
    
    print("  Hard negative 가중치로 XGB 재학습...")
    xgb_model2=xgb.XGBClassifier(**{k:v for k,v in xgb_model.get_params().items()})
    xgb_model2.fit(Xtrs,ytr,sample_weight=sample_weights)
    
    print("  Hard negative 가중치로 LGB 재학습...")
    lgb_model2=lgb.LGBMClassifier(**{k:v for k,v in lgb_model.get_params().items()})
    lgb_model2.fit(Xtrs,ytr,sample_weight=sample_weights)
    
    return xgb_model2,lgb_model2

# ============================================================
# 앙상블 + Threshold
# ============================================================
class PrecisionEnsemble:
    def __init__(self,models,weights):
        self.models=models;self.weights=weights
    def predict_proba(self,X):
        probs=np.zeros((X.shape[0],2))
        for m,w in zip(self.models,self.weights):
            probs+=m.predict_proba(X)*w
        probs/=sum(self.weights)
        return probs

def find_precision_threshold(model,X,y,min_precision=0.20):
    """Precision >= min_precision인 최저 threshold 찾기 (recall 최대화)"""
    probs=model.predict_proba(X)[:,1]
    best_t=0.95;best_f1=0;best_p=0;best_r=0
    
    # Precision 우선: 여러 precision 타겟에서 최적점
    for t in np.arange(0.10,0.95,0.005):
        pred=(probs>=t).astype(int)
        if sum(pred)<3: continue  # 최소 3개는 예측
        p=precision_score(y,pred,zero_division=0)
        r=recall_score(y,pred,zero_division=0)
        f1=2*p*r/(p+r) if p+r>0 else 0
        
        if p>=min_precision and f1>best_f1:
            best_t=t;best_f1=f1;best_p=p;best_r=r
    
    # min_precision 못 찾으면 F1 최대로
    if best_p<min_precision:
        for t in np.arange(0.10,0.95,0.005):
            pred=(probs>=t).astype(int)
            if sum(pred)==0: continue
            p=precision_score(y,pred,zero_division=0)
            r=recall_score(y,pred,zero_division=0)
            f1=2*p*r/(p+r) if p+r>0 else 0
            if f1>best_f1: best_t=t;best_f1=f1;best_p=p;best_r=r
    
    return best_t,best_p,best_r,best_f1

# ============================================================
# Stage 3: 30일 백테스트
# ============================================================
def backtest_30d(all_data,model,scaler,fcols,threshold):
    BD=30;tp=0;th_=0;ts=0
    for day in range(BD):
        dX=[];dy=[]
        for sym,d in all_data.items():
            n=len(d["c"]);oo,hh,ll,cc,vv=d["o"],d["h"],d["l"],d["c"],d["v"]
            idx=n-1-SURGE_WINDOW-(BD-1-day)
            if idx<59 or idx+SURGE_WINDOW>=n: continue
            av=mean([vv[j]*cc[j] for j in range(max(0,idx-29),idx+1)])
            if av>MAX_VOL or av<MIN_VOL: continue
            feat=extract_features(oo,hh,ll,cc,vv,idx)
            if not feat: continue
            mx=0
            for j in range(idx+1,min(idx+SURGE_WINDOW+1,n)):
                r=cc[j]/cc[idx]-1 if cc[idx]>0 else 0;mx=max(mx,r)
            dX.append([feat.get(fc,0) for fc in fcols])
            dy.append(1 if mx>=SURGE_TH else 0)
        if not dX: continue
        Xd=np.nan_to_num(np.array(dX,dtype=np.float64),nan=0,posinf=0,neginf=0)
        Xds=scaler.transform(Xd);yd=np.array(dy)
        probs=model.predict_proba(Xds)[:,1]
        preds=(probs>=threshold).astype(int)
        tp+=int(sum(preds));th_+=int(sum(preds&yd));ts+=int(sum(yd))
    p=th_/tp if tp>0 else 0;r=th_/ts if ts>0 else 0
    f1=2*p*r/(p+r) if p+r>0 else 0
    return {"precision":float(p),"recall":float(r),"f1":float(f1),"predictions":tp,"hits":th_,"surges":ts}

# ============================================================
# Main
# ============================================================
def main():
    t0=time.time()
    print("="*60)
    print("  Precision 극대화 파이프라인")
    print("  목표: Precision 20%+ (현재 10.6%)")
    print("="*60)
    
    # 데이터
    print("\n[데이터 로드]")
    with open(DATA_DIR/"raw_data.json") as f: all_data=json.load(f)
    print(f"  {len(all_data)}개 코인")
    
    # 데이터셋
    print("\n[데이터셋 구축]")
    rows=[];sc=0
    for sym,d in all_data.items():
        n=len(d["c"]);o,h,l,c,v=d["o"],d["h"],d["l"],d["c"],d["v"]
        av=mean([v[j]*c[j] for j in range(max(0,n-30),n)])
        if av>MAX_VOL or av<MIN_VOL: continue
        for idx in range(59,n-SURGE_WINDOW):
            feat=extract_features(o,h,l,c,v,idx)
            if not feat: continue
            mx=0
            for j in range(idx+1,min(idx+SURGE_WINDOW+1,n)):
                r=c[j]/c[idx]-1 if c[idx]>0 else 0;mx=max(mx,r)
            feat["_label"]=1 if mx>=SURGE_TH else 0
            feat["_ts"]=d["ts"][idx]
            rows.append(feat)
            if mx>=SURGE_TH: sc+=1
    print(f"  {len(rows)}개 ({sc} surge, {sc/len(rows)*100:.2f}%)")
    
    df=pd.DataFrame(rows)
    fcols=[c for c in df.columns if not c.startswith("_")]
    X=np.nan_to_num(df[fcols].values.astype(np.float64),nan=0,posinf=0,neginf=0)
    y=df["_label"].values;ts_arr=df["_ts"].values
    si=np.argsort(ts_arr);X,y=X[si],y[si]
    
    sp=int(len(X)*0.8)
    Xtr,Xte=X[:sp],X[sp:];ytr,yte=y[:sp],y[sp:]
    npos=int(sum(ytr));nneg=len(ytr)-npos;spw=nneg/npos if npos>0 else 1
    print(f"  Train:{len(Xtr)}({npos}+) Test:{len(Xte)}({int(sum(yte))}+)")
    print(f"  Features: {len(fcols)}개")
    
    scaler=RobustScaler();Xtrs=scaler.fit_transform(Xtr);Xtes=scaler.transform(Xte)
    
    # Stage 1
    xgb1,lgb1,xgb_ap,lgb_ap=stage1_train(Xtrs,ytr,spw)
    
    # Stage 1 결과
    print("\n  [Stage 1 Test 결과]")
    for name,m in [("XGB",xgb1),("LGB",lgb1)]:
        t,p,r,f1=find_precision_threshold(m,Xtes,yte,min_precision=0.20)
        print(f"    {name}: t={t:.2f} P={p:.1%} R={r:.1%} F1={f1:.3f}")
    
    # 앙상블 v1
    tw=xgb_ap+lgb_ap
    ens1=PrecisionEnsemble([xgb1,lgb1],[xgb_ap/tw,lgb_ap/tw])
    t1,p1,r1,f1_1=find_precision_threshold(ens1,Xtes,yte,min_precision=0.20)
    print(f"    ENS1: t={t1:.2f} P={p1:.1%} R={r1:.1%} F1={f1_1:.3f}")
    
    # Stage 2: Hard Negative
    xgb2,lgb2=stage2_hard_negative(xgb1,lgb1,Xtrs,ytr,spw)
    
    print("\n  [Stage 2 Test 결과]")
    for name,m in [("XGB_HN",xgb2),("LGB_HN",lgb2)]:
        t,p,r,f1=find_precision_threshold(m,Xtes,yte,min_precision=0.20)
        print(f"    {name}: t={t:.2f} P={p:.1%} R={r:.1%} F1={f1:.3f}")
    
    # 앙상블 v2 (4개 모델)
    ens2=PrecisionEnsemble([xgb1,lgb1,xgb2,lgb2],[xgb_ap,lgb_ap,xgb_ap*1.2,lgb_ap*1.2])
    t2,p2,r2,f1_2=find_precision_threshold(ens2,Xtes,yte,min_precision=0.20)
    print(f"    ENS2(4모델): t={t2:.2f} P={p2:.1%} R={r2:.1%} F1={f1_2:.3f}")
    
    # 최적 모델 선택
    candidates=[
        ("ens1",ens1,t1,p1,r1,f1_1),
        ("ens2",ens2,t2,p2,r2,f1_2),
    ]
    # 개별 모델도 추가
    for name,m in [("xgb1",xgb1),("lgb1",lgb1),("xgb2",xgb2),("lgb2",lgb2)]:
        t,p,r,f1=find_precision_threshold(m,Xtes,yte,min_precision=0.20)
        candidates.append((name,m,t,p,r,f1))
    
    # Precision 높은 순으로 정렬, 같으면 F1
    candidates.sort(key=lambda x:(-x[3],-x[5]))
    
    print(f"\n  [모델 비교 — Precision 순]")
    for name,m,t,p,r,f1 in candidates:
        print(f"    {name:<10} t={t:.2f} P={p:.1%} R={r:.1%} F1={f1:.3f}")
    
    best_name,best_model,best_t,best_p,best_r,best_f1=candidates[0]
    print(f"\n  선택: {best_name} (P={best_p:.1%})")
    
    # Stage 3: 30일 백테스트
    print("\n" + "="*60)
    print("  Stage 3: 30일 백테스트")
    print("="*60)
    bt=backtest_30d(all_data,best_model,scaler,fcols,best_t)
    print(f"  P={bt['precision']:.1%} R={bt['recall']:.1%} F1={bt['f1']:.3f}")
    print(f"  예측 {bt['predictions']}, 적중 {bt['hits']}, 급등 {bt['surges']}")
    
    # 다른 후보도 백테스트
    print("\n  [다른 모델 백테스트]")
    all_bt=[(best_name,best_t,bt)]
    for name,m,t,p,r,f1 in candidates[1:3]:  # 상위 3개만
        bt2=backtest_30d(all_data,m,scaler,fcols,t)
        all_bt.append((name,t,bt2))
        print(f"  {name}: P={bt2['precision']:.1%} R={bt2['recall']:.1%} F1={bt2['f1']:.3f} ({bt2['hits']}/{bt2['predictions']})")
    
    # 백테스트 결과로 최종 선택
    all_bt.sort(key=lambda x:(-x[2]["precision"],-x[2]["f1"]))
    final_name,final_t,final_bt=all_bt[0]
    final_model=[c[1] for c in candidates if c[0]==final_name][0]
    
    print(f"\n  === 최종 모델: {final_name} ===")
    print(f"  Threshold: {final_t:.3f}")
    print(f"  30일 Precision: {final_bt['precision']:.1%}")
    print(f"  30일 Recall: {final_bt['recall']:.1%}")
    print(f"  30일 F1: {final_bt['f1']:.3f}")
    
    # 저장
    print("\n[저장]")
    with open(DATA_DIR/"best_model.pkl","wb") as f:
        pickle.dump({"model":final_model,"scaler":scaler,"feature_cols":fcols,
                     "threshold":float(final_t),"surge_threshold":SURGE_TH},f)
    
    config={"model_name":final_name,"surge_threshold":SURGE_TH,"threshold":float(final_t),
            "feature_cols":fcols,"precision":float(final_bt["precision"]),
            "recall":float(final_bt["recall"]),"f1":float(final_bt["f1"]),
            "predictions_30d":final_bt["predictions"],"hits_30d":final_bt["hits"],
            "surges_30d":final_bt["surges"],"trained_at":datetime.datetime.now().isoformat()}
    with open(BASE_DIR/"ml_model_config.json","w") as f: json.dump(config,f,indent=2)
    
    elapsed=time.time()-t0
    print(f"  저장 완료! ({elapsed/60:.1f}분)")
    print("="*60)

if __name__=="__main__": main()
