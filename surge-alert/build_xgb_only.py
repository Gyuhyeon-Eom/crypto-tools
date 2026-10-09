#!/usr/bin/env python3
"""XGBoost만 빠르게 학습 → pkl 저장 → 즉시 스캔"""

import json,math,os,sys,time,datetime,pickle,warnings,random
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score,precision_score,recall_score
from sklearn.preprocessing import RobustScaler
import xgboost as xgb

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

BASE_DIR=Path(__file__).parent
DATA_DIR=BASE_DIR/"ml_data"
DATA_DIR.mkdir(exist_ok=True)

SURGE_TH=0.3; SURGE_WINDOW=7; MIN_VOL=50_000; MAX_VOL=10_000_000

STABLECOINS={"USDCUSDT","UUSDT","XUSDUSDT","PAXGUSDT","DAIUSDT","FDUSDUSDT","TUSDUSDT","BUSDUSDT","EURUSDT","USTCUSDT","USDTUSDT","USDDUSDT","RLUSDUSDT","USD1USDT"}
LARGE_CAPS={"BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","SOLUSDT","ADAUSDT","DOGEUSDT","TRXUSDT","DOTUSDT","MATICUSDT","LTCUSDT","BCHUSDT","LINKUSDT","AVAXUSDT","UNIUSDT","AAVEUSDT","ATOMUSDT"}

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
    g=[max(x,0) for x in d]; l=[max(-x,0) for x in d]
    ag=mean(g[:period]);al=mean(l[:period])
    rsi[period]=100.0 if al==0 else 100.0-100.0/(1.0+ag/al)
    for i in range(period,len(d)):
        ag=(ag*(period-1)+g[i])/period;al=(al*(period-1)+l[i])/period
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
    return f

# ============================================================
def bybit_get(ep,params=None):
    url="https://api.bybit.com"+ep
    if params: url+="?"+"&".join(f"{k}={v}" for k,v in params.items())
    req=urllib.request.Request(url,headers={"User-Agent":"xgb-scan/1.0"})
    for i in range(3):
        try:
            with urllib.request.urlopen(req,timeout=15) as r: return json.loads(r.read())
        except:
            if i==2: raise
            time.sleep(1)
def fetch_klines(sym,interval="D",limit=200):
    d=bybit_get("/v5/market/kline",{"category":"spot","symbol":sym,"interval":interval,"limit":str(limit)})
    rows=d.get("result",{}).get("list",[])
    return list(reversed(rows)) if rows else []
def fetch_tickers():
    d=bybit_get("/v5/market/tickers",{"category":"spot"})
    return d.get("result",{}).get("list",[])

# ============================================================
def main():
    t0=time.time()
    print("="*60)
    print("  XGBoost 학습 + 즉시 스캔")
    print("="*60)

    # 1. 캐시 로드
    cache=DATA_DIR/"raw_data.json"
    print("\n[1] 데이터 로드...")
    with open(cache) as f: all_data=json.load(f)
    print(f"  {len(all_data)}개 코인")

    # 2. 데이터셋
    print("\n[2] 데이터셋 구축...")
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
    print(f"  {len(rows)}개 ({sc} surge)")

    df=pd.DataFrame(rows)
    fcols=[c for c in df.columns if not c.startswith("_")]
    X=np.nan_to_num(df[fcols].values.astype(np.float64),nan=0,posinf=0,neginf=0)
    y=df["_label"].values;ts=df["_ts"].values
    si=np.argsort(ts);X,y=X[si],y[si]

    sp=int(len(X)*0.8)
    Xtr,Xte=X[:sp],X[sp:];ytr,yte=y[:sp],y[sp:]
    npos=sum(ytr);nneg=len(ytr)-npos;spw=nneg/npos if npos>0 else 1

    scaler=RobustScaler();Xtrs=scaler.fit_transform(Xtr);Xtes=scaler.transform(Xte)

    # 3. 이전 실행에서 찾은 best params 직접 사용
    print("\n[3] XGBoost 학습 (best params)...")
    # 60회 튜닝 결과 best AP=0.0629였던 설정 기반으로 몇 가지 시도
    best_model=None;best_ap=0;best_params=None
    
    param_sets=[
        {'n_estimators':500,'max_depth':6,'learning_rate':0.05,'min_child_weight':5,'subsample':0.8,'colsample_bytree':0.7,'gamma':0.1,'scale_pos_weight':spw*1.5},
        {'n_estimators':300,'max_depth':5,'learning_rate':0.1,'min_child_weight':3,'subsample':0.7,'colsample_bytree':0.8,'gamma':0,'scale_pos_weight':spw*2},
        {'n_estimators':500,'max_depth':7,'learning_rate':0.02,'min_child_weight':10,'subsample':0.8,'colsample_bytree':0.6,'gamma':0.5,'scale_pos_weight':spw},
        {'n_estimators':300,'max_depth':4,'learning_rate':0.05,'min_child_weight':5,'subsample':0.8,'colsample_bytree':0.7,'gamma':0.1,'scale_pos_weight':spw*2},
        {'n_estimators':500,'max_depth':5,'learning_rate':0.05,'min_child_weight':3,'subsample':0.7,'colsample_bytree':0.7,'gamma':0,'scale_pos_weight':spw*1.5},
    ]
    
    # + 랜덤 20개 추가
    random.seed(42)
    space={'n_estimators':[200,300,500],'max_depth':[4,5,6,7],'learning_rate':[0.02,0.05,0.1],
           'min_child_weight':[3,5,10],'subsample':[0.7,0.8],'colsample_bytree':[0.6,0.7,0.8],
           'gamma':[0,0.1,0.5],'scale_pos_weight':[spw,spw*1.5,spw*2]}
    for _ in range(20):
        param_sets.append({k:random.choice(v) for k,v in space.items()})
    
    tscv=TimeSeriesSplit(n_splits=3)  # 3-fold로 빠르게
    
    for i,params in enumerate(param_sets):
        p=params.copy()
        p.update({'objective':'binary:logistic','eval_metric':'aucpr','use_label_encoder':False,
                  'random_state':42,'tree_method':'hist','verbosity':0})
        try:
            scores=[]
            for ti,vi in tscv.split(Xtrs):
                m=xgb.XGBClassifier(**p);m.fit(Xtrs[ti],ytr[ti])
                pr=m.predict_proba(Xtrs[vi])[:,1]
                if sum(ytr[vi])>0: scores.append(average_precision_score(ytr[vi],pr))
            ap=mean(scores) if scores else 0
            if ap>best_ap:
                best_ap=ap;best_params=p
                print(f"    [{i+1}/25] AP={ap:.4f} *")
        except: continue
    
    print(f"  Best AP={best_ap:.4f}")
    model=xgb.XGBClassifier(**best_params)
    model.fit(Xtrs,ytr)

    # Threshold 최적화
    probs=model.predict_proba(Xtes)[:,1]
    best_t=0.5;best_f1=0;best_p=0;best_r=0
    for t in np.arange(0.05,0.95,0.01):
        pred=(probs>=t).astype(int)
        if sum(pred)==0: continue
        p=precision_score(yte,pred,zero_division=0)
        r=recall_score(yte,pred,zero_division=0)
        f1=2*p*r/(p+r) if p+r>0 else 0
        if p>=0.08 and f1>best_f1: best_t=t;best_f1=f1;best_p=p;best_r=r
    
    print(f"  Threshold={best_t:.2f} P={best_p:.1%} R={best_r:.1%} F1={best_f1:.3f}")

    # 4. 저장
    print("\n[4] 모델 저장...")
    with open(DATA_DIR/"best_model.pkl","wb") as f:
        pickle.dump({"model":model,"scaler":scaler,"feature_cols":fcols,
                     "threshold":float(best_t),"surge_threshold":SURGE_TH},f)
    
    config={"model_name":"xgboost","surge_threshold":SURGE_TH,"threshold":float(best_t),
            "feature_cols":fcols,"precision":float(best_p),"recall":float(best_r),"f1":float(best_f1)}
    with open(BASE_DIR/"ml_model_config.json","w") as f: json.dump(config,f,indent=2)
    print("  저장 완료!")

    # 5. 즉시 스캔
    print("\n[5] 현재 시점 스캔...")
    tickers=fetch_tickers()
    small_alts=[]
    for t in tickers:
        sym=t["symbol"]
        if not sym.endswith("USDT") or sym in STABLECOINS or sym in LARGE_CAPS: continue
        vol24=float(t.get("turnover24h",0))
        if MIN_VOL<=vol24<=MAX_VOL: small_alts.append(sym)
    
    print(f"  {len(small_alts)}개 소형 알트 스캔...")
    results=[]
    for i,sym in enumerate(small_alts):
        if (i+1)%50==0: print(f"    [{i+1}/{len(small_alts)}]...")
        try:
            klines=fetch_klines(sym,"D",100)
            if len(klines)<60: continue
            o=[float(k[1]) for k in klines];h=[float(k[2]) for k in klines]
            l=[float(k[3]) for k in klines];c=[float(k[4]) for k in klines]
            v=[float(k[5]) for k in klines]
            
            av=mean([v[j]*c[j] for j in range(max(0,len(c)-30),len(c))])
            if av>MAX_VOL or av<MIN_VOL: continue
            
            feat=extract_features(o,h,l,c,v,len(c)-1)
            if not feat: continue
            
            fv=np.array([[feat.get(fc,0) for fc in fcols]],dtype=np.float64)
            fv=np.nan_to_num(fv,nan=0,posinf=0,neginf=0)
            fvs=scaler.transform(fv)
            prob=model.predict_proba(fvs)[0,1]
            
            if prob>=best_t*0.7:  # threshold의 70%부터 표시
                results.append({
                    "symbol":sym,"coin":sym.replace("USDT",""),
                    "prob":float(prob),"price":c[-1],
                    "ret_7d":feat.get("ret_7d",0)*100,
                    "rsi":feat.get("rsi_14",0),
                    "vol_ratio":feat.get("vol_ratio_7d_30d",1),
                    "above_threshold":prob>=best_t,
                })
            time.sleep(0.03)
        except: continue
    
    results.sort(key=lambda x:x["prob"],reverse=True)
    
    print(f"\n{'='*60}")
    print(f"  스캔 결과 (threshold={best_t:.2f})")
    print(f"{'='*60}")
    print(f"\n  {'#':>2} {'코인':<10} {'확률':>7} {'현재가':<12} {'7d':>7} {'RSI':>5} {'Vol':>5} {'신호':>4}")
    print(f"  {'-'*60}")
    
    for i,r in enumerate(results[:20]):
        flag=">>>" if r["above_threshold"] else ""
        print(f"  {i+1:>2} {r['coin']:<10} {r['prob']:>6.1%} ${r['price']:<11.4f} {r['ret_7d']:>+6.1f}% {r['rsi']:>5.0f} {r['vol_ratio']:>5.2f} {flag}")
    
    alert_coins=[r for r in results if r["above_threshold"]]
    print(f"\n  급등 후보: {len(alert_coins)}개 (threshold 이상)")
    
    # 결과 저장
    with open(BASE_DIR/"scan_result_ml.json","w") as f:
        json.dump({"timestamp":datetime.datetime.now().isoformat(),"threshold":float(best_t),
                   "results":results[:30],"alert_coins":[r["coin"] for r in alert_coins]},f,indent=2)
    
    elapsed=time.time()-t0
    print(f"\n  소요: {elapsed/60:.1f}분")
    print("="*60)

if __name__=="__main__": main()
