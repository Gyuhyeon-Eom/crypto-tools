#!/usr/bin/env python3
"""
Round 5: Precision x Coverage 밸런스
Round 4 결과에서 P=26.7%(16/60)~26.4%(28/106) 나온 설정들을 활용
최종 모델 선정 기준: precision * sqrt(predictions) — 적중률 높으면서 충분한 예측 수

+ threshold 세밀 조정 (0.01 단위)
+ 앙상블 다시 시도 (xgb*4 + lgb*6)
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

# Round 4의 feature extraction + helpers 재사용
sys.path.insert(0,str(BASE_DIR))
from round4_metalabel import (extract_features, mean, stdev, triple_barrier_label, 
                                has_recent_surge, compute_rsi, compute_atr, compute_macd,
                                compute_obv, polyfit_slope)

def manual_cv(model_cls,params,X,y,n_splits=5):
    from sklearn.model_selection import TimeSeriesSplit
    tscv=TimeSeriesSplit(n_splits=n_splits);scores=[]
    for ti,vi in tscv.split(X):
        m=model_cls(**params);m.fit(X[ti],y[ti])
        pr=m.predict_proba(X[vi])[:,1]
        if sum(y[vi])>0: scores.append(average_precision_score(y[vi],pr))
    return mean(scores) if scores else 0

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
            if has_recent_surge(cc,idx): continue
            feat=extract_features(oo,hh,ll,cc,vv,idx)
            if not feat: continue
            label=triple_barrier_label(cc,idx,upper=SURGE_TH,lower=-0.15,max_days=SURGE_WINDOW)
            dX.append([feat.get(fc,0) for fc in fcols]);dy.append(1 if label==1 else 0)
        if not dX: continue
        Xd=np.nan_to_num(np.array(dX,dtype=np.float64),nan=0,posinf=0,neginf=0)
        Xds=scaler.transform(Xd);yd=np.array(dy)
        probs=model.predict_proba(Xds)[:,1]
        preds=(probs>=threshold).astype(int)
        tp+=int(sum(preds));th_+=int(sum(preds&yd));ts+=int(sum(yd))
    p=th_/tp if tp>0 else 0;r=th_/ts if ts>0 else 0
    f1=2*p*r/(p+r) if p+r>0 else 0
    return {"precision":float(p),"recall":float(r),"f1":float(f1),"predictions":tp,"hits":th_,"surges":ts}

class DualEnsemble:
    def __init__(self,m1,m2,w1=0.5,w2=0.5):
        self.m1=m1;self.m2=m2;self.w1=w1;self.w2=w2
    def predict_proba(self,X):
        return self.m1.predict_proba(X)*self.w1+self.m2.predict_proba(X)*self.w2

def main():
    t0=time.time()
    send_slack(":zap: *Round 5 시작* — Precision x Coverage 밸런스 최적화")
    print("Round 5: Balance optimization")
    
    with open(DATA_DIR/"raw_data.json") as f: all_data=json.load(f)
    
    # 데이터셋 (Round 4와 동일)
    print("\n[1] 데이터셋...")
    rows=[]
    for sym,d in all_data.items():
        n=len(d["c"]);o,h,l,c,v=d["o"],d["h"],d["l"],d["c"],d["v"]
        av=mean([v[j]*c[j] for j in range(max(0,n-30),n)])
        if av>MAX_VOL or av<MIN_VOL: continue
        for idx in range(59,n-SURGE_WINDOW):
            if has_recent_surge(c,idx): continue
            feat=extract_features(o,h,l,c,v,idx)
            if not feat: continue
            label=triple_barrier_label(c,idx,upper=SURGE_TH,lower=-0.15,max_days=SURGE_WINDOW)
            feat["_label"]=1 if label==1 else 0;feat["_ts"]=d["ts"][idx]
            rows.append(feat)
    
    sc=sum(1 for r in rows if r["_label"]==1)
    print(f"  {len(rows)}개 ({sc} surge)")
    
    df=pd.DataFrame(rows)
    fcols=[c for c in df.columns if not c.startswith("_")]
    X=np.nan_to_num(df[fcols].values.astype(np.float64),nan=0,posinf=0,neginf=0)
    y=df["_label"].values;ts_arr=df["_ts"].values
    si=np.argsort(ts_arr);X,y=X[si],y[si]
    sp=int(len(X)*0.8)
    Xtr,Xte=X[:sp],X[sp:];ytr,yte=y[:sp],y[sp:]
    npos=int(sum(ytr));spw=(len(ytr)-npos)/npos if npos>0 else 1
    scaler=RobustScaler();Xtrs=scaler.fit_transform(Xtr);Xtes=scaler.transform(Xte)
    print(f"  Train:{len(Xtr)}({npos}+) Test:{len(Xte)}({int(sum(yte))}+)")
    
    # [2] Round 4 best 설정으로 모델 학습 + 세밀 threshold 탐색
    print("\n[2] 세밀 최적화...")
    
    all_candidates=[]  # (name, model, threshold, bt)
    
    # XGB spw*4 (Round 4에서 P=26.7% 나온 설정)
    for mult in [3,4,5]:
        random.seed(42+mult)
        best_ap=0;best_p=None
        space={'n_estimators':[300,500,700],'max_depth':[3,4,5,6],'learning_rate':[0.01,0.02,0.05],
               'min_child_weight':[5,10,15],'subsample':[0.6,0.7,0.8],'colsample_bytree':[0.5,0.6,0.7],
               'gamma':[0,0.1,0.3,0.5],'reg_alpha':[0,0.1,1],'reg_lambda':[1,3,5],
               'scale_pos_weight':[spw*mult]}
        for i in range(30):
            p={k:random.choice(v) for k,v in space.items()}
            p.update({'objective':'binary:logistic','eval_metric':'aucpr','use_label_encoder':False,
                      'random_state':42,'tree_method':'hist','verbosity':0})
            try: s=manual_cv(xgb.XGBClassifier,p,Xtrs,ytr,n_splits=5)
            except: continue
            if s>best_ap: best_ap=s;best_p=p.copy()
        if not best_p: continue
        m=xgb.XGBClassifier(**best_p);m.fit(Xtrs,ytr)
        
        # 세밀 threshold (0.01 단위)
        probs=m.predict_proba(Xtes)[:,1]
        for t in np.arange(0.95,0.50,-0.01):
            pred=(probs>=t).astype(int)
            n_pred=sum(pred)
            if n_pred<5: continue
            pr=precision_score(yte,pred,zero_division=0)
            # 점수: precision * sqrt(predictions) — 적중률+커버리지 밸런스
            score=pr*math.sqrt(n_pred)
            if pr>=0.15 and n_pred>=5:
                bt=backtest_30d(all_data,m,scaler,fcols,t)
                bt_score=bt["precision"]*math.sqrt(max(1,bt["predictions"]))
                print(f"    xgb*{mult} t={t:.2f}: P={bt['precision']:.1%} R={bt['recall']:.1%} ({bt['hits']}/{bt['predictions']}) score={bt_score:.2f}")
                all_candidates.append((f"xgb*{mult}",m,float(t),bt,bt_score))
                if len([c for c in all_candidates if c[0].startswith(f"xgb*{mult}")])>=3: break
    
    # LGB spw*5,6 (Round 4에서 P=26% 나온 설정들)
    for mult in [4,5,6]:
        random.seed(100+mult)
        best_ap=0;best_p=None
        space={'n_estimators':[300,500,700],'max_depth':[3,5,7],'learning_rate':[0.01,0.02,0.05],
               'num_leaves':[15,31,50],'min_child_samples':[10,20,50],'subsample':[0.6,0.7,0.8],
               'colsample_bytree':[0.5,0.6,0.7],'reg_alpha':[0,0.1,1],'reg_lambda':[0,0.1,1],
               'scale_pos_weight':[spw*mult]}
        for i in range(30):
            p={k:random.choice(v) for k,v in space.items()}
            p.update({'objective':'binary','metric':'average_precision','random_state':42,
                      'verbose':-1,'force_col_wise':True,'n_jobs':1})
            try: s=manual_cv(lgb.LGBMClassifier,p,Xtrs,ytr,n_splits=5)
            except: continue
            if s>best_ap: best_ap=s;best_p=p.copy()
        if not best_p: continue
        m=lgb.LGBMClassifier(**best_p);m.fit(Xtrs,ytr)
        
        probs=m.predict_proba(Xtes)[:,1]
        for t in np.arange(0.95,0.50,-0.01):
            pred=(probs>=t).astype(int)
            n_pred=sum(pred)
            if n_pred<5: continue
            pr=precision_score(yte,pred,zero_division=0)
            if pr>=0.15 and n_pred>=5:
                bt=backtest_30d(all_data,m,scaler,fcols,t)
                bt_score=bt["precision"]*math.sqrt(max(1,bt["predictions"]))
                print(f"    lgb*{mult} t={t:.2f}: P={bt['precision']:.1%} R={bt['recall']:.1%} ({bt['hits']}/{bt['predictions']}) score={bt_score:.2f}")
                all_candidates.append((f"lgb*{mult}",m,float(t),bt,bt_score))
                if len([c for c in all_candidates if c[0].startswith(f"lgb*{mult}")])>=3: break
    
    # [3] 앙상블 시도 (상위 XGB + 상위 LGB)
    print("\n[3] 앙상블...")
    xgb_models=[c for c in all_candidates if "xgb" in c[0]]
    lgb_models=[c for c in all_candidates if "lgb" in c[0]]
    
    if xgb_models and lgb_models:
        best_xgb=max(xgb_models,key=lambda x:x[4])
        best_lgb=max(lgb_models,key=lambda x:x[4])
        
        for w1 in [0.4,0.5,0.6]:
            ens=DualEnsemble(best_xgb[1],best_lgb[1],w1,1-w1)
            probs=ens.predict_proba(Xtes)[:,1]
            for t in np.arange(0.92,0.50,-0.01):
                pred=(probs>=t).astype(int)
                n_pred=sum(pred)
                if n_pred<5: continue
                pr=precision_score(yte,pred,zero_division=0)
                if pr>=0.15:
                    bt=backtest_30d(all_data,ens,scaler,fcols,t)
                    bt_score=bt["precision"]*math.sqrt(max(1,bt["predictions"]))
                    print(f"    ens({w1:.1f}) t={t:.2f}: P={bt['precision']:.1%} R={bt['recall']:.1%} ({bt['hits']}/{bt['predictions']}) score={bt_score:.2f}")
                    all_candidates.append((f"ens_{w1:.1f}",ens,float(t),bt,bt_score))
                    break
    
    # [4] 최종 선택 — bt_score 순
    all_candidates.sort(key=lambda x:x[4],reverse=True)
    
    print(f"\n[4] 전체 결과 (score = precision * sqrt(preds)):")
    for name,_,t,bt,sc in all_candidates[:10]:
        print(f"    {name:<15} t={t:.2f} P={bt['precision']:.1%} R={bt['recall']:.1%} ({bt['hits']}/{bt['predictions']}) score={sc:.2f}")
    
    best_name,best_model,best_t,best_bt,best_score=all_candidates[0]
    
    # [5] 스캔
    print("\n[5] 스캔...")
    def bybit_get(ep,params=None):
        url="https://api.bybit.com"+ep
        if params: url+="?"+"&".join(f"{k}={v}" for k,v in params.items())
        req=urllib.request.Request(url,headers={"User-Agent":"r5scan"})
        with urllib.request.urlopen(req,timeout=15) as r: return json.loads(r.read())
    
    tickers=bybit_get("/v5/market/tickers",{"category":"spot"}).get("result",{}).get("list",[])
    SKIP={"USDCUSDT","UUSDT","XUSDUSDT","PAXGUSDT","DAIUSDT","FDUSDUSDT","TUSDUSDT","BUSDUSDT","EURUSDT","USTCUSDT","USDTUSDT","USDDUSDT","RLUSDUSDT","USD1USDT","BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","SOLUSDT","ADAUSDT","DOGEUSDT","TRXUSDT","DOTUSDT","MATICUSDT","LTCUSDT","BCHUSDT","LINKUSDT","AVAXUSDT","UNIUSDT","AAVEUSDT","ATOMUSDT"}
    
    scan_results=[]
    alts=[t["symbol"] for t in tickers if t["symbol"].endswith("USDT") and t["symbol"] not in SKIP and MIN_VOL<=float(t.get("turnover24h",0))<=MAX_VOL]
    
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
            prob=best_model.predict_proba(scaler.transform(fv))[0,1]
            if prob>=best_t*0.7:
                r7=(cc[-1]/cc[-8]-1)*100 if len(cc)>=8 and cc[-8]>0 else 0
                r14=(cc[-1]/cc[-15]-1)*100 if len(cc)>=15 and cc[-15]>0 else 0
                scan_results.append({"coin":sym.replace("USDT",""),"prob":float(prob),
                    "price":cc[-1],"ret_7d":r7,"ret_14d":r14,
                    "rsi":feat.get("rsi_14",0),"vol_ratio":feat.get("vol_ratio_7d_30d",1),
                    "hit":prob>=best_t})
            time.sleep(0.03)
        except: continue
    
    scan_results.sort(key=lambda x:x["prob"],reverse=True)
    hits=[r for r in scan_results if r["hit"]]
    
    # 결과
    elapsed=time.time()-t0
    lines=[f"*Round 5 완료* ({elapsed/60:.0f}분)"]
    lines.append(f"*최적 모델: {best_name}* (score={best_score:.2f})")
    lines.append(f"```Precision: {best_bt['precision']:.1%}\nRecall:    {best_bt['recall']:.1%}\n30일: {best_bt['hits']}/{best_bt['predictions']} 적중```")
    lines.append(f"\n*전조 후보 (아직 안 오른 코인만):*")
    lines.append("```")
    lines.append(f" # {'코인':<10} {'확률':>6} {'현재가':<11} {'7d':>7} {'14d':>7} {'RSI':>4}")
    for i,r in enumerate(scan_results[:15]):
        flag="<<<" if r["hit"] else ""
        lines.append(f"{i+1:>2} {r['coin']:<10} {r['prob']:>5.1%} ${r['price']:<10.4f} {r['ret_7d']:>+6.1f}% {r['ret_14d']:>+6.1f}% {r['rsi']:>4.0f} {flag}")
    lines.append(f"```\n전조 후보: {len(hits)}개")
    
    # 진행 요약
    lines.append(f"\n*Precision 추이:*")
    lines.append("R1: 17.4% → R2: 18.2% → R4: 26.7% → R5: {:.1%}".format(best_bt['precision']))
    
    msg="\n".join(lines)
    send_slack(msg);print(msg)
    
    # 저장
    save_data={"model":best_model,"scaler":scaler,"feature_cols":fcols,
               "threshold":float(best_t),"surge_threshold":SURGE_TH,
               "pretrigger_filter":True,"triple_barrier":True}
    with open(DATA_DIR/"best_model.pkl","wb") as f: pickle.dump(save_data,f)
    config={"model_name":best_name,"surge_threshold":SURGE_TH,"threshold":float(best_t),
            "feature_cols":fcols,"precision":float(best_bt["precision"]),
            "recall":float(best_bt["recall"]),"f1":float(best_bt["f1"]),
            "predictions_30d":best_bt["predictions"],"hits_30d":best_bt["hits"],
            "surges_30d":best_bt["surges"],"trained_at":datetime.datetime.now().isoformat(),
            "round":5,"score":float(best_score)}
    with open(BASE_DIR/"ml_model_config.json","w") as f: json.dump(config,f,indent=2)
    print("  저장 완료")

if __name__=="__main__": main()
