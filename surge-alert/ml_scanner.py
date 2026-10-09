#!/usr/bin/env python3
"""
ML 기반 급등 전조 스캐너 — Round 5 모델
매 1시간 실행, Slack DM으로 리포트
"""

import json,math,os,sys,time,datetime,pickle,warnings
import urllib.request
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

BASE_DIR=Path(__file__).parent
DATA_DIR=BASE_DIR/"ml_data"
STATE_FILE=BASE_DIR/"ml_scanner_state.json"
MIN_VOL=1_000_000;MAX_VOL=10_000_000

WEBHOOK_URL=""
_env=os.path.expanduser("~/.openclaw/.env.slack-surge")
if os.path.exists(_env):
    with open(_env) as f:
        for line in f:
            if line.startswith("SLACK_WEBHOOK_URL="): WEBHOOK_URL=line.strip().split("=",1)[1]

def send_slack(msg):
    if not WEBHOOK_URL: print(msg); return
    for chunk in [msg[i:i+3500] for i in range(0,len(msg),3500)]:
        payload=json.dumps({"text":chunk})
        req=urllib.request.Request(WEBHOOK_URL,data=payload.encode(),headers={"Content-Type":"application/json"},method="POST")
        try: urllib.request.urlopen(req,timeout=30)
        except Exception as e: print(f"Slack err: {e}")
        time.sleep(0.5)

# Feature extraction 재사용
sys.path.insert(0,str(BASE_DIR))
from round4_metalabel import extract_features, mean, has_recent_surge

SKIP={"USDCUSDT","UUSDT","XUSDUSDT","PAXGUSDT","DAIUSDT","FDUSDUSDT","TUSDUSDT","BUSDUSDT","EURUSDT","USTCUSDT","USDTUSDT","USDDUSDT","RLUSDUSDT","USD1USDT","BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","SOLUSDT","ADAUSDT","DOGEUSDT","TRXUSDT","DOTUSDT","MATICUSDT","LTCUSDT","BCHUSDT","LINKUSDT","AVAXUSDT","UNIUSDT","AAVEUSDT","ATOMUSDT"}

def bybit_get(ep,params=None):
    url="https://api.bybit.com"+ep
    if params: url+="?"+"&".join(f"{k}={v}" for k,v in params.items())
    req=urllib.request.Request(url,headers={"User-Agent":"ml-scanner/1.0"})
    for i in range(3):
        try:
            with urllib.request.urlopen(req,timeout=15) as r: return json.loads(r.read())
        except:
            if i==2: raise
            time.sleep(1+i)

def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f: return json.load(f)
    return {"prev_hits":[],"last_scan":""}

def save_state(state):
    with open(STATE_FILE,"w") as f: json.dump(state,f,indent=2)

def main():
    now=datetime.datetime.now()
    print(f"[{now.strftime('%H:%M')}] ML Scanner 시작")
    
    # 모델 로드
    pkl_path=DATA_DIR/"best_model.pkl"
    if not pkl_path.exists():
        print("모델 없음!"); return
    
    with open(pkl_path,"rb") as f: pkg=pickle.load(f)
    model=pkg["model"];scaler=pkg["scaler"];fcols=pkg["feature_cols"];threshold=pkg["threshold"]
    
    state=load_state()
    prev_hits=set(state.get("prev_hits",[]))
    
    # 티커
    tickers=bybit_get("/v5/market/tickers",{"category":"spot"}).get("result",{}).get("list",[])
    alts=[t["symbol"] for t in tickers if t["symbol"].endswith("USDT") and t["symbol"] not in SKIP and MIN_VOL<=float(t.get("turnover24h",0))<=MAX_VOL]
    
    results=[]
    for sym in alts:
        try:
            data=bybit_get("/v5/market/kline",{"category":"spot","symbol":sym,"interval":"D","limit":"100"})
            klines=list(reversed(data.get("result",{}).get("list",[])))
            if len(klines)<60: continue
            o=[float(k[1]) for k in klines];h=[float(k[2]) for k in klines]
            l=[float(k[3]) for k in klines];c=[float(k[4]) for k in klines]
            v=[float(k[5]) for k in klines]
            if has_recent_surge(c,len(c)-1): continue
            feat=extract_features(o,h,l,c,v,len(c)-1)
            if not feat: continue
            fv=np.nan_to_num(np.array([[feat.get(fc,0) for fc in fcols]],dtype=np.float64),nan=0,posinf=0,neginf=0)
            prob=model.predict_proba(scaler.transform(fv))[0,1]
            if prob>=threshold*0.7:
                r7=(c[-1]/c[-8]-1)*100 if len(c)>=8 and c[-8]>0 else 0
                r14=(c[-1]/c[-15]-1)*100 if len(c)>=15 and c[-15]>0 else 0
                results.append({"coin":sym.replace("USDT",""),"symbol":sym,"prob":float(prob),
                    "price":c[-1],"ret_7d":r7,"ret_14d":r14,"rsi":feat.get("rsi_14",0),
                    "vol_ratio":feat.get("vol_ratio_7d_30d",1),"hit":prob>=threshold})
            time.sleep(0.03)
        except: continue
    
    results.sort(key=lambda x:x["prob"],reverse=True)
    hits=[r for r in results if r["hit"]]
    hit_coins=set(r["coin"] for r in hits)
    new_coins=hit_coins-prev_hits
    
    # 리포트 생성
    lines=[f":chart_with_upwards_trend: *ML 전조 스캐너* ({now.strftime('%m/%d %H:%M')})"]
    lines.append(f"`P=24.8% | threshold={threshold:.2f} | Round5 LGB`")
    
    if new_coins:
        lines.append(f"\n:rotating_light: *신규 후보 {len(new_coins)}개:* {', '.join(new_coins)}")
    
    if hits:
        lines.append(f"\n*전조 후보 ({len(hits)}개):*")
        lines.append("```")
        lines.append(f" # {'코인':<10} {'확률':>6} {'현재가':<11} {'7d':>7} {'14d':>7} {'RSI':>4}")
        for i,r in enumerate(hits):
            new="NEW" if r["coin"] in new_coins else ""
            lines.append(f"{i+1:>2} {r['coin']:<10} {r['prob']:>5.1%} ${r['price']:<10.4f} {r['ret_7d']:>+6.1f}% {r['ret_14d']:>+6.1f}% {r['rsi']:>4.0f} {new}")
        lines.append("```")
    else:
        lines.append("\n후보 없음 (모든 코인 threshold 미달)")
    
    # 주시 목록 (threshold 미달이지만 높은 것들)
    watchlist=[r for r in results if not r["hit"]][:5]
    if watchlist:
        lines.append(f"\n*주시 목록:*")
        for r in watchlist:
            lines.append(f"  {r['coin']}: {r['prob']:.1%} (7d {r['ret_7d']:+.1f}%, RSI {r['rsi']:.0f})")
    
    lines.append(f"\n_스캔: {len(alts)}개 중 {len(results)}개 분석, {len(hits)}개 후보_")
    
    send_slack("\n".join(lines))
    print(f"  완료: {len(hits)}개 후보, {len(new_coins)}개 신규")
    
    # 상태 저장
    state["prev_hits"]=list(hit_coins)
    state["last_scan"]=now.isoformat()
    save_state(state)

if __name__=="__main__": main()
