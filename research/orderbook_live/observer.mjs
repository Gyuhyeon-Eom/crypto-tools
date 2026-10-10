import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
export const policy=JSON.parse(fs.readFileSync(new URL('./policy.json',import.meta.url)));
const finite=(r,keys)=>keys.every(k=>typeof r[k]==='number'&&Number.isFinite(r[k]));
export function classify(row){
  if(row.status!=='ready'||policy.excluded_symbols.includes(row.symbol)||row.target_already_above!==false)return null;
  if(!finite(row,['spread_bps','trade_count_60s','net_taker_fraction','buy_quote_60s','sell_quote_60s','mid_return_60s'])
    ||row.spread_bps>policy.max_spread_bps||row.trade_count_60s<policy.min_trade_count)return null;
  const a=policy.ask_depletion,b=policy.bid_recovery;
  if(finite(row,['ask_depth_change','ask_last10s_max_ratio','ask_cost_change'])
    &&row.buy_quote_60s>=policy.min_side_quote&&row.net_taker_fraction>=a.min_net_buy_fraction
    &&row.ask_depth_change<=a.max_depth_change&&row.ask_last10s_max_ratio<=a.max_last10s_ratio
    &&row.ask_cost_change<=a.max_cost_change&&row.mid_return_60s>=a.min_mid_return&&row.mid_return_60s<a.max_mid_return)
    return 'ask_depletion';
  if(finite(row,['bid_recovery'])&&row.sell_quote_60s>=policy.min_side_quote&&row.net_taker_fraction<=b.max_net_buy_fraction
    &&row.bid_recovery>=b.min_recovery&&row.mid_return_60s>=b.min_mid_return&&row.mid_return_60s<b.max_mid_return)
    return 'bid_recovery';
  return null;
}
export class Observer{
  constructor(stateFile){
    this.stateFile=stateFile;this.holds=new Map();this.last={};
    if(stateFile&&fs.existsSync(stateFile)){
      const state=JSON.parse(fs.readFileSync(stateFile));
      if(state.version!==1||!state.last||Array.isArray(state.last)||Object.values(state.last).some(t=>!Number.isFinite(t)||t<0))
        throw new Error('Invalid notification cooldown state; inspect locally before restarting');
      this.last=state.last;
    }
  }
  reset(){this.holds.clear();}
  observe(row,wallNow=Date.now()){
    const kind=classify(row),old=this.holds.get(row.symbol);
    if(!kind){this.holds.delete(row.symbol);return null;}
    if(!old||old.kind!==kind||old.generation!==row.generation||row.t-old.last>policy.max_sample_gap_ms||row.t<=old.last){
      this.holds.set(row.symbol,{kind,generation:row.generation,since:row.t,last:row.t});return null;
    }
    old.last=row.t;
    if(row.t-old.since<policy.hold_ms||wallNow-(this.last[row.symbol]??-Infinity)<policy.symbol_cooldown_ms)return null;
    // Persist before enqueueing: a delivery failure must not cause an alert every second.
    this.last[row.symbol]=wallNow;
    if(this.stateFile){
      fs.mkdirSync(path.dirname(this.stateFile),{recursive:true});
      const tmp=this.stateFile+'.tmp';fs.writeFileSync(tmp,JSON.stringify({version:1,last:this.last}));fs.renameSync(tmp,this.stateFile);
    }
    return {type:'observation',rule:kind,symbol:row.symbol,time:new Date(wallNow).toISOString(),measurements:{...row}};
  }
}
export function formatObservation(event){
  const r=event.measurements,pct=v=>(v*100).toFixed(1)+'%';
  const reason=event.rule==='ask_depletion'
    ?`매수 우위 + 매도 물량 감소 지속\n매도 잔량 변화 ${pct(r.ask_depth_change)} | 목표까지 매도 금액 변화 ${pct(r.ask_cost_change)}`
    :`매도 우위 속 매수 대기 물량 회복\n매수 잔량 회복 ${pct(r.bid_recovery)}`;
  return `[Orderbook v1 · 실험용 관찰] ${event.symbol}\n${event.time}\n${reason}\n최근 약60초 매수 ${r.buy_quote_60s.toFixed(0)} / 매도 ${r.sell_quote_60s.toFixed(0)} USDT\n가격 변화 ${pct(r.mid_return_60s)} | 스프레드 ${r.spread_bps.toFixed(2)}bp\n관찰 조건 ${policy.hold_ms/1000}초 유지 · 동일 종목 재알림 최소1시간\n급등 예측 모델 미적용 · 적중률/선행시간 미검증`;
}
