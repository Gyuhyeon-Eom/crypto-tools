export const HOUR=3600000;
export const mean=a=>a.reduce((s,x)=>s+x,0)/a.length;
export const median=a=>{const b=[...a].sort((x,y)=>x-y),i=Math.floor(b.length/2);return b.length%2?b[i]:(b[i-1]+b[i])/2;};
export const variance=a=>mean(a.map(x=>(x-mean(a))**2));
export const covariance=(a,b)=>mean(a.map((x,i)=>(x-mean(a))*(b[i]-mean(b))));
export const correlation=(a,b)=>covariance(a,b)/Math.sqrt(variance(a)*variance(b));
export const returns=c=>c.slice(1).map((x,i)=>Math.log(x.close/c[i].close));
export function residual(y,x){const beta=covariance(y,x)/variance(x);return y.map((v,i)=>v-beta*x[i]);}
export function ema(values,n){return values.reduce((v,x)=>v+(2/(n+1))*(x-v),values[0]);}
export function closedCandles(list,interval,asof){
  const duration=interval*60000;
  const rows=list.map(row=>{
    const [time,open,high,low,close,volume,turnover]=row.map(Number);
    if(![time,open,high,low,close,volume,turnover].every(Number.isFinite)||!Number.isSafeInteger(time)
      ||time%duration!==0||low<=0||high<Math.max(open,close)||low>Math.min(open,close)||volume<0||turnover<0)
      throw new Error('Invalid candle');
    return {time,open,high,low,close,volume,turnover};
  }).filter(x=>x.time+duration<=asof).sort((a,b)=>a.time-b.time);
  for(let i=1;i<rows.length;i++)if(rows[i].time-rows[i-1].time!==duration)throw new Error('Candle gap or duplicate');
  return rows;
}
export function requireWindow(rows,interval,count,asof){
  const duration=interval*60000,expected=Math.floor(asof/duration)*duration-duration;
  if(rows.length<count||rows.at(-1).time!==expected)throw new Error('Insufficient or stale closed candles');
  return rows.slice(-count);
}
export function similarity(near,coin,btc){
  if(near.length!==coin.length||coin.some((r,i)=>r.time!==near[i].time||r.time!==btc[i]?.time))throw new Error('Unaligned similarity window');
  const n=returns(near),c=returns(coin),b=returns(btc);
  return {correlation:correlation(n,c),residual_correlation:correlation(residual(n,b),residual(c,b)),
    vol_ratio:Math.sqrt(variance(c)/variance(n)),return_7d:coin.at(-1).close/coin.at(-169).close-1};
}
export function rankPeers(rows,policy){
  const p=policy.selection;
  return rows.filter(r=>[r.correlation,r.residual_correlation,r.vol_ratio,r.turnover24h].every(Number.isFinite)
    &&r.correlation>=p.min_correlation&&r.residual_correlation>=p.min_residual_correlation
    &&r.vol_ratio>=p.min_vol_ratio&&r.vol_ratio<=p.max_vol_ratio&&r.turnover24h>=p.min_daily_turnover)
    .map(r=>({...r,similarity_score:(r.correlation+r.residual_correlation)/2-0.1*Math.abs(Math.log(r.vol_ratio))}))
    .sort((a,b)=>b.similarity_score-a.similarity_score||a.symbol.localeCompare(b.symbol)).slice(0,p.additional_symbols);
}
export function assess(symbol,hour,quarter,btcHour,ticker,policy,direction='long'){
  if(!['long','short'].includes(direction))throw new Error('Invalid signal direction');
  const long=direction==='long';
  const h=hour.map(x=>x.close),q=quarter.at(-1),previous=quarter.slice(-21,-1),priorHigh=Math.max(...previous.map(x=>x.high)),priorLow=Math.min(...previous.map(x=>x.low));
  const bid=Number(ticker.bid1Price),ask=Number(ticker.ask1Price),last=Number(ticker.lastPrice);
  const f={symbol,direction,candle_time:q.time,close:q.close,reference_price:last,ema20:ema(h,20),ema50:ema(h,50),
    hour_return:h.at(-1)/h.at(-2)-1,btc_hour_return:btcHour.at(-1).close/btcHour.at(-2).close-1,
    turnover_ratio:q.turnover/median(previous.map(x=>x.turnover)),distance_from_high:q.close/priorHigh-1,distance_from_low:q.close/priorLow-1,
    spread_bps:(ask-bid)/((ask+bid)/2)*10000,turnover24h:Number(ticker.turnover24h)};
  const p=policy.signal;
  const checks={valid:[...Object.values(f)].filter(x=>typeof x==='number').every(Number.isFinite)&&bid>0&&ask>=bid&&last>0,
    liquid:f.turnover24h>=policy.selection.min_daily_turnover&&f.spread_bps<=p.max_spread_bps,
    trend:long?f.close>f.ema20&&f.ema20>f.ema50:f.close<f.ema20&&f.ema20<f.ema50,
    volume:f.turnover_ratio>=p.turnover_ratio,
    [long?'near_high':'near_low']:long
      ?f.distance_from_high>=-p.max_distance_below_high&&f.distance_from_high<=p.max_above_high
      :f.distance_from_low>=-p.max_below_low&&f.distance_from_low<=p.max_distance_above_low,
    early_move:long?f.hour_return>0&&f.hour_return<=p.max_hour_return:f.hour_return<0&&f.hour_return>=-p.max_hour_decline,
    relative_strength:long?f.hour_return-f.btc_hour_return>=p.min_btc_excess_hour_return:f.btc_hour_return-f.hour_return>=p.min_btc_underperformance};
  return {...f,checks,eligible:Object.values(checks).every(Boolean)};
}
export function takeSignal(state,row,now,policy){
  const old=state[row.symbol];
  if(!['long','short'].includes(row.direction)||!row.eligible||old&&(row.candle_time<=old.candle_time||now-old.notified_at<policy.signal.cooldown_hours*HOUR))return false;
  // Retain the symbol key so v1 state also suppresses repeats after upgrading.
  state[row.symbol]={direction:row.direction,candle_time:row.candle_time,notified_at:now};return true;
}
export function formatSignal(row){
  if(!['long','short'].includes(row.direction))throw new Error('Invalid signal direction');
  const long=row.direction==='long',sign=long?'+':'-',multiplier=long?1:-1;
  const pct=x=>(x*100).toFixed(2)+'%';
  return `[${row.symbol} · ${long?'LONG/롱':'SHORT/숏'} · 추세 관찰] Bybit USDT 무기한\n`
    +`기준 시각 ${new Date(row.observed_at).toISOString()} · 현재 참고 가격 ${row.reference_price}\n`
    +`확정 1시간봉 ${pct(row.hour_return)} / BTC 대비 ${pct(row.hour_return-row.btc_hour_return)}\n`
    +`15분 거래대금 평소 대비 ${row.turnover_ratio.toFixed(2)}배 · 최근 5시간 ${long?'고점':'저점'} 대비 ${pct(long?row.distance_from_high:row.distance_from_low)}\n`
    +`참고 가격 대비 ${sign}5% ${(row.reference_price*(1+multiplier*.05)).toPrecision(6)} / ${sign}10% ${(row.reference_price*(1+multiplier*.10)).toPrecision(6)}\n`
    +`${long?'상승':'하락'} 조건 관찰 · 전조 모델 검증 대기 · 목표 도달 확률/적중률 미검증 · 진입 추천 아님`;
}
