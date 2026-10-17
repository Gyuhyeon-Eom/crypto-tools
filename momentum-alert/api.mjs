import {closedCandles,requireWindow} from './core.mjs';
export async function get(endpoint,params={}){
  const url=new URL('https://api.bybit.com/v5/market/'+endpoint);
  for(const [k,v] of Object.entries(params))url.searchParams.set(k,String(v));
  const r=await fetch(url,{signal:AbortSignal.timeout(15000)});
  if(!r.ok)throw new Error('Bybit '+endpoint+' HTTP '+r.status);
  const d=await r.json();if(d.retCode!==0)throw new Error('Bybit '+endpoint+' code '+d.retCode);
  return d;
}
export async function clock(){const d=await get('time');return Number(d.result.timeSecond)*1000;}
export async function candles(symbol,interval,count,asof){
  const d=await get('kline',{category:'linear',symbol,interval,limit:count+2,end:asof-1});
  if(d.result.category!=='linear'||d.result.symbol!==symbol)throw new Error('Wrong candle market');
  return requireWindow(closedCandles(d.result.list,interval,asof),interval,count,asof);
}
export async function instrument(symbol){
  const d=await get('instruments-info',{category:'linear',symbol});
  const x=d.result.list.find(x=>x.symbol===symbol);
  if(!x||x.contractType!=='LinearPerpetual'||x.status!=='Trading'||x.settleCoin!=='USDT')throw new Error('Inactive USDT perpetual: '+symbol);
  return {symbol,contractType:x.contractType,status:x.status,settleCoin:x.settleCoin};
}
