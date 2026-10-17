import fs from 'node:fs';
export const profile=JSON.parse(fs.readFileSync(new URL('./profile.json',import.meta.url)));
export const symbols=[profile.target_symbol,...profile.reference_symbols];
export function parseArgs(args){
  const [duration='180',...rest]=args;
  const seconds=Number(duration),flags=new Set(rest);
  if(!Number.isInteger(seconds)||seconds<1||seconds>86400
    ||rest.some(x=>!['--slack','--signals','--dry-run'].includes(x))
    ||(flags.has('--slack')&&flags.has('--dry-run')))
    throw new Error('Usage: node research/near_futures/capture.mjs SECONDS[1..86400] [--slack | --dry-run] [--signals]');
  return {seconds,slackEnabled:flags.has('--slack'),signalsEnabled:flags.has('--signals')};
}
export function isTarget(symbol){return symbol===profile.target_symbol;}
export function marketIdentity(){
  return {exchange:profile.exchange,category:profile.category,contract_type:profile.contract_type,
    settle_coin:profile.settle_coin,study:profile.version};
}
export function observationText(event,format){
  return format(event).replace('[Orderbook v1 · 실험용 관찰]','[NEAR 선물 v1 · 실험용 관찰]')
    +'\nBybit USDT 무기한 선물 · 가격 +5%/+10% 도달 예측 미검증';
}
export async function checkInstruments(fetcher=fetch){
  const results=[];
  for(const symbol of symbols){
    const response=await fetcher('https://api.bybit.com/v5/market/instruments-info?category=linear&symbol='+symbol,
      {signal:AbortSignal.timeout(15000)});
    if(!response.ok)throw new Error('Instrument lookup HTTP '+response.status+' for '+symbol);
    const data=await response.json(),instrument=data.result?.list?.find(x=>x.symbol===symbol);
    if(data.retCode!==0||!instrument||instrument.contractType!==profile.contract_type
      ||instrument.status!=='Trading'||instrument.settleCoin!==profile.settle_coin)
      throw new Error('Expected active USDT linear perpetual: '+symbol);
    results.push({symbol,contractType:instrument.contractType,status:instrument.status,settleCoin:instrument.settleCoin});
  }
  return results;
}
