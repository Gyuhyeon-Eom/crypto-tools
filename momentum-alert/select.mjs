import fs from 'node:fs';
import {clock,get,candles,instrument} from './api.mjs';
import {similarity,rankPeers} from './core.mjs';
const base=new URL('./',import.meta.url),policy=JSON.parse(fs.readFileSync(new URL('policy.json',base)));
if(process.argv.slice(2).some(x=>x!=='--write-watchlist'))throw new Error('Usage: node momentum-alert/select.mjs [--write-watchlist]');
const asof=await clock(),tickers=(await get('tickers',{category:'linear'})).result.list;
const near=await candles(policy.primary,60,policy.selection.hours+1,asof);
const btc=await candles('BTCUSDT',60,policy.selection.hours+1,asof);
const rows=[],errors=[];
await instrument(policy.primary);
for(const symbol of policy.candidates){
  try{
    await instrument(symbol);
    const data=await candles(symbol,60,policy.selection.hours+1,asof);
    rows.push({symbol,...similarity(near,data,btc),turnover24h:Number(tickers.find(x=>x.symbol===symbol)?.turnover24h)});
  }catch(e){errors.push({symbol,error:e.message});}
}
const selected=rankPeers(rows,policy),watchlist={version:1,category:'linear',asof:new Date(asof).toISOString(),
  symbols:[policy.primary,...selected.map(x=>x.symbol)],references:policy.references,
  basis:'720 closed hourly returns: correlation, BTC-residual correlation, volatility ratio, liquidity. Similarity is not a forecast of rising prices.',selected};
fs.mkdirSync(new URL('output/',base),{recursive:true});
fs.writeFileSync(new URL('output/selection.json',base),JSON.stringify({watchlist,rows,errors,policy},null,2));
if(process.argv.includes('--write-watchlist'))fs.writeFileSync(new URL('watchlist.json',base),JSON.stringify(watchlist,null,2)+'\n');
console.log(JSON.stringify({watchlist,errors},null,2));
