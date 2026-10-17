// Public-data feasibility only. Does not fit/promote models or send alerts.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {gzipSync} from 'node:zlib';
import {createHash} from 'node:crypto';
const root=path.dirname(fileURLToPath(import.meta.url));
const out=path.join(root,'output',new Date().toISOString().replaceAll(':','-'));
fs.mkdirSync(out,{recursive:true});
const start=Date.parse('2024-10-17T00:00:00Z'),end=start+86400000;
const jobs=[
 ['chain_list','https://api.llama.fi/v2/chains'],
 ['near_tvl','https://api.llama.fi/v2/historicalChainTvl/Near'],
 ['near_stablecoins','https://stablecoins.llama.fi/stablecoincharts/Near'],
 ['near_bridge_flows','https://bridges.llama.fi/bridgevolume/Near'],
 ['near_dex_volume','https://api.llama.fi/overview/dexs/Near?excludeTotalDataChart=false&excludeTotalDataChartBreakdown=true'],
 ['binance_spot_sample','https://api.binance.com/api/v3/klines?symbol=NEARUSDT&interval=15m&startTime='+start+'&endTime='+end+'&limit=96'],
 ['bybit_account_ratio_sample','https://api.bybit.com/v5/market/account-ratio?category=linear&symbol=NEARUSDT&period=1h&startTime='+start+'&endTime='+end+'&limit=24'],
 ['near_categories','https://api.coingecko.com/api/v3/coins/near?localization=false&tickers=false&market_data=false&community_data=false&developer_data=false']
];
const results=[];
for(let i=0;i<jobs.length;i+=3){
 const batch=await Promise.allSettled(jobs.slice(i,i+3).map(async([name,url])=>{
  const begun=new Date().toISOString();const res=await fetch(url,{signal:AbortSignal.timeout(30000)});const body=await res.text();const received=new Date().toISOString();
  const raw=gzipSync(body);fs.writeFileSync(path.join(out,name+'.json.gz'),raw);
  let d;try{d=JSON.parse(body)}catch{}
  let rows=Array.isArray(d)?d:Array.isArray(d?.totalDataChart)?d.totalDataChart:Array.isArray(d?.result?.list)?d.result.list:[];
  const summary={name,url,status:res.status,request_started:begun,received_at:received,sha256:createHash('sha256').update(raw).digest('hex'),rows:rows.length,keys:d&&!Array.isArray(d)?Object.keys(d):[],first:rows[0],last:rows.at(-1)};
  if(name==='chain_list'){summary.near=rows.filter(r=>String(r.name).toLowerCase().includes('near'));delete summary.first;delete summary.last;}
  if(name==='near_categories')summary.categories=d?.categories;
  if(name==='bybit_account_ratio_sample')summary.retCode=d?.retCode;
  return summary;
 }));
 for(let k=0;k<batch.length;k++){
  const r=batch[k];results.push(r.status==='fulfilled'?r.value:{name:jobs[i+k][0],error:String(r.reason)});
  console.log(JSON.stringify(results.at(-1)));
 }
}
fs.writeFileSync(path.join(out,'manifest.json'),JSON.stringify({scope:'Availability/sample-schema check only, not full coverage or predictive validation',results},null,2));
fs.writeFileSync(path.join(root,'output','latest.json'),JSON.stringify({directory:out},null,2));
console.log(out);
