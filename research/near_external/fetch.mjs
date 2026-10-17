import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {gzipSync,gunzipSync} from 'node:zlib';
import {createHash} from 'node:crypto';
import {get,instrument} from '../../momentum-alert/api.mjs';
const root=path.dirname(fileURLToPath(import.meta.url));
const out=path.join(root,'output','study_v1');fs.mkdirSync(out,{recursive:true});
const sha=x=>createHash('sha256').update(x).digest('hex');
const protocolBytes=fs.readFileSync(path.join(root,'protocol.json')),p=JSON.parse(protocolBytes);
const basePath=path.join(root,p.base_snapshot),packed=fs.readFileSync(basePath);
if(sha(packed)!==p.base_sha256)throw Error('Base snapshot changed');
const base=JSON.parse(gunzipSync(packed));
const manifestPath=path.join(out,'manifest.json');
if(fs.existsSync(manifestPath))throw Error('Snapshot already complete; preserve it');
const start=base.start,end=base.end,entries=[];
async function external(url){
 const r=await fetch(url,{signal:AbortSignal.timeout(30000)});
 if(!r.ok)throw Error('HTTP '+r.status+' '+url);
 return await r.json();
}
async function save(name,fn){
 const file=path.join(out,name+'.json.gz');
 if(!fs.existsSync(file)){
  const started=new Date().toISOString();const data=await fn();const bytes=gzipSync(JSON.stringify({started,received_at:new Date().toISOString(),data}));
  fs.writeFileSync(file,bytes,{flag:'wx'});
 }
 const bytes=fs.readFileSync(file),d=JSON.parse(gunzipSync(bytes));
 const rows=d.data.rows||d.data;
 entries.push({name,file:path.basename(file),sha256:sha(bytes),received_at:d.received_at,count:Array.isArray(rows)?rows.length:null});
 console.log(name,entries.at(-1).count);
}
async function candles(symbol){
 const metadata=await instrument(symbol);let cursor=end-1;const rows=[],pages=[];
 while(cursor>=start){
  const params={category:'linear',symbol,interval:15,start,end:cursor,limit:1000};
  const d=await get('kline',params);pages.push({params,response:d,received_at:new Date().toISOString()});
  if(d.result.category!=='linear'||d.result.symbol!==symbol)throw Error('Wrong market');
  const batch=d.result.list;if(!batch.length)break;
  rows.push(...batch.filter(r=>Number(r[0])>=start&&Number(r[0])+900000<=end));
  const next=Math.min(...batch.map(r=>Number(r[0])))-1;if(next>=cursor)throw Error('Pagination stalled');cursor=next;
 }
 return {metadata,rows,pages};
}
const amendmentBytes=fs.readFileSync(path.join(root,'acquisition_amendment.json'));
const amendment=JSON.parse(amendmentBytes);
if(amendment.original_protocol_sha256!==sha(protocolBytes))throw Error('Wrong amendment');
const symbols=p.sector_baskets.l1.filter(s=>!base.candles[s]);
for(let i=0;i<symbols.length;i+=3){
 const tasks=await Promise.allSettled(symbols.slice(i,i+3).map(s=>save('sector_'+s,()=>candles(s))));
 for(const r of tasks)if(r.status==='rejected')throw r.reason;
}
async function spotCandles(symbol){
 const metadata=await external('https://api.binance.com/api/v3/exchangeInfo?symbol='+symbol);
 const m=metadata.symbols?.[0];if(m?.symbol!==symbol||m?.quoteAsset!=='USDT')throw Error('Wrong Binance market');
 let cursor=start;const rows=[],pages=[];
 while(cursor<end){
  const url='https://api.binance.com/api/v3/klines?symbol='+symbol+'&interval=15m&limit=1000&startTime='+cursor+'&endTime='+(end-1);
  const batch=await external(url);if(!Array.isArray(batch))throw Error('Invalid spot candles');
  pages.push({url,response:batch,received_at:new Date().toISOString()});if(!batch.length)break;
  rows.push(...batch.filter(r=>Number(r[0])>=start&&Number(r[6])<end));
  const next=Math.max(...batch.map(r=>Number(r[0])))+900000;if(next<=cursor)throw Error('Spot pagination stalled');cursor=next;
 }
 return {metadata,rows,pages};
}
for(const symbol of ['NEARUSDT',...p.sector_baskets.ai])await save('spot_'+symbol,()=>spotCandles(symbol));
await save('account_NEARUSDT',async()=>{
 let cursor='',seen=new Set();const rows=[],pages=[];
 do{
  const params={category:'linear',symbol:'NEARUSDT',period:'1h',startTime:start,endTime:end-1,limit:500};if(cursor)params.cursor=cursor;
  const d=await get('account-ratio',params);pages.push({params,response:d,received_at:new Date().toISOString()});
  rows.push(...d.result.list);cursor=d.result.nextPageCursor||'';
  if(cursor&&seen.has(cursor))throw Error('Ratio pagination stalled');seen.add(cursor);
 }while(cursor);
 return {rows,pages};
});
const probe=JSON.parse(fs.readFileSync(path.join(root,'output','latest.json'))).directory;
for(const name of ['near_stablecoins','near_tvl','near_dex_volume','near_categories']){
 const source=path.join(probe,name+'.json.gz');const bytes=fs.readFileSync(source);
 const original=JSON.parse(fs.readFileSync(path.join(probe,'manifest.json'))).results.find(r=>r.name===name);
 if(original.status!==200||sha(bytes)!==original.sha256)throw Error('Invalid source '+name);
 const file=name+'.json.gz';if(!fs.existsSync(path.join(out,file)))fs.copyFileSync(source,path.join(out,file));
 entries.push({name,file,sha256:sha(fs.readFileSync(path.join(out,file))),received_at:original.received_at,count:original.rows,source_manifest:path.join(probe,'manifest.json')});
}
const manifest={protocol_sha256:sha(protocolBytes),amendment_sha256:sha(amendmentBytes),base_sha256:p.base_sha256,start,end,entries,status:'Acquired, feature/data checks still required'};
fs.writeFileSync(manifestPath,JSON.stringify(manifest,null,2),{flag:'wx'});console.log('Saved',manifestPath);
