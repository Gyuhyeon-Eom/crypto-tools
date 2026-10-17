import fs from 'node:fs';
import {gzipSync} from 'node:zlib';
import {createHash} from 'node:crypto';
import {get,clock,instrument} from '../../momentum-alert/api.mjs';
const base=new URL('./',import.meta.url),protocol=JSON.parse(fs.readFileSync(new URL('protocol.json',base)));
const end=Math.floor((await clock())/86400000)*86400000,start=end-protocol.history_days*86400000;
const out=new URL('output/',base);fs.mkdirSync(out,{recursive:true});
if(fs.existsSync(new URL('data.json.gz',out)))throw new Error('Data already exists; preserve this frozen run');
const data={start,end,category:'linear',candles:{},mark:[],oi:[],funding:[],errors:[],raw:[]};
async function request(endpoint,params){
  for(let attempt=0;attempt<3;attempt++){
    try{const d=await get(endpoint,params);data.raw.push({endpoint,params,response:d});return d;}
    catch(e){if(attempt===2)throw e;await new Promise(r=>setTimeout(r,1000*(attempt+1)));}
  }
}
async function fetchCandles(symbol,mark=false){
  await instrument(symbol);const rows=[];let cursor=end-1;
  while(cursor>=start){
    const d=await request(mark?'mark-price-kline':'kline',{category:'linear',symbol,interval:15,limit:1000,start,end:cursor});
    if(d.result.category!=='linear'||d.result.symbol!==symbol)throw new Error('Wrong candle market');
    const batch=d.result.list;if(!batch.length)break;
    rows.push(...batch.filter(r=>Number(r[0])>=start&&Number(r[0])+900000<=Math.min(end,Number(d.time))));
    const next=Math.min(...batch.map(r=>Number(r[0])))-1;if(next>=cursor)throw new Error('Pagination stalled');cursor=next;
  }
  if(mark)data.mark=rows;else data.candles[symbol]=rows;console.log(symbol+(mark?' mark ':' candles ')+rows.length);
}
for(let i=0;i<protocol.symbols.length;i+=3)await Promise.all(protocol.symbols.slice(i,i+3).map(s=>fetchCandles(s)));
await fetchCandles('NEARUSDT',true);

try{
  let cursor='',seen=new Set();
  do{
    const params={category:'linear',symbol:'NEARUSDT',intervalTime:'15min',startTime:start,endTime:end-1,limit:200};
    if(cursor)params.cursor=cursor;
    const d=await request('open-interest',params);
    if(d.result.symbol!=='NEARUSDT'||d.result.category!=='linear')throw new Error('Wrong OI market');
    data.oi.push(...d.result.list);cursor=d.result.nextPageCursor||'';
    if(cursor&&seen.has(cursor))throw new Error('OI pagination stalled');seen.add(cursor);
  }while(cursor);
}catch(e){data.errors.push({source:'oi',error:e.message});}
try{
  let cursor=end-1;
  while(cursor>=start){
    const d=await request('funding/history',{category:'linear',symbol:'NEARUSDT',startTime:start,endTime:cursor,limit:200});
    const rows=d.result.list;if(!rows.length)break;data.funding.push(...rows);
    const next=Math.min(...rows.map(x=>Number(x.fundingRateTimestamp)))-1;
    if(next>=cursor)throw new Error('Funding pagination stalled');cursor=next;
  }
}catch(e){data.errors.push({source:'funding',error:e.message});}
const packed=gzipSync(JSON.stringify(data));fs.writeFileSync(new URL('data.json.gz',out),packed,{flag:'wx'});
const manifest={start:new Date(start).toISOString(),end:new Date(end).toISOString(),category:'linear',
  counts:Object.fromEntries(Object.entries(data.candles).map(([s,r])=>[s,r.length])),oi:data.oi.length,funding:data.funding.length,mark:data.mark.length,
  errors:data.errors,data_sha256:createHash('sha256').update(packed).digest('hex'),protocol_sha256:createHash('sha256').update(fs.readFileSync(new URL('protocol.json',base))).digest('hex')};
fs.writeFileSync(new URL('manifest.json',out),JSON.stringify(manifest,null,2));console.log(JSON.stringify(manifest,null,2));
