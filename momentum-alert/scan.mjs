import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {fileURLToPath} from 'node:url';
import {gzipSync} from 'node:zlib';
import {createHash} from 'node:crypto';
import {clock,get,candles,instrument} from './api.mjs';
import {assess,takeSignal,formatSignal} from './core.mjs';
import {Notifier,loadWebhooks} from '../research/orderbook_live/notify.mjs';

const base=path.dirname(fileURLToPath(import.meta.url));
const flags=new Set(process.argv.slice(2));
if([...flags].some(x=>!['--slack','--check'].includes(x)))throw new Error('Usage: node momentum-alert/scan.mjs [--slack] [--check]');
const enabled=flags.has('--slack'),check=flags.has('--check');
const policy=JSON.parse(fs.readFileSync(path.join(base,'policy.json')));
if(JSON.stringify(policy.directions)!==JSON.stringify(['long','short']))throw new Error('Expected LONG/SHORT policy');
const watchlist=JSON.parse(fs.readFileSync(path.join(base,'watchlist.json')));
if(watchlist.category!=='linear'||!Array.isArray(watchlist.symbols)||watchlist.symbols.length>4
  ||!watchlist.symbols.includes('NEARUSDT')||new Set(watchlist.symbols).size!==watchlist.symbols.length
  ||watchlist.symbols.some(s=>![policy.primary,...policy.candidates].includes(s)))throw new Error('Invalid futures watchlist');
const env={...process.env};
if(env.MOMENTUM_SLACK_WEBHOOK_URL)env.ORDERBOOK_SLACK_WEBHOOK_URL=env.MOMENTUM_SLACK_WEBHOOK_URL;
let urls=enabled?loadWebhooks(env):[];
if(enabled&&!urls.length)urls=loadWebhooks(env,path.join(os.homedir(),'.openclaw','.env.slack-crypto'));
if(enabled&&!urls.length)throw new Error('Slack webhook missing. Configure MOMENTUM_SLACK_WEBHOOK_URL or existing .env.slack-surge/.env.slack-crypto');
const output=path.join(base,'output');fs.mkdirSync(output,{recursive:true});
const mode=enabled?'live':'preview',lock=path.join(output,mode+'.lock');
try{fs.writeFileSync(lock,String(process.pid),{flag:'wx'});}
catch(e){
  if(e.code!=='EEXIST')throw e;
  const pid=Number(fs.readFileSync(lock,'utf8'));
  if(!Number.isSafeInteger(pid)||pid<=0)throw new Error('Invalid scan lock');
  try{process.kill(pid,0);throw new Error('Scan already running');}
  catch(err){if(err.code!=='ESRCH')throw err;fs.unlinkSync(lock);fs.writeFileSync(lock,String(process.pid),{flag:'wx'});}
}
process.on('exit',()=>{try{if(fs.readFileSync(lock,'utf8')===String(process.pid))fs.unlinkSync(lock);}catch{}});
const asof=await clock(),universe=[...new Set([...watchlist.symbols,...policy.references])];
const instruments=[];for(const symbol of universe)instruments.push(await instrument(symbol));
if(check){console.log(JSON.stringify({checked:true,category:'linear',directions:policy.directions,symbols:watchlist.symbols,references:policy.references,slack_configured:enabled&&urls.length>0},null,2));process.exit(0);}
const stateFile=path.join(output,mode+'_state.json');
const state=fs.existsSync(stateFile)?JSON.parse(fs.readFileSync(stateFile)):{};
if(!state||Array.isArray(state)||typeof state!=='object'||Object.values(state).some(v=>!v||!Number.isFinite(v.candle_time)||!Number.isFinite(v.notified_at)
  ||(v.direction!==undefined&&!['long','short'].includes(v.direction))))throw new Error('Invalid cooldown state');
const hourly={},quarters={},errors=[];
for(const symbol of universe){
  try{hourly[symbol]=await candles(symbol,60,200,asof);if(watchlist.symbols.includes(symbol))quarters[symbol]=await candles(symbol,15,96,asof);}
  catch(e){errors.push({symbol,error:e.message});}
}
if(!hourly.BTCUSDT||!hourly.ETHUSDT)throw new Error('Market reference candles unavailable; no alerts sent');
const tickerData=await get('tickers',{category:'linear'}),observedAt=Number(tickerData.time);
if(!Number.isFinite(observedAt)||observedAt<asof||observedAt-asof>120000)throw new Error('Scan took too long; no stale alerts sent');
const day=new Date(observedAt).toISOString().slice(0,10),journalPath=path.join(output,day+'_'+mode+'_notifications.jsonl');
const journal=e=>fs.appendFileSync(journalPath,JSON.stringify({time:Date.now(),...e})+'\n');
const notifier=new Notifier({urls,enabled,journal}),rows=[];
for(const symbol of watchlist.symbols){
  if(!hourly[symbol]||!quarters[symbol])continue;
  const ticker=tickerData.result.list.find(x=>x.symbol===symbol);
  if(!ticker){errors.push({symbol,error:'Ticker missing'});continue;}
  for(const direction of policy.directions){
    const row={...assess(symbol,hourly[symbol],quarters[symbol],hourly.BTCUSDT,ticker,policy,direction),observed_at:observedAt,
      category:'linear',rule_version:policy.version,eth_hour_return:hourly.ETHUSDT.at(-1).close/hourly.ETHUSDT.at(-2).close-1};
    row.notify=takeSignal(state,row,observedAt,policy);rows.push(row);
  }
}
// Save both qualifying and rejected observations, with source data before external delivery.
const scan={asof,observed_at:observedAt,policy,watchlist,instruments,rows,errors,hourly,quarters,tickers:tickerData.result.list.filter(x=>universe.includes(x.symbol)),
  code_hashes:Object.fromEntries(['scan.mjs','api.mjs','core.mjs','policy.json','watchlist.json'].map(n=>[n,createHash('sha256').update(fs.readFileSync(path.join(base,n))).digest('hex')]))};
fs.writeFileSync(path.join(output,'scan_'+observedAt+'_'+mode+'.json.gz'),gzipSync(JSON.stringify(scan)),{flag:'wx'});
fs.appendFileSync(path.join(output,day+'_'+mode+'_candidates.jsonl'),rows.map(r=>JSON.stringify(r)+'\n').join(''));
fs.writeFileSync(stateFile+'.tmp',JSON.stringify(state));fs.renameSync(stateFile+'.tmp',stateFile);
for(const row of rows)if(row.notify)notifier.send('momentum_observation',formatSignal(row));
const flushed=await notifier.flush(45000);notifier.close();
const result={asof:new Date(asof).toISOString(),mode,watchlist:watchlist.symbols,scanned:rows.length,
  scanned_symbols:new Set(rows.map(r=>r.symbol)).size,directions:policy.directions,
  observations:rows.filter(r=>r.notify).map(({symbol,direction})=>({symbol,direction})),errors,notifications:notifier.stats,flushed};
fs.writeFileSync(path.join(output,'last_'+mode+'_scan.json'),JSON.stringify(result,null,2));
console.log(JSON.stringify(result,null,2));
process.exitCode=errors.length||!flushed||notifier.stats.failed||notifier.stats.dropped?1:0;
