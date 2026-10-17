// Historical diagnostic of the actual core; missing ticker/spread history is explicit.
import fs from 'node:fs';
import zlib from 'node:zlib';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {assess,takeSignal} from '../../momentum-alert/core.mjs';
const base=path.dirname(fileURLToPath(import.meta.url)),out=path.join(base,'output');
const raw=JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join(out,'data.json.gz'))));
const policy=JSON.parse(fs.readFileSync(path.join(base,'../../momentum-alert/policy.json')));
const report=JSON.parse(fs.readFileSync(path.join(out,'report.json')));
const parse=s=>Date.parse(s.replace(' ','T')+'Z');
const windows=report.folds.map(f=>f.test.map(parse));
function bars(symbol){return raw.candles[symbol].map(r=>{const [time,open,high,low,close,volume,turnover]=r.map(Number);return {time,open,high,low,close,volume,turnover};}).sort((a,b)=>a.time-b.time);}
function hours(q){const a=[];for(let i=0;i<q.length;i+=4){const b=q.slice(i,i+4);if(b.length!==4||b[0].time%3600000)throw Error('Misaligned hour');a.push({time:b[0].time,open:b[0].open,high:Math.max(...b.map(x=>x.high)),low:Math.min(...b.map(x=>x.low)),close:b[3].close,volume:b.reduce((s,x)=>s+x.volume,0),turnover:b.reduce((s,x)=>s+x.turnover,0)});}return a;}
const near=bars('NEARUSDT'),btc=bars('BTCUSDT'),hour=hours(near),btcHour=hours(btc),state={},rows=[];
for(let i=800;i<near.length;i++){
 const now=near[i].time+900000;
 if(!windows.some(([a,b])=>now>=a&&now+73*3600000<b))continue;
 const h=Math.floor((i+1)/4),q=near.slice(i-95,i+1);
 const ticker={bid1Price:near[i].close,ask1Price:near[i].close,lastPrice:near[i].close,turnover24h:q.reduce((s,x)=>s+x.turnover,0)};
 for(const direction of policy.directions){
  const row=assess('NEARUSDT',hour.slice(h-200,h),q,btcHour.slice(h-200,h),ticker,policy,direction);
  if(takeSignal(state,row,now,policy))rows.push({time:now,direction,turnover_ratio:row.turnover_ratio,hour_return:row.hour_return});
 }
}
const result={scope:'Diagnostic only: actual assess/takeSignal core with15min observations, close as ticker last, synthetic zero spread (gate always passes), reconstructed24h turnover. Not a full historical reproduction of5min production checks or real fills. No model selection using these results.',core_sha256:crypto.createHash('sha256').update(fs.readFileSync(path.join(base,'../../momentum-alert/core.mjs'))).digest('hex'),rows};
fs.writeFileSync(path.join(out,'current_rules.json'),JSON.stringify(result,null,2));console.log('Current-rule diagnostic',Object.fromEntries(policy.directions.map(d=>[d,rows.filter(r=>r.direction===d).length])));
