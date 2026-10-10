// Offline deterministic replay of the stored public feed; no network calls.
import fs from 'node:fs';
import path from 'node:path';
import {gunzipSync} from 'node:zlib';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {TapeEngine} from './engine.mjs';

const dir=process.argv[2];
if(!dir)throw new Error('Usage: node audit.mjs RUN_DIRECTORY');
const base=path.dirname(fileURLToPath(import.meta.url));
const summary=JSON.parse(fs.readFileSync(path.join(dir,'summary.json')));
const readGzip=name=>gunzipSync(fs.readFileSync(path.join(dir,name))).toString().trim().split('\n').filter(Boolean).map(JSON.parse);
const rawFiles=summary.files.filter(f=>/^raw_\d+\.jsonl\.gz$/.test(f)).sort();
const records=rawFiles.flatMap(readGzip),features=readGzip('features.jsonl.gz');
const errors=[];
for(let i=1;i<records.length;i++)if(records[i].elapsed_ms<records[i-1].elapsed_ms)errors.push('Raw receive order decreased at '+i);
const ticks=new Map();
for(const row of features){if(!ticks.has(row.t))ticks.set(row.t,[]);ticks.get(row.t).push(row);}
const timeline=[...records.map((row,index)=>({t:row.elapsed_ms,index,kind:'raw',row})),
  ...[...ticks].map(([t,rows])=>({t,kind:'sample',rows}))].sort((a,b)=>a.t-b.t);
const engine=new TapeEngine(summary.symbols);
const topics={},statuses={},levels={},flow={};
let replayRows=0,mismatches=0,reconstructionErrors=0,connection=0;
function equivalent(a,b){
  if(typeof a==='number'&&typeof b==='number')return Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=1e-10*Math.max(1,Math.abs(a),Math.abs(b));
  return a===b;
}
for(const item of timeline){
  if(item.kind==='raw'){
    const {kind,data}=item.row;
    if(kind==='connecting'||kind==='gap')engine.reset();
    connection=item.row.connection;
    if(kind==='message'){
      if(data.topic)topics[data.topic]=(topics[data.topic]??0)+1;
      try{engine.onMessage(data,item.t);}catch{reconstructionErrors++;}
    }
    continue;
  }
  const actual=engine.sample(item.t);
  if(item.rows.length!==summary.symbols.length||new Set(item.rows.map(r=>r.symbol)).size!==summary.symbols.length)errors.push('Incomplete symbol sample at '+item.t);
  for(const expected of item.rows){
    replayRows++;
    const observed=actual.find(r=>r.symbol===expected.symbol);
    const keys=new Set([...Object.keys(expected),...Object.keys(observed??{})]);
    const bad=[...keys].filter(k=>!['received_ms','connection'].includes(k)&&!equivalent(expected[k],observed?.[k]));
    if(expected.connection!==connection)bad.push('connection');
    if(bad.length){mismatches++;if(errors.length<20)errors.push(`${expected.symbol} at ${item.t}: ${bad.join(',')}`);}
    const key=expected.symbol+':'+expected.status;statuses[key]=(statuses[key]??0)+1;
    const bounds=levels[expected.symbol]??={bid_min:Infinity,bid_max:0,ask_min:Infinity,ask_max:0};
    if(expected.bid_levels!==undefined){bounds.bid_min=Math.min(bounds.bid_min,expected.bid_levels);bounds.bid_max=Math.max(bounds.bid_max,expected.bid_levels);bounds.ask_min=Math.min(bounds.ask_min,expected.ask_levels);bounds.ask_max=Math.max(bounds.ask_max,expected.ask_levels);}
    const sums=flow[expected.symbol]??={feature_rows:0,rows_with_trades:0,partial_depth_rows:0,target_crossed_rows:0};
    if(expected.window_ms!==undefined){sums.feature_rows++;sums.rows_with_trades+=expected.trade_count_60s>0;sums.partial_depth_rows+=expected.status==='partial_depth';sums.target_crossed_rows+=expected.target_already_above;}
  }
}
for(const [name,want] of Object.entries(summary.source_hashes)){
  const got=createHash('sha256').update(fs.readFileSync(path.join(base,name))).digest('hex');
  if(got!==want)errors.push('Source changed since capture: '+name);
}
for(const [name,got,want] of [['topic_counts',topics,summary.raw_messages],['feature_status',statuses,summary.feature_status]]){
  for(const k of new Set([...Object.keys(got),...Object.keys(want)]))if(got[k]!==want[k])errors.push(`${name} differs at ${k}`);
}
if(records.at(-1)?.kind!=='end')errors.push('Missing end marker');
if(!features.length)errors.push('No feature rows');
const result={passed:errors.length===0,scope:'File integrity, count reconciliation and deterministic replay only; not independent evidence of feed completeness or predictive power.',
  raw_chunks:rawFiles.length,raw_records:records.length,feature_rows:features.length,replayed_rows:replayRows,mismatched_rows:mismatches,
  reconstruction_errors:reconstructionErrors,topics,statuses,levels,flow,errors};
fs.writeFileSync(path.join(dir,'audit.json'),JSON.stringify(result,null,2));
console.log(JSON.stringify(result,null,2));
process.exitCode=result.passed?0:1;
