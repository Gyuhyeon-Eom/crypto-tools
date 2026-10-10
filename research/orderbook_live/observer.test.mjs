import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {Observer,classify,formatObservation} from './observer.mjs';
import {Notifier} from './notify.mjs';
const row=(t=0)=>({symbol:'JTOUSDT',status:'ready',t,generation:1,target_already_above:false,
 spread_bps:5,trade_count_60s:12,buy_quote_60s:2000,sell_quote_60s:500,net_taker_fraction:.6,
 mid_return_60s:.001,ask_depth_change:-.4,ask_last10s_max_ratio:.6,ask_cost_change:-.3,bid_recovery:.5});
test('only eligible non-BTC observations with actual flow can qualify',()=>{
 assert.equal(classify(row()),'ask_depletion');
 for(const change of [{status:'warmup'},{status:'partial_depth'},{target_already_above:true},{ask_depth_change:null},{buy_quote_60s:0},{symbol:'BTCUSDT'},{spread_bps:30}])
  assert.equal(classify({...row(),...change}),null);
 assert.equal(classify({...row(),net_taker_fraction:-.5,buy_quote_60s:500,sell_quote_60s:2000}),'bid_recovery');
});
test('10sec continuous hold and independent per-symbol cooldown',()=>{
 const o=new Observer();let event;
 for(let t=0;t<=10000;t+=1000){event=o.observe(row(t),1e8+t);if(t<10000)assert.equal(event,null);}
 assert.equal(event.rule,'ask_depletion');assert.match(formatObservation(event),/실험용 관찰/);assert.match(formatObservation(event),/미검증/);
 assert.equal(o.observe(row(11000),1e8+11000),null);
 let other;for(let t=0;t<=10000;t+=1000)other=o.observe({...row(t),symbol:'WLDUSDT'},1e8+t);
 assert.equal(other.symbol,'WLDUSDT');
});
test('gap, invalid row or reconnect restarts hold',()=>{
 for(const action of [o=>o.observe({...row(6000),status:'warmup'},1e8+6000),o=>o.reset(),()=>{}]){
  const o=new Observer();for(let t=0;t<=5000;t+=1000)o.observe(row(t),1e8+t);action(o);
  assert.equal(o.observe(row(10000),1e8+10000),null);
 }
 const o=new Observer();for(let t=0;t<=9000;t+=1000)o.observe(row(t),1e8+t);
 assert.equal(o.observe({...row(10000),generation:2},1e8+10000),null);
});
test('cooldown persists through process restart; corrupted file fails closed',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'observation-state-')),file=path.join(dir,'state.json');
 try{
  const a=new Observer(file);for(let t=0;t<=10000;t+=1000)a.observe(row(t),1e8+t);
  const b=new Observer(file);for(let t=0;t<=10000;t+=1000)assert.equal(b.observe(row(t),1e8+20000+t),null);
  fs.writeFileSync(file,'not json');assert.throws(()=>new Observer(file));
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
test('qualifying observations reach Slack transport once with experiment label',async()=>{
 const payloads=[],o=new Observer();
 const n=new Notifier({enabled:true,urls:['https://hooks.slack.com/services/TEST/TEST/TEST'],wait:async()=>{},
  fetcher:async(_url,options)=>{payloads.push(JSON.parse(options.body));return {status:200,text:async()=>'ok'};}});
 for(let t=0;t<60000;t+=1000){const event=o.observe(row(t),1e8+t);if(event)n.send('observation',formatObservation(event));}
 await n.flush();assert.equal(payloads.length,1);assert.equal(n.stats.delivered,1);
 assert.match(payloads[0].text,/JTOUSDT/);assert.match(payloads[0].text,/급등 예측 모델 미적용/);
});
