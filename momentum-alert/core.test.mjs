import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {closedCandles,requireWindow,similarity,rankPeers,assess,takeSignal,formatSignal} from './core.mjs';
import {Notifier} from '../research/orderbook_live/notify.mjs';
const policy=JSON.parse(fs.readFileSync(new URL('./policy.json',import.meta.url)));
test('unfinished candles excluded; missing, duplicate, stale and malformed candles fail closed',()=>{
  const bar=t=>[t,100,102,99,101,5,500];
  const rows=closedCandles([bar(120000),bar(60000),bar(0)],1,120000);
  assert.equal(rows.length,2);assert.equal(rows.at(-1).time,60000);
  assert.equal(requireWindow(rows,1,2,120000).length,2);
  assert.throws(()=>requireWindow(rows,1,2,180000));
  assert.throws(()=>closedCandles([bar(0),bar(120000)],1,180000));
  assert.throws(()=>closedCandles([bar(0),bar(0)],1,60000));
  assert.throws(()=>closedCandles([[0,100,99,100,100,5,500]],1,60000));
});
test('selection uses aligned returns and rejects insufficient similarity or liquidity',()=>{
  const bars=Array.from({length:721},(_,i)=>({time:i*3600000,close:100*Math.exp(.02*Math.sin(i)+i*.0001)}));
  const btc=bars.map((x,i)=>({...x,close:100*Math.exp(.01*Math.sin(i/3))}));
  const match=similarity(bars,bars,btc);assert.ok(Math.abs(match.correlation-1)<1e-10);assert.ok(Math.abs(match.residual_correlation-1)<1e-10);
  assert.throws(()=>similarity(bars,bars.slice(1),btc));
  const good={symbol:'SUIUSDT',...match,turnover24h:2e7};
  const ranked=rankPeers([good,{...good,symbol:'APTUSDT',residual_correlation:-.1},{...good,symbol:'SEIUSDT',turnover24h:100}],policy);
  assert.deepEqual(ranked.map(x=>x.symbol),['SUIUSDT']);
  assert.deepEqual(rankPeers([],policy),[]);
});
function fixture(){
  const hour=Array.from({length:200},(_,i)=>({close:100+i*.1}));hour.at(-1).close=121;
  const quarter=Array.from({length:96},(_,i)=>({time:i*900000,close:120,high:121,low:119,turnover:100}));
  quarter.at(-1).close=121;quarter.at(-1).turnover=250;
  const ticker={bid1Price:'120.99',ask1Price:'121.01',lastPrice:'121',turnover24h:'20000000'};
  return {hour,quarter,ticker,btcHour:[{close:100},{close:100}]};
}
test('signal requires conjunction of trend, volume, limited rise, market-relative strength and liquidity',()=>{
  const run=x=>assess('NEARUSDT',x.hour,x.quarter,x.btcHour,x.ticker,policy);
  assert.equal(run(fixture()).eligible,true);
  for(const mutate of [x=>x.quarter.at(-1).turnover=100,x=>x.hour.at(-1).close=130,
    x=>x.btcHour.at(-1).close=110,x=>x.ticker.ask1Price='125',x=>x.ticker.lastPrice='NaN',x=>x.quarter.at(-1).close=90]){
    const x=fixture();mutate(x);assert.equal(run(x).eligible,false);
  }
});
test('cooldown survives serialization, rejects repeated bars, and is per symbol',()=>{
  const state={},row={symbol:'NEARUSDT',direction:'long',eligible:true,candle_time:1000};
  assert.equal(takeSignal(state,row,1e8,policy),true);
  const loaded=JSON.parse(JSON.stringify(state));
  assert.equal(takeSignal(loaded,row,2e8,policy),false);
  assert.equal(takeSignal(loaded,{...row,candle_time:2000},1e8+3600000,policy),false);
  assert.equal(takeSignal(loaded,{...row,symbol:'SUIUSDT'},1e8,policy),true);
  assert.equal(takeSignal(loaded,{...row,candle_time:2000},1e8+6*3600000,policy),true);
});
test('message uses current reference price, explicit unvalidated label and underlying targets',()=>{
  const x=fixture(),row={...assess('NEARUSDT',x.hour,x.quarter,x.btcHour,x.ticker,policy),reference_price:100,observed_at:1000};
  const text=formatSignal(row);assert.match(text,/LONG\/롱/);assert.match(text,/\+5% 105/);assert.match(text,/\+10% 110/);assert.match(text,/미검증/);
});
function shortFixture(){
  const x=fixture();
  x.hour=x.hour.map(r=>({...r,close:200-r.close}));
  x.quarter=x.quarter.map(r=>({...r,close:200-r.close,high:200-r.low,low:200-r.high}));
  x.ticker={...x.ticker,bid1Price:'78.99',ask1Price:'79.01',lastPrice:'79'};
  return x;
}
test('short requires falling trend, rising turnover, proximity to low and BTC underperformance',()=>{
  const run=x=>assess('NEARUSDT',x.hour,x.quarter,x.btcHour,x.ticker,policy,'short');
  const result=run(shortFixture());assert.equal(result.eligible,true);assert.equal(result.direction,'short');assert.equal(result.checks.near_low,true);
  for(const mutate of [x=>x.quarter.at(-1).turnover=100,x=>x.hour.at(-1).close=70,
    x=>x.btcHour.at(-1).close=90,x=>x.ticker.ask1Price='85',x=>x.quarter.at(-1).close=95,
    x=>x.quarter.at(-1).close=70,x=>x.ticker.lastPrice='NaN']){
    const x=shortFixture();mutate(x);assert.equal(run(x).eligible,false);
  }
});
test('long and short do not both qualify and neutral prices produce neither',()=>{
  for(const x of [fixture(),shortFixture()]){
    const r=['long','short'].map(d=>assess('NEARUSDT',x.hour,x.quarter,x.btcHour,x.ticker,policy,d));
    assert.equal(r.filter(x=>x.eligible).length,1);
  }
  const x=fixture();x.hour=x.hour.map(r=>({...r,close:120}));
  for(const d of ['long','short'])assert.equal(assess('NEARUSDT',x.hour,x.quarter,x.btcHour,x.ticker,policy,d).eligible,false);
  assert.throws(()=>assess('NEARUSDT',x.hour,x.quarter,x.btcHour,x.ticker,policy,'invalid'));
});
test('opposite direction and legacy v1 state share cooldown after restart',()=>{
  const state={NEARUSDT:{candle_time:1000,notified_at:1e8}},row={symbol:'NEARUSDT',direction:'short',eligible:true,candle_time:2000};
  assert.equal(takeSignal(state,row,1e8+3600000,policy),false);
  assert.equal(takeSignal(state,row,1e8+6*3600000,policy),true);
  const loaded=JSON.parse(JSON.stringify(state));assert.equal(loaded.NEARUSDT.direction,'short');
  assert.equal(takeSignal(loaded,{...row,direction:'long',candle_time:3000},1e8+7*3600000,policy),false);
  assert.equal(takeSignal(loaded,{...row,direction:'invalid',candle_time:3000},3e8,policy),false);
});
test('short message identifies direction and negative underlying-price targets',()=>{
  const x=shortFixture(),row={...assess('NEARUSDT',x.hour,x.quarter,x.btcHour,x.ticker,policy,'short'),reference_price:100,observed_at:1000};
  const text=formatSignal(row);assert.match(text,/SHORT\/숏/);assert.match(text,/-5% 95/);assert.match(text,/-10% 90/);
  assert.match(text,/저점 대비/);assert.match(text,/하락 조건 관찰/);assert.doesNotMatch(text,/\+5%|고점 대비|상승 조건/);
  assert.throws(()=>formatSignal({...row,direction:undefined}));
});
test('short observation reaches mocked Slack with its own direction label',async()=>{
  const sent=[],x=shortFixture(),row={...assess('NEARUSDT',x.hour,x.quarter,x.btcHour,x.ticker,policy,'short'),observed_at:1e8};
  const n=new Notifier({enabled:true,urls:['https://hooks.slack.com/services/TEST/TEST/TEST'],wait:async()=>{},
    fetcher:async(_url,options)=>{sent.push(JSON.parse(options.body).text);return {status:200,text:async()=>'ok'};}});
  assert.equal(takeSignal({},row,row.observed_at,policy),true);
  n.send('momentum_observation',formatSignal(row));await n.flush();n.close();
  assert.equal(sent.length,1);assert.match(sent[0],/SHORT\/숏/);assert.match(sent[0],/-5%/);
});
test('qualified momentum observation reaches mocked Slack only once across repeated scans',async()=>{
  const sent=[],state={},x=fixture();
  const notifier=new Notifier({enabled:true,urls:['https://hooks.slack.com/services/TEST/TEST/TEST'],wait:async()=>{},
    fetcher:async(_url,options)=>{sent.push(JSON.parse(options.body).text);return {status:200,text:async()=>'ok'};}});
  for(let i=0;i<12;i++){
    const row={...assess('NEARUSDT',x.hour,x.quarter,x.btcHour,x.ticker,policy),observed_at:1e8+i*300000};
    if(takeSignal(state,row,row.observed_at,policy))notifier.send('momentum_observation',formatSignal(row));
  }
  await notifier.flush();assert.equal(sent.length,1);assert.equal(notifier.stats.delivered,1);
  assert.match(sent[0],/NEAR Momentum/);assert.match(sent[0],/적중률 미검증/);notifier.close();
});
