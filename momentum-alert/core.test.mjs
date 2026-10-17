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
  const quarter=Array.from({length:96},(_,i)=>({time:i*900000,close:120,high:121,turnover:100}));
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
  const state={},row={symbol:'NEARUSDT',eligible:true,candle_time:1000};
  assert.equal(takeSignal(state,row,1e8,policy),true);
  const loaded=JSON.parse(JSON.stringify(state));
  assert.equal(takeSignal(loaded,row,2e8,policy),false);
  assert.equal(takeSignal(loaded,{...row,candle_time:2000},1e8+3600000,policy),false);
  assert.equal(takeSignal(loaded,{...row,symbol:'SUIUSDT'},1e8,policy),true);
  assert.equal(takeSignal(loaded,{...row,candle_time:2000},1e8+6*3600000,policy),true);
});
test('message uses current reference price, explicit unvalidated label and underlying targets',()=>{
  const x=fixture(),row={...assess('NEARUSDT',x.hour,x.quarter,x.btcHour,x.ticker,policy),reference_price:100,observed_at:1000};
  const text=formatSignal(row);assert.match(text,/\+5% 105/);assert.match(text,/\+10% 110/);assert.match(text,/미검증/);
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
