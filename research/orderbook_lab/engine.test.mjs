import test from 'node:test';
import assert from 'node:assert/strict';
import {Book,TapeEngine,band} from './engine.mjs';

function message(type='snapshot',u=10,b=[[99.9,10],[99,20]],a=[[100.1,10],[101,20]]) {
 return {topic:'orderbook.1000.TESTUSDT',type,cts:u,data:{s:'TESTUSDT',u,seq:u,b,a}};
}
function trade(id,side='Buy',rpi=false) {
 return {topic:'publicTrade.TESTUSDT',data:[{s:'TESTUSDT',i:id,S:side,p:'100',v:'2',T:1000,seq:5,RPI:rpi}]};
}
test('snapshot resets state;delta inserts,updates,deletes',()=>{
 const b=new Book();b.apply(message(),0);
 b.apply(message('delta',11,[[99.9,0],[99.8,5]],[]),1);
 assert.equal(b.bids.has(99.9),false);assert.equal(b.bids.get(99.8),5);
 b.apply(message('snapshot',1,[[98,1]],[[102,1]]),2);
 assert.equal(b.bids.size,1);assert.equal(b.u,1);
});
test('gaps in IDs are allowed but reordered or crossed books rejected',()=>{
 const b=new Book();assert.throws(()=>b.apply(message('delta'),0));b.apply(message(),0);
 b.apply(message('delta',100,[],[]),1);assert.equal(b.u,100);
 assert.throws(()=>b.apply(message('delta',99,[],[]),2));
 assert.throws(()=>b.apply(message('delta',101,[[102,1]],[]),3));
 assert.equal(b.u,100); // failed mutation does not partially replace known state
});
test('exact duplicate delta ignored',()=>{
 const b=new Book();b.apply(message(),0);const m=message('delta',11,[],[]);
 b.apply(m,1);assert.equal(b.apply(m,2),'duplicate');
});
test('insufficient depth is unavailable not zero; crossed target not zero-cost opportunity',()=>{
 const b=new Book();b.apply(message('snapshot',10,[[99.9,1]],[[100.1,1]]),0);
 const v=band(b.snapshot(0),100);assert.equal(v.ask_quote,null);assert.equal(v.bid_quote,null);
 b.apply(message('snapshot',11,[[100.4,1],[99,1]],[[100.5,1],[101,1]]),1);
 const z=band(b.snapshot(1),100);assert.equal(z.target_already_above,true);assert.equal(z.ask_cost,null);
});
test('same seq with different trade IDs retained,IDs deduped,RPI separate',()=>{
 const e=new TapeEngine(['TESTUSDT']);e.onMessage(message(),0);
 e.onMessage(trade('a'),100);e.onMessage(trade('b','Sell'),101);
 assert.equal(e.onMessage(trade('a'),102).duplicates,1);
 e.onMessage(trade('c','Buy',true),103);
 assert.equal(e.states.get('TESTUSDT').trades.length,3);
 for(let t=0;t<=60000;t+=1000){if(t>0)e.onMessage(message('delta',10+t,[],[]),t);var rows=e.sample(t);}
 assert.equal(rows[0].buy_quote_60s,200);assert.equal(rows[0].sell_quote_60s,200);assert.equal(rows[0].rpi_quote_60s,200);
});
test('fixed-anchor depth changes measured only after warmup',()=>{
 const e=new TapeEngine(['TESTUSDT']);e.onMessage(message(),0);
 for(let t=0;t<60000;t+=1000){if(t>0)e.onMessage(message('delta',10+t,[],[]),t);assert.equal(e.sample(t)[0].status,'warmup');}
 e.onMessage(message('delta',60010,[[99.9,15]],[[100.1,5]]),60000);
 const r=e.sample(60000)[0];assert.equal(r.status,'ready');assert.equal(r.ask_depth_change,-.5);assert.equal(r.bid_recovery,.5);
});
test('connection reset and stale books invalidate windows',()=>{
 const e=new TapeEngine(['TESTUSDT']);e.onMessage(message(),0);
 for(let t=0;t<=60000;t+=1000)var r=e.sample(t)[0];
 assert.equal(r.status,'stale_book_or_sample_gap');
 e.reset();assert.equal(e.sample(61000)[0].status,'no_snapshot');
 e.onMessage(message(),62000);assert.equal(e.sample(63000)[0].status,'warmup');
});
test('a sampling gap makes60sec measures unavailable',()=>{
 const e=new TapeEngine(['TESTUSDT']);e.onMessage(message(),0);e.sample(0);
 for(let t=5000;t<=60000;t+=1000){e.onMessage(message('delta',t+10,[],[]),t);var r=e.sample(t)[0];}
 assert.equal(r.status,'stale_book_or_sample_gap');
});
test('future messages cannot alter previously emitted measurements',()=>{
 const e=new TapeEngine(['TESTUSDT']);e.onMessage(message(),0);
 const r=e.sample(0)[0],before=JSON.stringify(r);
 e.onMessage(message('delta',11,[[99.9,50]],[]),1000);
 assert.equal(JSON.stringify(r),before);
});
