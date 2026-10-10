import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {Notifier,loadWebhooks,parseEnv} from './notify.mjs';
const url='https://hooks.slack.com/services/TEST/TEST/TEST';
const reply=(status,body='ok',retry=null)=>({status,text:async()=>body,headers:{get:()=>retry}});

test('quoted legacy config, environment precedence, dedup and URL validation',()=>{
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'slack-config-'));
  try{
    const file=path.join(dir,'env');fs.writeFileSync(file,`export SLACK_WEBHOOK_URL="${url}"\nSLACK_WEBHOOK_URL_2='${url}'\n`);
    assert.equal(parseEnv(fs.readFileSync(file,'utf8')).SLACK_WEBHOOK_URL,url);
    assert.deepEqual(loadWebhooks({},file),[url]);
    assert.deepEqual(loadWebhooks({ORDERBOOK_SLACK_WEBHOOK_URL:url},'/nonexistent'),[url]);
    assert.deepEqual(loadWebhooks({},'/nonexistent'),[]);
    assert.throws(()=>loadWebhooks({ORDERBOOK_SLACK_WEBHOOK_URL:'https://example.com/secret'},file),/configuration/);
  }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
test('preview never sends and actual delivery requires credentials',async()=>{
  let calls=0;const rows=[];const n=new Notifier({fetcher:async()=>{calls++;},journal:r=>rows.push(r)});
  n.send('test','<@everyone> status');await n.flush();assert.equal(calls,0);assert.equal(rows[0].status,'preview');
  assert.match(rows[0].text,/&lt;/);assert.throws(()=>new Notifier({enabled:true}),/requires/);
});
test('only HTTP200 ok is counted delivered; no secret in journal',async()=>{
  const rows=[],bodies=[];
  const n=new Notifier({enabled:true,urls:[url],wait:async()=>{},journal:r=>rows.push(r),fetcher:async(u,init)=>{bodies.push(JSON.parse(init.body));assert.equal(init.redirect,'error');return reply(200);}});
  n.send('start','collector started');assert.equal(await n.flush(),true);
  assert.equal(n.stats.delivered,1);assert.equal(bodies[0].text,'collector started');assert.ok(!JSON.stringify(rows).includes(url));
});
test('429 respects Retry-After and retries, permanent rejection does not',async()=>{
  const waits=[];let calls=0;
  const n=new Notifier({enabled:true,urls:[url],wait:async ms=>waits.push(ms),fetcher:async()=>++calls===1?reply(429,'limited','2'):reply(200)});
  n.send('test','status');await n.flush();assert.equal(calls,2);assert.ok(waits.includes(2000));assert.equal(n.stats.delivered,1);
  const denied=new Notifier({enabled:true,urls:[url],wait:async()=>{},fetcher:async()=>reply(403,'secret body')});
  denied.send('test','status');await denied.flush();assert.equal(denied.stats.failed,1);assert.equal(denied.stats.delivered,0);
});
test('ambiguous timeout is not retried or counted successful',async()=>{
  let calls=0;const rows=[];
  const n=new Notifier({enabled:true,urls:[url],wait:async()=>{},journal:r=>rows.push(r),fetcher:async()=>{calls++;throw new Error(url);}});
  n.send('test','status');await n.flush();assert.equal(calls,1);assert.equal(n.stats.delivery_unknown,1);
  assert.equal(n.stats.delivered,0);assert.ok(!JSON.stringify(rows).includes(url));
});
test('bounded retries on server errors and non-ok200 are failures',async()=>{
  let calls=0;
  const n=new Notifier({enabled:true,urls:[url],wait:async()=>{},fetcher:async()=>{calls++;return reply(500,'error');}});
  n.send('test','status');await n.flush();assert.equal(calls,3);assert.equal(n.stats.failed,1);
  const bad=new Notifier({enabled:true,urls:[url],wait:async()=>{},fetcher:async()=>reply(200,'not ok')});
  bad.send('test','status');await bad.flush();assert.equal(bad.stats.delivered,0);
});
test('queue is bounded and shutdown cancels pending requests',async()=>{
  const n=new Notifier({enabled:true,urls:[url],wait:async()=>{},fetcher:async()=>reply(200)});
  for(let i=0;i<40;i++)n.send('test','status');
  assert.equal(n.stats.dropped,10);n.close();await n.flush();assert.equal(n.stats.delivered,0);
});
