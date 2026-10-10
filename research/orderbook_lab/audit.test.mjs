import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {gzipSync} from 'node:zlib';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {TapeEngine} from './engine.mjs';

const base=path.dirname(fileURLToPath(import.meta.url));
function fixture(dir,corrupt){
  const symbols=['TESTUSDT'],engine=new TapeEngine(symbols);engine.reset();
  const records=[{elapsed_ms:0,connection:1,kind:'connecting',data:{}}],features=[],statuses={};
  for(let second=0;second<=61;second++){
    const t=second*1000;
    const m={topic:'orderbook.1000.TESTUSDT',type:second===0?'snapshot':'delta',cts:t,
      data:{s:'TESTUSDT',u:second+10,seq:second+10,b:second===0?[[99.9,10],[99,20]]:[],a:second===0?[[100.1,10],[101,20]]:[]}};
    records.push({elapsed_ms:t,connection:1,kind:'message',data:m});engine.onMessage(m,t);
    for(const r of engine.sample(t+1)){
      features.push({...r,connection:1,received_ms:100000+t});
      const key=r.symbol+':'+r.status;statuses[key]=(statuses[key]??0)+1;
    }
  }
  records.push({elapsed_ms:62000,connection:1,kind:'end',data:{status:'complete'}});
  if(corrupt)features.at(-1).ask_depth_change=0.1;
  const pack=rows=>gzipSync(rows.map(r=>JSON.stringify(r)+'\n').join(''));
  fs.writeFileSync(path.join(dir,'raw_0000.jsonl.gz'),pack(records));
  fs.writeFileSync(path.join(dir,'features.jsonl.gz'),pack(features));
  fs.writeFileSync(path.join(dir,'summary.json'),JSON.stringify({symbols,files:['raw_0000.jsonl.gz','features.jsonl.gz'],feature_status:statuses,
    raw_messages:{'orderbook.1000.TESTUSDT':62},source_hashes:Object.fromEntries(['engine.mjs','capture.mjs','protocol.json'].map(p=>[p,createHash('sha256').update(fs.readFileSync(path.join(base,p))).digest('hex')]))}));
}
for(const corrupt of [false,true])test(corrupt?'audit rejects a modified saved measurement':'audit reproduces complete capture fixture',()=>{
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'book-audit-'));
  try{
    fixture(dir,corrupt);
    const result=spawnSync(process.execPath,[path.join(base,'audit.mjs'),dir],{encoding:'utf8'});
    const audit=JSON.parse(fs.readFileSync(path.join(dir,'audit.json')));
    assert.equal(result.status,corrupt?1:0,result.stderr);
    assert.equal(audit.passed,!corrupt);
    assert.equal(audit.mismatched_rows,corrupt?1:0);
    assert.equal(audit.replayed_rows,62);
  }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
