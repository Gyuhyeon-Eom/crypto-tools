// Node22+, built-in WebSocket. No credentials or exchange account actions.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createGzip} from 'node:zlib';
import {pipeline} from 'node:stream/promises';
import {createHash} from 'node:crypto';
import {TapeEngine} from './engine.mjs';

const base=path.dirname(fileURLToPath(import.meta.url));
const seconds=Number(process.argv[2]??180);
const symbols=(process.argv[3]??'JTOUSDT,TREEUSDT,WLDUSDT,BTCUSDT').split(',');
if (!Number.isInteger(seconds)||seconds<1||seconds>86400||symbols.length>5||!symbols.every(s=>/^[A-Z0-9]+USDT$/.test(s))||new Set(symbols).size!==symbols.length) {
  throw new Error('Usage: node capture.mjs SECONDS[1..86400] SYMBOLS[max5 comma separated]');
}
const dir=path.join(base,'output','run_'+new Date().toISOString().replaceAll(':','-'));
fs.mkdirSync(dir,{recursive:true});
const summary={started_at:new Date().toISOString(),seconds_requested:seconds,symbols,depth:1000,
  status:'starting',connections:0,subscriptions:0,reconnects:0,raw_messages:{},book_snapshots:{},
  trade_records:0,duplicate_trades:0,feature_status:{},errors:[],raw_uncompressed_bytes:0,files:[],
  source_hashes:Object.fromEntries(['engine.mjs','capture.mjs','protocol.json'].map(p=>[p,createHash('sha256').update(fs.readFileSync(path.join(base,p))).digest('hex')])),
  scope:'Receive-time measurements only; no predictive model, labels or unattended scheduler.'};
const engine=new TapeEngine(symbols);
const start=performance.now();
const elapsed=()=>performance.now()-start;
let finished=false,ws=null,raw=null,chunk=-1,connection=0,backoff=1000,retry=null,heartbeat=null;
let lastReceive=performance.now(),handshakeDeadline=null;
const pending=[];
function stream(filename) {
  const z=createGzip();summary.files.push(filename);
  const p=pipeline(z,fs.createWriteStream(path.join(dir,filename),{flags:'wx'}));
  // Attach the rejection handler immediately; retain settled status for final audit.
  pending.push(p.then(()=>true,error=>{summary.errors.push('Storage: '+error.message);finish('storage_error');return false;}));
  return z;
}
const featureStream=stream('features.jsonl.gz');
function write(z,obj) {
  const line=JSON.stringify(obj)+'\n';
  if (!z.write(line)&&z.writableLength>8*1024*1024) finish('writer_backlog');
  return Buffer.byteLength(line);
}
function record(kind,data,now=elapsed()) {
  const c=Math.floor(now/60000);
  if (c!==chunk) {
    raw?.end();chunk=c;raw=stream(`raw_${String(c).padStart(4,'0')}.jsonl.gz`);
  }
  summary.raw_uncompressed_bytes+=write(raw,{received_ms:Date.now(),elapsed_ms:now,connection,kind,data});
  if (summary.raw_uncompressed_bytes>1024**3&&!finished) finish('storage_budget_1GiB');
}
async function finish(status) {
  if (finished)return;
  finished=true;summary.status=status;
  clearTimeout(deadline);clearTimeout(retry);clearTimeout(handshakeDeadline);clearInterval(sampleTimer);clearInterval(heartbeat);
  record('end',{status});raw?.end();featureStream.end();
  try {ws?.close();}catch {}
  const writes=await Promise.all(pending);
  if (writes.some(ok=>!ok)) summary.status='storage_error';
  summary.finished_at=new Date().toISOString();summary.elapsed_seconds=elapsed()/1000;
  summary.total_disk_bytes=summary.files.reduce((s,f)=>s+(fs.existsSync(path.join(dir,f))?fs.statSync(path.join(dir,f)).size:0),0);
  fs.writeFileSync(path.join(dir,'summary.json'),JSON.stringify(summary,null,2));
  console.log(JSON.stringify({...summary,output_directory:dir},null,2));
  process.exitCode=summary.status==='complete'&&summary.subscriptions>0?0:1;
  setTimeout(()=>process.exit(process.exitCode),2000).unref();
}
function reconnect(reason) {
  if (finished||retry)return;
  summary.errors.push(reason);record('gap',{reason});engine.reset();
  clearInterval(heartbeat);clearTimeout(handshakeDeadline);
  const old=ws;ws=null;
  try {old?.close();}catch {}
  summary.reconnects++;
  retry=setTimeout(()=>{retry=null;connect();},backoff);
  backoff=Math.min(backoff*2,15000);
}
function connect() {
  if(finished)return;
  connection++;summary.connections++;engine.reset();
  record('connecting',{});
  const socket=new WebSocket('wss://stream.bybit.com/v5/public/spot');ws=socket;
  handshakeDeadline=setTimeout(()=>reconnect('Connection/subscription timeout'),15000);
  socket.addEventListener('open',()=>{
    if(ws!==socket||finished)return;
    lastReceive=performance.now();record('connection_open',{});
    socket.send(JSON.stringify({op:'subscribe',args:symbols.flatMap(s=>[`orderbook.1000.${s}`,`publicTrade.${s}`])}));
    heartbeat=setInterval(()=>{
      if(performance.now()-lastReceive>45000){reconnect('No server messages/pongs45sec');return;}
      if(socket.readyState!==WebSocket.OPEN){reconnect('Socket not open at heartbeat');return;}
      try {socket.send(JSON.stringify({op:'ping'}));}catch(error){reconnect('Ping: '+error.message);}
    },15000);
  });
  socket.addEventListener('message',event=>{
    if(ws!==socket||finished)return;
    lastReceive=performance.now();
    let m;
    try {m=JSON.parse(String(event.data));}catch {reconnect('Invalid JSON');return;}
    const now=elapsed();
    record('message',m,now);
    if(m.op==='subscribe') {
      if(m.success!==true){finish('subscription_rejected');return;}
      clearTimeout(handshakeDeadline);summary.subscriptions++;backoff=1000;
    }
    if(m.topic)summary.raw_messages[m.topic]=(summary.raw_messages[m.topic]??0)+1;
    try {
      const v=engine.onMessage(m,now);
      if(v.kind==='snapshot')summary.book_snapshots[m.data.s]=(summary.book_snapshots[m.data.s]??0)+1;
      if(v.kind==='trade'){summary.trade_records+=v.accepted;summary.duplicate_trades+=v.duplicates;}
    }catch(error){reconnect('Reconstruction: '+error.message);}
  });
  socket.addEventListener('error',event=>{if(ws===socket&&!finished)reconnect(event.message??'WebSocket error');});
  socket.addEventListener('close',event=>{if(ws===socket&&!finished)reconnect(`Closed ${event.code}: ${event.reason}`);});
}
const sampleTimer=setInterval(()=>{
  if(finished)return;
  for(const row of engine.sample(elapsed())) {
    row.received_ms=Date.now();row.connection=connection;
    write(featureStream,row);
    const key=row.symbol+':'+row.status;summary.feature_status[key]=(summary.feature_status[key]??0)+1;
  }
},1000);
const deadline=setTimeout(()=>finish('complete'),seconds*1000);
process.on('SIGINT',()=>finish('interrupted'));
process.on('SIGTERM',()=>finish('interrupted'));
connect();
