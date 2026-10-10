// Bounded public-data capture only. Node 22+; no keys, notifications or orders.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createGzip} from 'node:zlib';
import {pipeline} from 'node:stream/promises';

const base = path.dirname(fileURLToPath(import.meta.url));
const seconds = Number(process.argv[2] ?? 60);
const symbols = (process.argv[3] ?? 'JTOUSDT,TREEUSDT').split(',');
if (!Number.isInteger(seconds) || seconds < 1 || seconds > 3600 || symbols.length > 5 ||
    !symbols.every(s => /^[A-Z0-9]+USDT$/.test(s)) || new Set(symbols).size !== symbols.length) {
  throw new Error('Usage: node capture_orderbook.mjs SECONDS[1..3600] SYMBOLS[1..5 comma-separated]');
}
const stamp = new Date().toISOString().replaceAll(':', '-');
const dir = path.join(base, 'output', `capture_${stamp}`);
fs.mkdirSync(dir, {recursive: true});
const gzip = createGzip();
const writing = pipeline(gzip, fs.createWriteStream(path.join(dir, 'messages.jsonl.gz'), {flags: 'wx'}));
const summary = {started_at: new Date().toISOString(), seconds_requested: seconds, symbols,
  endpoint: 'wss://stream.bybit.com/v5/public/spot', subscribed: false,
  messages: {}, book_snapshots: {}, trade_records: {}, bytes_before_compression: 0,
  status: 'connecting', errors: [], note: 'Bounded connectivity capture; no predictive validation or continuous collection.'};
const started = performance.now();
let finished = false;
let ping;
let forceClose;
const ws = new WebSocket(summary.endpoint);
function record(kind, data) {
  const line = JSON.stringify({received_ms: Date.now(), elapsed_ms: performance.now()-started, kind, data})+'\n';
  summary.bytes_before_compression += Buffer.byteLength(line);
  if (!gzip.write(line) && gzip.writableLength > 16*1024*1024) {
    summary.errors.push('Writer backlog exceeded 16 MiB; stopped to avoid unbounded memory.');
    finish('writer_backlog');
  }
}
async function finish(status) {
  if (finished) return;
  finished = true;
  clearTimeout(deadline);
  clearInterval(ping);
  summary.status = status;
  summary.finished_at = new Date().toISOString();
  summary.elapsed_seconds = (performance.now()-started)/1000;
  record('end', {status});
  ws.close();
  gzip.end();
  await writing;
  fs.writeFileSync(path.join(dir, 'summary.json'), JSON.stringify(summary, null, 2));
  console.log(JSON.stringify({...summary, output_directory: dir}, null, 2));
  process.exitCode = status === 'complete' && summary.subscribed ? 0 : 1;
  // Bound a stalled closing handshake after the output is safely flushed.
  forceClose = setTimeout(() => process.exit(process.exitCode), 3000);
  forceClose.unref();
}
const deadline = setTimeout(() => finish('complete'), seconds*1000);
ws.addEventListener('open', () => {
  record('connection_open', {});
  ws.send(JSON.stringify({op: 'subscribe', args: symbols.flatMap(s => [`orderbook.50.${s}`, `publicTrade.${s}`])}));
  ping = setInterval(() => ws.send(JSON.stringify({op: 'ping'})), 20000);
});
ws.addEventListener('message', event => {
  if (finished) return;
  const raw = String(event.data);
  let m;
  try { m = JSON.parse(raw); } catch {
    record('unparsed', raw);
    return;
  }
  record('message', m);
  if (m.op === 'subscribe') {
    summary.subscribed = m.success === true;
    if (!summary.subscribed) {
      summary.errors.push(m.ret_msg ?? 'Subscription rejected');
      finish('subscription_rejected');
    }
  }
  if (m.topic) summary.messages[m.topic] = (summary.messages[m.topic] ?? 0)+1;
  if (m.topic?.startsWith('orderbook.') && m.type === 'snapshot') {
    summary.book_snapshots[m.topic] = (summary.book_snapshots[m.topic] ?? 0)+1;
  }
  if (m.topic?.startsWith('publicTrade.')) {
    summary.trade_records[m.topic] = (summary.trade_records[m.topic] ?? 0)+(m.data?.length ?? 0);
  }
});
ws.addEventListener('error', event => {
  summary.errors.push(event.message ?? 'WebSocket connection error');
  finish('connection_error');
});
ws.addEventListener('close', event => {
  clearTimeout(forceClose);
  if (!finished) {
    record('connection_closed', {code: event.code, reason: event.reason});
    finish('closed_early');
  }
});
process.on('SIGINT', () => finish('interrupted'));
