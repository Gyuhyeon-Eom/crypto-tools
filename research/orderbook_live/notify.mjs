import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

export function parseEnv(text){
  const result={};
  for(const line of text.split(/\r?\n/)){
    const m=line.match(/^\s*(?:export\s+)?(SLACK_WEBHOOK_URL(?:_\d+)?)\s*=\s*(.*?)\s*$/);
    if(!m)continue;
    let value=m[2];
    if((value.startsWith('"')&&value.endsWith('"'))||(value.startsWith("'")&&value.endsWith("'")))value=value.slice(1,-1);
    if(value)result[m[1]]=value;
  }
  return result;
}
export function loadWebhooks(env=process.env,file=path.join(os.homedir(),'.openclaw','.env.slack-surge')){
  const named=Object.entries(env).filter(([k,v])=>/^SLACK_WEBHOOK_URL(?:_\d+)?$/.test(k)&&v).map(([,v])=>v);
  const urls=env.ORDERBOOK_SLACK_WEBHOOK_URL?[env.ORDERBOOK_SLACK_WEBHOOK_URL]:named.length?named:
    fs.existsSync(file)?Object.values(parseEnv(fs.readFileSync(file,'utf8'))):[];
  const unique=[...new Set(urls.map(x=>x.trim()))];
  if(unique.length>5)throw new Error('At most5 Slack webhook destinations supported');
  for(const value of unique){
    let u;try{u=new URL(value);}catch{throw new Error('Invalid Slack webhook configuration');}
    if(u.protocol!=='https:'||!['hooks.slack.com','hooks.slack-gov.com'].includes(u.hostname)||u.port||u.username||u.password
      ||!/^\/services\/[^/]+\/[^/]+\/[^/]+$/.test(u.pathname)||u.search||u.hash)
      throw new Error('Invalid Slack webhook configuration');
  }
  return unique;
}
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
export class Notifier{
  constructor({urls=[],enabled=false,fetcher=fetch,wait=sleep,journal=()=>{},spacingMs=1100,timeoutMs=8000}={}){
    if(enabled&&!urls.length)throw new Error('--slack requires ORDERBOOK_SLACK_WEBHOOK_URL or existing Slack surge webhook configuration');
    this.urls=urls;this.enabled=enabled;this.fetcher=fetcher;this.wait=wait;this.journal=journal;
    this.spacingMs=spacingMs;this.timeoutMs=timeoutMs;this.tail=Promise.resolve();this.queued=0;this.closed=false;
    this.abort=new AbortController();this.stats={delivered:0,failed:0,preview:0,dropped:0,delivery_unknown:0};
  }
  send(kind,text){
    const safe=text.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
    if(this.closed||this.queued>=30){this.stats.dropped++;this.journal({kind,status:'queue_full_or_closed'});return;}
    if(!this.enabled){this.stats.preview++;this.journal({kind,status:'preview',text:safe});return;}
    this.queued++;
    this.journal({kind,status:'queued',text:safe});
    this.tail=this.tail.then(async()=>{
      for(let index=0;index<this.urls.length&&!this.closed;index++){
        const result=await this.deliver(this.urls[index],safe);
        this.stats[result==='delivered'?'delivered':'failed']++;
        if(result==='delivery_unknown')this.stats.delivery_unknown++;
        this.journal({kind,destination:index+1,status:result});
      }
    }).catch(()=>{this.stats.failed++;this.journal({kind,status:'internal_error'});}).finally(()=>{this.queued--;});
  }
  async deliver(url,text){
    for(let attempt=0;attempt<3&&!this.closed;attempt++){
      await this.wait(this.spacingMs);
      if(this.closed)return 'cancelled';
      let response;
      try{
        response=await this.fetcher(url,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({text}),
          redirect:'error',signal:AbortSignal.any([this.abort.signal,AbortSignal.timeout(this.timeoutMs)])});
        if(response.status===429){
          const seconds=Number(response.headers.get('retry-after')??1);
          await response.text();
          if(!Number.isFinite(seconds)||seconds<0||seconds>30||attempt===2)return 'rate_limited';
          await this.wait(seconds*1000);continue;
        }
        if(response.status>=500){await response.text();if(attempt===2)return 'server_error';await this.wait(1000*2**attempt);continue;}
        const body=(await response.text()).trim();
        return response.status===200&&body==='ok'?'delivered':`rejected_http_${response.status}`;
      }catch{return 'delivery_unknown';} // Server may have accepted the request. Avoid duplicate retries.
    }
    return 'cancelled';
  }
  async flush(timeoutMs=15000){
    let timer;const timeout=new Promise(resolve=>{timer=setTimeout(()=>resolve(false),timeoutMs);});
    const done=await Promise.race([this.tail.then(()=>true),timeout]);clearTimeout(timer);return done;
  }
  close(){this.closed=true;this.abort.abort();}
}
