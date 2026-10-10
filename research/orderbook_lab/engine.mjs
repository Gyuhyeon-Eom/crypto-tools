// Receive-time causal book reconstruction and60s measurements; no predictions.
export class Book {
  constructor() { this.reset(); }
  reset() { this.bids=new Map(); this.asks=new Map(); this.valid=false; this.u=null; this.seq=null; this.cts=null; this.received=null; this.signature=null; }
  apply(message, received) {
    const d=message.data;
    if (!d || !['snapshot','delta'].includes(message.type)) throw new Error('Malformed book message');
    const snapshot=message.type==='snapshot';
    if (d.u===1 && !snapshot) throw new Error('Service reset without snapshot');
    if (!snapshot && !this.valid) throw new Error('Delta before snapshot');
    const signature=JSON.stringify(d);
    if (!snapshot && d.u===this.u && signature===this.signature) return 'duplicate';
    if (!Number.isSafeInteger(d.u) || !Number.isSafeInteger(d.seq) || !Number.isFinite(message.cts)) throw new Error('Missing book ordering metadata');
    if (!snapshot && (d.u<=this.u || d.seq<this.seq || message.cts<this.cts)) throw new Error('Non-monotone book update');
    const bids=snapshot?new Map():new Map(this.bids),asks=snapshot?new Map():new Map(this.asks);
    for (const [side, levels] of [[bids,d.b],[asks,d.a]]) {
      if (!Array.isArray(levels)) throw new Error('Missing price levels');
      for (const level of levels) {
        const [p,q]=level.map(Number);
        if (!(Number.isFinite(p)&&p>0&&Number.isFinite(q)&&q>=0)) throw new Error('Invalid price or quantity');
        if (q===0) side.delete(p); else side.set(p,q);
      }
    }
    if (!bids.size || !asks.size || Math.max(...bids.keys())>=Math.min(...asks.keys())) throw new Error('Empty or crossed book');
    this.bids=bids;this.asks=asks;this.u=d.u;this.seq=d.seq;this.cts=message.cts;
    this.received=received;this.signature=signature;this.valid=true;
    return snapshot?'snapshot':'delta';
  }
  snapshot(now) {
    if (!this.valid) return null;
    const bids=[...this.bids].sort((a,b)=>b[0]-a[0]),asks=[...this.asks].sort((a,b)=>a[0]-b[0]);
    const mid=(bids[0][0]+asks[0][0])/2;
    return {t:now,bids,asks,mid,spread_bps:(asks[0][0]-bids[0][0])/mid*10000,book_age_ms:now-this.received,u:this.u,seq:this.seq,cts:this.cts};
  }
}

export function band(book, anchor, bps=25) {
  const lower=anchor*(1-bps/10000),upper=anchor*(1+bps/10000);
  const bidCovered=book.bids.at(-1)[0]<=lower,askCovered=book.asks.at(-1)[0]>=upper;
  const sum=(levels,lo,hi)=>levels.reduce((s,[p,q])=>s+(p>=lo&&p<=hi?p*q:0),0);
  return {bid_covered:bidCovered,ask_covered:askCovered,
    bid_quote:bidCovered?sum(book.bids,lower,anchor):null,
    ask_quote:askCovered?sum(book.asks,anchor,upper):null,
    ask_cost:askCovered&&book.asks[0][0]<=upper?sum(book.asks,0,upper):null,
    target_already_above:book.asks[0][0]>upper};
}

export class TapeEngine {
  constructor(symbols) { this.symbols=symbols;this.generation=0;this.reset(); }
  reset() {
    this.generation++;
    this.states=new Map(this.symbols.map(s=>[s,{book:new Book(),history:[],trades:[],seen:new Map(),warmStart:null}]));
  }
  onMessage(message,now) {
    if (message.topic?.startsWith('orderbook.')) {
      const s=this.states.get(message.data?.s);
      if (!s) throw new Error('Unsubscribed book symbol');
      try {
        const result=s.book.apply(message,now);
        if (result==='snapshot') {s.history=[];s.trades=[];s.seen.clear();s.warmStart=now;}
        return {kind:result};
      } catch(e) {s.book.reset();s.history=[];s.trades=[];s.seen.clear();s.warmStart=null;throw e;}
    }
    if (message.topic?.startsWith('publicTrade.')) {
      let accepted=0,duplicates=0;
      if (!Array.isArray(message.data)) throw new Error('Malformed trade message');
      for (const t of message.data) {
        const s=this.states.get(t.s),p=Number(t.p),q=Number(t.v);
        if (!s || !t.i || !['Buy','Sell'].includes(t.S) || !(p>0&&q>0&&Number.isFinite(p*q)&&Number.isFinite(t.T))) throw new Error('Malformed trade');
        if (s.seen.has(t.i)) {duplicates++;continue;}
        s.seen.set(t.i,now);
        // Keep different trade IDs even when their exchange seq is the same.
        s.trades.push({t:now,exchange_ms:t.T,quote:p*q,side:t.S,rpi:t.RPI===true});accepted++;
      }
      return {kind:'trade',accepted,duplicates};
    }
    return {kind:'control'};
  }
  sample(now) {
    const rows=[];
    for (const [symbol,s] of this.states) {
      const current=s.book.snapshot(now);
      s.trades=s.trades.filter(t=>now-t.t<=120000);
      for (const [id,t] of s.seen) if (now-t>300000) s.seen.delete(id);
      if (!current) {rows.push({symbol,t:now,generation:this.generation,status:'no_snapshot'});continue;}
      // Time since last update is only a conservative freshness gate, not proof of missing data.
      s.history.push(current);s.history=s.history.filter(x=>now-x.t<=62000);
      const anchors=s.history.filter(x=>x.t<=now-60000);
      const anchor=anchors.at(-1);
      const row={symbol,t:now,generation:this.generation,status:'warmup',mid:current.mid,spread_bps:current.spread_bps,
        book_age_ms:current.book_age_ms,exchange_cts:current.cts,update_id:current.u,seq:current.seq,
        bid_levels:current.bids.length,ask_levels:current.asks.length};
      if (!anchor || now-anchor.t>62000) {rows.push(row);continue;}
      const hist=s.history.filter(x=>x.t>=anchor.t);
      if (hist.some((x,i)=>x.book_age_ms>30000||(i>0&&x.t-hist[i-1].t>2500))) {
        rows.push({...row,status:'stale_book_or_sample_gap'});continue;
      }
      const bands=hist.map(x=>band(x,anchor.mid));
      const first=bands[0],last=bands.at(-1);
      const trades=s.trades.filter(t=>t.t>anchor.t&&t.t<=now);
      const buy=trades.filter(t=>!t.rpi&&t.side==='Buy').reduce((s,t)=>s+t.quote,0);
      const sell=trades.filter(t=>!t.rpi&&t.side==='Sell').reduce((s,t)=>s+t.quote,0);
      const rpi=trades.filter(t=>t.rpi).reduce((s,t)=>s+t.quote,0);
      const askCovered=bands.every(b=>b.ask_covered),bidCovered=bands.every(b=>b.bid_covered);
      rows.push({...row,status:askCovered&&bidCovered?'ready':'partial_depth',anchor_mid:anchor.mid,window_ms:now-anchor.t,
        mid_return_60s:current.mid/anchor.mid-1,buy_quote_60s:buy,sell_quote_60s:sell,rpi_quote_60s:rpi,
        net_taker_fraction:buy+sell>0?(buy-sell)/(buy+sell):null,trade_count_60s:trades.length,
        ask_band_covered:askCovered,bid_band_covered:bidCovered,
        ask_depth_change:askCovered&&first.ask_quote>0?last.ask_quote/first.ask_quote-1:null,
        ask_last10s_max_ratio:askCovered&&first.ask_quote>0?Math.max(...bands.filter((_,i)=>hist[i].t>=now-10000).map(b=>b.ask_quote))/first.ask_quote:null,
        bid_recovery:bidCovered&&first.bid_quote>0?(last.bid_quote-Math.min(...bands.map(b=>b.bid_quote)))/first.bid_quote:null,
        ask_cost_to_anchor25bp:last.ask_cost,
        ask_cost_change:first.ask_cost>0&&last.ask_cost!==null?last.ask_cost/first.ask_cost-1:null,
        target_already_above:last.target_already_above});
    }
    return rows;
  }
}
