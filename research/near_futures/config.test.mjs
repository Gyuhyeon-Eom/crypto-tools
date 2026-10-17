import test from 'node:test';
import assert from 'node:assert/strict';
import {profile,symbols,parseArgs,isTarget,marketIdentity,observationText,checkInstruments} from './config.mjs';

test('collection is fixed to linear NEAR with BTC/ETH as references only',()=>{
  assert.equal(profile.websocket,'wss://stream.bybit.com/v5/public/linear');
  assert.deepEqual(symbols,['NEARUSDT','BTCUSDT','ETHUSDT']);
  assert.deepEqual(symbols.filter(isTarget),['NEARUSDT']);
  assert.equal(marketIdentity().category,'linear');
  assert.deepEqual(profile.price_return_targets,[.05,.10]);
});
test('default cannot send Slack; reject invalid durations, symbol overrides and conflicting flags',()=>{
  assert.deepEqual(parseArgs([]),{seconds:180,slackEnabled:false,signalsEnabled:false});
  assert.deepEqual(parseArgs(['86400','--slack','--signals']),{seconds:86400,slackEnabled:true,signalsEnabled:true});
  for(const args of [['0'],['86401'],['NaN'],['2.5'],['180','NEARUSDT'],['180','--slack','--dry-run']])
    assert.throws(()=>parseArgs(args));
});
test('instrument lookup rejects wrong market, inactive contract, errors and missing symbol',async()=>{
  const good=symbol=>({symbol,contractType:'LinearPerpetual',status:'Trading',settleCoin:'USDT'});
  const mock=transform=>async url=>{
    assert.equal(new URL(url).searchParams.get('category'),'linear');
    const symbol=new URL(url).searchParams.get('symbol');
    return {ok:true,json:async()=>({retCode:0,result:{list:[transform(good(symbol))]}})};
  };
  assert.equal((await checkInstruments(mock(x=>x))).length,3);
  for(const change of [{contractType:'Spot'},{status:'Settled'},{settleCoin:'USDC'},{symbol:'OTHERUSDT'}])
    await assert.rejects(checkInstruments(mock(x=>({...x,...change}))));
  await assert.rejects(checkInstruments(async()=>({ok:false,status:503})));
  await assert.rejects(checkInstruments(async()=>({ok:true,json:async()=>({retCode:10001})})));
});
test('observations explicitly identify futures and unvalidated price targets',()=>{
  const text=observationText({},()=> '[Orderbook v1 · 실험용 관찰] NEARUSDT');
  assert.match(text,/NEAR 선물/);assert.match(text,/\+5%\/\+10% 도달 예측 미검증/);
});
