import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from dataset import aggregate, prices, legacy, MICRO, TARGETS
from study import register, fit, evaluate, MODELS


def tape(at='2026-10-10 03:00'):
    end=pd.Timestamp(at)
    times=pd.date_range(end-pd.Timedelta(hours=1)+pd.Timedelta(milliseconds=500),periods=3600,freq='s')
    return pd.DataFrame({'received_time':times,'run':'test','connection':1,'generation':1,'status':'ready',
                         'ask_depth_change':-.5,'bid_recovery':.3,'ask_cost_change':-.1,'mid_return_60s':.002,
                         'spread_bps':2.,'buy_quote_60s':100.,'sell_quote_60s':0.,'net_taker_fraction':1.})


def candles():
    times=pd.date_range('2026-01-01',periods=820,freq='h')
    raw=pd.DataFrame({'time':times,'symbol':'TESTUSDT','open':100.,'high':100.,'low':100.,'close':100.,'turnover':1000.})
    raw.loc[751:,'close']=106.;raw.loc[751:,'high']=106.;raw.loc[751,'high']=120.
    return raw


class DatasetTests(unittest.TestCase):
    def test_fixed_causal_aggregation(self):
        x=tape();at=pd.Timestamp('2026-10-10 03:00');before=aggregate(x,at)
        self.assertTrue(before['micro_eligible']);self.assertAlmostEqual(before['ask_depletion_buy_60m'],.5)
        future=x.iloc[[-1]].copy();future.received_time=at+pd.Timedelta(milliseconds=1);future.ask_depth_change=-100
        self.assertEqual(before,aggregate(pd.concat([x,future]),at))

    def test_short_sample_is_not_a_full_hour(self):
        r=aggregate(tape().tail(180),'2026-10-10 03:00')
        self.assertFalse(r['micro_eligible']);self.assertEqual(r['micro_reason'],'coverage_5m')

    def test_small_missing_fraction_still_rejects_long_gap(self):
        x=tape().drop(index=range(1000,1010))
        self.assertEqual(aggregate(x,'2026-10-10 03:00')['micro_reason'],'gap_60m')

    def test_reset_boundaries_not_stitched(self):
        x=tape();x.loc[3599,'generation']=2
        self.assertEqual(aggregate(x,'2026-10-10 03:00')['micro_reason'],'reset_or_run_boundary')

    def test_zero_flow_and_unknown_flow_are_distinct(self):
        x=tape();x['buy_quote_60s']=0.;x['net_taker_fraction']=np.nan
        r=aggregate(x,'2026-10-10 03:00');self.assertTrue(r['micro_eligible']);self.assertEqual(r['no_trade_60m'],1.)
        x['buy_quote_60s']=100.
        self.assertFalse(aggregate(x,'2026-10-10 03:00')['micro_eligible'])

    def test_exact_legacy_event_and_lead_labels(self):
        raw=candles();d=prices(raw,raw.time.max()).set_index('time')
        expected=legacy.outcomes(legacy.hard.census.labels(legacy.hard.features(raw)))
        mature=d.label_status.eq('mature')
        self.assertEqual(d.loc[mature,'hit'].astype(bool).tolist(),expected.loc[mature,'hit'].tolist())
        self.assertTrue(d.iloc[744].hit);self.assertTrue(d.iloc[749].hit)
        self.assertFalse(d.iloc[743].hit);self.assertFalse(d.iloc[750].hit)
        self.assertEqual(d.iloc[744].lead_hours,6.)

    def test_future_and_missing_labels_remain_null(self):
        raw=candles();d=prices(raw,raw.time.iloc[749])
        self.assertEqual(d.iloc[-1].label_status,'pending_time');self.assertTrue(pd.isna(d.iloc[-1].hit))
        missing=prices(raw.drop(index=760),raw.time.max())
        self.assertEqual(missing.iloc[744].label_status,'missing_price');self.assertTrue(pd.isna(missing.iloc[744].hit))
        self.assertFalse(missing.iloc[700].hit)

    def test_future_candles_cannot_change_current_context(self):
        raw=candles();cut=raw.time.iloc[740]
        a=prices(raw,cut);raw.loc[741:,'turnover']=1e9;b=prices(raw,cut)
        pd.testing.assert_frame_equal(a,b)


def model_fixture(reg):
    times=pd.date_range(reg['train'][0],reg['evaluation'][1],freq='h',inclusive='left')
    d=pd.MultiIndex.from_product([TARGETS,times],names=['symbol','time']).to_frame(index=False)
    i=np.arange(len(d));y=i%10==0
    d['base_candidate']=True;d['micro_eligible']=True;d['label_status']='mature';d['label_complete']=True
    d['hit']=y;d['target_event']=d.time.where(y)+pd.Timedelta(hours=1)
    d['lead_hours']=np.where(y,1.,np.nan);d['event']=i%10==1;d['return24']=.01;d['min24']=-.01
    for j,col in enumerate(legacy.CONTEXT):d[col]=np.sin(i/(j+1))
    for j,col in enumerate(MICRO):d[col]=np.cos(i/(j+1))
    return d


class StudyTests(unittest.TestCase):
    def setUp(self):
        # Tiny synthetic universe exercises mechanics only, not statistical evidence.
        self.universe = patch('study.UNIVERSE', TARGETS)
        self.universe.start()
        self.addCleanup(self.universe.stop)

    def test_registration_requires_future_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'reg.json'
            with self.assertRaises(ValueError):register('2026-01-01',path,now='2026-01-01')
            reg=register('2026-01-02',path,now='2026-01-01')
            self.assertEqual(pd.Timestamp(reg['evaluation'][1])-pd.Timestamp(reg['evaluation'][0]),pd.Timedelta(days=56))
            with self.assertRaises(FileExistsError):register('2026-01-02',path,now='2026-01-01')

    def test_small_data_cannot_fit_or_report_precision(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg=register('2026-01-02',Path(tmp)/'reg.json',now='2026-01-01')
            d=model_fixture(reg).iloc[:10]
            lock=fit(d,reg,now=pd.Timestamp(reg['evaluation'][0])-pd.Timedelta(hours=1))
            self.assertEqual(lock['status'],'insufficient_data');self.assertEqual(lock['models'],{})
            self.assertIsNone(evaluate(d,lock,now='2027-01-01')['precision'])

    def test_fit_excludes_test_labels_and_maturity_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg=register('2026-01-02',Path(tmp)/'reg.json',now='2026-01-01');d=model_fixture(reg)
            now=pd.Timestamp(reg['evaluation'][0])-pd.Timedelta(hours=1)
            lock=fit(d,reg,now=now);self.assertEqual(lock['status'],'locked')
            changed=d.copy();changed.loc[changed.time.ge(reg['evaluation'][0]),'hit']=False
            second=fit(changed,reg,now=now);self.assertEqual(lock['models'],second['models'])
            self.assertEqual(evaluate(d,lock,now=reg['evaluation'][1])['status'],'pending_outcomes')
            result=evaluate(d,lock,now=reg['evaluation_mature_at'])
            self.assertEqual(result['status'],'evaluated');self.assertEqual(set(result['models']),set(MODELS))
            late=fit(d,reg,now=reg['evaluation'][0]);self.assertEqual(late['status'],'insufficient_data')


if __name__=='__main__':unittest.main()
