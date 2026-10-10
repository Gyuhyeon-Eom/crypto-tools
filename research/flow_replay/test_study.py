import unittest
import gzip
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from study import flow_features,price_features,add_outcomes,replay,comparisons
import forward


class ReplayTests(unittest.TestCase):
    def bars(self):
        t=pd.date_range('2026-08-01',periods=288,freq='5min')
        buy=np.full(288,20.);buy[-24:]=80.
        return pd.DataFrame({'bucket':t,'quote':100.,'buy':buy,'sell':100-buy,
                             'last_ms':t.as_unit('ms').astype('int64')+299999})

    def test_exact_flow_windows_and_future_exclusion(self):
        b=self.bars();start=pd.Timestamp('2026-08-01');end=start+pd.Timedelta(days=1)
        f=flow_features(b,[start],start,end)
        self.assertAlmostEqual(f.loc[end,'net2'],.6)
        self.assertAlmostEqual(f.loc[end,'buy_shift'],1.2)
        self.assertAlmostEqual(f.loc[end,'raw_quote24'],28800)
        extra=b.copy();extra.bucket+=pd.Timedelta(days=1);extra['buy']=1e9
        pd.testing.assert_frame_equal(f,flow_features(pd.concat([b,extra]),[start],start,end))

    def test_missing_file_is_not_zero_activity(self):
        b=self.bars();start=pd.Timestamp('2026-08-01');end=start+pd.Timedelta(days=1)
        f=flow_features(b,[],start,end)
        self.assertFalse(f.loc[end,'files_complete'])

    def test_price_features_prefix_invariance(self):
        t=pd.date_range('2026-03-01',periods=1000,freq='h')
        h=pd.DataFrame({'time':t,'open':100.,'high':101.,'low':99.,'close':100.,'turnover':100000.,'symbol':'X'})
        short=price_features(h.iloc[:800]);full=price_features(h)
        pd.testing.assert_frame_equal(short,full.loc[short.index])

    def test_label_maturity_and_stop_first(self):
        t=pd.date_range('2026-08-01',periods=200,freq='h')
        h=pd.DataFrame({'open':100.,'high':100.,'low':99.,'close':100.},index=t)
        h.loc[t[49],'high']=131.;h.loc[t[49],'low']=89.
        f=add_outcomes(h)
        self.assertEqual(f.y.iloc[0],1);self.assertEqual(f.y_clean.iloc[0],0)
        self.assertEqual(f.next_open_y.iloc[0],1);self.assertEqual(f.next_open_clean.iloc[0],0)
        self.assertEqual(f.lead_hours.iloc[0],49)
        self.assertFalse(f.mature.iloc[-1])
        h.loc[t[1],'high']=111.
        self.assertEqual(add_outcomes(h).y.iloc[0],0)

    def test_causal_daily_cap_and_cooldown(self):
        start=pd.Timestamp('2026-07-31')
        d=pd.DataFrame({'symbol':['A','B','C','D','A','A'],
                        'time':[start,start,start,start+pd.Timedelta(hours=4),start+pd.Timedelta(days=1),start+pd.Timedelta(days=7)],
                        'buy_shift':[.2,.3,.4,.99,.99,.2],'trigger':True})
        a=replay(d)
        self.assertEqual(a.symbol.tolist(),['C','B','A','A'])
        self.assertEqual(a.time.iloc[-1],start+pd.Timedelta(days=7))

    def test_random_comparison_does_not_fake_unfillable_slots(self):
        t=pd.Timestamp('2026-08-01')
        d=pd.DataFrame({'time':[t,t+pd.Timedelta(days=1)],'symbol':['A','A'],'y':[0,1],'volume_ratio6':[1.,2.]})
        _,e,r=comparisons(d,d,repetitions=10)
        self.assertEqual(e.expected_count.tolist(),[1.,0.])
        self.assertEqual(r[-1]['mean_alerts'],1.)

    def test_no_signals_is_a_valid_result(self):
        t=pd.Timestamp('2026-08-01')
        d=pd.DataFrame({'time':[t],'symbol':['A'],'y':[0],'volume_ratio6':[1.]})
        simple,e,r=comparisons(d,d.iloc[:0],repetitions=10)
        self.assertEqual(len(simple),0);self.assertEqual(len(e),0)
        self.assertIn('period',e.columns)
        self.assertEqual(r[-1]['mean_alerts'],0)

    def test_future_cache_filters_unfinished_boundary_and_duplicates(self):
        first=int((forward.START-pd.Timedelta(days=31)).timestamp()*1000)
        end=int(forward.MATURE.timestamp()*1000)
        rows=[[ms,'100','101','99','100','1','100'] for ms in [first-3600000,first,first,end-3600000,end]]
        with tempfile.TemporaryDirectory() as temp,patch.object(forward,'DEST',Path(temp)):
            cache=Path(temp)/'hourly_cache';cache.mkdir()
            with gzip.open(cache/'TEST.json.gz','wt') as f:
                json.dump({'rows':rows},f)
            h=forward.collect_hourly('TEST')
            self.assertEqual(len(h),2)
            self.assertEqual(h.time.max(),forward.MATURE)

    def test_future_run_refuses_changed_frozen_code(self):
        with patch.object(forward,'freeze',return_value={'source_hashes':{'wrong':'hash'}}):
            with self.assertRaisesRegex(RuntimeError,'Frozen research code'):
                forward.run()


if __name__=='__main__':
    unittest.main()
