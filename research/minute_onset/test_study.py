import unittest
import numpy as np
import pandas as pd
from study import build_features,label,split


class MinuteTests(unittest.TestCase):
    def test_event_anchor_and_pre_event_lead(self):
        t=pd.date_range('2026-04-01',periods=100,freq='5min')
        f=pd.DataFrame({'close':100.,'high':100.,'low':99.,'known':True,'active':True,'eligible_price':True},index=t)
        f.loc[t[40]:,'close']=103.;f.loc[t[40]:,'high']=103.;f.loc[t[44],'high']=109.
        z=label(f)
        self.assertTrue(z.event.loc[t[39]])
        self.assertFalse(z.event.loc[t[40]])
        self.assertEqual(z.lead_minutes.loc[t[27]],60)
        self.assertTrue(z.strict_hit.loc[t[38]])
        self.assertFalse(z.strict_hit.loc[t[39]])
        self.assertFalse(z.hit6.loc[t[26]])
        self.assertTrue(z.delay5_hit.loc[t[26]])
        self.assertFalse(z.label_complete.iloc[-1])

    def test_missing_future_file_cannot_produce_event(self):
        t=pd.date_range('2026-04-01',periods=100,freq='5min')
        f=pd.DataFrame({'close':100.,'high':100.,'low':99.,'known':True,'active':True,'eligible_price':True},index=t)
        f.loc[t[40]:,'close']=103.;f.loc[t[40]:,'high']=103.;f.loc[t[44],'high']=109.
        f.loc[t[45],'known']=False
        self.assertEqual(label(f).event.sum(),0)

    def fixtures(self):
        ht=pd.date_range('2026-03-01',periods=840,freq='h')
        h=pd.DataFrame({'time':ht,'close':100.,'turnover':12000.})
        bt=pd.date_range('2026-04-01',periods=288*3,freq='5min')
        b=pd.DataFrame({'bucket':bt,'open':100.,'high':100.2,'low':99.8,'close':100.,'quote':1000.,'buy':550.,'sell':450.,'trades':10,
                        'last_ms':bt.as_unit('ms').astype('int64')+299999})
        return h,b,list(pd.date_range('2026-04-01','2026-04-03'))

    def test_exact_normalization_and_causal_hourly_context(self):
        h,b,days=self.fixtures();f=build_features(b,h,days)
        t=pd.Timestamp('2026-04-02 00:05')
        self.assertTrue(f.quality.loc[t])
        for k in ['quote_accel','count_accel','size_accel','net_change','range_accel','price_accel']:
            self.assertAlmostEqual(f.loc[t,k],0)
        # A future hourly candle must not affect the00:05 observation.
        changed=h.copy();changed.loc[changed.time>t,'turnover']=1e12
        z=build_features(b,changed,days)
        pd.testing.assert_series_equal(f.loc[t],z.loc[t])

    def test_future_ticks_do_not_change_prior_features(self):
        h,b,days=self.fixtures();f=build_features(b,h,days)
        cutoff=pd.Timestamp('2026-04-02 00:05')
        changed=b.copy();changed.loc[changed.bucket>=cutoff,['quote','buy','sell','close']]=[1e8,9e7,1e7,200.]
        z=build_features(changed,h,days)
        pd.testing.assert_frame_equal(f.loc[:cutoff],z.loc[:cutoff])

    def test_missing_past_file_is_not_quiet_trading(self):
        h,b,days=self.fixtures();days.remove(pd.Timestamp('2026-04-02'))
        f=build_features(b,h,days)
        self.assertFalse(f.quality.loc['2026-04-02 12:00'])

    def test_discovery_purge(self):
        d=pd.DataFrame({'time':pd.date_range('2026-06-30',periods=288,freq='5min')})
        self.assertTrue((split(d,'discovery').time+pd.Timedelta(hours=2)<pd.Timestamp('2026-07-01')).all())


if __name__=='__main__':
    unittest.main()
