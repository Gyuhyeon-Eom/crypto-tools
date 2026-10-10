import unittest
import numpy as np
import pandas as pd
from study import features,label_events,replay,window


class OnsetTests(unittest.TestCase):
    def test_anchor_precedes_breakout_hour_and_strict_lead(self):
        t=pd.date_range('2026-04-01',periods=100,freq='h')
        f=pd.DataFrame({'close':100.,'high':100.,'low':99.,'eligible_price':True},index=t)
        f.loc[t[50]:,'close']=105.;f.loc[t[50]:,'high']=105.
        f.loc[t[52],'high']=116.
        z=label_events(f)
        self.assertTrue(z.event.loc[t[49]])
        self.assertFalse(z.event.loc[t[50]])
        self.assertTrue(z.hit6.loc[t[43]])
        self.assertEqual(z.lead_hours.loc[t[43]],6)
        self.assertFalse(z.hit6.loc[t[42]])
        self.assertFalse(z.hit6_strict.loc[t[49]])
        self.assertTrue(z.hit6_strict.loc[t[48]])
        self.assertFalse(z.label_complete.iloc[-1])

    def test_qualifying_breakout_without_followthrough_is_not_event(self):
        t=pd.date_range('2026-04-01',periods=100,freq='h')
        f=pd.DataFrame({'close':100.,'high':100.,'low':99.,'eligible_price':True},index=t)
        f.loc[t[50]:,'close']=106.;f.loc[t[50]:,'high']=106.
        self.assertEqual(label_events(f).event.sum(),0)

    def test_future_trades_and_prices_do_not_change_features(self):
        t=pd.date_range('2026-03-01',periods=840,freq='h')
        h=pd.DataFrame({'time':t,'symbol':'TEST','close':100.,'open':100.,'high':101.,'low':99.,'turnover':12000.})
        bt=pd.date_range(t.min()-pd.Timedelta(hours=1),t.max(),freq='5min',inclusive='left')
        b=pd.DataFrame({'bucket':bt,'quote':1000.,'buy':550.,'sell':450.,'trades':10,
                        'last_ms':bt.as_unit('ms').astype('int64')+299999})
        cutoff=t[800]
        days=list(pd.date_range(bt.min().floor('D'),bt.max().floor('D')))
        full=features(h,b,days)
        short=features(h[h.time<=cutoff],b[b.bucket<cutoff],days)
        pd.testing.assert_frame_equal(full.loc[short.index],short)

    def test_day_cap_cannot_replace_early_signals_and_24h_cooldown(self):
        t=pd.Timestamp('2026-08-01')
        d=pd.DataFrame({'time':[t,t,t,t+pd.Timedelta(hours=1),t+pd.Timedelta(hours=23),t+pd.Timedelta(hours=24)],
                        'symbol':['A','B','C','D','A','A']})
        a=replay(d,np.ones(6,dtype=bool),np.array([1,2,3,99,99,1]))
        self.assertEqual(a.symbol.tolist(),['C','B','A','A'])
        self.assertEqual(a.time.iloc[-1],t+pd.Timedelta(hours=24))

    def test_discovery_future_label_does_not_cross_split(self):
        d=pd.DataFrame({'time':pd.date_range('2026-06-29',periods=48,freq='h')})
        z=window(d,'discovery')
        self.assertTrue((z.time+pd.Timedelta(hours=30)<pd.Timestamp('2026-07-01')).all())


if __name__=='__main__':
    unittest.main()
