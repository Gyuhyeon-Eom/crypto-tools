import unittest
import numpy as np
import pandas as pd
from study import price_features,labels,match


class CensusTests(unittest.TestCase):
    def raw(self):
        return pd.DataFrame({'time':pd.date_range('2026-03-01',periods=1100,freq='h'),
                             'symbol':'TEST','open':100.,'high':101.,'low':99.,'close':100.,'turnover':100.})

    def test_pre_features_unchanged_by_future_prices(self):
        a=self.raw();b=a.copy();b.loc[901:,['close','high','turnover']]=999
        fa,fb=price_features(a),price_features(b)
        cols=['volume_build','compression','near_high','prior_runup','ret6','recent_spike','match_vol','match_liquidity']
        pd.testing.assert_frame_equal(fa.iloc[:901][cols],fb.iloc[:901][cols])

    def test_volume_baseline_excludes_recent_six_hours(self):
        d=self.raw();d.loc[795:800,'turnover']=200.
        f=price_features(d)
        self.assertEqual(f.iloc[800].volume_build,2.)

    def test_event_anchor_before_first_jump_and_no_volume_gate(self):
        d=self.raw();d.loc[801,'close']=106.;d.loc[801,'high']=116.
        f=labels(price_features(d))
        self.assertTrue(f.iloc[800].event)
        self.assertEqual(f.iloc[800].volume_build,1.)
        self.assertFalse(f.iloc[799].event)
        self.assertFalse(f.iloc[801].event)

    def test_peak_alone_not_event_and_future_missing_not_labelled(self):
        d=self.raw();d.loc[801,'high']=120.
        self.assertFalse(labels(price_features(d)).event.any())
        d.loc[801,'close']=106.;d.loc[810,'close']=np.nan
        self.assertFalse(labels(price_features(d)).iloc[800].event)

    def test_refractory_prevents_repeated_surge_count(self):
        d=self.raw()
        for i in [801,810,827]:
            d.loc[i,'close']=106.;d.loc[i,'high']=116.
        f=labels(price_features(d))
        self.assertEqual(np.flatnonzero(f.event).tolist(),[800,826])

    def test_controls_same_coin_hour_and_not_near_event(self):
        times=pd.to_datetime(['2026-04-20 10:00','2026-04-17 10:00','2026-04-23 10:00','2026-04-26 10:00','2026-04-29 10:00'])
        d=pd.DataFrame({'time':times,'symbol':'TEST','event':[True,False,False,False,False],
                        'past_complete':True,'future_complete':True,'event_nearby':[True,False,False,False,True],
                        'max24':0.,'match_vol':.01,'match_liquidity':10000.,'context':'flat'})
        m,drops=match(d,'discovery',np.array([1.,1.]))
        self.assertFalse(drops)
        self.assertEqual(len(m),4)
        self.assertTrue(m.time.dt.hour.eq(10).all())
        self.assertNotIn(times[-1],m.time.tolist())


if __name__=='__main__':unittest.main()
