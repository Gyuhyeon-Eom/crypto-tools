import unittest
import numpy as np
import pandas as pd
from analyze import lagged_derivatives,symbol_features,label_symbol,replay


def fixtures(n=1100):
    times=pd.date_range('2026-03-01 01:00',periods=n,freq='h')
    rng=np.random.default_rng(7)
    close=100*np.exp(np.cumsum(rng.normal(0,.002,n)))
    y=pd.DataFrame({'time':times,'symbol':'TESTUSDT','open':close,'close':close,
                    'high':close*1.01,'low':close*.99,'turnover':rng.uniform(30000,70000,n)})
    b=y.copy();b['buy_turnover']=b.turnover*rng.uniform(.3,.7,n)
    oi=pd.DataFrame({'time':times[::4],'oi':np.arange(len(times[::4]))+10000})
    funding=pd.DataFrame({'time':times[::8],'funding':.0001})
    return y,b,oi,funding


class Integrity(unittest.TestCase):
    def test_oi_delayed_and_stale_values_unavailable(self):
        index=pd.date_range('2026-01-01',periods=20,freq='h')
        oi=pd.DataFrame({'time':[index[0],index[4]],'oi':[100,200]})
        f=lagged_derivatives(index,oi,pd.DataFrame())
        self.assertTrue(pd.isna(f.iloc[0].oi))
        self.assertEqual(f.iloc[4].oi,100)
        self.assertEqual(f.iloc[5].oi,200)
        self.assertTrue(pd.isna(f.iloc[14].oi))

    def test_future_observations_cannot_change_features(self):
        y,b,oi,funding=fixtures()
        cutoff=y.time.iloc[990]
        full=symbol_features(y,b,oi,funding).set_index('time')
        prefix=symbol_features(*(d[d.time<=cutoff] for d in [y,b,oi,funding])).set_index('time')
        pd.testing.assert_frame_equal(prefix,full.loc[prefix.index],check_exact=False,rtol=1e-12,atol=1e-12)

    def test_funding_intervals_normalized_by_actual_time(self):
        index=pd.date_range('2026-01-01',periods=110,freq='h')
        f4=pd.DataFrame({'time':index[::4],'funding':.0001})
        f8=pd.DataFrame({'time':index[::8],'funding':.0002})
        a=lagged_derivatives(index,pd.DataFrame(),f4)
        b=lagged_derivatives(index,pd.DataFrame(),f8)
        self.assertAlmostEqual(a.iloc[96].funding24,.0006)
        self.assertAlmostEqual(a.iloc[96].funding24,b.iloc[96].funding24)
        self.assertTrue(pd.isna(a.iloc[0].funding24))

    def test_label_requires_quiet_lead_and_conservative_stop_order(self):
        y,_,_,_=fixtures(1000)
        y[['open','close']]=100.;y['high']=101.;y['low']=99.
        y.loc[786,'high']=140.;y.loc[786,'low']=85.
        a=label_symbol(y)
        self.assertEqual(a.loc[750,'y'],1)
        self.assertEqual(a.loc[750,'first_hit_hour'],36)
        self.assertEqual(a.loc[750,'y_clean'],0)
        y.loc[760,'high']=112.
        b=label_symbol(y)
        self.assertEqual(b.loc[750,'y'],0)

    def test_missing_future_hour_invalidates_outcome(self):
        y,_,_,_=fixtures(1000)
        y.loc[770,'high']=np.nan
        labels=label_symbol(y)
        self.assertFalse(labels.loc[750,'mature'])

    def test_daily_cap_is_causal_and_cooldown_is_elapsed_hours(self):
        d=pd.DataFrame({'time':pd.to_datetime(['2026-01-01 00:00','2026-01-01 04:00','2026-01-02 00:00','2026-01-08 00:00']),
                        'symbol':['A','B','A','A']})
        selected=replay(d,[True]*4,[.2,.9,.9,.3],cap=1)
        self.assertEqual(selected.index.tolist(),[0,3])


if __name__=='__main__':unittest.main()
