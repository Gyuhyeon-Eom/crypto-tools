import unittest
import numpy as np
import pandas as pd
from study import tape_features,gap_median,impact,comparison


class TapeTests(unittest.TestCase):
    def raw(self):
        start=pd.Timestamp('2026-08-01')
        times=start.value//1000000+np.arange(600)*6000
        return pd.DataFrame({'timestamp':times,'price':100.,'volume':1.,'side':np.where(np.arange(600)%2,'sell','buy')}),start+pd.Timedelta(hours=1)

    def test_anchor_trade_is_excluded_and_future_is_invariant(self):
        x,t=self.raw();f=tape_features(x,t,60000.)
        self.assertNotIn('quality_error',f)
        extra=pd.DataFrame({'timestamp':[t.value//1000000,t.value//1000000+1],'price':[1000.,2000.],'volume':[1e6,1e6],'side':['buy','buy']})
        g=tape_features(pd.concat([x,extra]),t,60000.)
        self.assertEqual(f['raw_trades'],600);self.assertEqual(f['quote60'],60000.)
        for key in f:
            if isinstance(f[key],float) and np.isnan(f[key]):self.assertTrue(np.isnan(g[key]))
            else:self.assertEqual(f[key],g[key])

    def test_flow_order_is_measured_in_correct_windows(self):
        x,t=self.raw();x.loc[450:549,'side']='sell';x.loc[550:,'side']='buy'
        f=tape_features(x,t,60000.)
        self.assertAlmostEqual(f['buy_acceleration'],2.)
        self.assertAlmostEqual(f['buy_persistence'],1/3)

    def test_timestamp_batches_do_not_create_zero_gap(self):
        timestamps=np.repeat(np.arange(20)*100,4)
        self.assertEqual(gap_median(pd.DataFrame({'timestamp':timestamps})),100.)

    def test_no_sell_minutes_is_unavailable_not_absorption(self):
        b=pd.DataFrame({'buy':[10.]*10,'sell':[0.]*10,'quote':[10.]*10,'ret':[.01]*10})
        self.assertTrue(np.isnan(impact(b,3)))

    def test_sparse_or_bad_quote_windows_fail_quality(self):
        x,t=self.raw();self.assertIn('quality_error',tape_features(x,t,30000.))
        self.assertIn('quality_error',tape_features(x.iloc[445:],t,15500.))

    def test_missing_feature_drops_whole_matched_set(self):
        d=pd.DataFrame({'group':['a']*4+['b']*4,'role':['case','control','control','control']*2,'offset':0,
                        'anchor':[pd.Timestamp('2026-08-01')]*8,'feature':[2,1,2,3,10,0,np.nan,0]})
        r=comparison(d,'feature');self.assertEqual(r['n'],1);self.assertEqual(r['concordance'],.5)

    def test_known_inactive_minutes_are_not_missing_data(self):
        x,t=self.raw()
        # Trade only every other minute, with a complete reconciled source window.
        minute=((x.timestamp-x.timestamp.min())//60000).astype(int)
        x=x[minute%2==1]
        f=tape_features(x,t,float((x.price*x.volume).sum()))
        self.assertNotIn('quality_error',f)
        self.assertLess(f['baseline_active'],.8)
        self.assertAlmostEqual(f['buy_acceleration'],0.)


if __name__=='__main__':
    unittest.main()
