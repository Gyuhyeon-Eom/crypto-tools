import unittest
import numpy as np
import pandas as pd
from study import features,choose_controls,compare,FEATURES


class HardControlTests(unittest.TestCase):
 def raw(self):
  return pd.DataFrame({'time':pd.date_range('2026-03-01',periods=1000,freq='h'),
                       'symbol':'TEST','open':100.,'high':102.,'low':99.,'close':101.,'turnover':100.})

 def test_future_changes_do_not_change_features(self):
  a=self.raw();b=a.copy();b.loc[801:,['close','high','turnover']]=500.
  pd.testing.assert_frame_equal(features(a).iloc[:801][FEATURES],features(b).iloc[:801][FEATURES])

 def test_volume_acceleration_disjoint_windows(self):
  d=self.raw();d.loc[799:800,'turnover']=200
  self.assertAlmostEqual(features(d).iloc[800].volume_acceleration,np.log(2))

 def test_flat_path_missing_and_monotone_path_efficiency(self):
  d=self.raw();self.assertTrue(np.isnan(features(d).iloc[800].price_efficiency))
  d['close']=100*np.exp(np.arange(len(d))*.001)
  self.assertAlmostEqual(features(d).iloc[800].price_efficiency,1.)

 def test_retention_compares_closed_candle_giveback(self):
  d=self.raw();d.loc[798:800,'close']=102.
  self.assertAlmostEqual(features(d).iloc[800].close_retention_change,1/3)

 def test_calipers_and_distinct_days(self):
  times=pd.to_datetime(['2026-05-20 10:00','2026-05-17 10:00','2026-05-17 11:00','2026-05-23 10:00','2026-05-26 10:00','2026-05-29 10:00'])
  d=pd.DataFrame({'time':times,'symbol':'TEST','past_complete':True,'future_complete':True,
                  'prior_runup':.10,'volume_build':2.,'near_high':.98,'match_vol':.01,
                  'match_liquidity':10000.,'max24':.01,'event_nearby':False})
  d.loc[3,'volume_build']=4.
  d.loc[4,'prior_runup']=.20
  c=next(d.iloc[:1].itertuples(index=False));p=choose_controls(c,d,np.array([1.,1.]))
  self.assertEqual(len(p),2)
  self.assertEqual(p.time.dt.floor('D').nunique(),2)
  self.assertNotIn(times[3],p.time.tolist());self.assertNotIn(times[4],p.time.tolist())

 def test_matching_does_not_depend_on_test_features(self):
  d=pd.DataFrame({'time':pd.to_datetime(['2026-05-20','2026-05-17','2026-05-23']),
                  'symbol':'TEST','past_complete':True,'future_complete':True,'prior_runup':.1,
                  'volume_build':2.,'near_high':.98,'match_vol':.01,'match_liquidity':10000.,
                  'max24':.01,'event_nearby':False,'volume_acceleration':[0.,100.,-100.]})
  c=next(d.iloc[:1].itertuples(index=False));a=choose_controls(c,d,np.ones(2))
  d.volume_acceleration=-d.volume_acceleration
  b=choose_controls(c,d,np.ones(2))
  self.assertEqual(a.time.tolist(),b.time.tolist())

 def test_each_case_equal_weight_despite_number_of_controls(self):
  d=pd.DataFrame({'group':['a','a','b','b','b','b'],'anchor':pd.Timestamp('2026-05-20'),
                  'role':['case','control','case','control','control','control'],
                  'volume_acceleration':[2.,1.,0.,1.,1.,1.]})
  r,_=compare(d,'volume_acceleration');self.assertEqual(r['concordance'],.5)
  d.loc[5,'volume_acceleration']=np.nan
  r,_=compare(d,'volume_acceleration');self.assertEqual(r['n'],1)


if __name__=='__main__':unittest.main()
