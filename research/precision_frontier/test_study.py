import unittest
import numpy as np
import pandas as pd
from study import outcomes,replay,windows,select_policy,evidence_gate,candidate_mask,FULL


class FrontierTests(unittest.TestCase):
 def test_strict_lead_and_right_censoring(self):
  t=pd.date_range('2026-04-02',periods=100,freq='h')
  f=pd.DataFrame({'open':100.,'high':101.,'low':99.,'close':100.,'event':False},index=t)
  f.loc[t[10],'event']=True
  o=outcomes(f)
  self.assertFalse(o.iloc[3].hit)
  self.assertTrue(o.iloc[4].hit);self.assertEqual(o.iloc[4].lead_hours,6)
  self.assertTrue(o.iloc[9].hit);self.assertEqual(o.iloc[9].lead_hours,1)
  self.assertFalse(o.iloc[10].hit)
  self.assertTrue(o.iloc[69].label_complete)
  self.assertFalse(o.iloc[70].label_complete)
  f.loc[t[29],'open']=np.nan
  self.assertFalse(outcomes(f).iloc[0].label_complete)

 def test_replay_is_causal_and_cooldown_is_per_coin(self):
  t=pd.date_range('2026-04-02',periods=50,freq='h')
  f=pd.DataFrame({'time':t,'symbol':'A','score':1.,'hit':False})
  self.assertEqual(replay(f,.5),[0,24,48])
  f['hit']=True;f['future_price']=10000
  self.assertEqual(replay(f,.5),[0,24,48])
  f.loc[1,'symbol']='B'
  self.assertEqual(replay(f,.5),[0,1,24,48])

 def test_threshold_uses_greater_equal_and_does_not_rank_future(self):
  d=pd.DataFrame({'time':pd.date_range('2026-04-02',periods=3,freq='h'),
                  'symbol':'A','score':[.4,.5,.99]})
  self.assertEqual(replay(d,.5),[1])

 def test_split_purges_full_outcome_horizon(self):
  d=pd.DataFrame({'time':pd.date_range('2026-05-29',periods=100,freq='h')})
  tr=windows(d,'train');ca=windows(d,'calibration')
  self.assertTrue((tr.time+pd.Timedelta(hours=30)<pd.Timestamp('2026-06-01')).all())
  self.assertTrue(ca.time.ge('2026-06-01').all())

 def test_policy_selection_rejects_tiny_perfect_sample(self):
  rows=[{'policy_id':'tiny','model':'a','threshold':.9,'alerts':2,'all_recall':.1,'precision':1.},
        {'policy_id':'valid','model':'b','threshold':.2,'alerts':35,'all_recall':.2,'precision':.2}]
  self.assertEqual(select_policy(rows),'valid')
  self.assertIsNone(select_policy(rows[:1]))

 def test_candidate_mask_never_uses_outcomes(self):
  d=pd.DataFrame({key:[.2,.2] for key in FULL})
  d['symbol']=['A','BTCUSDT'];d['past_complete']=True;d['hit']=False;d['label_complete']=False
  self.assertEqual(candidate_mask(d).tolist(),[True,False])
  d.hit=True;d.label_complete=True
  self.assertEqual(candidate_mask(d).tolist(),[True,False])

 def test_eighty_percent_gate_requires_support(self):
  m={'precision':.9,'alerts':100,'all_recall':.2,'active_weeks':8,'block_lower':.81,'iid_lower':.82}
  self.assertTrue(evidence_gate(m))
  for k,v in [('alerts',10),('all_recall',.01),('active_weeks',2),('block_lower',.7),('iid_lower',.7)]:
   self.assertFalse(evidence_gate({**m,k:v}))


if __name__=='__main__':unittest.main()
