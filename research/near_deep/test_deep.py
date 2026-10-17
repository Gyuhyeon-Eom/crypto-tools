import unittest
import numpy as np
import pandas as pd
from data import zigzag, precursor_labels, touch_outcomes, fold_windows, split_index, HOUR, make_features, PRICE, DERIV
from evaluate import replay, calibrate
from stops import levels, evaluate_stop

class DeepTests(unittest.TestCase):
    def frame(self,n=400):
        return pd.DataFrame(100.,index=pd.date_range('2025-01-01',periods=n,freq='15min'),columns=['open','high','low','close'])
    def test_all_features_are_causal_including_derivatives(self):
        import copy
        n=900;ms=1735689600000;times=ms+np.arange(n)*900000
        prices=100+np.sin(np.arange(n)/10)
        rows=[[int(t),p,p+.1,p-.1,p,100,10000] for t,p in zip(times,prices)]
        raw={'start':ms,'end':int(ms+n*900000),'candles':{s:copy.deepcopy(rows) for s in ('NEARUSDT','BTCUSDT','ETHUSDT')},
             'oi':[{'timestamp':str(t),'openInterest':str(100+i)} for i,t in enumerate(times)],
             'funding':[{'fundingRateTimestamp':str(times[0]),'fundingRate':'.0001'}], 'mark':[r[:5] for r in rows]}
        a=make_features(raw);changed=copy.deepcopy(raw)
        for symbol in changed['candles']:
            for row in changed['candles'][symbol][800:]:row[1:]=[v*2 for v in row[1:]]
        for row in changed['mark'][800:]:row[1:]=[v*3 for v in row[1:]]
        for row in changed['oi'][800:]:row['openInterest']='999999'
        changed['funding'].append({'fundingRateTimestamp':str(times[800]),'fundingRate':'.5'})
        b=make_features(changed)
        pd.testing.assert_frame_equal(a.iloc[:800][PRICE+DERIV],b.iloc[:800][PRICE+DERIV])

    def test_touch_exact_boundaries_and_censoring(self):
        f=self.frame();f.iloc[2,f.columns.get_loc('high')]=105
        f.iloc[2,f.columns.get_loc('low')]=90
        for d,t in [('long',.05),('short',.10)]:
            a=touch_outcomes(f,d,t,24)
            self.assertEqual(a.touch.iloc[0],1)
            self.assertEqual(a.first_touch_hours.iloc[0],.5)
            self.assertTrue(a.touch.iloc[-96:].isna().all())
        self.assertEqual(touch_outcomes(f,'long',.05,24).before_adverse_2.iloc[0],0)
    def test_same_bar_stop_wins_and_gap_is_worse(self):
        f=self.frame();f.iloc[1]=[96,106,94,100]
        a=evaluate_stop(f,'long',.05,24,pd.Series(97.,index=f.index))
        # Stop above next open is invalid, never silently turned into a valid trade.
        self.assertFalse(a.valid_stop.iloc[0])
        f.iloc[1]=[100,106,94,100]
        a=evaluate_stop(f,'long',.05,24,pd.Series(97.,index=f.index))
        self.assertEqual(a.target_before_stop.iloc[0],0)
        self.assertTrue(a.ambiguous.iloc[0]);self.assertAlmostEqual(a.net20.iloc[0],-.032)
        f.iloc[1]=[100,101,99,100];f.iloc[2]=[95,96,94,95]
        a=evaluate_stop(f,'long',.05,24,pd.Series(97.,index=f.index))
        self.assertAlmostEqual(a.net20.iloc[0],-.052)
    def test_short_stop_and_funding_direction(self):
        f=self.frame();f.iloc[2,f.columns.get_loc('high')]=104
        a=evaluate_stop(f,'short',.1,24,pd.Series(103.,index=f.index),np.arange(len(f))*.0001)
        self.assertEqual(a.stopped.iloc[0],1);self.assertAlmostEqual(a.net20.iloc[0],-.03+.0002-.002)
    def test_confirmed_pivot_available_only_two_bars_later(self):
        f=self.frame();f.iloc[10,f.columns.get_loc('low')]=95
        a=levels(f,'long')['confirmed_pivot']
        self.assertTrue(np.isnan(a.iloc[11]));self.assertEqual(a.iloc[12],95)
        changed=f.copy();changed.iloc[20:]=500
        for key,s in levels(f,'long').items():
            pd.testing.assert_series_equal(s.iloc[:20],levels(changed,'long')[key].iloc[:20])
    def test_zigzag_and_pre_onset_labels(self):
        f=self.frame();f.close=100.;f.iloc[20:,f.columns.get_loc('close')]=106
        f.iloc[30:,f.columns.get_loc('close')]=94
        e=zigzag(f,.05)
        self.assertEqual(e.direction.tolist(),['long','short'])
        self.assertEqual(e.iloc[1].onset,f.index[20])
        y,ids,_=precursor_labels(f,e,'short')
        self.assertEqual(y.iloc[19],1);self.assertEqual(y.iloc[20],0)
    def test_splits_purge_and_embargo(self):
        f=self.frame(70080);fold=fold_windows(f.index[-1])[0]
        self.assertEqual(fold['calibration'][0]-fold['train'][1],72*HOUR)
        idx=split_index(f,fold['train'],pd.Series(True,index=f.index))
        self.assertTrue((idx+73*HOUR<fold['train'][1]).all())
    def test_calibration_and_cooldown(self):
        idx=pd.date_range('2025-01-01',periods=720,freq='h');s=pd.Series(np.arange(720),index=idx)
        self.assertTrue((replay(s,0).to_series().diff().dropna()>=6*HOUR).all())
        c=calibrate(s,pd.Series((np.arange(720)%2).astype(float),index=idx))
        self.assertGreaterEqual(c['alerts'],20)

if __name__=='__main__':unittest.main()
