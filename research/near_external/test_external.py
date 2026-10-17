import unittest
import numpy as np
import pandas as pd
from features import align_available,sector_features,spot_frame,spot_features,chain_features,account_features

class ExternalTests(unittest.TestCase):
    def test_availability_not_backfilled_and_stale_rejected(self):
        t=pd.Timestamp('2025-01-01');index=pd.date_range(t,periods=100,freq='h')
        s=pd.DataFrame({'x':[7.]},index=pd.DatetimeIndex([t]))
        a=align_available(s,index,'48h','25h')
        self.assertTrue(a.x.iloc[:48].isna().all());self.assertEqual(a.x.iloc[48],7)
        self.assertTrue(a.x.iloc[74:].isna().all())
    def test_daily_missing_date_not_silently_forward_filled(self):
        days=pd.date_range('2025-01-01',periods=70,freq='D');secs=[int(x.timestamp()) for x in days]
        records={'near_stablecoins':[{'date':t,'totalCirculating':{'peggedUSD':100+i}} for i,t in enumerate(secs) if i!=40],
                 'near_tvl':[{'date':t,'tvl':100+i} for i,t in enumerate(secs)],
                 'near_dex_volume':{'totalDataChart':[[t,100+i] for i,t in enumerate(secs)]}}
        index=pd.date_range(days[0],days[-1]+pd.Timedelta(days=2),freq='h');f,_=chain_features(records,index)
        self.assertTrue(np.isnan(f.loc[days[42],'chain48_stable1']))
        self.assertTrue(np.isnan(f.loc[days[43],'chain48_stable1']))
        self.assertTrue(np.isfinite(f.loc[days[44],'chain48_stable1']))
    def test_sector_features_causal_and_near_excluded(self):
        idx=pd.date_range('2025-01-01',periods=3400,freq='15min');x=np.arange(len(idx))
        frames={s:pd.DataFrame({'close':100+np.sin(x/13+k)+x*.001,'turnover':100+x%7},index=idx) for k,s in enumerate(['BTCUSDT','A','B'])}
        f=pd.DataFrame({'ret4':frames['A'].close.pct_change(16)},index=idx)
        baskets={'l1':['A','B']};old=sector_features(frames,f,baskets)
        mutated={k:v.copy() for k,v in frames.items()}
        for v in mutated.values():v.iloc[3200:]*=5
        new=sector_features(mutated,f,baskets)
        pd.testing.assert_frame_equal(old.iloc[:3200],new.iloc[:3200])
        self.assertTrue(np.isfinite(old.iloc[3100]).all())
        with self.assertRaises(ValueError):sector_features(frames,f,{'l1':['NEARUSDT']})
    def test_spot_delta_is_taker_imbalance_and_zero_volume_unknown(self):
        idx=pd.date_range('2025-01-01',periods=120,freq='15min')
        s=pd.DataFrame({'quote':100.,'buy_quote':75.,'close':100.},index=idx)
        f=pd.DataFrame({'close':100.,'ret1':0.,'turnover':100.},index=idx)
        x=spot_features(s,f);self.assertEqual(x.spot_delta15.iloc[0],.5);self.assertEqual(x.spot_share1.iloc[4],.5)
        s.iloc[0,s.columns.get_loc('quote')]=0;s.iloc[0,s.columns.get_loc('buy_quote')]=0
        self.assertTrue(np.isnan(spot_features(s,f).spot_delta15.iloc[0]))
    def test_spot_rejects_side_volume_and_timestamp_errors(self):
        start=1735689600000
        row=[start,100,101,99,100,2,start+899999,200,10,1,100,0]
        self.assertEqual(len(spot_frame([row],start,start+900000)),1)
        bad=row.copy();bad[10]=201
        with self.assertRaises(ValueError):spot_frame([bad],start,start+900000)
        with self.assertRaises(ValueError):spot_frame([row,row],start,start+1800000)
    def test_account_ratio_lag_and_missing_hour(self):
        t=pd.Timestamp('2025-01-01');idx=pd.date_range(t,periods=10,freq='h')
        rows=[{'symbol':'NEARUSDT','timestamp':str(int(x.timestamp()*1000)),'buyRatio':str(.5+i*.01),'sellRatio':str(.5-i*.01)} for i,x in enumerate(idx) if i!=3]
        x=account_features(rows,idx)
        self.assertTrue(np.isnan(x.accounts_long.iloc[0]));self.assertEqual(x.accounts_long.iloc[1],.5)
        self.assertTrue(np.isnan(x.accounts_change1.iloc[5]))

class ComparisonTests(unittest.TestCase):
    def test_paired_brier_direction_and_correction(self):
        from train import paired_week_statistics,holm
        idx=pd.date_range('2025-01-01',periods=40,freq='7D')
        positive=paired_week_statistics(pd.Series(.01,index=idx))
        negative=paired_week_statistics(pd.Series(-.01,index=idx))
        zero=paired_week_statistics(pd.Series(0.,index=idx))
        self.assertAlmostEqual(positive['mean_brier_improvement'],.01)
        self.assertLess(positive['week_sign_p'],.001)
        self.assertEqual(negative['week_sign_p'],1.)
        self.assertEqual(zero['week_sign_p'],1.)
        np.testing.assert_allclose(holm([.01,.04,.2]),[.03,.08,.2])

    def test_all_groups_share_baseline_and_24h_is_only_chain_lag_sensitivity(self):
        from features import groups,CHAIN
        g=groups()
        for cols in g.values():self.assertTrue(set(g['baseline'])<=set(cols))
        self.assertEqual(set(g['all24'])-set(CHAIN(24)),set(g['all48'])-set(CHAIN(48)))
        self.assertNotIn('target_before_stop',set(sum(g.values(),[])))

if __name__=='__main__':unittest.main()
