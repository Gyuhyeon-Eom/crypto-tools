import unittest
import numpy as np
import pandas as pd
from analyze import features, compare
from fetch import aggregate


class TradeflowTests(unittest.TestCase):
    def test_bucket_boundaries_and_notional_conservation(self):
        x=pd.DataFrame({'timestamp':[0,299999,300000],'price':[2,3,4], 'volume':[5,7,11], 'side':['buy','sell','buy'], 'rpi':[0,1,0]})
        b=aggregate(x)
        self.assertEqual(len(b),2)
        self.assertEqual(b.quote.sum(),75)
        self.assertEqual(b.buy.sum(),54)
        self.assertEqual(b.sell.sum(),21)
        self.assertEqual(b.rpi_quote.sum(),21)
        self.assertEqual(b.iloc[0].close,3)

    def bars(self):
        t=pd.date_range('2026-08-01', periods=288, freq='5min')
        price=100+np.sin(np.arange(288)/3)
        return pd.DataFrame({'bucket':t,'open':price,'high':price+.1,'low':price-.1,'close':price,
                             'quote':100.,'buy':30.,'sell':70.,'trades':10,'last_ms':t.as_unit('ms').astype('int64')+299999,'rpi_quote':0.})

    def test_future_trades_do_not_change_features(self):
        b=self.bars();anchor=pd.Timestamp('2026-08-02')
        extra=b.copy();extra.bucket+=pd.Timedelta(days=1);extra['quote']=1e12
        f=features(b,anchor,.01);g=features(pd.concat([b,extra]),anchor,.01)
        self.assertNotIn('quality_error',f)
        self.assertEqual(f,g)

    def test_sparse_data_not_backfilled(self):
        b=self.bars().iloc[100:]
        f=features(b,pd.Timestamp('2026-08-02'),.01)
        self.assertIn('quality_error',f)

    def test_absorption_then_reversal(self):
        b=self.bars()
        b[['open','close']]=100.;b['high']=100.01;b['low']=99.99
        b.loc[216:263,['quote','buy','sell']]=[200.,40.,160.]
        b.loc[264:,['quote','buy','sell']]=[200.,160.,40.]
        b.loc[264:,'close']=np.linspace(100.,101.,24)
        f=features(b,pd.Timestamp('2026-08-02'),.01)
        self.assertEqual(f['sequence'],1)
        b.loc[264:,'close']=99.
        self.assertEqual(features(b,pd.Timestamp('2026-08-02'),.01)['sequence'],0)

    def test_concordance_ties_and_whole_set_missingness(self):
        d=pd.DataFrame({'group':['a']*4+['b']*4,'role':['case','control','control','control']*2,
                        'time':[pd.Timestamp('2026-08-01')]*8,'score':[2,1,2,3,9,0,np.nan,0]})
        r=compare(d,'score')
        self.assertEqual(r['n'],1)
        self.assertEqual(r['concordance'],.5)


if __name__=='__main__':
    unittest.main()
