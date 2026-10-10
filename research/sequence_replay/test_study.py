import unittest
import numpy as np
import pandas as pd
from study import run_symbol, flow_features


class SequenceTests(unittest.TestCase):
    def sample(self, n=50):
        d = pd.DataFrame({'time': pd.date_range('2026-04-02', periods=n, freq='5min'),
                          'symbol': 'TEST', 'close': 100., 'low': 100., 'quality': True,
                          'eligible_price': True, 'quote_accel': 0., 'net5': .3, 'netprev10': -.1})
        d.loc[6:18, 'quote_accel'] = 1.
        return d

    def test_sequence_and_single_alert_per_episode(self):
        d = self.sample()
        alerts, audit = run_symbol(d)
        self.assertEqual(audit['episodes'], 1)
        self.assertEqual(len(alerts), 4)
        self.assertEqual({a['policy']: a['wait_minutes'] for a in alerts},
                         {'entry': 0, 'delay15': 15, 'hold15': 15, 'hold_rebuy': 15})

    def test_drawdown_disqualifies_hold_but_not_delay(self):
        d = self.sample()
        d.loc[8, 'low'] = 98.9
        self.assertEqual({a['policy'] for a in run_symbol(d)[0]}, {'entry','delay15'})

    def test_requires_lull_then_rebuy_not_continuous_buy(self):
        d = self.sample()
        d['netprev10'] = .1
        self.assertNotIn('hold_rebuy', {a['policy'] for a in run_symbol(d)[0]})
        d.loc[12, 'netprev10'] = -.1
        a = [a for a in run_symbol(d)[0] if a['policy']=='hold_rebuy']
        self.assertEqual(a[0]['wait_minutes'], 30)

    def test_current_candle_open_low_does_not_precede_entry(self):
        d = self.sample()
        d.loc[6, 'low'] = 90.
        self.assertIn('hold15', {a['policy'] for a in run_symbol(d)[0]})

    def test_gap_and_bad_quality_invalidate_episode(self):
        for gap in [True, False]:
            d = self.sample()
            if gap:
                d = d.drop(index=8)
            else:
                d.loc[8, 'quality'] = False
            self.assertEqual([a['policy'] for a in run_symbol(d)[0]], ['entry'])

    def test_needs_rearm_and_does_not_repeat_on_persistent_volume(self):
        d = self.sample()
        d.loc[6:, 'quote_accel'] = 1.
        self.assertEqual(run_symbol(d)[1]['episodes'], 1)
        d.loc[19:24, 'quote_accel'] = 0.
        self.assertEqual(run_symbol(d)[1]['episodes'], 2)

    def test_future_and_labels_cannot_change_past_alerts(self):
        d = self.sample()
        first = run_symbol(d)[0]
        d.loc[20:, ['close','low']] = 900
        d['strict_hit'] = True
        d['next_event'] = pd.Timestamp('2099-01-01')
        self.assertEqual(first, run_symbol(d)[0])

    def test_closed_bar_alignment_and_no_current_in_prior_flow(self):
        t = pd.date_range('2026-04-02', periods=4, freq='5min')
        bars = pd.DataFrame({'bucket': t, 'quote': [100.]*4,
                             'buy': [0.,0.,100.,0.], 'sell': [100.,100.,0.,100.]})
        f = flow_features(bars, t+pd.Timedelta(minutes=5))
        self.assertEqual(f.iloc[2].net5, 1.)
        self.assertEqual(f.iloc[2].netprev10, -1.)
        self.assertTrue(np.isnan(f.iloc[0].netprev10))


if __name__ == '__main__':
    unittest.main()
