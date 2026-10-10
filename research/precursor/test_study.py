"""Temporal integrity tests for the research pipeline (offline)."""
import unittest

import numpy as np
import pandas as pd

from study import FEATURES, alert_policy, compute_features, label_panel


def bars(n=120):
    rng = np.random.default_rng(7)
    rows = []
    for symbol in ['AAAUSDT', 'BBBUSDT', 'BTCUSDT']:
        close = 100 * np.exp(np.cumsum(rng.normal(0, .006, n)))
        rows.append(pd.DataFrame({'symbol': symbol, 'date': pd.date_range('2026-01-01', periods=n),
            'open': close, 'high': close*1.01, 'low': close*.99, 'close': close,
            'usd_volume_proxy': rng.uniform(700000, 1300000, n)}))
    return pd.concat(rows, ignore_index=True)


class TemporalIntegrity(unittest.TestCase):
    def test_future_bars_cannot_change_past_features(self):
        data = bars()
        cutoff = pd.Timestamp('2026-04-10')
        prefix = compute_features(data[data.date <= cutoff]).set_index(['symbol', 'date'])
        full = compute_features(data).set_index(['symbol', 'date']).loc[prefix.index]
        pd.testing.assert_frame_equal(prefix[FEATURES], full[FEATURES], check_exact=False, atol=1e-12, rtol=1e-12)

    def test_early_pump_is_not_a_precursor_hit_and_same_day_stop_is_failure(self):
        data = bars()
        # Fix a quiet past, then a large high on the second future day.
        data.loc[:, ['open', 'close']] = 100.0
        data.loc[:, 'high'] = 101.0
        data.loc[:, 'low'] = 99.0
        anchor = pd.Timestamp('2026-04-05')
        target = anchor + pd.Timedelta(days=2)
        data.loc[data.symbol.eq('AAAUSDT') & data.date.eq(target), 'high'] = 140
        data.loc[data.symbol.eq('AAAUSDT') & data.date.eq(target), 'low'] = 85
        data.loc[data.symbol.eq('BBBUSDT') & data.date.eq(target), 'high'] = 140
        data.loc[data.symbol.eq('BBBUSDT') & data.date.eq(anchor + pd.Timedelta(days=1)), 'high'] = 112
        labels = label_panel(compute_features(data)).set_index(['symbol', 'date'])
        a = labels.loc[('AAAUSDT', anchor)]
        b = labels.loc[('BBBUSDT', anchor)]
        self.assertEqual(a.y, 1)
        self.assertEqual(a.y_clean, 0)
        self.assertEqual(b.y, 0)

    def test_missing_future_day_invalidates_label(self):
        data = bars()
        anchor = pd.Timestamp('2026-04-05')
        data = data[~(data.symbol.eq('AAAUSDT') & data.date.eq(anchor + pd.Timedelta(days=3)))]
        result = label_panel(compute_features(data))
        self.assertFalse(((result.symbol == 'AAAUSDT') & (result.date == anchor)).any())

    def test_purge_uses_label_end_and_whole_dates(self):
        result = label_panel(compute_features(bars(310)))
        self.assertTrue((result.loc[result.split.eq('train'), 'label_end'] < pd.Timestamp('2026-05-01')).all())
        self.assertTrue((result.loc[result.split.eq('validation'), 'label_end'] < pd.Timestamp('2026-08-01')).all())
        self.assertTrue((result.loc[result.split.eq('test'), 'feature_at'] >= pd.Timestamp('2026-08-01')).all())

    def test_alerts_respect_threshold_cooldown_and_daily_budget(self):
        d = pd.DataFrame({'symbol': ['A', 'B', 'A', 'B', 'A'],
                          'date': pd.to_datetime(['2026-01-01', '2026-01-01', '2026-01-02', '2026-01-02', '2026-01-08'])})
        selected = alert_policy(d, [.9, .8, .99, .4, .8], threshold=.5, daily_cap=1)
        self.assertEqual(selected.index.tolist(), [0, 4])


if __name__ == '__main__':
    unittest.main()
