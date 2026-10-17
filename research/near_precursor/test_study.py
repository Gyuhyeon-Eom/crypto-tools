import unittest
import numpy as np
import pandas as pd
from study import features, future_extreme, census, outcomes, replay, metrics, PRICE, DERIV


class StudyTests(unittest.TestCase):
    def frame(self, n=600):
        idx = pd.date_range('2026-01-01', periods=n, freq='15min')
        close = 100 + np.sin(np.arange(n) / 10)
        return pd.DataFrame({'open': close, 'close': close, 'high': close + .2, 'low': close - .2,
                             'volume': 100., 'turnover': 10000.}, index=idx)

    def test_future_extreme_excludes_current_and_requires_full_horizon(self):
        s = pd.Series([100, 1, 2, 3, 4])
        x = future_extreme(s, 2)
        self.assertEqual(x.iloc[0], 2)
        self.assertEqual(x.iloc[2], 4)
        self.assertTrue(x.iloc[-2:].isna().all())

    def test_future_prices_cannot_change_past_features(self):
        frame = self.frame()
        frames = {s: frame.copy() for s in ('NEARUSDT', 'BTCUSDT', 'ETHUSDT')}
        old = features(frames, [], [])
        frames['NEARUSDT'].loc[frame.index[400]:, ['close', 'high', 'low', 'turnover']] *= 10
        new = features(frames, [], [])
        pd.testing.assert_frame_equal(old.iloc[:400][PRICE], new.iloc[:400][PRICE])

    def test_derivatives_available_only_after_lag(self):
        frame = self.frame()
        t = frame.index[200]
        records = [{'timestamp': str(int(t.timestamp() * 1000)), 'openInterest': '100'}]
        f = features({s: frame.copy() for s in ('NEARUSDT', 'BTCUSDT', 'ETHUSDT')}, records, [])
        self.assertTrue(np.isnan(f.loc[t, 'oi']))
        self.assertEqual(f.loc[t + pd.Timedelta(minutes=15), 'oi'], 100)
        self.assertTrue(f.funding.isna().all())

    def test_labels_are_before_onset_and_pending_is_unknown(self):
        f = self.frame()
        event = f.index[300]
        y, target = outcomes(f, pd.DatetimeIndex([event]))
        self.assertEqual(y.loc[event], 0)
        self.assertEqual(y.loc[event - pd.Timedelta(minutes=15)], 1)
        self.assertEqual(y.loc[event - pd.Timedelta(hours=1)], 1)
        self.assertEqual(y.loc[event - pd.Timedelta(minutes=75)], 0)
        self.assertTrue(y.iloc[-100:].isna().all())
        self.assertEqual(target.loc[event - pd.Timedelta(minutes=15)], event)

    def test_census_keeps_events_separated_and_detects_both_directions(self):
        f = self.frame()
        f[['open', 'close']] = 100.
        f.high = 100.1
        f.low = 99.9
        for direction, value in [('long', 106.), ('short', 94.)]:
            x = f.copy()
            x.loc[x.index[200]:, 'close'] = value
            x.loc[x.index[200]:, 'high'] = value + .1
            x.loc[x.index[200]:, 'low'] = value - .1
            events, raw = census(x, direction, .05, x.index[0])
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0], x.index[196])
            self.assertGreaterEqual(len(raw), 1)

    def test_replay_is_chronological_and_shared_within_direction(self):
        f = self.frame(100)
        result = replay(pd.Series(1., index=f.index), .5)
        self.assertTrue((result.to_series().diff().dropna() >= pd.Timedelta(hours=6)).all())

    def test_short_price_touch_is_not_onset_precision(self):
        f = self.frame()
        f[['open', 'close', 'high', 'low']] = 100.
        t = f.index[100]
        f.loc[f.index[104], 'low'] = 90
        y, event = outcomes(f, pd.DatetimeIndex([]))
        result = metrics(pd.DatetimeIndex([t]), y, event, [], f, 'short', .10)
        self.assertEqual(result['onset_precision'], 0)
        self.assertEqual(result['target_touch_24h_rate'], 1)
        self.assertEqual(result['median_target_touch_hours'], 1)


if __name__ == '__main__':
    unittest.main()
