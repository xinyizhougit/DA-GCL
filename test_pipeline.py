import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
from config import Config
from preprocessing import dtw_distance, DTWGMM
from sampling import origins, batch
from metrics import all_horizons
from graph_construction import structural_neighbors
from search import optimize


class PipelineTests(unittest.TestCase):
    def test_dtw_shift_and_identity(self):
        self.assertEqual(dtw_distance(np.array([0., 1., 0.]), np.array([0., 1., 0.])), 0)
        a, b = np.array([0., 0., 1., 0.]), np.array([0., 1., 0., 0.])
        self.assertLess(dtw_distance(a, b), np.linalg.norm(a-b))

    def test_cleaning_freezes_statistics(self):
        rng = np.random.default_rng(1)
        train = rng.poisson(3, (25, 3)).astype(float)
        c = replace(Config(), dtw_reference='full')
        cleaner = DTWGMM().fit(train, np.arange(25), c)
        original = cleaner.gmm.means_.copy()
        distance = cleaner.distances.copy()
        future = np.array([[np.nan, 4., 5.], [1e8, 0., np.nan]])
        output = cleaner.transform(future)
        self.assertTrue(np.isfinite(output).all())
        np.testing.assert_array_equal(original, cleaner.gmm.means_)
        np.testing.assert_array_equal(distance, cleaner.distances)

    def test_windows_no_overnight_or_label_crossing(self):
        times = pd.date_range('2023-09-01', periods=96*6, freq='15min')
        times = times[times.hour >= 6]
        n = len(times)
        d = dict(times=times, split=(int(n*.7), int(n*.85)), raw=np.ones((n, 3)),
                 observed=np.ones((n, 3), bool), x=np.ones((n, 3, 3)))
        c = Config()
        for split, lower, upper in [('train', 0, d['split'][0]), ('val', *d['split']), ('test', d['split'][1], n)]:
            for t in origins(d, c, split):
                self.assertGreaterEqual(t+1, lower); self.assertLess(t+c.horizon, upper)
                self.assertTrue(np.all(np.diff(times[t-c.history+1:t+c.horizon+1].asi8) == 900*10**9))

    def test_metrics_raw_horizons(self):
        y = np.array([[[0., 4.], [2., 8.]]]); p = y+1
        result = all_horizons(y, p, np.ones_like(y, bool))
        self.assertEqual(result['overall']['MAE'], 1.)
        self.assertEqual(result['horizons']['15']['count'], 2)
        with self.assertRaises(ValueError):
            all_horizons(y, p*np.nan, np.ones_like(y, bool))

    def test_region_positive_not_negative(self):
        p, n = structural_neighbors('MRCLM', ['MRCLM', 'CITPI', 'LRRS', 'CITLA', 'CITHSL'])
        self.assertEqual(p, ['CITPI'])
        self.assertFalse(set(p) & set(n))
        self.assertNotIn('MRCLM', p+n)

    def test_search_evaluation_budget(self):
        for method in ('ga', 'de', 'pso', 'gwo', 'sa', 'rs', 'grid'):
            c = replace(Config(), population=4, generations=3, search_patience=10, search_method=method)
            _, _, trace = optimize(lambda x, i: np.square(x-.3).sum(), c)
            self.assertEqual(len(trace), 12)
            best = [r['best_fitness'] for r in trace]
            self.assertTrue(np.all(np.diff(best) <= 0))

    def test_csv_graph_causality_and_no_transfers(self):
        from synthetic_data import create
        from data_processing import prepare
        from graph_construction import build_graphs
        with tempfile.TemporaryDirectory() as root:
            path = str(Path(root, 'input.csv'))
            create(path, days=4)
            c = replace(Config(), data=path, cache=root, mode='exploratory', dtw_reference='daily_profile')
            datasets = prepare(c); g = build_graphs(datasets, c)['MRCLM']
            before = g.at(0).toarray()
            cutoff = (g.data['times'][0]+pd.Timedelta(minutes=15)).value
            for value in g.events.values():
                value[value[:, 0] >= cutoff, 2] = 1e8
            np.testing.assert_array_equal(before, g.at(0).toarray())
            # 两条线路在站1相交，DBC不得通过换乘拼接出0->2。
            stamp = g.data['times'][0].value
            g.paths = [('a', (0, 1)), ('b', (1, 2))]
            g.events = {('a', 0, 1): np.array([[stamp, 1., 1.]]),
                        ('b', 1, 2): np.array([[stamp, 1., 1.]])}
            matrix = g.at(0).toarray()
            self.assertEqual(matrix[0, 1], 1)
            self.assertEqual(matrix[1, 2], 1)
            self.assertEqual(matrix[0, 2], 0)

    def test_torch_forward_backward(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch is not installed')
        from scipy.sparse import csr_matrix
        from model import DAGCL, chebyshev_apply
        from losses import intra_region
        c = replace(Config(), history=4, hidden=8, spatial_layers=1, ista_steps=2, ser_window=3, tcn_layers=1)
        model = DAGCL(3, c)
        x = torch.rand(4, 5, 3)
        graphs = [csr_matrix(np.ones((5, 5))-np.eye(5))]*4
        active = torch.ones(4, 5, dtype=torch.bool)
        prediction, za, zp = model(x, graphs, active, contrastive=True)
        self.assertEqual(tuple(prediction.shape), (5, 4))
        loss = prediction.square().mean()+intra_region(za[None], zp[None], torch.eye(5), active[-1][None], .1, .8)
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        prediction, _, _ = model(x, graphs, active*False)
        self.assertTrue(torch.isfinite(prediction).all())


if __name__ == '__main__':
    unittest.main()
