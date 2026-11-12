from pathlib import Path
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp
from scipy.stats import norm


def dtw_distance(a, b, band=-1):
    # 精确动态规划；band=-1是不限制路径的DTW。可选numba仅加速，不改变公式。
    n, m = len(a), len(b)
    width = max(n, m) if band < 0 else max(band, abs(n-m))
    previous = np.full(m+1, np.inf); previous[0] = 0.
    for i in range(1, n+1):
        current = np.full(m+1, np.inf)
        for j in range(max(1, i-width), min(m, i+width)+1):
            current[j] = (a[i-1]-b[j-1])**2 + min(previous[j], current[j-1], previous[j-1])
        previous = current
    return np.sqrt(previous[m])


try:
    from numba import njit
    dtw_distance = njit(cache=True)(dtw_distance)
except ImportError:
    pass


class DTWGMM:
    def fit(self, train, slots, c):
        self.c = c
        self.median = np.nanmedian(train, axis=0)
        self.median = np.nan_to_num(self.median)
        filled = np.where(np.isfinite(train), train, self.median)
        reference = filled
        if c.dtw_reference == 'daily_profile':
            # 仅探索模式可用的日剖面加速；与完整历史DTW不等价，配置中显式记录。
            reference = np.stack([filled[slots == s].mean(0) for s in np.unique(slots)])
        self.distances = np.zeros((train.shape[1], train.shape[1]))
        for i in range(train.shape[1]):
            for j in range(i):
                d = dtw_distance(reference[:, i], reference[:, j], c.dtw_band)
                self.distances[i, j] = self.distances[j, i] = d
        np.fill_diagonal(self.distances, np.inf)
        self.neighbors = np.argsort(self.distances, axis=1)
        imputed = self.impute(train)
        self.gmm = GaussianMixture(n_components=min(c.gmm_components, imputed.size),
                                   reg_covar=1e-6, random_state=c.seed).fit(imputed.reshape(-1, 1))
        means = self.gmm.means_.ravel(); stds = np.sqrt(self.gmm.covariances_.ravel())
        # 混合分布的3sigma阈值未被论文唯一规定：采用两侧0.135%分位区间。
        from scipy.optimize import brentq
        lo, hi = float(np.min(means-12*stds)), float(np.max(means+12*stds))
        def cdf(x):
            return float(np.dot(self.gmm.weights_, norm.cdf((x-means)/stds)))
        tail = norm.cdf(-3)
        self.lower = brentq(lambda x: cdf(x)-tail, lo, hi)
        self.upper = brentq(lambda x: cdf(x)-(1-tail), lo, hi)
        self.nll_threshold = float(max(-self.gmm.score_samples([[self.lower]])[0],
                                       -self.gmm.score_samples([[self.upper]])[0]))
        return self

    def impute(self, x):
        result = x.copy()
        for t, node in zip(*np.where(~np.isfinite(x))):
            candidates = self.neighbors[node]
            candidates = candidates[np.isfinite(x[t, candidates]) & np.isfinite(self.distances[node, candidates])][:self.c.knn]
            if len(candidates):
                logits = -self.distances[node, candidates]/self.c.dtw_sigma**2
                weights = np.exp(logits-logsumexp(logits))
                result[t, node] = weights @ x[t, candidates]
            else:
                result[t, node] = self.median[node]
        return result

    def transform(self, x):
        z = self.impute(x)
        score = -self.gmm.score_samples(z.reshape(-1, 1)).reshape(z.shape)
        bad = score > self.nll_threshold
        means = self.gmm.means_.ravel()
        nearest = means[np.abs(z[..., None]-means).argmin(-1)]
        z = np.where(bad, (1-self.c.gmm_shrinkage)*z+self.c.gmm_shrinkage*nearest, z)
        return np.maximum(z, 0).astype('float32')
