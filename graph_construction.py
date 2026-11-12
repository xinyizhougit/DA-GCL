from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from config import REGIONS


def haversine(a, b):
    lon1, lat1 = np.deg2rad(a); lon2, lat2 = np.deg2rad(b)
    v = np.sin((lat2-lat1)/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 2*6371*np.arcsin(np.sqrt(np.clip(v, 0, 1)))


def structural_neighbors(target, available):
    # 按论文表I选择相似线路数/站点数；并列按区域名，重复正区域只保留一次。
    others = sorted(set(available)-{target})
    positives = list(dict.fromkeys(min(others, key=lambda x: (abs(REGIONS[x][k]-REGIONS[target][k]), x)) for k in (0, 1)))
    negatives = [r for r in others if r not in positives]
    negatives.sort(key=lambda r: -sum(abs(np.log((REGIONS[r][k]+1)/(REGIONS[target][k]+1))) for k in (0, 1)))
    if not negatives:
        raise ValueError('No topologically dissimilar negative region')
    return positives, negatives


class PhysicalGraphs:
    def __init__(self, data, region, c):
        self.data, self.region, self.c = data, region, c
        self.index = {v: i for i, v in enumerate(data['nodes'])}
        self.n = len(self.index)
        self.activity = None
        if c.activity_file:
            a = pd.read_csv(c.activity_file, dtype={'arret_code': str})
            a['time'] = pd.to_datetime(a.time)
            a = a[a.region == region].pivot(index='time', columns='arret_code', values='active')
            a = a.reindex(index=data['times'], columns=data['nodes'])
            if a.isna().any().any():
                raise ValueError('Activity file must cover the full time/node grid')
            self.activity = a.to_numpy().astype(bool)
        elif c.mode == 'strict':
            raise ValueError('Strict mode requires independent --activity-file; missing flow is not a closure')
        roads = {}
        if c.road_edges:
            r = pd.read_csv(c.road_edges, dtype=str)
            for _, row in r[r.region == region].iterrows():
                roads[(row.route, row.from_stop, row.to_stop)] = float(row.distance_km)
        elif c.mode == 'strict':
            raise ValueError('Strict DBC requires directed AVL/road distances via --road-edges')
        rows = data['records']
        key = ['date_operationnelle', 'ligne', 'voyage', 'vehicule']
        cutoff = data['times'][data['split'][0]]
        self.paths, events = set(), {}
        for _, trip in rows.groupby(key, sort=False):
            trip = trip.sort_values('seq')
            stops, seq = trip.arret_code.tolist(), trip.seq.to_numpy()
            line = str(trip.ligne.iloc[0])
            times = trip.time.to_numpy(dtype='datetime64[ns]')
            segment = []
            for k in range(len(stops)-1):
                i, j = self.index[stops[k]], self.index[stops[k+1]]
                if seq[k+1] != seq[k]+1 or i == j:
                    if segment and pd.Timestamp(times[k]) < cutoff:
                        self.paths.add((line, tuple(segment)))
                    segment = []
                    continue
                if not segment:
                    segment.append(i)
                segment.append(j)
                road = roads.get((line, stops[k], stops[k+1]))
                if road is None:
                    if c.mode == 'strict':
                        raise ValueError(f'Missing directed road distance: {line} {stops[k]} {stops[k+1]}')
                    road = haversine(data['coords'][i], data['coords'][j])
                minutes = float((times[k+1]-times[k])/np.timedelta64(1, 'm'))
                if road > 0 and minutes > 0:
                    # 可用时刻为车辆到达下游之后；查询严格限制在预测发出之前。
                    events.setdefault((line, i, j), []).append((times[k+1].astype('int64'), road, road/minutes))
            if segment and pd.Timestamp(times[-1]) < cutoff:
                self.paths.add((line, tuple(segment)))
        self.events = {key: np.array(sorted(value), dtype=float) for key, value in events.items()}
        self.paths = sorted(self.paths)
        self.static = self.at(data['split'][0]-1, training_template=True)
        save_npz(Path(c.cache, f'{region}_physical_training_template_{c.topology_filter}.npz'), self.static)

    def active(self, t):
        return self.activity[t] if self.activity is not None else self.data['service'][t]

    def at(self, t, training_template=False):
        cutoff = (self.data['times'][t]+pd.Timedelta(minutes=15)).value
        active = self.active(t)
        if training_template:
            active = self.data['service'][:self.data['split'][0]].any(0)
        rr, cc = [], []
        for route, path in self.paths:
            costs = []
            for i, j in zip(path[:-1], path[1:]):
                e = self.events.get((route, i, j))
                k = np.searchsorted(e[:, 0], cutoff, side='left')-1 if e is not None else -1
                costs.append(e[k, 1]/e[k, 2] if k >= 0 else np.inf)
            # 仅沿一条训练期已知的线路顺序扩展，不跨线路拼接换乘路径。
            for start in range(len(path)-1):
                total = 0.
                for end in range(start+1, len(path)):
                    total += costs[end-1]
                    if total > self.c.dbc_minutes:
                        break
                    i, j = path[start], path[end]
                    if i != j and active[i] and active[j]:
                        rr.append(i); cc.append(j)
        out = csr_matrix((np.ones(len(rr), dtype='float32'), (rr, cc)), shape=(self.n, self.n))
        out.data[:] = 1
        if self.c.topology_filter == 'dropedge':
            rng = np.random.default_rng(self.c.seed+int(t))
            out.data *= rng.random(len(out.data)) >= self.c.edge_drop
            out.eliminate_zeros()
        elif self.c.topology_filter == 'svd':
            # 独立快照截断SVD对照，不宣称它与EFT特征滤波等价。
            u, values, vh = np.linalg.svd(out.toarray(), full_matrices=False)
            rank = min(self.c.svd_rank, self.n)
            reconstructed = np.maximum((u[:, :rank]*values[:rank]) @ vh[:rank], 0)
            out = csr_matrix((reconstructed*(active[:, None] & active[None, :])).astype('float32'))
        return out


def build_graphs(datasets, c):
    return {r: PhysicalGraphs(d, r, c) for r, d in datasets.items()}
