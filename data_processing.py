from pathlib import Path
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
from config import ALIASES, REGIONS
from preprocessing import DTWGMM


REQUIRED = ['organisation', 'ligne', 'voyage', 'vehicule', 'date_operationnelle',
            'sequence_arret', 'arret_code', 'montant', 'arret_geom', 'date_heure_depart_prevue',
            'date_heure_depart_reelle']


def load_csv(c):
    # 与用户同一份法语字段CSV兼容；保留站号和线路字符串，不把整数编码当距离。
    d = pd.read_csv(c.data, dtype=str)
    missing = set(REQUIRED)-set(d)
    if missing:
        raise ValueError(f'Missing CSV fields: {sorted(missing)}')
    d['region'] = d.organisation.replace(ALIASES)
    planned = pd.to_datetime(d.date_heure_depart_prevue, errors='coerce')
    full = d.date_heure_depart_reelle.str.contains(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', na=False)
    actual = pd.to_datetime(d.date_heure_depart_reelle.where(full), errors='coerce')
    if c.mode == 'strict' and actual.notna().sum() != len(d):
        raise ValueError('Strict DBC needs complete actual timestamps; strings like 34:49.2 lack hours. '
                         'Supply full AVL timestamps or explicitly use exploratory planned-time fallback.')
    d['time'] = actual.fillna(planned)
    d['time_source'] = np.where(actual.notna(), 'actual', 'scheduled_fallback')
    d['count'] = pd.to_numeric(d.montant, errors='coerce')
    d.loc[(d['count'] < 0) | ~np.isfinite(d['count']), 'count'] = np.nan
    d['seq'] = pd.to_numeric(d.sequence_arret, errors='coerce')
    xy = d.arret_geom.str.extract(r'POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)')
    d['lon'], d['lat'] = pd.to_numeric(xy[0], errors='coerce'), pd.to_numeric(xy[1], errors='coerce')
    d = d.dropna(subset=['region', 'time', 'seq', 'arret_code', 'lon', 'lat'])
    d['bin'] = d.time.dt.floor('15min')
    return d.sort_values('time').reset_index(drop=True)


def prepare(c):
    d = load_csv(c)
    root = Path(c.cache); root.mkdir(parents=True, exist_ok=True)
    times = pd.date_range(d.bin.min().normalize(), d.bin.max().normalize()+pd.Timedelta(days=1), freq='15min', inclusive='left')
    times = times[times.hour >= c.start_hour]
    a, b = int(len(times)*c.train_fraction), int(len(times)*(c.train_fraction+c.val_fraction))
    if a < c.history+c.horizon or len(times)-b < c.history+c.horizon:
        raise ValueError('Insufficient time bins for chronological training/validation/test')
    if c.mode == 'strict' and (d.time.min().date() > pd.Timestamp('2023-09-01').date()
                              or d.time.max().date() < pd.Timestamp('2023-10-31').date()):
        raise ValueError('Strict paper replication requires full September-October coverage')
    weather = None
    if c.weather_file:
        weather = pd.read_csv(c.weather_file)
        for col in ('time', 'available_at'):
            weather[col] = pd.to_datetime(weather[col])
        if (weather.available_at > weather.time+pd.Timedelta(minutes=15)).any():
            raise ValueError('Weather observations must be available by bin-end issuance')
        weather = weather.set_index('time').drop(columns='available_at').reindex(times)
        if weather.isna().any().any():
            raise ValueError('Weather features must cover all input times')
        maximum = np.abs(weather.iloc[:a]).max().clip(lower=1)
        weather = (weather/maximum).to_numpy(dtype='float32')
    datasets, report = {}, {}
    for region in REGIONS:
        rows = d.loc[(d.region == region) & (d.time.dt.hour >= c.start_hour)].copy()
        if rows.empty:
            continue
        train_rows = rows.loc[rows.bin < times[a]]
        nodes = sorted(train_rows.arret_code.unique())
        if c.max_nodes:
            nodes = nodes[:c.max_nodes]
        if not nodes:
            continue
        rows = rows.loc[rows.arret_code.isin(nodes)]
        grouped = rows.groupby(['bin', 'arret_code'])['count'].sum(min_count=1).unstack()
        raw = grouped.reindex(index=times, columns=nodes).to_numpy(dtype=float)
        observed = np.isfinite(raw)
        service = rows.groupby(['bin', 'arret_code']).size().unstack(fill_value=0).reindex(index=times, columns=nodes, fill_value=0).fillna(0).to_numpy() > 0
        if c.zero_missing:
            raw = np.where(service, raw, 0.)
            observed = np.isfinite(raw)
        slots = times[:a].hour*4+times[:a].minute//15
        cleaner = DTWGMM().fit(raw[:a], np.array(slots), c)
        clean = cleaner.transform(raw)
        if c.ablation == 'raw':
            # 无清洗消融：只以0填充输入缺测，增加缺测指示特征，不改标签。
            clean = np.nan_to_num(raw).astype('float32')
        finite_training = raw[:a][np.isfinite(raw[:a])]
        if not len(finite_training):
            raise ValueError(f'No observed training boarding labels for {region}')
        scale = max(float(finite_training.max()), 1.)
        peak = np.asarray(((times.hour >= 7) & (times.hour < 10)) | ((times.hour >= 16) & (times.hour < 19)), dtype='float32')
        # 论文没有给出peak定义；采用显式默认并记录。未提供天气时不伪造天气。
        features = [clean/scale, np.broadcast_to(peak[:, None], clean.shape), observed.astype('float32')]
        if weather is not None:
            features.extend(np.broadcast_to(weather[:, k:k+1], clean.shape) for k in range(weather.shape[1]))
        x = np.stack(features, axis=-1).astype('float32')
        coords = train_rows.groupby('arret_code')[['lon', 'lat']].median().reindex(nodes).to_numpy()
        profile = np.stack([x[:a][slots == slot].mean(0) for slot in np.arange(c.start_hour*4, 96)])
        joblib.dump(cleaner, root/f'{region}_cleaner.joblib')
        np.savez_compressed(root/f'{region}.npz', x=x, raw=raw, observed=observed, service=service,
                            scale=scale, nodes=np.array(nodes), times=times.astype(str).to_numpy(dtype=str),
                            coords=coords, split=[a, b], training_profile=profile)
        datasets[region] = dict(x=x, raw=raw, observed=observed, service=service, scale=scale, nodes=nodes,
                                times=times, coords=coords, split=(a, b), profile=profile, records=rows)
        report[region] = dict(rows=len(rows), nodes=len(nodes), routes=int(train_rows.ligne.nunique()),
                              raw_boarding_sum=float(rows['count'].sum()), table_routes_stops=REGIONS[region],
                              actual_timestamp_fraction=float((rows.time_source == 'actual').mean()),
                              evaluation_observed_fraction=float(observed.mean()), scale_training_only=scale)
    if len(datasets) < 4:
        raise ValueError('Cross-region positives and dissimilar negatives require at least 4 available regions')
    report['protocol'] = dict(first=str(times[0]), last=str(times[-1]), train_end=str(times[a]), val_end=str(times[b]),
                              training_fraction=c.train_fraction, validation_fraction=c.val_fraction,
                              timestamp_policy=c.mode, weather_provided=bool(c.weather_file),
                              target_policy='raw observed labels; absent grid zero only if explicitly enabled',
                              data_sha256=hashlib.sha256(Path(c.data).read_bytes()).hexdigest())
    (root/'data_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return datasets
