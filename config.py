from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import json


REGIONS = {'MRCLM': (30, 752), 'OMITSJ': (10, 139), 'CTLSV': (15, 251),
           'MRCLASSO': (13, 521), 'LRRS': (34, 520), 'CITVR': (13, 283),
           'CITCRC': (15, 270), 'CITLA': (43, 1344), 'CITSO': (16, 372),
           'CITPI': (28, 639), 'CITHSL': (2, 107)}
ALIASES = {'OMITSJU': 'OMITSJ', 'CITSV': 'CTLSV'}


@dataclass
class Config:
    data: str = r'D:\PH.D\data\EXO_APC_AVL_202309_202310.csv'
    output: str = 'runs/DA_GCL'
    cache: str = 'cache/DA_GCL'
    target: str = 'MRCLM'
    mode: str = 'strict'
    history: int = 12
    horizon: int = 4
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    start_hour: int = 6
    # 以下未在论文指定的参数属于显式实现默认，不能作为作者实测最优配置。
    hidden: int = 64
    spatial_layers: int = 2
    cheb_order: int = 3
    tcn_layers: int = 3
    tcn_kernel: int = 3
    dropout: float = 0.1
    epochs: int = 200
    patience: int = 15
    batch_size: int = 32
    micro_batch: int = 1
    optimizer: str = 'adam'
    lr: float = 0.001
    eta: float = 0.0001
    seed: int = 42
    seeds: str = '42,123,1024,2026,3407'
    device: str = 'cpu'
    beta: float = 0.5
    temperature: float = 0.11
    balance_prior: float = 1.0
    gamma: float = 0.5
    lambda_max: float = 10.0
    static_cl_weight: float = 1.0
    eps: float = 1e-8
    lambda_ser: float = 0.01
    ista_steps: int = 10
    ser_window: int = 12
    ser_guidance: str = 'reconstruction'
    gmm_components: int = 3
    gmm_shrinkage: float = 0.5
    knn: int = 5
    dtw_sigma: float = 1.0
    dtw_band: int = -1
    dtw_reference: str = 'full'
    edge_drop: float = 0.2
    feature_drop: float = 0.1
    drop_min: float = 0.05
    drop_max: float = 0.7
    false_negative_phi: float = 0.8
    negative_regions: int = 3
    dbc_minutes: float = 60.0
    road_edges: str = ''
    activity_file: str = ''
    weather_file: str = ''
    zero_missing: int = 0
    max_nodes: int = 0
    ablation: str = 'full'
    search_method: str = 'ga'
    population: int = 20
    generations: int = 100
    search_patience: int = 15
    crossover: float = 0.2
    mutation: float = 0.05
    search_epochs: int = 200
    benchmark_warmup: int = 100
    benchmark_iterations: int = 1000
    checkpoint: str = ''
    topology_filter: str = 'none'
    svd_rank: int = 32


def parse():
    p = argparse.ArgumentParser(description='DA-GCL reproducible reference implementation')
    for key, value in asdict(Config()).items():
        p.add_argument('--'+key.replace('_', '-'), type=type(value), default=value)
    p.add_argument('--action', choices=['prepare', 'train', 'predict', 'repeat', 'search', 'ablate',
                                      'robustness', 'sensitivity', 'benchmark', 'search-compare', 'filter-ablate'], default='train')
    args = vars(p.parse_args()); action = args.pop('action'); c = Config(**args)
    validate(c)
    Path(c.output).mkdir(parents=True, exist_ok=True)
    Path(c.output, 'requested_config.json').write_text(json.dumps(asdict(c), indent=2), encoding='utf-8')
    return c, action


def validate(c):
    if c.mode not in ('strict', 'exploratory') or c.optimizer not in ('adam', 'sgd'):
        raise ValueError('invalid mode/optimizer')
    if c.target != 'ALL' and c.target not in REGIONS:
        raise ValueError('unknown target region')
    if not 0 < c.train_fraction < c.train_fraction+c.val_fraction < 1:
        raise ValueError('invalid chronological split')
    for key in ('history', 'horizon', 'hidden', 'spatial_layers', 'epochs', 'patience',
                'batch_size', 'micro_batch', 'knn', 'ista_steps', 'ser_window', 'population', 'generations'):
        if getattr(c, key) < 1:
            raise ValueError(f'{key} must be positive')
    if c.micro_batch > c.batch_size or c.temperature <= 0 or c.dtw_sigma <= 0:
        raise ValueError('invalid micro batch / temperature / DTW sigma')
    if c.micro_batch != 1:
        raise ValueError('This reference executes one physical sample at a time; micro_batch must equal 1')
    if not 0 < c.beta < 1 or c.negative_regions < 1:
        raise ValueError('beta must lie in (0,1), negative_regions must be positive')
    if c.dtw_reference not in ('full', 'daily_profile'):
        raise ValueError('dtw_reference must be full or daily_profile')
    if c.ser_guidance not in ('reconstruction', 'support'):
        raise ValueError('unknown SER guidance')
    if c.mode == 'strict' and (c.max_nodes or c.dtw_reference != 'full' or c.dtw_band != -1):
        raise ValueError('strict mode disallows node truncation and approximate DTW reference/band')
    if c.search_method not in ('ga', 'de', 'pso', 'gwo', 'sa', 'rs', 'grid'):
        raise ValueError('unknown search method')
    if c.topology_filter not in ('none', 'dropedge', 'svd') or c.svd_rank < 1:
        raise ValueError('invalid topology filter / SVD rank')
