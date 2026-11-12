from pathlib import Path
from dataclasses import replace
import json
import time
import numpy as np


ABLATIONS = ['full', 'raw', 'static_gft', 'no_ser', 'physical_only', 'random_mask',
             'no_cl', 'static_weight', 'no_ga']


def repeat_runs(datasets, graphs, c):
    from training import train
    root = Path(c.output); root.mkdir(parents=True, exist_ok=True)
    reports = []
    for seed in map(int, c.seeds.split(',')):
        cc = replace(c, seed=seed, output=str(root/f'seed_{seed}'))
        train(datasets, graphs, cc)
        reports.append(json.loads(Path(cc.output, 'test_metrics.json').read_text()))
    summary = {}
    for name in ['overall']+[str(15*(h+1)) for h in range(c.horizon)]:
        summary[name] = {}
        for key in ('MAE', 'RMSE'):
            values = [r['overall'][key] if name == 'overall' else r['horizons'][name][key] for r in reports]
            summary[name][key] = dict(mean=float(np.mean(values)) if all(v is not None for v in values) else None,
                std=float(np.std(values, ddof=1)) if len(values) > 1 and all(v is not None for v in values) else None, runs=len(values))
    (root/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def raw_datasets(datasets, c):
    result = {}
    for r, data in datasets.items():
        d = dict(data); d['x'] = data['x'].copy()
        d['x'][..., 0] = np.nan_to_num(d['raw'])/d['scale']
        a = d['split'][0]
        slots = d['times'][:a].hour*4+d['times'][:a].minute//15
        d['profile'] = np.stack([d['x'][:a][slots == s].mean(0) for s in range(c.start_hour*4, 96)])
        result[r] = d
    return result


def ablations(datasets, graphs, c):
    from search import run_search
    root = Path(c.output)
    for variant in ABLATIONS:
        cc = replace(c, ablation='full' if variant == 'no_ga' else variant, output=str(root/variant))
        data = raw_datasets(datasets, cc) if variant == 'raw' else datasets
        # GA消融必须真正更换外层搜索；其他结构消融的配置和搜索策略明确保留。
        method = 'grid' if variant == 'no_ga' else c.search_method
        run_search(data, graphs, replace(cc, search_method=method), refit=True)


def robustness(datasets, graphs, c):
    from training import load_checkpoint, evaluate
    if not c.checkpoint:
        raise ValueError('--checkpoint required')
    root = Path(c.output); root.mkdir(parents=True, exist_ok=True)
    d, g = datasets[c.target], graphs[c.target]
    model = load_checkpoint(c.checkpoint, c, d)
    adjacency = g.static.maximum(g.static.T)
    degree = np.asarray(adjacency.sum(1)).ravel()
    n = len(degree); rng = np.random.default_rng(c.seed)
    hub = np.argsort(degree)[-max(1, int(np.ceil(0.1*n))):]
    random_nodes = rng.choice(n, max(1, int(np.ceil(0.2*n))), replace=False)
    center = rng.integers(n)
    # 文中未指定地理簇规模：显式采用最近20%站点，保存选中站点，便于复验。
    cluster = np.argsort(np.square(d['coords']-d['coords'][center]).sum(1))[:max(1, int(np.ceil(0.2*n)))]
    reports = {}
    for name, nodes in [('normal', None), ('hub_10pct', hub), ('random_20pct', random_nodes), ('geographic_20pct', cluster)]:
        cc = replace(c, output=str(root/name)); Path(cc.output).mkdir(parents=True, exist_ok=True)
        reports[name] = dict(failed_nodes=[] if nodes is None else [d['nodes'][i] for i in nodes],
                            result=evaluate(model, d, g, cc, 'test', save=True, failure=nodes))
    (root/'robustness.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    return reports


def sensitivity(datasets, graphs, c):
    for temperature in (0.01, 0.05, 0.10, 0.11, 0.15, 0.50, 1.00):
        repeat_runs(datasets, graphs, replace(c, temperature=temperature, output=str(Path(c.output, f'tau_{temperature}'))))
    for weight in (1e-4, 1e-3, 1e-2, 1e-1, 1.):
        repeat_runs(datasets, graphs, replace(c, lambda_ser=weight, output=str(Path(c.output, f'ser_{weight}'))))


def benchmark(datasets, graphs, c):
    import torch
    from sampling import origins
    from training import load_checkpoint, sample_inputs
    if not c.checkpoint:
        raise ValueError('--checkpoint required')
    d, g = datasets[c.target], graphs[c.target]
    model = load_checkpoint(c.checkpoint, c, d).eval()
    ids = origins(d, c, 'test')
    if len(ids) < c.batch_size:
        raise ValueError('Need enough distinct test windows for the logical inference batch')
    # 离线DBC已准备；计时完整模型前向，包含按方法论必须在线的SER。
    inputs = [sample_inputs(d, g, c, t)[:3] for t in ids[:c.batch_size]]
    def sync():
        if c.device.startswith('cuda'):
            torch.cuda.synchronize()
    records = []
    with torch.no_grad():
        for step in range(c.benchmark_warmup+c.benchmark_iterations):
            sync(); begin = time.perf_counter()
            for x, gs, active in inputs:
                model(x, gs, active)
            sync()
            if step >= c.benchmark_warmup:
                records.append((time.perf_counter()-begin)*1000)
    report = dict(logical_batch=c.batch_size, physical_execution='sequential single-sample reference',
        historical_steps=c.history, output_horizons=c.horizon,
        ms_per_logical_batch=float(np.mean(records)), amortized_ms_per_sample=float(np.mean(records)/c.batch_size),
        warmup=c.benchmark_warmup, iterations=c.benchmark_iterations,
        includes='EFT, online SER, semantic graph, spatial encoder, TCN, prediction head',
        excludes='CSV, DTW, GMM, DBC preprocessing, GA; no training-only contrastive branches',
        paper_physical_batch_comparable=False)
    Path(c.output, 'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def search_comparison(datasets, graphs, c):
    from search import run_search
    # 各法按相同候选评价数预算，不把SA一次移动与GA整个人口混为同一代。
    for method in ('ga', 'de', 'pso', 'gwo', 'sa', 'rs'):
        for seed in map(int, c.seeds.split(',')):
            cc = replace(c, search_method=method, seed=seed, output=str(Path(c.output, method, f'seed_{seed}')))
            run_search(datasets, graphs, cc, refit=False)


def filter_ablation(datasets, graphs, c):
    from graph_construction import build_graphs
    # 主预测结构/参数保持固定，只替换论文中五类滤波方案。
    for name, ablation, topology in [('raw', 'no_filter', 'none'), ('dropedge', 'no_filter', 'dropedge'),
                                     ('svd', 'no_filter', 'svd'), ('static_gft', 'static_gft', 'none'),
                                     ('eft', 'full', 'none')]:
        cc = replace(c, ablation=ablation, topology_filter=topology, output=str(Path(c.output, name)))
        repeat_runs(datasets, build_graphs(datasets, cc), cc)
