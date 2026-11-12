from dataclasses import replace
from pathlib import Path
from config import parse, REGIONS


def main():
    c, action = parse()
    from data_processing import prepare
    data = prepare(c)
    if action == 'prepare':
        print(f'Prepared regions: {list(data)}; audit: {Path(c.cache, "data_audit.json")}')
        return
    from graph_construction import build_graphs
    graphs = build_graphs(data, c)
    from training import train, load_checkpoint, evaluate
    from search import run_search
    from experiments import repeat_runs, ablations, robustness, sensitivity, benchmark, search_comparison, filter_ablation
    dispatch = dict(train=train, repeat=repeat_runs, search=run_search, ablate=ablations,
                    robustness=robustness, sensitivity=sensitivity, benchmark=benchmark,
                    **{'search-compare': search_comparison, 'filter-ablate': filter_ablation})
    targets = [r for r in REGIONS if r in data] if c.target == 'ALL' else [c.target]
    for target in targets:
        if target not in data:
            raise ValueError(f'Target region missing: {target}')
        cc = replace(c, target=target, output=str(Path(c.output, target)) if c.target == 'ALL' else c.output)
        Path(cc.output).mkdir(parents=True, exist_ok=True)
        if action == 'predict':
            if not cc.checkpoint:
                raise ValueError('--checkpoint required')
            model = load_checkpoint(cc.checkpoint, cc, data[target])
            evaluate(model, data[target], graphs[target], cc, 'test', save=True)
        else:
            dispatch[action](data, graphs, cc)


if __name__ == '__main__':
    main()
