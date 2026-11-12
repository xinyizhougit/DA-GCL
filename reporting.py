from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd


def aggregate_searches(root, output):
    # 对真实search_trace汇总，既按候选评价数，也按共同覆盖的墙钟时间对齐。
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    methods = ['ga', 'de', 'pso', 'gwo', 'sa', 'rs']
    results = []
    for method in methods:
        paths = sorted((root/method).glob('seed_*/search_trace.json'))
        if not paths:
            continue
        traces = [json.loads(p.read_text()) for p in paths]
        count = min(len(t) for t in traces)
        by_count = np.array([[row['best_fitness'] for row in t[:count]] for t in traces])
        count_frame = pd.DataFrame(dict(evaluations=np.arange(1, count+1),
            mean=by_count.mean(0), std=by_count.std(0, ddof=1) if len(traces)>1 else np.full(count, np.nan)))
        count_frame.to_csv(output/f'{method}_by_evaluations.csv', index=False)
        lower = max(t[0]['seconds'] for t in traces)
        upper = min(t[-1]['seconds'] for t in traces)
        if upper >= lower:
            timeline = np.linspace(lower, upper, 100)
            values = []
            for t in traces:
                seconds = np.array([r['seconds'] for r in t])
                ids = np.searchsorted(seconds, timeline, side='right')-1
                values.append(np.array([r['best_fitness'] for r in t])[ids])
            values = np.array(values)
            pd.DataFrame(dict(seconds=timeline, mean=values.mean(0),
                std=values.std(0, ddof=1) if len(traces)>1 else np.full(100, np.nan))).to_csv(output/f'{method}_by_seconds.csv', index=False)
        results.append(dict(method=method, completed_runs=len(traces), common_evaluations=count,
                            best_per_run=[t[-1]['best_fitness'] for t in traces]))
    (output/'search_summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--output', required=True)
    args = p.parse_args(); print(aggregate_searches(args.root, args.output))
