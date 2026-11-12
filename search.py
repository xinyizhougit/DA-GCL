from pathlib import Path
from dataclasses import replace
import itertools
import json
import time
import numpy as np


def decode(vector, c):
    # 五基因与论文范围一致；学习率在对数空间交叉/变异。
    return replace(c, beta=float(0.1+0.8*vector[0]), temperature=float(0.01+0.19*vector[1]),
                   balance_prior=float(0.5+1.5*vector[2]), gamma=float(0.1+0.9*vector[3]),
                   lr=float(10**(-4+2*vector[4])))


def optimize(objective, c):
    rng = np.random.default_rng(c.seed)
    p, dim = c.population, 5
    if p < 4:
        raise ValueError('Population must be at least 4 for comparative optimizers')
    population = rng.random((p, dim)); scores = np.full(p, np.inf)
    personal, personal_scores = population.copy(), scores.copy()
    velocity = np.zeros_like(population)
    global_x, global_score = population[0].copy(), np.inf
    accepted, accepted_score = population[0].copy(), np.inf
    budget = p*c.generations
    grid = None
    if c.search_method == 'grid':
        resolution = int(np.ceil(budget**(1/dim)))
        grid = np.array(list(itertools.product(np.linspace(0, 1, resolution), repeat=dim)))
        rng.shuffle(grid)
    trace, stagnant, start = [], 0, time.perf_counter()
    for generation in range(c.generations):
        old = global_score
        if generation == 0 or c.search_method in ('rs', 'grid', 'sa'):
            candidates = rng.random((p, dim))
        elif c.search_method == 'ga':
            elite = population[np.argsort(scores)[:max(2, p//2)]]
            children = [global_x.copy()]
            while len(children) < p:
                a, b = elite[rng.integers(len(elite), size=2)]
                child = a.copy()
                if rng.random() < c.crossover:
                    mixing = rng.random(dim)
                    child = mixing*a+(1-mixing)*b
                mutation = rng.random(dim) < c.mutation
                child += mutation*rng.normal(0, 0.1, dim)
                children.append(np.clip(child, 0, 1))
            candidates = np.stack(children)
        elif c.search_method == 'de':
            children = []
            for i in range(p):
                ids = rng.choice(np.delete(np.arange(p), i), 3, replace=False)
                mutant = population[ids[0]]+0.8*(population[ids[1]]-population[ids[2]])
                cross = rng.random(dim) < 0.7; cross[rng.integers(dim)] = True
                children.append(np.clip(np.where(cross, mutant, population[i]), 0, 1))
            candidates = np.stack(children)
        elif c.search_method == 'pso':
            velocity = 0.7*velocity+1.5*rng.random((p, dim))*(personal-population)+1.5*rng.random((p, dim))*(global_x-population)
            candidates = np.clip(population+velocity, 0, 1)
        elif c.search_method == 'gwo':
            leaders = population[np.argsort(scores)[:3]]
            coefficient = 2*(1-generation/max(c.generations-1, 1))
            positions = []
            for leader in leaders:
                aa = coefficient*(2*rng.random((p, dim))-1)
                cc = 2*rng.random((p, dim))
                positions.append(leader-aa*np.abs(cc*leader-population))
            candidates = np.clip(np.mean(positions, axis=0), 0, 1)
        else:
            raise ValueError(c.search_method)
        candidate_scores = []
        for i in range(p):
            count = len(trace)
            vector = candidates[i]
            if c.search_method == 'grid':
                vector = grid[count]
            if c.search_method == 'sa':
                temperature = max(0.01, 1-count/max(budget-1, 1))
                vector = np.clip(accepted+rng.normal(0, 0.2*temperature, dim), 0, 1)
            value = float(objective(vector.copy(), count))
            if not np.isfinite(value):
                raise ValueError('Nonfinite validation fitness; candidate failures are not silently ignored')
            if c.search_method == 'sa':
                difference = value-accepted_score
                if difference <= 0 or rng.random() < np.exp(-difference/max(temperature, 1e-8)):
                    accepted, accepted_score = vector.copy(), value
            candidates[i] = vector
            candidate_scores.append(value)
            if value < global_score:
                global_x, global_score = vector.copy(), value
            trace.append(dict(evaluations=count+1, generation=generation+1, fitness=value,
                              best_fitness=global_score, genome=vector.tolist(), seconds=time.perf_counter()-start))
        candidate_scores = np.array(candidate_scores)
        if c.search_method == 'de' and generation > 0:
            improved = candidate_scores < scores
            population[improved], scores[improved] = candidates[improved], candidate_scores[improved]
        else:
            population, scores = candidates, candidate_scores
        improved = scores < personal_scores
        personal[improved], personal_scores[improved] = population[improved], scores[improved]
        stagnant = 0 if global_score < old else stagnant+1
        if stagnant >= c.search_patience:
            break
    return global_x, global_score, trace


def run_search(datasets, graphs, c, refit=True):
    from training import train
    root = Path(c.output); root.mkdir(parents=True, exist_ok=True)
    def objective(vector, number):
        candidate = decode(vector, c)
        candidate = replace(candidate, output=str(root/f'candidate_{number:04d}'), epochs=c.search_epochs)
        return train(datasets, graphs, candidate, test=False)
    vector, score, trace = optimize(objective, c)
    (root/'search_trace.json').write_text(json.dumps(trace, indent=2), encoding='utf-8')
    winner = decode(vector, c)
    (root/'winner.json').write_text(json.dumps(dict(beta=winner.beta, temperature=winner.temperature,
        balance_prior=winner.balance_prior, gamma=winner.gamma, lr=winner.lr, validation_mae=score,
        evaluations=len(trace)), indent=2), encoding='utf-8')
    if refit:
        from experiments import repeat_runs
        repeat_runs(datasets, graphs, replace(winner, output=str(root/'winner_repeats')))
    return vector, score, trace
