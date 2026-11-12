import numpy as np


def origins(data, c, split):
    a, b = data['split']
    lo, hi = {'train': (0, a), 'val': (a, b), 'test': (b, len(data['times']))}[split]
    valid = []
    for t in range(max(c.history-1, lo-1), hi-c.horizon):
        if t+1 < lo:
            continue
        sequence = data['times'][t-c.history+1:t+c.horizon+1].asi8
        # 排除24:00到次日06:00的运营间断，不能把跨夜拼接当连续15分钟。
        if not np.all(np.diff(sequence) == 15*60*10**9):
            continue
        if data['observed'][t+1:t+c.horizon+1].any():
            valid.append(t)
    if not valid:
        raise ValueError(f'No valid continuous windows in {split}')
    return np.array(valid, dtype=np.int64)


def batch(data, c, origin):
    x = data['x'][origin-c.history+1:origin+1]
    y = data['raw'][origin+1:origin+c.horizon+1].T.astype('float32')
    mask = data['observed'][origin+1:origin+c.horizon+1].T & np.isfinite(y)
    return x, np.nan_to_num(y), mask


def training_similarity(data):
    flow = data['x'][:data['split'][0], :, 0]
    std = flow.std(0)
    z = (flow-flow.mean(0))/np.maximum(std, 1e-8)
    return np.clip(z.T @ z/max(len(z), 1), -1, 1).astype('float32')
